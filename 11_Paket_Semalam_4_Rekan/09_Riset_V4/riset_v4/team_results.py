"""Export/import teammate results without caches, credentials, or unpickling."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
from .data import fingerprint
from .evaluate import atomic_json
from .experiment_io import output_lock, verified_complete
from .registry import sync_registry


def export_results(project, destination, member, session):
    project,destination=Path(project),Path(destination)
    if member not in range(1,5) or not re.fullmatch(r'[A-Za-z0-9_-]{1,60}',session):
        raise ValueError('Invalid member/session')
    output=project/'outputs'; prefix=f'rekan_{member}_{session}_fold_'
    folders=sorted(p for p in output.glob(prefix+'*') if p.is_dir())
    if not folders: raise ValueError('No experiment outputs yet; do not send an empty result archive')
    queue=output/f'queue_rekan_{member}_{session}'
    with output_lock(output,'team_computer.lock'):
        files=[]; runs=[]
        for folder in folders:
            if not re.fullmatch(re.escape(prefix)+r'[0-4]',folder.name): raise ValueError('Unexpected run folder')
            complete=verified_complete(folder,30 if member==1 else 60)
            runs.append(dict(name=folder.name,complete=complete))
            # Checkpoints retained locally for resume; deployment candidates travel.
            for file in folder.iterdir():
                if file.is_file() and (file.suffix in ('.json','.csv') or file.name in
                        ('candidate.joblib','variant_candidate.joblib','ensemble_candidate.joblib')):
                    files.append(file)
        if queue.is_dir(): files.extend(p for p in queue.iterdir() if p.suffix in ('.json','.log'))
        audit=output/'audit_publik_beku'
        if member==1 and audit.is_dir(): files.extend(p for p in audit.iterdir() if p.suffix in ('.json','.csv'))
        hashes={p.relative_to(output).as_posix():fingerprint(p) for p in files}
        index=dict(format='team_results_v1',member=member,session=session,runs=runs,sha256=hashes,
            created_utc=datetime.now(timezone.utc).isoformat(),
            note='Partial runs allowed, never silently marked complete. No target GT or tracking included.')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists(): raise FileExistsError('Choose a new archive name; not overwriting results')
        temporary=destination.with_suffix('.tmp')
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for file in files: archive.write(file,file.relative_to(output).as_posix())
            archive.writestr('transfer_manifest.json',json.dumps(index,indent=2,allow_nan=False))
        temporary.replace(destination)
    return index


def import_results(archive_path, project):
    project=Path(project); output=project/'outputs'; output.mkdir(parents=True,exist_ok=True)
    with output_lock(output,'team_import.lock'), tempfile.TemporaryDirectory(prefix='team_import_',dir=output) as temp:
        stage=Path(temp)
        with zipfile.ZipFile(archive_path) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)): raise ValueError('Duplicate archive entries')
            if sum(i.file_size for i in archive.infolist())>2*1024**3: raise ValueError('Archive exceeds 2 GiB safety limit')
            index=json.loads(archive.read('transfer_manifest.json'))
            if index.get('format')!='team_results_v1': raise ValueError('Unsupported transfer format')
            member=index['member']; session=index['session']
            if member not in range(1,5) or not re.fullmatch(r'[A-Za-z0-9_-]{1,60}',session): raise ValueError('Bad transfer identity')
            prefix=f'rekan_{member}_{session}_fold_'
            allowed={f'queue_rekan_{member}_{session}'}
            if member==1: allowed.add('audit_publik_beku')
            runs=index['runs']
            if not runs or len({r['name'] for r in runs}) != len(runs):
                raise ValueError('Missing or duplicate transfer runs')
            for run in runs:
                if not re.fullmatch(re.escape(prefix)+r'[0-4]',run['name']): raise ValueError('Unexpected run identity')
                if type(run['complete']) is not bool: raise ValueError('Invalid completion flag')
                allowed.add(run['name'])
            hashes=index['sha256']
            if set(names)!=set(hashes)|{'transfer_manifest.json'}: raise ValueError('Transfer file list mismatch')
            for name,digest in hashes.items():
                parts=name.split('/')
                if (len(parts)!=2 or parts[0] not in allowed or any(c in name for c in ('\\',':'))
                        or '..' in parts or not re.fullmatch(r'[A-Za-z0-9_-]+\.(json|csv|log|joblib)',parts[-1])):
                    raise ValueError('Unsafe archive path')
                path=stage.joinpath(*parts); path.parent.mkdir(exist_ok=True)
                path.write_bytes(archive.read(name))
                if fingerprint(path)!=digest: raise ValueError('Transfer hash mismatch: '+name)
            for run in runs:
                folder=stage/run['name']
                if run['complete'] != verified_complete(folder,30 if member==1 else 60):
                    raise ValueError('Transferred completion status is inconsistent')
                manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
                fold=int(run['name'][-1])
                if manifest.get('folds') != [fold]: raise ValueError('Transferred manifest fold mismatch')
                if member != 1 and manifest.get('track') != {2:'tcn',3:'capacity',4:'augment'}[member]:
                    raise ValueError('Transferred manifest track mismatch')
                detail=folder/'performa_detail.csv'
                if run['complete'] and not detail.is_file(): raise ValueError('Complete run lacks detail CSV')
                if detail.exists():
                    from .data import DEV, SCENARIOS, split
                    from .experiment_io import validate_rows
                    from .team_training import methods_for
                    fold=int(run['name'][-1]); methods=(['frozen_v44_ensemble','calibration_selected_decoder']
                        if member==1 else methods_for({2:'tcn',3:'capacity',4:'augment'}[member]))
                    with detail.open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))
                    seen=set()
                    for row in rows:
                        row.update(fold=int(row['fold']),seed=int(row['seed']),
                                   spellF1=float(row['spellF1']),macroF1=float(row['macroF1']))
                        if row['match'] not in split(fold)[2] or row['scenario'] not in SCENARIOS or row['method'] not in methods:
                            raise ValueError('Unexpected evaluation key')
                        key=(row['match'],row['scenario'],row['method'])
                        if key in seen: raise ValueError('Duplicate evaluation key')
                        seen.add(key)
                        validate_rows([row],[row['method']],fold,row['match'],row['scenario'],432)
                        if row['split']!=('development' if row['match'] in DEV else 'known_retest'): raise ValueError('Bad split')
                    expected={(m,s,method) for m in split(fold)[2] for s in SCENARIOS for method in methods}
                    if run['complete'] and seen != expected: raise ValueError('Complete evaluation matrix has missing keys')
            # Refuse changed existing results BEFORE copying any file.
            for name,digest in hashes.items():
                target=output/name
                if target.is_symlink() or target.parent.is_symlink(): raise ValueError('Refusing symlink destination')
                if target.exists() and fingerprint(target)!=digest: raise FileExistsError('Existing result differs: '+name)
            for name in hashes:
                target=output/name
                if not target.exists(): target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(stage/name,target)
    sync_registry(project,strict=True)
    return dict(imported_files=len(hashes),runs=index['runs'],member=member,session=session)


def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    export=commands.add_parser('export'); export.add_argument('--member',type=int,required=True)
    export.add_argument('--session',required=True); export.add_argument('--destination',type=Path,required=True)
    export.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    receive=commands.add_parser('import'); receive.add_argument('--archive',type=Path,required=True)
    receive.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    result=(export_results(args.project,args.destination,args.member,args.session) if args.command=='export'
            else import_results(args.archive,args.project))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
