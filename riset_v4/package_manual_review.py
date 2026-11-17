"""Package two existing label-free candidates for manual review, without promotion."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from .data import fingerprint
from .evaluate import atomic_json, write_csv
from evaluation.possession_metrics import read_label_csv, validate_submission


def package(source, viewer, output):
    source, viewer, output = Path(source), Path(viewer), Path(output)
    m = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if not m.get('complete') or m.get('target_labels_used') is not False:
        raise ValueError('Candidates incomplete or used target labels')
    specifications = [('CSV_1_Fusi_V46.csv','calibration_fusion_v46_equal5.csv'),
                      ('CSV_2_Deep_V45.csv','deep_v45_equal5.csv')]
    arrays=[]
    for _,name in specifications:
        path=source/name
        if fingerprint(path)!=m['outputs_sha256'][name]: raise ValueError('Candidate hash differs')
        error=validate_submission(read_label_csv(str(path)),m['frames'])
        if error: raise ValueError(error)
        with path.open(encoding='utf-8-sig',newline='') as handle:
            rows=list(csv.DictReader(handle))
        if len(rows)!=m['frames'] or any(int(r['frame'])!=i for i,r in enumerate(rows)):
            raise ValueError('Candidate frame sequence differs')
        arrays.append(np.asarray([r['label'] for r in rows]))
    difference=arrays[0]!=arrays[1]
    edges=np.diff(np.r_[False,difference,False].astype(np.int8))
    starts,ends=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
    segments=[dict(first_frame=int(a),last_frame=int(b-1),frames=int(b-a),
                   start_s=float(a/m['fps']),end_s_exclusive=float(b/m['fps'])) for a,b in zip(starts,ends)]
    output.mkdir(parents=True,exist_ok=False)
    for dest,src in specifications: shutil.copyfile(source/src,output/dest)
    shutil.copyfile(viewer,output/'Viewer_2_CSV.html')
    if segments: write_csv(output/'BAGIAN_BERBEDA.csv',segments)
    receipt=dict(complete=True,frames=m['frames'],fps=m['fps'],
        source_manifest_sha256=fingerprint(source/'manifest.json'),
        target_labels_used=False,promoted=False,original_submission_changed=False,
        target_metrics=dict(spellF1=None,macroF1=None),ensemble_external_score_measured=False,
        differing_frames=int(difference.sum()),differing_percent=float(difference.mean()*100),
        differing_intervals=len(segments),
        labels={dest:{label:int((array==label).sum()) for label in 'HADX'}
                for (dest,_),array in zip(specifications,arrays)},
        files_sha256={p.name:fingerprint(p) for p in output.iterdir() if p.is_file()})
    atomic_json(output/'VERIFIKASI.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--viewer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(package(args.source,args.viewer,args.output),indent=2))


if __name__=='__main__': main()
