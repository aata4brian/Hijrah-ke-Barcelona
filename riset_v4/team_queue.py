"""One independent serial experiment queue per teammate's computer."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from .data import discover, fingerprint
from .evaluate import atomic_json
from .experiment_io import output_lock, verified_complete
from .registry import sync_registry

TRACKS = {1:'decoder',2:'tcn',3:'capacity',4:'augment'}


def job_arguments(member, fold, source, cache, output, device, candidates=192, epochs=20, min_free=6.):
    common = ['--source',str(source),'--cache',*[str(p) for p in cache],
              '--output',str(output),'--fold',str(fold)]
    if member == 1:
        return ['-m','riset_v4.search_decoder',*common,'--candidates',str(candidates)]
    return ['-m','riset_v4.team_training',*common,'--track',TRACKS[member],
            '--epochs',str(epochs),'--device',device,'--min-free-gib',str(min_free)]


def launch_allowed(queue, elapsed, hours):
    return not (queue/'STOP_AFTER_JOB').exists() and elapsed < hours*3600


def run(args):
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    queue = output/f'queue_rekan_{args.member}_{args.session}'
    with output_lock(output,'team_computer.lock'), output_lock(queue):
        return _run(args,project,output,queue)


def _run(args,project,output,queue):
    started = time.monotonic(); device = args.device
    if device=='auto': device = 'cuda' if args.member==2 else 'cpu'
    if args.member==2 and device!='cuda' and not args.allow_cpu_tcn:
        raise ValueError('Rekan 2 requires CUDA; explicit --allow-cpu-tcn is needed for slower CPU training')
    if device=='cuda':
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable. Assign this task to a CUDA computer.')
        x = torch.ones((16,16),device='cuda')
        if float((x@x).sum())!=4096.: raise RuntimeError('CUDA arithmetic self-test failed')
        del x; torch.cuda.empty_cache()
    cache = discover(args.cache)
    manifest = dict(member=args.member,track=TRACKS[args.member],session=args.session,folds=args.folds,
        source_manifest_sha256=fingerprint(args.source/'manifest.json'),
        cache_sha256={m:fingerprint(p) for m,p in cache.items()},device=device,
        candidates=args.candidates,epochs=args.epochs,queue_code_sha256=fingerprint(Path(__file__)))
    file = queue/'queue_manifest.json'
    if file.exists() and json.loads(file.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Queue configuration changed: choose another --session, never overwrite old results')
    if not file.exists(): atomic_json(file,manifest)
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='4',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1')
    if device=='cpu': env['CUDA_VISIBLE_DEVICES']=''
    def status(stage,**details):
        info = dict(stage=stage,member=args.member,pid=os.getpid(),
            time=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,**details)
        atomic_json(queue/'queue_status.json',info); print(json.dumps(info),flush=True)
    completed=[]
    if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        status('preflight_ok',device=device,free_gib=shutil.disk_usage(output).free/1024**3)
        for fold in args.folds:
            root = output/f'rekan_{args.member}_{args.session}_fold_{fold}'
            expected = 30 if args.member==1 else 60
            already_complete = verified_complete(root,expected)
            # Even complete runs re-enter the child: it compares current source,
            # code, package and input hashes BEFORE accepting its saved result.
            if not already_complete and not launch_allowed(queue,time.monotonic()-started,args.hours):
                status('paused',completed_folds=completed,reason='STOP_AFTER_JOB or launch budget reached'); return
            if not already_complete and shutil.disk_usage(output).free < args.min_free_gib*1024**3:
                status('paused_disk',completed_folds=completed,required_free_gib=args.min_free_gib); return
            arguments=job_arguments(args.member,fold,args.source,args.cache,root,device,
                                    args.candidates,args.epochs,args.min_free_gib)
            logfile=queue/f'fold_{fold}.log'
            with logfile.open('a',encoding='utf-8') as handle:
                print('RUN',datetime.now(timezone.utc).isoformat(),file=handle,flush=True)
                process=subprocess.Popen([sys.executable,'-u',*arguments],cwd=project,env=env,
                    stdout=handle,stderr=subprocess.STDOUT)
                status('running',fold=fold,child_pid=process.pid,log=str(logfile))
                code=process.wait()
            sync_registry()
            if code: raise RuntimeError(f'Fold {fold} exited {code}; inspect {logfile}; rerun same launcher to resume')
            if not verified_complete(root,expected): raise ValueError('Child exited without verified complete matrix')
            completed.append(fold); status('fold_complete',fold=fold,completed_folds=completed)
        sync_registry(strict=True); status('complete',completed_folds=completed)
    except Exception as error:
        status('failed',error=str(error),completed_folds=completed); raise
    finally:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--member',type=int,choices=range(1,5),required=True)
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--cache',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    parser.add_argument('--session',default='semalam_20261116')
    parser.add_argument('--folds',type=int,nargs='+',default=[0,2,1,3,4])
    parser.add_argument('--hours',type=float,default=8.); parser.add_argument('--min-free-gib',type=float,default=6.)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--allow-cpu-tcn',action='store_true')
    parser.add_argument('--candidates',type=int,default=192); parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}',args.session): parser.error('Invalid session slug')
    if len(args.folds)!=len(set(args.folds)) or any(f not in range(5) for f in args.folds): parser.error('Distinct folds 0..4 required')
    if not math.isfinite(args.hours) or args.hours<=0 or not math.isfinite(args.min_free_gib) or args.min_free_gib<0: parser.error('Invalid hours or disk limit')
    run(args)


if __name__=='__main__': main()
