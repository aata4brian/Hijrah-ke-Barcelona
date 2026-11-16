"""Run the verified experiment queue on AATA, with logs and a single-run lock."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from .evaluate import atomic_json


def completed_run(output, folds, epochs, steps):
    paths = [output / name for name in ('queue_status.json', 'result.json', 'manifest.json')]
    if not all(path.is_file() for path in paths):
        return False
    queue, result, manifest = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if queue.get('stage') != 'complete' or result.get('complete') is not True:
        return False
    if (manifest['folds'], manifest['epochs'], manifest['steps']) != (folds, epochs, steps):
        raise ValueError('Completed output has another configuration; use a new --output directory')
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('outputs/riset_v44_full'))
    parser.add_argument('--folds',type=int,nargs='+',default=[0,2,1,3,4])
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--rerun-complete',action='store_true',
                        help='Explicitly allow regenerating predictions in a completed queue')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if not args.rerun_complete and completed_run(output,args.folds,args.epochs,args.steps):
        print('Queue already complete. Auditing saved artifacts; predictions will not be overwritten.',flush=True)
        subprocess.run([sys.executable,'-m','riset_v4.audit','--output',str(output)],check=True)
        from .registry import sync_registry
        sync_registry(strict=True)
        return
    lock=(output/'queue.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    if os.name=='nt':
        import msvcrt
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('This experiment queue is already running')
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',LOKY_MAX_CPU_COUNT='4',PYTHONUNBUFFERED='1')
    def status(stage,**kw):
        state=dict(time=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stage=stage,**kw)
        atomic_json(output/'queue_status.json',state);print(json.dumps(state),flush=True)
    def command(stage,arguments,optional=False):
        logfile=output/(stage+'.log');status(stage,command=arguments,log=str(logfile))
        with logfile.open('a',encoding='utf-8') as log:
            log.write('\nRUN '+datetime.now(timezone.utc).isoformat()+'\n');log.flush()
            process=subprocess.Popen([sys.executable,'-u',*arguments],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            status(stage,child_pid=process.pid,log=str(logfile));code=process.wait()
        if code:
            status(stage+'_failed',exit_code=code,log=str(logfile))
            if not optional:raise RuntimeError(f'{stage} failed: exit {code}; see {logfile}')
        return code==0
    try:
        command('tests',['-m','unittest','riset_v4.test_v4','riset_v4.test_v44',
                         'riset_v4.test_artifact_checks','riset_v4.test_audit','riset_v4.test_queue','-v'])
        parent=root.parent
        command('benchmark',['-m','riset_v4.run','--cache',str(parent/'04_Eksperimen/overnight_20261109/cache_idsse_publik'),
            str(parent/'04_Eksperimen/lanjutan_20261109/cache_public'),'--output',str(output),
            '--folds',*[str(f) for f in args.folds],'--epochs',str(args.epochs),'--steps',str(args.steps),'--device','cuda'])
        command('report',['-m','riset_v4.report','--output',str(output)])
        local=json.loads((parent/'08_Paket_Tim/.mtah-local.json').read_text(encoding='utf-8-sig'))
        dataset=local['data_dir'];prediction=output/'target_candidates';prediction.mkdir(exist_ok=True)
        # Save explicit candidates. Choice is documented from external-match evidence.
        for fold in args.folds:
            for family in ('tree','tcn','ensemble'):
                model=output/f'fold_{fold}'/(family+'_candidate.joblib')
                if model.exists():
                    command(f'inference_{fold}_{family}',['-m','riset_v4.predict','--model',str(model),
                        '--data-dir',dataset,'--output',str(prediction/f'fold{fold}_{family}.csv')])
        video=root/'video_cache/BAR-ATH_tactical_buffered.mp4'
        metadata=video.with_suffix('.mp4.copy.json')
        if not video.exists() and metadata.exists():
            info=json.loads(metadata.read_text(encoding='utf-8'))
            command('cache_video',['-m','riset_v4.cache_video','--source',info['source_identity']['source'],
                '--destination',str(video)],optional=True)
        if video.exists():
            # Fixed fold/model for prioritisation; never pick using target human labels.
            probability=prediction/f'fold{args.folds[0]}_ensemble.probabilities.npz'
            params=['-m','riset_v4.video','--data-dir',dataset,'--video',str(video),'--output',str(output/'video'),
                '--top','80','--selftest','30','--max-gap-frames','250']
            if probability.exists():params+=['--probabilities',str(probability)]
            ok=command('video',params,optional=True)
            if ok and (output/'video/ball_recovery.npz').exists():
                model=output/f'fold_{args.folds[0]}/ensemble_candidate.joblib'
                command('inference_video',['-m','riset_v4.predict','--model',str(model),'--data-dir',dataset,
                    '--recovery',str(output/'video/ball_recovery.npz'),'--output',str(prediction/'ensemble_with_video.csv')])
        else:
            status('video_blocked',reason='Local video copy is incomplete; benchmark and predictions are saved')
        command('report_final',['-m','riset_v4.report','--output',str(output)])
        result_path=output/'video/result.json'
        video_complete=bool(result_path.exists() and json.loads(result_path.read_text(encoding='utf-8')).get('complete'))
        status('complete' if video_complete else 'complete_with_video_limit',benchmark_complete=True,video_complete=video_complete)
    except Exception as error:
        # Preserve any partial evaluation scores even when a later queue stage fails.
        from .registry import sync_registry
        sync_registry()
        status('failed',error=str(error));raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__=='__main__':main()
