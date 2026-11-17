"""One immutable candidate, once: public GT only checks the frozen pipeline."""
import argparse
import json
from pathlib import Path
from .data import fingerprint
from .evaluate import atomic_json, write_csv
from .experiment_io import output_lock
from evaluation import possession_metrics as pm


def run(assets, output):
    assets,output = Path(assets),Path(output)
    with output_lock(output):
        specification = json.loads((assets/'audit_spec.json').read_text(encoding='utf-8'))
        if specification.get('purpose') != 'pipeline_check_only_not_model_selection':
            raise ValueError('Invalid frozen audit purpose')
        paths = {name:assets/name for name in ('frozen_candidate.csv','BAR-ATH_possession_gt_public.csv')}
        for name,path in paths.items():
            if fingerprint(path) != specification['sha256'][name]:
                raise ValueError('Frozen audit asset differs: '+name)
        result = output/'audit_report.json'
        if result.exists():
            report=json.loads(result.read_text(encoding='utf-8'))
            if report['specification'] != specification: raise ValueError('Audit already locked to different inputs')
            print('Frozen public audit already recorded; not evaluating another candidate',flush=True)
            return report
        gt=pm.read_label_csv(str(paths['BAR-ATH_possession_gt_public.csv']))
        pred=pm.read_label_csv(str(paths['frozen_candidate.csv']))
        for name,frame in [('reference',gt),('candidate',pred)]:
            error=pm.validate_submission(frame,specification['frames'])
            if error: raise ValueError(name+': '+error)
        y,p=pm.label_codes(gt),pm.label_codes(pred)
        if not ((y<3).any() and (y==3).any()): raise ValueError('Expected partially graded public slice')
        metrics=pm.evaluate(y,p,taus=(12,25),bootstrap_n=0)
        report=dict(complete=True,purpose=specification['purpose'],specification=specification,
            target_public_gt_read=True,target_gt_used_for_training_or_selection=False,
            graded_frames=int((y<3).sum()),public_spellF1=metrics['spell_f1_mean'],
            public_macroF1=metrics['macro_f1'],
            note='Diagnostic only. No offset search, H/A trial, candidate comparison, training or promotion.')
        atomic_json(result,report)
        write_csv(output/'diagnostik_audit.csv',[dict(method='frozen_pipeline_check',
            split='diagnostic_public_only',match='BAR-ATH_public_5min',scenario='original',
            spellF1=metrics['spell_f1_mean'],macroF1=metrics['macro_f1'],graded_frames=int((y<3).sum()))])
        print('Frozen public pipeline audit recorded; not a model ranking',flush=True)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.assets,args.output)


if __name__=='__main__': main()
