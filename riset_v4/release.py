"""Regenerate the frozen equal-five submission from licensed raw tracking.

No target labels, Smart Tagging, video, cue sidecar or feature cache are read.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from .artifact_checks import validate_probabilities
from .decode import Settings, decode
from .features import pair
from .fusion_bundle import local_tracking
from .fusion_model import combine_probabilities

ROOT = Path(__file__).resolve().parents[1]


def trusted_model(folder, spec):
    payload = base64.b64decode((folder / spec['file']).read_bytes())
    if len(payload) != spec['bytes'] or hashlib.sha256(payload).hexdigest() != spec['sha256']:
        raise ValueError('Frozen model integrity mismatch: ' + spec['file'])
    # joblib is executable serialization. Only load audited weights from this release.
    return joblib.load(io.BytesIO(payload))


def run(data_dir, output, method='fusion', device='auto'):
    import games
    from starter.baseline import write_submission
    from evaluation.possession_metrics import read_label_csv, validate_submission
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to replace an existing CSV')
    if not Path(data_dir).is_dir():
        raise FileNotFoundError('Licensed data directory not found')
    folder = ROOT / 'models/equal5'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if [f['fold'] for f in manifest['folds']] != list(range(5)):
        raise ValueError('Exactly the five frozen folds are required')
    if manifest.get('target_manual_labels_used') is not False or manifest['specialist_weight'] != 0:
        raise ValueError('Unexpected release contract')
    torch.set_num_threads(4)
    models = [(f, trusted_model(folder, f['deep']),
               trusted_model(folder, f['tree']) if method == 'fusion' and f['tree_weight'] else None)
              for f in manifest['folds']]
    n = games.GAMES[games.GAME]['n_video_frames']
    labels = np.full(n, 2, np.int8)
    started = time.monotonic()
    for half, first, end, data, provenance in local_tracking(data_dir):
        dense, names = pair(data)
        if len(names) != manifest['feature_columns']:
            raise ValueError('Feature schema changed')
        features = []
        for x in dense:
            finite = x[np.isfinite(x)]
            if finite.size and np.max(np.abs(finite)) > 60000:
                raise ValueError('Feature cannot be represented by frozen precision contract')
            features.append(x.astype(np.float16).astype(np.float32))
        del dense, data
        ps = np.zeros((end-first, 3), np.float32)
        qs = np.zeros((end-first, 6), np.float32)
        for spec, deep, tree in models:
            p, q = deep.raw(tuple(features), chunk=8192, device=device)
            p, q = deep.calibration.state(p), deep.calibration.edge(q)
            if tree is not None:
                tp, tq = tree.predict(tuple(features))
                p, q = combine_probabilities((p, q), (tp, tq), p, spec['tree_weight'], 0.)
            ps += p / 5
            qs += q / 5
        validate_probabilities(ps, qs, end-first)
        labels[first:end] = decode(ps, qs, Settings(**manifest['decoder']))
        print(json.dumps(dict(half=half, complete=True, elapsed_seconds=time.monotonic()-started)), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.pending')
    if temporary.exists():
        raise FileExistsError('Previous pending output exists; preserve it for inspection')
    write_submission(str(temporary), np.asarray(['H', 'A', 'D'])[labels])
    error = validate_submission(read_label_csv(str(temporary)), n)
    if error:
        raise ValueError(error)
    actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
    key = 'expected_csv_sha256' if method == 'fusion' else 'expected_deep_csv_sha256'
    if actual != manifest[key]:
        raise ValueError('CSV differs from frozen release; pending output retained, not promoted: ' + actual)
    # Exclusive creation also protects against another process producing the destination.
    with output.open('xb') as handle:
        handle.write(temporary.read_bytes())
    temporary.unlink()
    receipt = dict(complete=True, method=method, frames=n, csv_sha256=actual,
                   byte_identical=True, target_manual_labels_used=False,
                   smart_tagging_read=False, cues_used=False, **provenance)
    with output.with_suffix('.verification.json').open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', choices=('fusion', 'deep'), default='fusion')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    a = p.parse_args()
    run(a.data_dir, a.output, a.method, a.device)


if __name__ == '__main__':
    main()
