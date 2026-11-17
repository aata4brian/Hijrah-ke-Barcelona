"""Audit saved research artifacts locally, without loading any target ground truth."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from games import GAME, GAMES, half_video_blocks
from evaluation import possession_metrics as metrics
from .artifact_checks import load_recovery, validate_probabilities
from .data import DEV, fingerprint, split
from .decode import Settings, decode
from .evaluate import aggregate, atomic_json


METHODS = ('legacy_refit', 'switch_refit', 'relative_25fps', 'episode_weighted',
           'boundary_raw', 'boundary_calibrated', 'tcn_dual', 'ensemble')


def validate_matrix(rows, manifest):
    methods = METHODS if manifest['epochs'] else METHODS[:-2]
    expected = {(fold, match, scenario, method)
                for fold in manifest['folds'] for match in split(fold)[2]
                for scenario in manifest['test_scenarios'] for method in methods}
    found = set()
    for row in rows:
        key = (int(row['fold']), row['match'], row['scenario'], row['method'])
        if key in found:
            raise ValueError(f'Duplicate benchmark row: {key}')
        found.add(key)
        required_split = 'development' if row['match'] in DEV else 'known_retest'
        if row['split'] != required_split or int(row['seed']) != manifest['seed'] + 101:
            raise ValueError(f'Incorrect split or evaluation seed: {key}')
        for name in ('spellF1', 'macroF1'):
            if not np.isfinite(float(row[name])) or not 0 <= float(row[name]) <= 1:
                raise ValueError(f'Invalid metric: {key}, {name}')
    if found != expected:
        raise ValueError(f'Incomplete benchmark matrix: {len(expected-found)} missing, '
                         f'{len(found-expected)} unexpected rows')
    return len(expected)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def audit(root):
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    result = read_json(root / 'result.json')
    if result.get('complete') is not True:
        raise ValueError('Benchmark has not completed')
    with (root / 'performa_detail.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = validate_matrix(rows, manifest)
    for row in rows:
        for key in ('spellF1', 'macroF1', 'predicted_spells', 'reference_spells'):
            row[key] = float(row[key])
    summary = aggregate(rows)
    saved = {item['method']: item for item in result['summary']}
    if result['rows'] != expected_rows or set(saved) != {r['method'] for r in summary}:
        raise ValueError('Result summary does not match the benchmark matrix')
    for row in summary:
        for key in ('matches', 'rows', 'spellF1', 'macroF1', 'predicted_spells', 'reference_spells'):
            if not np.isclose(row[key], saved[row['method']][key], atol=1e-12, rtol=0):
                raise ValueError(f'Stale result summary: {row["method"]}, {key}')
    families = ('tree', 'tcn', 'ensemble') if manifest['epochs'] else ('tree',)
    candidates = []
    for fold in manifest['folds']:
        folder = root / f'fold_{fold}'
        saved_split = read_json(folder / 'split.json')
        train, calibration, test = split(fold)
        if saved_split != dict(train=list(train), calibration=calibration, test=list(test)):
            raise ValueError(f'Saved match split differs from protocol: fold {fold}')
        for family in families:
            candidates.append((f'fold{fold}_{family}', folder, family))
    video_result = root / 'video/result.json'
    video = read_json(video_result) if video_result.exists() else None
    if video and video.get('complete'):
        recovery = load_recovery(root / 'video/ball_recovery.npz', GAMES[GAME]['n_video_frames'])
        if int(recovery['recovered'].sum()) != video['recovered_frames']:
            raise ValueError('Video report disagrees with the recovery mask')
        candidates.append(('ensemble_with_video', root / f'fold_{manifest["folds"][0]}', 'ensemble'))
    reports = []
    n = GAMES[GAME]['n_video_frames']
    for name, folder, family in candidates:
        path = root / 'target_candidates' / (name + '.csv')
        table = metrics.read_label_csv(str(path))
        error = metrics.validate_submission(table, n)
        if error:
            raise ValueError(f'{name}: {error}')
        codes = metrics.label_codes(table)
        meta = read_json(path.with_suffix('.json'))
        model_sha = fingerprint(folder / (family + '_candidate.joblib'))
        if meta['model_sha256'] != model_sha or meta['frames'] != n:
            raise ValueError(f'{name}: model hash or row count differs')
        if meta.get('target_labels_used') is not False:
            raise ValueError(f'{name}: target-label provenance is not explicit')
        for index, label in enumerate(metrics.LABELS):
            if meta['labels'].get(label, 0) != int((codes == index).sum()):
                raise ValueError(f'{name}: metadata label counts differ')
        with np.load(path.with_suffix('.probabilities.npz'), allow_pickle=False) as arrays:
            p, q = arrays['probabilities'], arrays['boundaries']
        validate_probabilities(p, q, n)
        decoder_name = 'tcn' if family == 'tcn' else 'trees'
        settings = Settings(**read_json(folder / f'decoder_{decoder_name}.json')['chosen'])
        replay = np.full(n, metrics.D, np.int8)
        for _, first, last in half_video_blocks(GAME):
            replay[first:last+1] = decode(p[first:last+1], q[first:last+1], settings)
        if not np.array_equal(replay, codes):
            raise ValueError(f'{name}: CSV differs from its probability/decoder replay')
        csv_sha = fingerprint(path)
        probability_sha = fingerprint(path.with_suffix('.probabilities.npz'))
        for key, actual in (('csv_sha256', csv_sha), ('probabilities_sha256', probability_sha)):
            if key in meta and meta[key] != actual:
                raise ValueError(f'{name}: {key} differs')
        reports.append(dict(candidate=name, frames=n, validator='ok', decoder_replay=True,
                            csv_sha256=csv_sha, model_sha256=model_sha,
                            probabilities_sha256=probability_sha))
    if video and video.get('complete') and video['recovered_frames'] == 0:
        paths = root / 'target_candidates'
        if fingerprint(paths / 'ensemble_with_video.csv') != fingerprint(
                paths / f'fold{manifest["folds"][0]}_ensemble.csv'):
            raise ValueError('Zero recovered frames should not change the candidate CSV')
    return dict(complete=True, checked_at=datetime.now(timezone.utc).isoformat(),
                benchmark_rows=expected_rows, unique_matches=len({r['match'] for r in rows}),
                candidates_checked=len(reports), candidates=reports,
                video_recovered_frames=video['recovered_frames'] if video else None,
                target_ground_truth_read=False, promoted=False,
                limitation='Decoder replay is not end-to-end model inference or a target accuracy score.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.output / 'audit_artifacts.json'
    try:
        result = audit(args.output)
    except Exception as error:
        atomic_json(path, dict(complete=False, checked_at=datetime.now(timezone.utc).isoformat(),
                               error=str(error)))
        raise
    atomic_json(path, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'candidates'}))


if __name__ == '__main__':
    main()
