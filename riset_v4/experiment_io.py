"""Locks and integrity checks scoped to new overnight research outputs."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
from .data import fingerprint
from .evaluate import atomic_json


@contextmanager
def output_lock(root, name='experiment.lock'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    handle = (root/name).open('a+b')
    locked = False
    try:
        if handle.seek(0, 2) == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise RuntimeError('Another process owns this output: '+str(root)) from error
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def code_fingerprints(names):
    root = Path(__file__).resolve().parents[1]
    files = [root/'riset_v4'/name for name in names]
    files += sorted((root/'eksperimen'/'pipeline').glob('*.py'))
    files += [root/'evaluation'/'possession_metrics.py']
    return {p.relative_to(root).as_posix(): fingerprint(p) for p in files}


def validate_rows(rows, methods, fold, match, scenario, seed):
    if not isinstance(rows, list) or len(rows) != len(methods):
        raise ValueError('Incomplete evaluation checkpoint')
    if {r.get('method') for r in rows} != set(methods):
        raise ValueError('Unexpected or duplicate evaluation method')
    for row in rows:
        from .data import DEV
        if row.get('split') != ('development' if match in DEV else 'known_retest'):
            raise ValueError('Evaluation checkpoint split mismatch')
        if any(row.get(k) != v for k, v in dict(fold=fold, match=match, scenario=scenario, seed=seed).items()):
            raise ValueError('Evaluation checkpoint metadata mismatch')
        for metric in ('spellF1', 'macroF1'):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid evaluation score: '+metric)
    return rows


def verified_complete(root, expected_rows):
    root = Path(root)
    if not (root/'result.json').exists():
        return False
    info = json.loads((root/'result.json').read_text(encoding='utf-8'))
    if info.get('complete') is not True:
        return False
    if info.get('rows') != expected_rows or not info.get('artifacts_sha256'):
        raise ValueError('Completion record lacks expected rows or integrity hashes')
    for name, digest in info['artifacts_sha256'].items():
        path = root/name
        if path.parent != root or not path.is_file() or fingerprint(path) != digest:
            raise ValueError('Completed artifact missing or modified: '+name)
    return True


def finish(root, names, **info):
    root = Path(root)
    atomic_json(root/'result.json', dict(info, complete=True, promoted=False, target_gt_read=False,
        artifacts_sha256={name: fingerprint(root/name) for name in names}))
