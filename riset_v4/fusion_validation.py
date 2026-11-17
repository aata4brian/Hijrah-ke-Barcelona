"""Fail-closed contracts for isolated fusion inference artifacts."""
import json
from pathlib import Path
import numpy as np
from .cue_features import CUE_NAMES
from .data import fingerprint
from .fusion_features import SCHEMA


def expected_feature_names():
    from night_tracking import Half
    from .features import pair
    from .fusion_features import extra_pair
    data = Half(ball=np.ones((100,2),np.float32),
        players=tuple(np.ones((100,11,2),np.float32) for _ in (0,1)),
        pitch=np.asarray([105,68],np.float32))
    return pair(data)[1]+extra_pair(data)[1]+CUE_NAMES


def source_model_digest(manifest, filename):
    matches = [digest for name, digest in manifest['source_models_sha256'].items()
               if name.replace('\\', '/').rsplit('/', 1)[-1] == filename]
    if len(matches) != 1:
        raise ValueError('Missing/ambiguous source model hash: '+filename)
    return matches[0]


def validate_bundle(folder):
    import games
    folder = Path(folder)
    m = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    target = games.GAMES[games.GAME]
    if (m.get('complete') is not True or m.get('target_labels_used') is not False
            or m.get('smart_tagging_read') is not False or m.get('schema') != SCHEMA
            or m.get('frames') != target['n_video_frames'] or m.get('fps') != target['fps']):
        raise ValueError('Invalid/unfinished label-free target feature bundle')
    names = m.get('feature_names', [])
    # Some old/new display names coincide; positional schema is authoritative.
    if names != expected_feature_names():
        raise ValueError('Invalid feature names/order')
    expected = [(h, first, last+1) for h, first, last in games.half_video_blocks(games.GAME)]
    actual = [(h['half'], h['first_frame'], h['end_frame_exclusive']) for h in m['halves']]
    if actual != expected:
        raise ValueError('Missing, duplicated, reordered or misaligned target halves')
    for half in m['halves']:
        h, first, end = half['half'], half['first_frame'], half['end_frame_exclusive']
        if not 0 <= first < end <= m['frames']:
            raise ValueError('Half lies outside target timeline')
        files = half['files']
        if [f['name'] for f in files] != [f'half_{h}_side_{s}.npy' for s in (0, 1)]:
            raise ValueError('Feature sides missing, duplicated or reordered')
        for f in files:
            path = folder/f['name']
            if fingerprint(path) != f['sha256']:
                raise ValueError('Feature bundle missing/modified')
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            try:
                if (a.dtype != np.float16 or a.shape != (end-first, 290)
                        or list(a.shape) != f['shape'] or np.isinf(a).any()):
                    raise ValueError('Feature shape/dtype/value contract differs')
            finally:
                a._mmap.close()
    return m
