"""Sparse whistle/flag sidecars; missing evidence is not evidence of live play."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np
from .data import fingerprint

CUE_NAMES = ['audio_covered', 'whistle_recent_score', 'whistle_age_s',
             'visual_sample_valid', 'flag_associated', 'raised_candidate_score',
             'raised_persistent_score', 'lowered_candidate_score', 'lowered_persistent_score']


def load_cues(folder, frames, fps=25):
    """Point samples only for visual; never forward-fill through unobserved frames."""
    folder = Path(folder)
    m = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if m.get('status') != 'complete' or m['metadata']['fps'] != fps or m['metadata']['frames'] != frames:
        raise ValueError('Cue source incomplete or timeline differs')
    required = {'whistles.csv'} if m.get('audio_status','').startswith('processed') else set()
    if m.get('visual_status') == 'processed':
        required.update(('visual_samples.csv','flags.csv'))
    if not required <= set(m.get('output_sha256', {})):
        raise ValueError('Required cue integrity hashes are missing')
    for name, expected in m['output_sha256'].items():
        if Path(name).name != name or fingerprint(folder/name) != expected:
            raise ValueError('Cue CSV missing/modified or invalid path: '+name)
    out = np.full((frames, len(CUE_NAMES)), np.nan, np.float32)
    out[:, 0] = 0; out[:, 3] = 0
    times = np.arange(frames)/fps
    audit = m.get('audio_audit', {})
    if m.get('audio_status', '').startswith('processed'):
        # Actual decoded samples, not rounded container duration.
        a = float(audit['start_s'])
        b = min(float(audit['end_s']), a+audit['decoded_samples']/m['whistle_config']['sample_rate'])
        covered = (times >= a) & (times < b)
        out[covered, 0] = 1; out[covered, 1] = 0
        with (folder/'whistles.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                start, end, value = float(row['start_s']), float(row['end_s']), float(row['score'])
                if not (a <= start < end <= b+.001 and 0 <= value <= 1):
                    raise ValueError('Invalid whistle interval/score')
                age = times-end
                current = covered & (age >= .096) & (age <= 2)
                stronger = current & (value >= out[:, 1])
                out[stronger, 1] = value; out[stronger, 2] = age[stronger]
    if m.get('visual_status') == 'processed':
        with (folder/'visual_samples.csv').open(encoding='utf-8-sig', newline='') as handle:
            seen = set()
            for row in csv.DictReader(handle):
                frame = int(row['frame'])
                if frame in seen or not 0 <= frame < frames or abs(float(row['time_s'])-frame/fps) > 1e-6:
                    raise ValueError('Invalid/duplicate visual sample')
                seen.add(frame)
                if int(row['homography_valid']):
                    out[frame, 3] = 1; out[frame, 4:] = 0
        states = {'RAISED_CANDIDATE': 5, 'RAISED_SUPPORTED': 6,
                  'LOWERED_CANDIDATE': 7, 'LOWERED_SUPPORTED': 8}
        with (folder/'flags.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                frame = int(row['frame']); value = float(row['score'])
                if not 0 <= frame < frames or out[frame, 3] != 1 or row['state'] not in states or not 0 <= value <= 1:
                    raise ValueError('Flag not aligned with valid visual sample')
                out[frame, 4] = 1
                col = states[row['state']]
                out[frame, col] = max(out[frame, col], value)
    return out, dict(source_manifest_sha256=fingerprint(folder/'manifest.json'),
                     source_name=m['source_name'], source_bytes=m['source_bytes'],
                     feature_names=CUE_NAMES, semantics='unvalidated_cues_not_live_dead_labels',
                     visual_forward_filled=False, learned_weight=None)


def merge_cues(arrays):
    """Combine disjoint runs; reject conflicts rather than silently overwrite."""
    if not arrays:
        raise ValueError('No cue arrays')
    result = arrays[0].copy()
    for array in arrays[1:]:
        if array.shape != result.shape:
            raise ValueError('Cue timeline shape mismatch')
        for gate, cols in ((0, slice(0, 3)), (3, slice(3, 9))):
            both = (result[:, gate] == 1) & (array[:, gate] == 1)
            if not np.allclose(result[both, cols], array[both, cols], equal_nan=True):
                raise ValueError('Overlapping cue runs disagree')
            fresh = (result[:, gate] != 1) & (array[:, gate] == 1)
            result[fresh, cols] = array[fresh, cols]
    return result


def export_cues(folders, output):
    """Create a new auditable sidecar, not possession predictions or training labels."""
    folders = [Path(p) for p in folders]
    if not folders:
        raise ValueError('At least one completed cue run is required')
    manifests = [json.loads((p/'manifest.json').read_text(encoding='utf-8')) for p in folders]
    def identity(m):
        return (m['source_name'], m['source_bytes'], m['metadata']['frames'],
                m['metadata']['fps'], m['metadata']['width'], m['metadata']['height'])
    expected = identity(manifests[0])
    if any(identity(m) != expected for m in manifests):
        raise ValueError('Cue runs refer to different source videos/timelines')
    frames, fps = expected[2:4]
    if not isinstance(frames, int) or frames <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid cue timeline')
    pairs = [load_cues(p, frames, fps) for p in folders]
    features = merge_cues([p[0] for p in pairs])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'cue_features.npz').open('wb') as handle:
        np.savez_compressed(handle, features=features, feature_names=np.asarray(CUE_NAMES),
                            fps=np.asarray(fps), frame_count=np.asarray(frames))
    audit = dict(version='v46-cues-v1', complete=True, frames=frames, fps=fps,
        source_identity='name_size_metadata_only_not_full_media_hash',
        sources=[dict(folder=str(p), **info) for p, (_, info) in zip(folders, pairs)],
        audio_covered_frames=int((features[:,0] == 1).sum()),
        visual_valid_sample_frames=int((features[:,3] == 1).sum()),
        associated_flag_sample_frames=int(np.nansum(features[:,4])),
        nonzero_whistle_frames=int(np.nansum(features[:,1] > 0)),
        feature_names=CUE_NAMES, npz_sha256=fingerprint(output/'cue_features.npz'),
        learned_weight=None, labels_used=False, submission_changed=False,
        note='NaN=unavailable; zero within covered intervals=no candidate, NOT proof of live play. Visual samples are not forward-filled. No learned cue contribution has been validated.')
    (output/'manifest.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_cues(args.runs, args.output), indent=2))


if __name__ == '__main__':
    main()
