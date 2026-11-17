"""Join target tracking and sparse cues without reading Smart Tagging or target GT."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import fingerprint
from .evaluate import atomic_json
from .features import pair
from .fusion_features import extra_pair, SCHEMA
from .cue_features import CUE_NAMES


def local_tracking(data_dir):
    """Tracking-only loader. Never import/load target annotation files."""
    import games
    from night_tracking import Half
    from starter.baseline import team_of_each_track
    meta, raw = games.require(['atd_meta','atd_raw'], data_dir=str(data_dir), source='tactical')
    teams = team_of_each_track(meta)
    if set(teams) != {0,1}:
        raise ValueError('Metadata must identify both teams')
    frames = games.GAMES[games.GAME]['n_video_frames']
    pitch = np.asarray(games.GAMES[games.GAME]['pitch'],np.float32)
    players = np.empty((frames,len(teams),2),np.float32)
    ball = np.empty((frames,2),np.float32)
    count = 0
    with Path(raw).open(encoding='utf-8-sig') as handle:
        for count,line in enumerate(handle,1):
            number,group,ball_group = line.split(':')
            frame = int(number)-1
            if frame != count-1 or frame >= frames:
                raise ValueError('ATD frame sequence differs from video timeline')
            xyz = np.fromstring(group.replace(';',','),sep=',',dtype=np.float32)
            if xyz.size != 3*len(teams):
                raise ValueError('ATD slot count differs from metadata')
            players[frame] = xyz.reshape(-1,3)[:,:2]*pitch
            x,y,_ = ball_group.split(',')
            ball[frame] = np.asarray([float(x),float(y)],np.float32)*pitch
    if count != frames:
        raise ValueError('ATD tracking shorter than video')
    provenance = dict(atd_meta_sha256=fingerprint(meta),atd_raw_sha256=fingerprint(raw))
    for half,first,last in games.half_video_blocks(games.GAME):
        end = last+1
        yield half,first,end,Half(ball[first:end],tuple(players[first:end,teams==s] for s in (0,1)),pitch),provenance


def export_bundle(data_dir, cues, output):
    cues, output = Path(cues), Path(output)
    m = json.loads((cues/'manifest.json').read_text())
    if not m.get('complete') or fingerprint(cues/'cue_features.npz') != m['npz_sha256']:
        raise ValueError('Cue sidecar incomplete or modified')
    from games import GAME,GAMES
    if m['frames'] != GAMES[GAME]['n_video_frames'] or m['fps'] != GAMES[GAME]['fps']:
        raise ValueError('Cue timeline differs from target tracking')
    with np.load(cues/'cue_features.npz',allow_pickle=False) as z:
        cue_values = z['features']
        if cue_values.shape != (m['frames'],len(CUE_NAMES)) or list(z['feature_names']) != CUE_NAMES:
            raise ValueError('Cue schema differs')
    output.mkdir(parents=True,exist_ok=False)
    receipt = dict(complete=False,schema=SCHEMA,frames=m['frames'],fps=m['fps'],
        cue_manifest_sha256=fingerprint(cues/'manifest.json'),halves=[],target_labels_used=False,
        smart_tagging_read=False,submission_changed=False,cue_learned_weight=None,
        offline_context=True,visual_samples_not_forward_filled=True,
        note='Feature bundle only, not a trained 290-column possession model; NaN denotes unavailable evidence.')
    atomic_json(output/'manifest.json',receipt)
    for half,first,end,data,provenance in local_tracking(data_dir):
        old,old_names = pair(data); added,added_names = extra_pair(data)
        names = old_names+added_names+CUE_NAMES
        files = []
        for side in range(2):
            x = np.column_stack([old[side],added[side],cue_values[first:end]])
            finite = x[np.isfinite(x)]
            if finite.size and np.max(np.abs(finite)) > 60000:
                raise ValueError('Feature exceeds safe compact storage range')
            name = f'half_{half}_side_{side}.npy'
            with (output/name).open('wb') as handle:
                np.save(handle,x.astype(np.float16))
            files.append(dict(name=name,sha256=fingerprint(output/name),shape=list(x.shape)))
        receipt.update(provenance)
        receipt['feature_names'] = names
        receipt['halves'].append(dict(half=half,first_frame=first,end_frame_exclusive=end,files=files))
        atomic_json(output/'manifest.json',receipt)
        print(json.dumps(dict(half=half,frames=end-first,columns=len(names),complete=True)),flush=True)
    receipt['complete'] = True
    receipt['code_sha256'] = {name:fingerprint(Path(__file__).with_name(name))
        for name in ('fusion_bundle.py','fusion_features.py','cue_features.py','features.py')}
    atomic_json(output/'manifest.json',receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--cues',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); export_bundle(args.data_dir,args.cues,args.output)


if __name__ == '__main__':
    main()
