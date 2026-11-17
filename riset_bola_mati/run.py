"""Run local whistle/flag detectors without changing any possession submission."""
import argparse
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import math
from pathlib import Path
import sys
import time

from deadball import __version__
from deadball.audio import WhistleConfig, detect_media, FIELDS as AUDIO_FIELDS
from deadball.common import digest, save_csv, save_json
from deadball.flags import FlagDetector, FIELDS as FLAG_FIELDS
from deadball.fusion import timeline, FIELDS as TIMELINE_FIELDS
from deadball.media import probe

ROOT = Path(__file__).resolve().parent


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['probe', 'audio', 'flags', 'all'])
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='New directory; existing results are never overwritten')
    parser.add_argument('--start', type=float, default=0, help='Seconds on tactical video timeline')
    parser.add_argument('--duration', type=float, help='Seconds; omitted means to end of file')
    parser.add_argument('--sample-hz', type=float, default=2, help='Visual samples/sec, default 2')
    parser.add_argument('--homography', type=Path, help='Matching tactical per-frame homography CSV')
    parser.add_argument('--model', type=Path, default=ROOT/'models/yolov8n-pose.onnx')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--overlay', action='store_true', help='Local time-compressed diagnostic video')
    return parser.parse_args(argv)


def execute(args):
    before = time.perf_counter()
    metadata = probe(args.video)
    if args.mode == 'probe':
        import json
        print(json.dumps(metadata, indent=2))
        return metadata
    if args.out is None:
        raise ValueError('--out is required for detection')
    if not math.isfinite(args.start) or not 0 <= args.start < metadata['duration']:
        raise ValueError('--start must be within the video')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise ValueError('--duration must be finite and positive')
    duration = metadata['duration']-args.start if args.duration is None else args.duration
    if args.start+duration > metadata['duration']+0.001:
        raise ValueError('Requested interval extends past the video; shorten --duration')
    if not math.isfinite(args.sample_hz) or not 0.5 <= args.sample_hz <= min(10,metadata['fps']):
        raise ValueError('--sample-hz must be between 0.5 and min(10, video fps)')
    if not 1 <= args.threads <= 16:
        raise ValueError('--threads must be between 1 and 16')
    needs_visual = args.mode in ('flags', 'all')
    if needs_visual and (args.homography is None or not args.homography.is_file()):
        raise ValueError('Flag mode requires the matching --homography CSV; no manual target ROI')
    if needs_visual and not args.model.is_file():
        raise ValueError('Missing model; run setup_model.py (no pip install required)')
    # Current decoder contract is verified for zero-based, aligned streams only.
    if abs(metadata['video_start']) > 0.001:
        raise ValueError('Nonzero video start time unsupported; align streams first')
    if metadata['audio_channels'] and (abs(metadata['audio_start']) > 0.001):
        raise ValueError('Nonzero audio start time unsupported; align streams first')
    args.out.mkdir(parents=True, exist_ok=False)
    config = WhistleConfig()
    manifest = dict(version=__version__, status='running', mode=args.mode,
        source_name=args.video.name, source_bytes=args.video.stat().st_size,
        source_identity='name_size_metadata_only_not_full_media_hash',
        metadata=metadata, start_s=args.start, end_s=args.start+duration,
        sample_hz=args.sample_hz, visual_sampled_not_dense=True, threads=args.threads,
        whistle_config=asdict(config), model_sha256=digest(args.model) if needs_visual else None,
        homography_sha256=digest(args.homography) if needs_visual else None,
        code_sha256={str(p.relative_to(ROOT)):digest(p)
            for p in sorted((ROOT/'deadball').glob('*.py'))+[ROOT/'run.py']},
        output_semantics='heuristic_evidence_not_calibrated_probabilities_or_possession_labels',
        target_ground_truth_read=False, submission_changed=False, raw_media_uploaded=False,
        metrics=dict(precision=None, recall=None, f1=None, spellF1=None),
        limitations=['Referee identity not established from sound',
            'Pose/color/touchline association is not a trained assistant-referee classifier',
            'Restart/live state cannot be established by absence of cues'])
    versions = {}
    for name in ['numpy','scipy','opencv-python','opencv-python-headless','opencv-contrib-python','opencv-contrib-python-headless']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest['packages'] = versions
    import cv2
    manifest['opencv_runtime_version'] = cv2.__version__
    save_json(args.out/'manifest.json', manifest)
    try:
        audio, flags, audit = [], [], []
        audio_status = 'disabled'
        audio_audit = {}
        if args.mode in ('audio','all'):
            if metadata['audio_channels'] == 0:
                audio_status = 'unavailable_no_audio_stream'
            else:
                audio_end = min(args.start+duration, metadata['audio_duration']) if metadata['audio_duration'] is not None else args.start+duration
                audio_duration = max(0, audio_end-args.start)
                if audio_duration > 0:
                    audio, audio_audit = detect_media(args.video, args.start, audio_duration, config,
                        progress=lambda i,n: print(f'Audio chunks {i}/{n}', flush=True)
                            if i == 1 or i % 10 == 0 or i == n else None)
                    audio_audit.update(start_s=args.start,end_s=audio_end,
                        uncovered_tail_seconds=max(0,args.start+duration-audio_end))
                    audio_status = 'processed_partial_coverage' if audio_end < args.start+duration-0.001 else 'processed'
                else:
                    audio_status = 'unavailable_outside_audio_stream'
        save_csv(args.out/'whistles.csv', audio, AUDIO_FIELDS)
        if needs_visual:
            from deadball.pose import PoseDetector
            from deadball.video import observe
            detector = FlagDetector(PoseDetector(args.model,args.threads))
            flags, audit = observe(args.video, metadata, args.start, args.start+duration,
                detector, args.homography, args.sample_hz,
                progress=lambda i,n: print(f'Visual samples {i}/{n}',flush=True),
                overlay=args.out/'overlay_local.mp4' if args.overlay else None)
        save_csv(args.out/'flags.csv', flags, FLAG_FIELDS)
        save_csv(args.out/'visual_samples.csv', audit, ['frame','time_s','homography_valid',
            'color_candidates','pose_tiles','associated_flags','visual_state','inference_seconds'])
        fused = timeline(audio,flags,args.start,args.start+duration,metadata['fps'],args.sample_hz)
        save_csv(args.out/'deadball_evidence.csv',fused,TIMELINE_FIELDS)
        manifest.update(status='complete', elapsed_seconds=time.perf_counter()-before,
            audio_status=audio_status, audio_audit=audio_audit, whistle_candidates=len(audio),
            visual_status='processed' if needs_visual else 'disabled', visual_samples=len(audit),
            visual_samples_with_associations=sum(r['associated_flags']>0 for r in audit),
            flag_observations=len(flags), flag_states=dict(Counter(r['state'] for r in flags)),
            fusion_states=dict(Counter(r['state'] for r in fused)),
            output_sha256={p.name:digest(p) for p in sorted(args.out.glob('*.csv'))})
        save_json(args.out/'manifest.json',manifest)
        print(f'COMPLETE {args.out} | whistles={len(audio)} | flag_observations={len(flags)} | '
            f'runtime={manifest["elapsed_seconds"]:.2f}s',flush=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc),
            elapsed_seconds=time.perf_counter()-before)
        save_json(args.out/'manifest.json',manifest)
        raise


if __name__ == '__main__':
    try:
        execute(arguments())
    except Exception as exc:
        print(f'FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
