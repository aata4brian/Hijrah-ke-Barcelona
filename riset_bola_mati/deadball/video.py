"""Sequential decode, sparse inference, exact CFR frame indices."""
import time

import cv2

from .common import sample_frames
from .flags import HomographyReader, TemporalFlags


def observe(path, metadata, start, end, detector, homography=None, sample_hz=2,
            progress=None, overlay=None):
    frames = sample_frames(start, end, metadata['fps'], sample_hz)
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 20000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 20000])
    reader = None
    writer = None
    observations, sample_audit = [], []
    tracking = TemporalFlags(max_gap=max(1.6, 1.5/sample_hz))
    try:
        if not cap.isOpened():
            raise RuntimeError('Video cannot be opened')
        if abs(cap.get(cv2.CAP_PROP_FPS)-metadata['fps']) > 0.001:
            raise RuntimeError('OpenCV and ffprobe FPS disagree')
        if homography:
            reader = HomographyReader(homography)
        if overlay:
            writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*'mp4v'),
                sample_hz, (metadata['width'], metadata['height']))
            if not writer.isOpened():
                raise RuntimeError('Overlay VideoWriter cannot open output')
        cursor = frames[0]
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, cursor):
            raise RuntimeError('Video seek failed')
        for number, index in enumerate(frames, 1):
            while cursor <= index:
                if not cap.grab():
                    raise RuntimeError(f'Video decode failed at frame {cursor}')
                cursor += 1
            actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if abs(actual-index-1) > 0.5:
                raise RuntimeError(f'Frame alignment failed: requested {index}, next={actual}')
            ok, image = cap.retrieve()
            if not ok or image is None:
                raise RuntimeError(f'Cannot retrieve video frame {index}')
            if image.shape[:2] != (metadata['height'], metadata['width']):
                raise RuntimeError('Unexpected resolution change')
            matrix = reader.at(index) if reader else None
            before = time.perf_counter()
            detections, colors, tiles = detector(image, matrix)
            tracked = tracking.update(index/metadata['fps'], detections)
            for d in tracked:
                x,y,w,h = d['person']['box']
                observations.append(dict(frame=index, time_s=index/metadata['fps'],
                    track=d['track'], state=d['state'], score=d['score'], reason=d['reason'],
                    person_x=x, person_y=y, person_w=w, person_h=h,
                    flag_x=d['flag']['x'], flag_y=d['flag']['y']))
                if writer:
                    cv2.rectangle(image, (round(x),round(y)), (round(x+w),round(y+h)), (0,220,255), 2)
                    cv2.putText(image, d['state'], (round(x),max(25,round(y)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,255), 2)
            sample_audit.append(dict(frame=index, time_s=index/metadata['fps'],
                homography_valid=int(matrix is not None), color_candidates=colors,
                pose_tiles=tiles, associated_flags=len(tracked),
                visual_state='OBSERVED_CANDIDATE' if tracked else 'UNKNOWN',
                inference_seconds=time.perf_counter()-before))
            if writer:
                cv2.putText(image, f'frame {index} | {index/metadata["fps"]:.3f}s | '
                    + ('FLAG CANDIDATE' if tracked else 'UNKNOWN'), (25,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,255), 2)
                writer.write(image)
            if progress and (number == 1 or number % 20 == 0 or number == len(frames)):
                progress(number, len(frames))
    finally:
        cap.release()
        if reader:
            reader.close()
        if writer:
            writer.release()
    return observations, sample_audit
