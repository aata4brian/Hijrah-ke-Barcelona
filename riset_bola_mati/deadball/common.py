"""Small output helpers; manifests never contain raw media or labels."""
import csv
import hashlib
import json
import math
from pathlib import Path


def sample_frames(start, end, fps, sample_hz):
    if not (0 <= start < end and fps > 0 and 0 < sample_hz <= fps):
        raise ValueError('Invalid sampling bounds/rate')
    first, stop = math.ceil(start*fps-1e-7), math.ceil(end*fps-1e-7)
    result = []
    for i in range(math.ceil((stop-first)*sample_hz/fps)):
        frame = first + int(i*fps/sample_hz+0.5)
        if frame < stop and (not result or frame != result[-1]):
            result.append(frame)
    if not result:
        raise ValueError('Requested interval contains no video frames')
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
