"""Fetch a pinned public pose model, validate hash, never pip-install anything."""
import argparse
from pathlib import Path
import urllib.request

from deadball.common import digest, save_json
from deadball.pose import MODEL_SHA256, MODEL_URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'models/yolov8n-pose.onnx')
    args = parser.parse_args()
    if args.out.exists():
        if digest(args.out) != MODEL_SHA256:
            raise ValueError('Existing model has another hash; move it explicitly before retrying')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        partial = args.out.with_suffix('.download')
        with partial.open('xb') as handle, urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            size = 0
            while block := response.read(1024*1024):
                size += len(block)
                if size > 16*1024*1024:
                    raise ValueError('Model download exceeds expected budget')
                handle.write(block)
        if digest(partial) != MODEL_SHA256:
            raise ValueError('Downloaded model hash mismatch; partial retained for inspection')
        partial.replace(args.out)
    save_json(args.out.with_suffix('.source.json'), dict(url=MODEL_URL,sha256=MODEL_SHA256,
        license='AGPL-3.0',source='Xenova/yolov8n-pose ONNX export of Ultralytics YOLOv8n-pose',
        note='Review upstream license obligations before distributing or deploying.'))
    print('MODEL_HASH_OK',MODEL_SHA256)


if __name__ == '__main__':
    main()
