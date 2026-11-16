"""Standard-library verification before running any transferred model."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'payload_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Missing/unsafe payload path: '+name)
        h=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
        if h.hexdigest()!=digest: raise ValueError('Payload modified: '+name)
    print('PAYLOAD_HASHES_OK',len(manifest['sha256']))


if __name__=='__main__': main()
