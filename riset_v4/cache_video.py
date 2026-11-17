"""Salinan video bertahap yang dapat dilanjutkan tanpa menahan seluruh file di RAM."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--retries',type=int,default=12)
    parser.add_argument('--retry-delay',type=float,default=5)
    args=parser.parse_args()
    if args.retries<1 or not 0<=args.retry_delay<=30:parser.error('Invalid retry budget')
    src,dst=args.source,args.destination
    dst.parent.mkdir(parents=True,exist_ok=True)
    info=src.stat();identity={'source':str(src),'bytes':info.st_size,'mtime_ns':info.st_mtime_ns}
    part=dst.with_suffix(dst.suffix+'.part');meta=dst.with_suffix(dst.suffix+'.copy.json')
    if dst.exists():
        if dst.stat().st_size==info.st_size and meta.exists() and json.loads(meta.read_text()).get('complete'):
            print('CACHE_ALREADY_COMPLETE',flush=True);return
        raise ValueError('Destination exists without a verified completed copy; choose a new destination')
    if part.exists():
        saved=json.loads(meta.read_text())
        if saved['source_identity']!=identity:raise ValueError('Source changed; use a fresh destination')
    state={'source_identity':identity,'complete':False}
    meta.write_text(json.dumps(state,indent=2),encoding='utf-8')
    offset=part.stat().st_size if part.exists() else 0
    if offset>info.st_size:raise ValueError('Partial file is longer than source')
    attempts=0;last_report=0
    while offset<info.st_size:
        try:
            with src.open('rb',buffering=0) as incoming,part.open('ab',buffering=1024*1024) as output:
                incoming.seek(offset)
                while offset<info.st_size:
                    block=incoming.read(min(1024*1024,info.st_size-offset))
                    if not block:raise OSError('Unexpected source EOF')
                    output.write(block);offset+=len(block)
                    if time.monotonic()-last_report>15:
                        output.flush();print(json.dumps({'copied_bytes':offset,'total_bytes':info.st_size,'percent':round(100*offset/info.st_size,2)}),flush=True)
                        last_report=time.monotonic()
        except OSError as error:
            attempts+=1;offset=part.stat().st_size if part.exists() else 0
            print(json.dumps({'retry':attempts,'offset':offset,'error':str(error)}),flush=True)
            if attempts>=args.retries:raise
            time.sleep(min(30,args.retry_delay*attempts))
    if src.stat().st_size!=info.st_size or src.stat().st_mtime_ns!=info.st_mtime_ns:
        raise ValueError('Source changed during transfer')
    digest=hashlib.sha256()
    with part.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    part.replace(dst);state.update(complete=True,sha256=digest.hexdigest())
    meta.write_text(json.dumps(state,indent=2),encoding='utf-8')
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()

