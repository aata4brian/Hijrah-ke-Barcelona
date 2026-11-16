"""Split official tactical evaluation windows, never read video or labels."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, half_video_blocks

MEMBERS = ('Affan', 'Mosi', 'Khansa', 'Faiq', 'Brian')


def timestamp(frame, fps):
    seconds, sub = divmod(frame, fps)
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f'{hours:02}:{minute:02}:{second:02}.{sub*1000//fps:03}'


def assignments():
    blocks = half_video_blocks(GAME)
    total = sum(last-first+1 for _,first,last in blocks)
    base, extra = divmod(total,len(MEMBERS))
    rows = []
    cursor = 0
    fps = GAMES[GAME]['fps']
    for i,name in enumerate(MEMBERS):
        stop = cursor+base+(i<extra)
        block_start = 0
        segment = 0
        for half,first,last in blocks:
            block_stop = block_start+last-first+1
            a,b = max(cursor,block_start),min(stop,block_stop)
            if a<b:
                start = first+a-block_start
                end = first+b-block_start-1
                segment += 1
                rows.append(dict(anggota=name,bagian=i+1,segmen=segment,babak=half,
                    frame_awal_inklusif=start,frame_akhir_inklusif=end,
                    jumlah_frame=end-start+1,fps=fps,
                    waktu_awal_video=timestamp(start,fps),
                    waktu_akhir_eksklusif_video=timestamp(end+1,fps),
                    tujuan='inspeksi_teknis_bukan_anotasi'))
            block_start=block_stop
        cursor=stop
    return rows


def main(output):
    rows=assignments()
    output.mkdir(parents=True,exist_ok=True)
    with (output/'pembagian_frame.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary=dict(game=GAME,source='tactical',index='zero_based_inclusive',fps=GAMES[GAME]['fps'],
        video_frames=GAMES[GAME]['n_video_frames'],
        evaluation_windows=[dict(babak=h,first=a,last=b) for h,a,b in half_video_blocks(GAME)],
        total_assigned_frames=sum(r['jumlah_frame'] for r in rows),
        note='Official published windows, not private label mask; no GT/video read',
        members={m:sum(r['jumlah_frame'] for r in rows if r['anggota']==m) for m in MEMBERS})
    (output/'pembagian_frame.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    main(parser.parse_args().output)
