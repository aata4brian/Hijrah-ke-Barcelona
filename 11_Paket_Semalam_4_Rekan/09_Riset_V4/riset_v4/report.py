"""Summarise saved external evaluation without reading target labels."""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .evaluate import aggregate
from .data import SCENARIOS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output
    with (root/'performa_detail.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    numeric=('spellF1','macroF1','predicted_spells','reference_spells','recall_under2s','recall_2to5s','recall_over5s')
    for row in rows:
        for key in numeric:
            if row[key]!='':row[key]=float(row[key])
    summary=aggregate(rows);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    finished=(root/'result.json').exists()
    text=['# Hasil riset penguasaan bola V4.4','',f"Diperbarui: {datetime.now(timezone.utc).isoformat()}.",
          f"Status benchmark: {'selesai' if finished else 'masih berlangsung'}. {len(rows)} baris hasil tersimpan.",'',
          'Evaluasi memakai pertandingan publik IDSSE yang sudah dikenal dalam riset sebelumnya. Angka ini bukan skor pertandingan BAR–ATH dan bukan hasil pada holdout baru.',
          'Pelatihan, kalibrasi, dan uji dipisahkan per pertandingan di setiap fold. Tidak ada label manusia pertandingan target yang dibaca untuk memilih model.','',
          '## Perbandingan keseluruhan','',
          '| Metode | spellF1 | macroF1 | Pertandingan unik | Episode prediksi | Episode acuan |',
          '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        text.append(f"| {r['method']} | {r['spellF1']:.4f} | {r['macroF1']:.4f} | {r['matches']} | {r['predicted_spells']:.1f} | {r['reference_spells']:.1f} |")
    text+=['','Skenario dirata-rata dengan bobot yang ditetapkan sebelumnya. Pengulangan fold untuk pertandingan yang sama dirata-rata dahulu; satu pertandingan tetap dihitung satu kali.']
    for split in ('development','known_retest'):
        part=[r for r in rows if r['split']==split]
        if not part:continue
        text+=['',f'## {split}','','| Metode | spellF1 | macroF1 |','|---|---:|---:|']
        for r in aggregate(part):text.append(f"| {r['method']} | {r['spellF1']:.4f} | {r['macroF1']:.4f} |")
    text+=['','## Per pertandingan','','| Pertandingan | Metode | spellF1 | Selisih dari switch |','|---|---|---:|---:|']
    for match in sorted({r['match'] for r in rows}):
        entries=aggregate([r for r in rows if r['match']==match])
        base=next((r['spellF1'] for r in entries if r['method']=='switch_refit'),None)
        for r in entries:
            difference=f"{r['spellF1']-base:+.4f}" if base is not None else ''
            text.append(f"| {match} | {r['method']} | {r['spellF1']:.4f} | {difference} |")
    text+=['','## Recall episode pendek','','| Metode | <2 detik | 2–5 detik | ≥5 detik |','|---|---:|---:|---:|']
    for method in sorted({r['method'] for r in rows}):
        values=[]
        for key in numeric[-3:]:
            by_match=[]
            for match in sorted({r['match'] for r in rows}):
                part=[r for r in rows if r['method']==method and r['match']==match and r[key]!='']
                if part:by_match.append(np.average([r[key] for r in part],weights=[SCENARIOS[r['scenario']] for r in part]))
            values.append(f'{np.mean(by_match):.4f}' if by_match else '')
        text.append('| '+method+' | '+' | '.join(values)+' |')
    text+=['','## Konfigurasi','',f"- Fold: {manifest['folds']}; seed: {manifest['seed']}.",
          f"- Pohon status: {manifest['iterations']} iterasi. TCN: {manifest['epochs']} epoch × {manifest['steps']} langkah.",
          '- Kalibrasi sigmoid batas dan suhu status memakai pertandingan kalibrasi saja.',
          '- Decoder memakai 25 fps dan dapat membentuk batas baru di seluruh urutan.',
          '- Fitur gerak relatif dan kontinuitas kontak; pembobotan episode pendek dan sekitar batas.',
          '- Pembanding switch dilatih pada pertandingan serta skenario yang sama.',
          '- Prediksi kandidat dan pemulihan video tersimpan terpisah dari paket tim 1–15.','',
          '## Pemulihan video','']
    video=root/'video/result.json'
    if video.exists():
        v=json.loads(video.read_text(encoding='utf-8'))
        text += [f"Percobaan celah: {v['attempted_gaps']}; celah diterima: {v['accepted_gaps']}; frame pulih: {v['recovered_frames']}; galat pembacaan video: {v['video_io_errors']}.",
                 f"Uji koordinat disembunyikan: {v['selftest_accepted']}/{v['selftest_attempts']} diterima."]
    else:text+=['Belum ada hasil video lengkap di folder benchmark ini.']
    text+=['','Koordinat hasil optical flow diuji dari dua jangkar, kemiripan patch, homografi, dan batas gerak. Kesesuaian dengan tracking masukan bukan pengukuran spellF1. Jika tidak ada celah yang lolos, jumlah pemulihan dilaporkan nol.',
           '', 'Belum ada penggantian otomatis terhadap submission utama. Peringkat kandidat harus dibaca bersama hasil tiap pertandingan, kondisi celah panjang, dan recall episode pendek.']
    path=root/'LAPORAN_HASIL.md';temp=path.with_suffix('.tmp');temp.write_text('\n'.join(text)+'\n',encoding='utf-8');temp.replace(path)
    print(path)
    from .registry import sync_registry
    sync_registry()


if __name__=='__main__':main()
