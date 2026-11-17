"""Summarize all V4.6 tests, including negative results, without selecting test winners."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .data import SCENARIOS, fingerprint, split
from .evaluate import aggregate, atomic_json, write_csv
from .experiment_io import verified_complete, validate_rows

METHODS = ['frozen_deep_v45', 'frozen_ensemble_deep_v45', 'control_old_features',
           'expanded_tracking_features', 'calibration_selected_v46']

COMPARISONS = [('expanded_tracking_features','control_old_features'),
               ('calibration_selected_v46','frozen_deep_v45'),
               ('calibration_selected_v46','frozen_ensemble_deep_v45')]


def report(folders, output):
    if not folders:
        raise ValueError('At least one completed fold is required')
    rows = []; provenance = []; selections = []
    for folder in folders:
        folder = Path(folder)
        if not verified_complete(folder,75):
            raise ValueError('Incomplete or modified fold: '+str(folder))
        with (folder/'performa_detail.csv').open(encoding='utf-8-sig',newline='') as handle:
            records = list(csv.DictReader(handle))
        for row in records:
            row['fold'] = int(row['fold']); row['seed'] = int(row['seed'])
            for k,v in row.items():
                if k not in ('fold','seed','split','match','scenario','method') and v != '':
                    row[k] = float(v)
        rows.extend(records)
        m = json.loads((folder/'manifest.json').read_text())
        s = json.loads((folder/'calibration_selection.json').read_text())
        fold = m['folds'][0]
        train, calibration, tests = split(fold)
        if (m['split'] != dict(train=list(train),calibration=calibration,test=list(tests))
                or s['calibration_match'] != calibration or len(records) != 75):
            raise ValueError('Fold split/calibration/row count differs')
        expected = {(fold,match,scenario,method) for match in tests
                    for scenario in SCENARIOS for method in METHODS}
        if {(r['fold'],r['match'],r['scenario'],r['method']) for r in records} != expected:
            raise ValueError('Incomplete/unexpected fold evaluation grid')
        for match in tests:
            for scenario in SCENARIOS:
                validate_rows([r for r in records if r['match']==match and r['scenario']==scenario],
                              METHODS,fold,match,scenario,432)
        selections.append(dict(fold=m['folds'][0], calibration_match=s['calibration_match'],
            **s['chosen'], old_ablation=s['by_ablation']['old'], expanded_ablation=s['by_ablation']['expanded']))
        provenance.append(dict(folder=str(folder), result_sha256=fingerprint(folder/'result.json'),
                               rows=len(records), fold=m['folds'][0]))
    identities = [(r['fold'],r['match'],r['scenario'],r['method']) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError('Duplicate evaluation rows')
    methods = sorted({r['method'] for r in rows})
    pairs = []; pair_summary = []
    index = {(r['fold'],r['match'],r['scenario'],r['method']):r for r in rows}
    keys = sorted({(r['fold'],r['match'],r['scenario']) for r in rows})
    for candidate,baseline in COMPARISONS:
        subset = []
        for key in keys:
            c,b = index[(*key,candidate)],index[(*key,baseline)]
            row = dict(fold=key[0],match=key[1],scenario=key[2],candidate=candidate,baseline=baseline,
                spellF1_candidate=c['spellF1'],spellF1_baseline=b['spellF1'],
                delta_spellF1=c['spellF1']-b['spellF1'],delta_macroF1=c['macroF1']-b['macroF1'])
            pairs.append(row); subset.append(row)
        by_match = []
        for match in sorted({r['match'] for r in subset}):
            group = [r for r in subset if r['match'] == match]
            by_match.append(float(np.average([r['delta_spellF1'] for r in group],
                                               weights=[SCENARIOS[r['scenario']] for r in group])))
        pair_summary.append(dict(candidate=candidate,baseline=baseline,pairs=len(subset),
            unique_matches=len(by_match),delta_spellF1_match_mean=float(np.mean(by_match)),
            positive_pairs=sum(r['delta_spellF1'] > 1e-12 for r in subset),
            negative_pairs=sum(r['delta_spellF1'] < -1e-12 for r in subset),
            tied_pairs=sum(abs(r['delta_spellF1']) <= 1e-12 for r in subset),
            worst_pair_delta=min(r['delta_spellF1'] for r in subset)))
    summary = aggregate(rows)
    by_match = []
    for match in sorted({r['match'] for r in rows}):
        for row in aggregate([r for r in rows if r['match']==match]):
            by_match.append(dict(match=match,**row))
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'evaluation_rows.csv',rows); write_csv(output/'summary.csv',summary)
    write_csv(output/'by_match.csv',by_match); write_csv(output/'paired_deltas.csv',pairs)
    write_csv(output/'comparison_summary.csv',pair_summary)
    atomic_json(output/'calibration_choices.json',selections)
    lines = ['# Hasil fusi fitur V4.6', '',
        f"**{len(rows)} baris uji nyata**, {len(provenance)} fold, {len({r['match'] for r in rows})} laga unik.",
        'Known controlled retest, bukan holdout baru. Fold/skenario bukan sampel independen.', '',
        '| Metode | SpellF1 | MacroF1 |', '| --- | ---: | ---: |']
    for r in summary:
        lines.append(f"| {r['method']} | {r['spellF1']:.6f} | {r['macroF1']:.6f} |")
    lines += ['', '## Perbandingan berpasangan', '',
        '| Kandidat vs pembanding | Delta SpellF1 per laga | Naik/turun/seri | Terburuk |',
        '| --- | ---: | --- | ---: |']
    for r in pair_summary:
        lines.append(f"| {r['candidate']} vs {r['baseline']} | {r['delta_spellF1_match_mean']:+.6f} | "
            f"{r['positive_pairs']}/{r['negative_pairs']}/{r['tied_pairs']} | {r['worst_pair_delta']:+.6f} |")
    lines += ['', '## Pilihan sebelum test', '']
    for s in selections:
        lines.append(f"- Fold {s['fold']}, calibration {s['calibration_match']}: specialist={s['specialist']}, "
            f"tree_weight={s['tree_weight']}, specialist_weight={s['specialist_weight']}.")
    lines += ['', '## Interpretasi dan batas', '',
        '- Kontrol 203 fitur dan expanded 281 fitur memakai sampel/anggaran pohon identik.',
        '- Deep V4.5 dan ensemble deep adalah baseline beku; skor uji direproduksi hingga toleransi 1e-12.',
        '- Pemilihan suhu dan campuran hanya dari calibration. Tidak memilih fold dari test.',
        '- Jangan menyebut campuran paling baik hanya karena ditambah fitur; baca delta terhadap kontrol.',
        '- Audio/flag telah tersedia sebagai sidecar, tetapi bobotnya belum dipelajari: data eksternal sejajar tidak tersedia.',
        '- Tidak ada training deep baru, pembacaan GT target, promosi otomatis, atau perubahan submission.',
        '- Ringkasan memakai bobot skenario lalu rerata laga. Pengulangan dua retest bukan laga baru.',
        '- Precision/recall peluit/flag dan SpellF1 target tidak diukur.', '']
    (output/'HASIL_V46.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(output/'verification.json',dict(complete=True,folds=provenance,rows=len(rows),
        unique_matches=sorted({r['match'] for r in rows}),methods=methods,target_gt_read=False,promoted=False,
        artifacts_sha256={p.name:fingerprint(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(rows=len(rows),summary=summary,comparisons=pair_summary),indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); report(args.runs,args.output)


if __name__ == '__main__':
    main()
