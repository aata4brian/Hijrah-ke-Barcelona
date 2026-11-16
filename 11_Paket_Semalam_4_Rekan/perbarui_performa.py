"""Persistent local performance ledger; standard library only, no training or network."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
DETAIL_NAME = 'Performa_Semua_Eksperimen.csv'
SUMMARY_NAME = 'Ringkasan_Performa_Semua.csv'
STATUS_NAME = 'Status_Pencatatan_Performa.json'
META = ['ID_Evaluasi', 'ID_Hasil', 'Versi', 'Aktif', 'Run', 'Protokol_SHA256',
        'Dataset', 'Cakupan', 'Split', 'Fold', 'Pertandingan', 'Skenario', 'Seed',
        'Metode', 'Status_Run', 'Waktu_Pertama_UTC', 'Waktu_Pembaruan_UTC',
        'Sumber', 'Sumber_SHA256', 'Bobot_Skenario_JSON', 'Gabung_Split', 'Detail_JSON']
ALIASES = {'spellF1': 'SpellF1', 'spell_f1_mean': 'SpellF1', 'macroF1': 'MacroF1',
           'macro_f1': 'MacroF1', 'deadF1': 'DeadF1', 'f1_D': 'DeadF1',
           'spells': 'Episode_Prediksi', 'pred_spells': 'Episode_Prediksi',
           'predicted_spells': 'Episode_Prediksi', 'ref_spells': 'Episode_Acuan',
           'reference_spells': 'Episode_Acuan', 'n_gt_spells': 'Episode_Acuan'}
PREFERRED_METRICS = ['SpellF1', 'MacroF1', 'DeadF1', 'Episode_Prediksi', 'Episode_Acuan']
IDENTIFIERS = {'split', 'match', 'scenario', 'input', 'seed', 'fold', 'method',
               'submission', 'rank', 'status', 'reason'}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def registry_lock(root):
    path = root / '.performa.lock'
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('Pencatatan performa sedang dipakai proses lain; coba lagi.') from error
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_csv(path):
    raw = Path(path).read_bytes()
    encoding = 'utf-8-sig'
    if raw.startswith((b'\xff\xfe\x00\x00', b'\x00\x00\xfe\xff')):
        encoding = 'utf-32'
    elif raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        encoding = 'utf-16'
    with io.StringIO(raw.decode(encoding), newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f'Header ganda: {path}')
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f'Baris CSV terpotong/rusak: {path}')
    return fields, rows


def atomic_csv(path, rows, fields):
    with tempfile.NamedTemporaryFile('w', newline='', encoding='utf-8-sig',
                                     dir=path.parent, prefix=path.stem + '_',
                                     suffix='.tmp', delete=False) as handle:
        temp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='raise')
        writer.writeheader(); writer.writerows(rows)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix=path.stem + '_', suffix='.tmp', delete=False) as handle:
        temp = Path(handle.name); handle.write(text)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def metric_value(value):
    if value == '':
        return ''
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Metrik tidak finite')
    return number


def default_split(path):
    stem = path.stem.lower()
    if 'diagnostik' in stem:
        return 'diagnostic'
    if 'retest' in stem:
        return 'known_retest'
    if 'holdout' in stem:
        return 'holdout_historis_bukan_baru_saat_ini'
    if 'stres' in stem or 'stress' in stem:
        return 'stress'
    return 'development' if 'dev' in stem else 'tidak_dinyatakan'


def parse_source(root, path, rule):
    fields, rows = read_csv(path)
    diagnostic = rule.get('kind') == 'diagnostic'
    if not diagnostic and not any(key in fields for key in ('spellF1', 'spell_f1_mean')):
        raise ValueError(f'Bukan CSV performa yang didukung: {path}')
    if not any(key in fields for key in ('method', 'submission')):
        raise ValueError(f'Kolom metode tidak ada: {path}')
    relative = path.relative_to(root).as_posix()
    run = path.parent.relative_to(root).as_posix()
    manifest = {}
    if rule.get('manifest'):
        manifest_path = path.parent / rule['manifest']
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    weights = rule.get('weights')
    if weights is not None:
        weights = {str(key): float(value) for key, value in weights.items()}
        if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
            raise ValueError(f'Bobot skenario harus positif: {relative}')
    protocol = digest(dict(run=run, manifest=manifest, weights=weights,
                           dataset=rule['dataset'], scope=rule['scope']))
    source_hash = file_hash(path)
    status = 'hasil_tersimpan'
    result = path.parent / 'result.json'
    if result.exists() and manifest:
        info = json.loads(result.read_text(encoding='utf-8-sig'))
        status = 'selesai_tercatat' if info.get('complete') and info.get('rows') == len(rows) else 'parsial'
    elif manifest:
        status = 'parsial'
    output = []
    for index, raw in enumerate(rows, start=2):
        try:
            metrics, extra = {}, {}
            for key, value in raw.items():
                if key in IDENTIFIERS:
                    continue
                name = ALIASES.get(key, key)
                try:
                    parsed = metric_value(value)
                except (TypeError, ValueError):
                    if key in ALIASES or key.startswith(('recall_', 'count_', 'F1_', 'boundary_')):
                        raise ValueError(f'Metrik numerik tidak sah: {key}={value!r}')
                    extra[key] = value
                    continue
                if name in metrics and metric_value(metrics[name]) != parsed:
                    raise ValueError(f'Alias metrik bertentangan: {name}')
                metrics[name] = value
            if not diagnostic and (not metrics.get('SpellF1') or not metrics.get('MacroF1')):
                raise ValueError('SpellF1/MacroF1 kosong; hasil gagal tidak diberi skor nol')
            if diagnostic and not any(value != '' for value in metrics.values()):
                raise ValueError('Diagnostik harus memiliki setidaknya satu metrik numerik')
            for name, value in metrics.items():
                if value != '' and ('f1' in name.lower() or 'recall' in name.lower() or
                                    'precision' in name.lower() or name == 'accuracy'):
                    if not 0 <= float(value) <= 1:
                        raise ValueError(f'Skor di luar [0,1]: {name}')
            split = raw.get('split') or rule.get('default_split') or default_split(path)
            identity = dict(Run=run, Protokol_SHA256=protocol, Dataset=rule['dataset'],
                            Cakupan=rule['scope'], Split=split, Fold=raw.get('fold', ''),
                            Pertandingan=raw.get('match') or rule.get('default_match', ''),
                            Skenario=raw.get('scenario') or raw.get('input') or 'original',
                            Seed=raw.get('seed', ''), Metode=raw.get('method') or raw.get('submission'),
                            Detail_JSON=canonical(extra))
            evaluation = digest(identity)
            record = dict(identity, ID_Evaluasi=evaluation,
                          ID_Hasil=digest(dict(evaluation=evaluation, metrics={
                              key: metric_value(value) for key, value in metrics.items()})),
                          Status_Run=status, Sumber=relative, Sumber_SHA256=source_hash,
                          Bobot_Skenario_JSON=canonical(weights),
                          Gabung_Split='1' if rule.get('combine_splits') else '0', **metrics)
            output.append(record)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{relative}:{index}: {error}') from error
    return output


def discover(root, config):
    found = {}
    excluded = set(config.get('exclude_parts', []))
    for rule in config['sources']:
        for path in sorted(root.glob(rule['glob'])):
            if excluded.intersection(path.relative_to(root).parts):
                continue
            if path.is_file() and path.suffix.lower() == '.csv':
                found.setdefault(path, rule)
    return found


def metric_columns(records):
    keys = set().union(*(set(row) for row in records)) if records else set()
    metrics = keys - set(META)
    return [name for name in PREFERRED_METRICS if name in metrics] + sorted(metrics - set(PREFERRED_METRICS))


def merge_records(existing, incoming, now):
    by_id = {row['ID_Hasil']: dict(row) for row in existing}
    if len(by_id) != len(existing):
        raise ValueError('ID_Hasil ganda di ledger; file tidak diubah')
    active = {}
    latest_version = defaultdict(int)
    for row in by_id.values():
        evaluation = row['ID_Evaluasi']
        latest_version[evaluation] = max(latest_version[evaluation], int(row['Versi']))
        if row['Aktif'] == '1':
            if row['ID_Evaluasi'] in active:
                raise ValueError('Lebih dari satu versi aktif di ledger')
            active[row['ID_Evaluasi']] = row
    unique = {}
    for row in incoming:
        previous = unique.get(row['ID_Evaluasi'])
        if previous and previous['ID_Hasil'] != row['ID_Hasil']:
            raise ValueError('Sumber bertentangan untuk evaluasi yang sama: '
                             f'{previous["Sumber"]}, {row["Sumber"]}')
        unique.setdefault(row['ID_Evaluasi'], row)
    added = changed = 0
    for evaluation, row in sorted(unique.items()):
        current = active.get(evaluation)
        if current and current['ID_Hasil'] == row['ID_Hasil']:
            # An unchanged refresh must not create a new revision or timestamp.
            for key in ('Status_Run', 'Sumber', 'Sumber_SHA256'):
                current[key] = row[key]
            continue
        if current:
            current['Aktif'] = '0'; changed += 1
        record = by_id.get(row['ID_Hasil'])
        if record is None:
            latest_version[evaluation] += 1
            record = dict(row, Versi=str(latest_version[evaluation]), Aktif='1',
                          Waktu_Pertama_UTC=now, Waktu_Pembaruan_UTC=now)
            by_id[row['ID_Hasil']] = record; added += 1
        else:
            record.update(row, Aktif='1', Waktu_Pembaruan_UTC=now)
        active[evaluation] = record
    return list(by_id.values()), added, changed


def summarize(records, metrics):
    groups = defaultdict(list)
    for row in records:
        if row['Aktif'] != '1':
            continue
        base = tuple(row[key] for key in ('Run', 'Protokol_SHA256', 'Dataset', 'Cakupan', 'Metode'))
        groups[base + (row['Split'],)].append(row)
        if row['Gabung_Split'] == '1':
            groups[base + ('semua_split',)].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        weights = json.loads(rows[0]['Bobot_Skenario_JSON'])
        scenarios = sorted({row['Skenario'] for row in rows})
        known_weights = weights is not None and all(name in weights for name in scenarios)
        chosen_weights = weights if known_weights else {name: 1. for name in scenarios}
        result = dict(zip(('Run', 'Protokol_SHA256', 'Dataset', 'Cakupan', 'Metode', 'Split'), key))
        result.update(Jumlah_Baris=len(rows),
                      Pertandingan_Unik=len({row['Pertandingan'] for row in rows if row['Pertandingan']}),
                      Skenario=';'.join(scenarios),
                      Status_Run=';'.join(sorted({row['Status_Run'] for row in rows})),
                      Agregasi='rerata_replikasi_per_skenario_lalu_bobot_skenario_lalu_rerata_laga'
                      if known_weights else 'rerata_replikasi_lalu_skenario_setara_lalu_laga',
                      Catatan=('Diagnostik non-possession: SpellF1/MacroF1 tidak diukur.'
                               if all(row.get('SpellF1', '') == '' for row in rows) else
                               'Hanya bandingkan Run/Protokol/Split/Cakupan yang sama; bukan skor privat target.'))
        for metric in metrics:
            by_match = defaultdict(lambda: defaultdict(list))
            for row in rows:
                if row.get(metric, '') != '':
                    by_match[row['Pertandingan']][row['Skenario']].append(float(row[metric]))
            values = []
            for scenario_rows in by_match.values():
                total_weight = sum(chosen_weights[name] for name in scenario_rows)
                values.append(sum(sum(numbers) / len(numbers) * chosen_weights[name]
                                  for name, numbers in scenario_rows.items()) / total_weight)
            result[metric] = sum(values) / len(values) if values else ''
        output.append(result)
    return sorted(output, key=lambda row: (row['Run'], row['Split'],
                                           -float(row.get('SpellF1') or 0), row['Metode']))


def refresh(root=ROOT, config_path=None):
    root = Path(root).resolve()
    config_path = Path(config_path) if config_path else root / 'konfigurasi_performa.json'
    with registry_lock(root):
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        sources = discover(root, config)
        incoming = []
        for path, rule in sources.items():
            incoming.extend(parse_source(root, path, rule))
        path = root / DETAIL_NAME
        existing = read_csv(path)[1] if path.exists() else []
        now = utcnow()
        merged, added, revised = merge_records(existing, incoming, now)
        metrics = metric_columns(merged)
        # Keep historical metric columns when new sources have fewer metrics.
        for row in merged:
            for metric in metrics:
                row.setdefault(metric, '')
        merged.sort(key=lambda row: (row['Run'], row['Split'], row['Metode'], row['Pertandingan'],
                                     row['Skenario'], row['Fold'], row['Seed'], int(row['Versi'])))
        summary = summarize(merged, metrics)
        summary_meta = ['Run', 'Protokol_SHA256', 'Dataset', 'Cakupan', 'Split', 'Metode',
                        'Jumlah_Baris', 'Pertandingan_Unik', 'Skenario', 'Status_Run', 'Agregasi', 'Catatan']
        # All parsing/validation happens before replacing either output.
        atomic_csv(path, merged, META + metrics)
        atomic_csv(root / SUMMARY_NAME, summary, summary_meta + metrics)
        state = dict(complete=True, updated_utc=now, sources=len(sources),
                     source_rows=len(incoming), stored_rows=len(merged),
                     active_rows=sum(row['Aktif'] == '1' for row in merged),
                     new_records=added, revised_evaluations=revised, summary_rows=len(summary),
                     detail=DETAIL_NAME, summary=SUMMARY_NAME, target_private_score_known=False)
        atomic_json(root / STATUS_NAME, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--watch', type=float, metavar='SECONDS',
                        help='Refresh berkala selama terminal ini tetap terbuka, minimal 5 detik')
    args = parser.parse_args()
    if args.watch is not None and (not math.isfinite(args.watch) or args.watch < 5):
        parser.error('--watch harus minimal 5 detik')
    while True:
        try:
            print(json.dumps(refresh(args.root, args.config), ensure_ascii=False), flush=True)
        except Exception as error:
            print(f'GAGAL mencatat performa: {error}. Sumber skor tidak diubah.', file=sys.stderr, flush=True)
            if args.watch is None:
                raise SystemExit(1)
        if args.watch is None:
            return
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    main()
