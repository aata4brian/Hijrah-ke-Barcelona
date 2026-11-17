import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import perbarui_performa as registry


class PerformanceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = {'exclude_parts': ['backups'], 'sources': [
            {'glob': 'runs/*/scores*.csv', 'dataset': 'synthetic_test_only',
             'scope': 'unit_test', 'combine_splits': True,
             'weights': {'original': .25, 'long_gaps': .75}}]}
        (self.root / 'konfigurasi_performa.json').write_text(json.dumps(config), encoding='utf-8')
        self.path = self.root / 'runs/a/scores.csv'
        self.path.parent.mkdir(parents=True)
        self.rows = [dict(split='development', match='test_match', scenario='original',
                          fold='0', seed='1', method='test_model', spellF1='.8', macroF1='.9'),
                     dict(split='development', match='test_match', scenario='long_gaps',
                          fold='0', seed='1', method='test_model', spellF1='.4', macroF1='.5')]
        self.write(self.rows)

    def write(self, rows, path=None, encoding='utf-8-sig'):
        path = path or self.path
        with path.open('w', newline='', encoding=encoding) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def detail(self):
        return registry.read_csv(self.root / registry.DETAIL_NAME)[1]

    def test_initial_scores_and_weighted_summary(self):
        result = registry.refresh(self.root)
        self.assertEqual(result['active_rows'], 2)
        rows = registry.read_csv(self.root / registry.SUMMARY_NAME)[1]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(float(row['SpellF1']), .5)
            self.assertAlmostEqual(float(row['MacroF1']), .6)
            self.assertEqual(row['Pertandingan_Unik'], '1')

    def test_refresh_is_idempotent(self):
        registry.refresh(self.root)
        before = (self.root / registry.DETAIL_NAME).read_bytes()
        result = registry.refresh(self.root)
        self.assertEqual(result['new_records'], 0)
        self.assertEqual(result['revised_evaluations'], 0)
        self.assertEqual(before, (self.root / registry.DETAIL_NAME).read_bytes())

    def test_new_result_is_added(self):
        registry.refresh(self.root)
        self.write(self.rows + [dict(self.rows[0], method='new_model')])
        result = registry.refresh(self.root)
        self.assertEqual(result['active_rows'], 3)
        self.assertEqual(result['new_records'], 1)

    def test_revision_retains_old_value_without_double_counting(self):
        registry.refresh(self.root)
        self.write([dict(self.rows[0], spellF1='.6'), self.rows[1]])
        result = registry.refresh(self.root)
        self.assertEqual(result['stored_rows'], 3)
        self.assertEqual(result['active_rows'], 2)
        self.assertEqual(result['revised_evaluations'], 1)
        revised = [row for row in self.detail() if row['Skenario'] == 'original']
        self.assertEqual({(row['SpellF1'], row['Aktif']) for row in revised}, {('.8', '0'), ('.6', '1')})
        summary = registry.read_csv(self.root / registry.SUMMARY_NAME)[1]
        self.assertAlmostEqual(float(summary[0]['SpellF1']), .45)

    def test_source_disappearance_does_not_delete_history(self):
        registry.refresh(self.root)
        self.path.rename(self.path.with_suffix('.archived'))
        result = registry.refresh(self.root)
        self.assertEqual(result['stored_rows'], 2)
        self.assertEqual(result['active_rows'], 2)

    def test_corrupt_source_does_not_replace_existing_csv(self):
        registry.refresh(self.root)
        before = {name: (self.root / name).read_bytes()
                  for name in (registry.DETAIL_NAME, registry.SUMMARY_NAME)}
        self.write([dict(self.rows[0], spellF1='nan')])
        with self.assertRaises(ValueError):
            registry.refresh(self.root)
        for name, data in before.items():
            self.assertEqual((self.root / name).read_bytes(), data)

    def test_utf16_archive_supported(self):
        self.write(self.rows, encoding='utf-16')
        self.assertEqual(registry.refresh(self.root)['active_rows'], 2)

    def test_new_numeric_metric_is_preserved(self):
        registry.refresh(self.root)
        self.write([dict(row, recall_under2s='.3') for row in self.rows])
        registry.refresh(self.root)
        active = [row for row in self.detail() if row['Aktif'] == '1']
        self.assertTrue(all(row['recall_under2s'] == '.3' for row in active))
        self.assertEqual(len(self.detail()), 4)

    def test_conflicting_sources_fail_before_writes(self):
        registry.refresh(self.root)
        self.write([dict(self.rows[0], spellF1='.1')], path=self.path.with_name('scores_conflict.csv'))
        before = (self.root / registry.DETAIL_NAME).read_bytes()
        with self.assertRaises(ValueError):
            registry.refresh(self.root)
        self.assertEqual((self.root / registry.DETAIL_NAME).read_bytes(), before)

    def test_same_row_in_multiple_files_is_not_duplicated(self):
        self.write(self.rows, path=self.path.with_name('scores_copy.csv'))
        self.assertEqual(registry.refresh(self.root)['active_rows'], 2)

    def test_replicates_do_not_count_as_additional_matches(self):
        self.write(self.rows + [dict(self.rows[0], fold='1', seed='2')])
        registry.refresh(self.root)
        rows = registry.read_csv(self.root / registry.SUMMARY_NAME)[1]
        for row in rows:
            self.assertAlmostEqual(float(row['SpellF1']), .5)
            self.assertEqual(row['Pertandingan_Unik'], '1')

    def test_explicit_diagnostic_does_not_invent_possession_scores(self):
        config_path = self.root / 'konfigurasi_performa.json'
        config = json.loads(config_path.read_text())
        config['sources'][0]['kind'] = 'diagnostic'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        self.write([dict(method='video_test', match='test_match',
                         scenario='real_gap', recovered_frames='0', attempted_gaps='4')])
        result = registry.refresh(self.root)
        self.assertEqual(result['active_rows'], 1)
        row = self.detail()[0]
        self.assertNotIn('SpellF1', row)
        self.assertEqual(row['recovered_frames'], '0')
        summary = registry.read_csv(self.root / registry.SUMMARY_NAME)[1][0]
        self.assertIn('tidak diukur', summary['Catatan'])

    def test_diagnostic_without_numeric_metrics_is_rejected(self):
        config_path = self.root / 'konfigurasi_performa.json'
        config = json.loads(config_path.read_text())
        config['sources'][0]['kind'] = 'diagnostic'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        self.write([dict(method='video_test', reason='no measurement')])
        with self.assertRaises(ValueError):
            registry.refresh(self.root)

    def test_registry_sync_skips_standalone_project(self):
        import sys
        research = Path(__file__).resolve().parent / '09_Riset_V4'
        sys.path.insert(0, str(research))
        from riset_v4.registry import sync_registry
        self.assertIsNone(sync_registry(project=self.root / 'standalone'))

    def test_registry_sync_invokes_script_and_surfaces_failure(self):
        import sys
        research = Path(__file__).resolve().parent / '09_Riset_V4'
        sys.path.insert(0, str(research))
        from riset_v4.registry import sync_registry
        (self.root / 'perbarui_performa.py').write_text('# test fixture', encoding='utf-8')
        with patch('riset_v4.registry.subprocess.run') as execute:
            execute.return_value.returncode = 0
            execute.return_value.stdout = 'test-ok'
            self.assertEqual(sync_registry(self.root / 'research', strict=True), 'test-ok')
            self.assertIn(str(self.root / 'perbarui_performa.py'), execute.call_args.args[0])
            execute.return_value.returncode = 1
            execute.return_value.stderr = 'test error'
            with self.assertRaises(RuntimeError):
                sync_registry(self.root / 'research', strict=True)


if __name__ == '__main__':
    unittest.main()
