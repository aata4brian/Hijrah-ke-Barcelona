import json
from pathlib import Path
import tempfile
import unittest
from .continue_all import completed_run


class QueueTests(unittest.TestCase):
    def test_missing_output_is_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(completed_run(Path(folder), [0], 5, 120))

    def test_complete_queue_requires_same_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, value in (('queue_status.json', {'stage': 'complete'}),
                                ('result.json', {'complete': True}),
                                ('manifest.json', {'folds': [0], 'epochs': 5, 'steps': 120})):
                (root / name).write_text(json.dumps(value), encoding='utf-8')
            self.assertTrue(completed_run(root, [0], 5, 120))
            with self.assertRaises(ValueError):
                completed_run(root, [1], 5, 120)
            (root / 'queue_status.json').write_text('{"stage": "failed"}', encoding='utf-8')
            self.assertFalse(completed_run(root, [0], 5, 120))


if __name__ == '__main__':
    unittest.main()
