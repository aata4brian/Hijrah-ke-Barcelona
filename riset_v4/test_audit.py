import unittest
from .audit import METHODS, validate_matrix
from .data import DEV, split


class MatrixTests(unittest.TestCase):
    def fixture(self):
        manifest = dict(folds=[0], epochs=5, seed=331, test_scenarios=['original'])
        rows = [dict(fold=0, match=match, scenario='original', method=method,
                     split='development' if match in DEV else 'known_retest',
                     seed=432, spellF1=.5, macroF1=.8)
                for match in split(0)[2] for method in METHODS]
        return rows, manifest

    def test_complete_matrix(self):
        self.assertEqual(validate_matrix(*self.fixture()), 24)

    def test_missing_row(self):
        rows, manifest = self.fixture()
        with self.assertRaises(ValueError):
            validate_matrix(rows[:-1], manifest)

    def test_duplicate_row(self):
        rows, manifest = self.fixture()
        with self.assertRaises(ValueError):
            validate_matrix(rows + [rows[0]], manifest)

    def test_invalid_metric_and_split(self):
        for update in (dict(spellF1=float('nan')), dict(macroF1=1.1),
                       dict(split='known_retest'), dict(seed=1)):
            rows, manifest = self.fixture()
            rows[0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_matrix(rows, manifest)


if __name__ == '__main__':
    unittest.main()
