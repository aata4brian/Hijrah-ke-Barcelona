import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from importlib import metadata
from .reproduce import environment_evidence


class ReproductionEnvironmentTests(unittest.TestCase):
    def evidence(self, inherit, external=False, missing=(), user_site=False):
        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp)
            (prefix / 'pyvenv.cfg').write_text(
                'include-system-site-packages = ' + str(inherit).lower() + '\n', encoding='utf-8')
            class Distribution:
                version = 'test'
                def locate_file(self, name):
                    return prefix.parent if external else prefix / 'Lib/site-packages'
            def distribution(name):
                if name in missing:
                    raise metadata.PackageNotFoundError(name)
                return Distribution()
            with patch('riset_v4.reproduce.sys.prefix', str(prefix)), \
                 patch('riset_v4.reproduce.sys.base_prefix', str(prefix.parent)), \
                 patch('riset_v4.reproduce.site.ENABLE_USER_SITE', user_site), \
                 patch('riset_v4.reproduce.metadata.distribution', side_effect=distribution):
                return environment_evidence()

    def test_inherited_packages_are_not_clean(self):
        self.assertFalse(self.evidence(True)['clean_environment'])

    def test_isolation_is_not_proof_of_fresh_installation(self):
        evidence = self.evidence(False)
        self.assertTrue(evidence['isolated_environment'])
        self.assertFalse(evidence['fresh_installation_verified'])

    def test_external_distributions_are_not_clean(self):
        self.assertFalse(self.evidence(False, external=True)['clean_environment'])

    def test_missing_required_package_is_not_clean(self):
        evidence = self.evidence(False, missing=('torch',))
        self.assertFalse(evidence['isolated_environment'])
        self.assertFalse(evidence['clean_environment'])
        self.assertEqual(evidence['missing_required_packages'], ['torch'])
        self.assertNotIn('torch', evidence['packages'])

    def test_all_missing_packages_cannot_pass_vacuously(self):
        missing = ('numpy', 'scipy', 'scikit-learn', 'joblib', 'pandas',
                   'threadpoolctl', 'numba', 'llvmlite', 'torch', 'opencv-contrib-python')
        evidence = self.evidence(False, missing=missing)
        self.assertFalse(evidence['clean_environment'])
        self.assertEqual(set(evidence['missing_required_packages']), set(missing))
        self.assertEqual(evidence['packages'], {})

    def test_user_site_packages_are_not_clean(self):
        self.assertFalse(self.evidence(False, user_site=True)['clean_environment'])


if __name__ == '__main__':
    unittest.main()
