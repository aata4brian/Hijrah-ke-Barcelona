"""Storage-only calibration changes must not change the experiment."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .data import SCENARIOS
from .deep_temporal import DeepTemporal
from .local_deep import StreamingCalibrationCases, calibration_value, close_cases, feature_cases
from .test_v4 import fixture
from .tcn import Temporal
from .decode import Settings


class CalibrationStorageTests(unittest.TestCase):
    def test_streaming_matches_disk_and_is_repeatable(self):
        torch.set_num_threads(2)
        data, y = fixture(600)
        with tempfile.TemporaryDirectory(prefix='deep_storage_test_') as temp:
            root = Path(temp)
            cache = root/'J03WN1.npz'
            arrays = {'pitch': data.pitch}
            for half in (0, 1):
                arrays.update({f'ball{half}':data.ball, f'home{half}':data.players[0],
                    f'away{half}':data.players[1], f'truth{half}':y})
            np.savez(cache, **arrays)
            paths = {'J03WN1':cache}
            disk = feature_cases(paths, ['J03WN1'], list(SCENARIOS), 348, root/'disk', lambda *a, **k:None)
            stream = StreamingCalibrationCases(paths, ['J03WN1'], list(SCENARIOS), 348)
            try:
                for repeat in range(2):
                    count = 0
                    for actual, expected in zip(stream, disk):
                        for a, e in zip(actual[0], expected[0]):
                            self.assertEqual(a.dtype, np.float16)
                            np.testing.assert_array_equal(a, e)
                        np.testing.assert_array_equal(actual[1], expected[1])
                        self.assertEqual(actual[2], expected[2])
                        count += 1
                    self.assertEqual(count, 2*len(SCENARIOS))
                features, truth, weight = disk[0]
                base = Temporal().fit([(features,truth,weight)], epochs=1, steps=1, device='cpu', progress=lambda e:None)
                base.decoder = Settings()
                model = DeepTemporal(base)
                self.assertEqual(calibration_value(model,disk,base.decoder),
                                 calibration_value(model,stream,base.decoder))
                close_cases(stream)
            finally:
                close_cases(disk)


if __name__ == '__main__':
    unittest.main()
