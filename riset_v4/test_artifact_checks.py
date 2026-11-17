import tempfile
import unittest
from pathlib import Path
import numpy as np
from .artifact_checks import load_recovery, validate_probabilities


class ArtifactChecksTests(unittest.TestCase):
    def recovery(self, **updates):
        original = np.array([[1., 2.], [np.nan, np.nan], [3., 4.]], np.float32)
        ball = original.copy(); ball[1] = [2., 3.]
        data = dict(original_ball=original, ball=ball,
                    recovered=np.array([False, True, False]),
                    confidence=np.array([0., .8, 0.]), fps=np.array(25))
        data.update(updates)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'recovery.npz'
            np.savez(path, **data)
            return load_recovery(path, 3)

    def test_valid_recovery(self):
        self.assertEqual(int(self.recovery()['recovered'].sum()), 1)

    def test_reject_fractional_fps_and_wrong_shape(self):
        for updates in (dict(fps=np.array(25.9)), dict(fps=np.array([25])),
                        dict(original_ball=np.zeros((2, 2)))):
            with self.subTest(updates=list(updates)), self.assertRaises(ValueError):
                self.recovery(**updates)

    def test_reject_nonboolean_mask(self):
        with self.assertRaises(ValueError):
            self.recovery(recovered=np.array([0, 1, 0]))

    def test_reject_overwrite_observation(self):
        with self.assertRaises(ValueError):
            self.recovery(recovered=np.array([True, True, False]))
        with self.assertRaises(ValueError):
            self.recovery(ball=np.array([[9., 2.], [2., 3.], [3., 4.]]))

    def test_reject_nonfinite_recovery_and_bad_confidence(self):
        for updates in (dict(ball=np.array([[1., 2.], [np.nan, 3.], [3., 4.]])),
                        dict(confidence=np.array([0., np.nan, 0.])),
                        dict(confidence=np.array([0., 1.1, 0.]))):
            with self.subTest(updates=list(updates)), self.assertRaises(ValueError):
                self.recovery(**updates)

    def test_reject_changes_outside_mask(self):
        with self.assertRaises(ValueError):
            self.recovery(recovered=np.zeros(3, bool))

    def test_valid_probabilities(self):
        validate_probabilities(np.tile([.2, .3, .5], (4, 1)), np.zeros((4, 6)), 4)

    def test_reject_bad_probabilities(self):
        for p, q in ((np.zeros((4, 3)), np.zeros((4, 6))),
                     (np.full((4, 3), np.nan), np.zeros((4, 6))),
                     (np.tile([.2, .3, .5], (4, 1)), np.full((4, 6), -1.)),
                     (np.ones((4, 2)), np.zeros((4, 6)))):
            with self.subTest(shape=p.shape), self.assertRaises(ValueError):
                validate_probabilities(p, q, 4)


if __name__ == '__main__':
    unittest.main()
