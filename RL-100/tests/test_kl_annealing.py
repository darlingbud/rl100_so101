import unittest

from rl_100.common.kl_annealing import kl_annealing_progress


class KlAnnealingProgressTest(unittest.TestCase):
    def test_none_uses_full_training_duration(self):
        self.assertEqual(kl_annealing_progress(0, 500, None), 0.0)
        self.assertAlmostEqual(kl_annealing_progress(250, 500, None), 250 / 499)
        self.assertEqual(kl_annealing_progress(499, 500, None), 1.0)

    def test_configured_duration_reaches_target_and_stays_there(self):
        self.assertEqual(kl_annealing_progress(0, 500, 100), 0.0)
        self.assertAlmostEqual(kl_annealing_progress(50, 500, 100), 50 / 99)
        self.assertEqual(kl_annealing_progress(99, 500, 100), 1.0)
        self.assertEqual(kl_annealing_progress(100, 500, 100), 1.0)

    def test_one_epoch_duration_starts_at_target(self):
        self.assertEqual(kl_annealing_progress(0, 500, 1), 1.0)

    def test_invalid_duration_is_rejected(self):
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "kl_annealing_epoch"):
                    kl_annealing_progress(0, 500, value)

    def test_negative_epoch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "epoch"):
            kl_annealing_progress(-1, 500, None)


if __name__ == "__main__":
    unittest.main()
