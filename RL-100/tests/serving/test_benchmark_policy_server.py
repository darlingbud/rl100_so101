from __future__ import annotations

import unittest

import numpy as np

from benchmark_policy_server import synthetic_observation


class BenchmarkPolicyServerTest(unittest.TestCase):
    def test_synthetic_observation_uses_metadata(self):
        metadata = {
            "observation_spec": {
                "image": {"policy_input_shape": [2, 3, 8, 10], "dtype": "uint8"},
                "state": {"policy_input_shape": [2, 6], "dtype": "float32"},
            }
        }
        observation = synthetic_observation(metadata)
        self.assertEqual(observation["image"].shape, (2, 3, 8, 10))
        self.assertEqual(observation["image"].dtype, np.uint8)
        self.assertEqual(observation["state"].shape, (2, 6))
        self.assertEqual(observation["state"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
