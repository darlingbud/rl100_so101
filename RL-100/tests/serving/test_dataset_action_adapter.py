import pathlib
import tempfile
import unittest

import numpy as np
import zarr

from rl_100.serving.dataset_action_adapter import DatasetActionAdapter


class DatasetActionAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset = pathlib.Path(self.temp_dir.name) / "test.zarr"
        root = zarr.open_group(str(self.dataset), mode="w")
        data = root.create_group("data")
        meta = root.create_group("meta")
        actions = np.arange(60, dtype=np.float32).reshape(10, 6)
        data.create_dataset("action", data=actions, chunks=(5, 6))
        data.create_dataset("state", data=actions + 0.5, chunks=(5, 6))
        data.create_dataset(
            "image_front", data=np.zeros((10, 2, 3, 3), dtype=np.uint8)
        )
        data.create_dataset(
            "image_side", data=np.zeros((10, 2, 3, 3), dtype=np.uint8)
        )
        meta.create_dataset("episode_ends", data=np.array([5, 10], dtype=np.int64))
        root.attrs["fps"] = 30
        self.actions = actions

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_replay_uses_episode_and_client_step(self):
        adapter = DatasetActionAdapter(
            self.dataset,
            mode="replay",
            episode_index=1,
            start_step=1,
            action_horizon=4,
        )
        result = adapter.infer(self._observation(), step_id=2, episode_id="test")
        np.testing.assert_array_equal(
            result["actions"], self.actions[[8, 9, 9, 9]]
        )
        self.assertEqual(result["replay"]["source_step"], 3)
        self.assertTrue(result["replay"]["done"])

    def test_hold_repeats_latest_observed_state(self):
        adapter = DatasetActionAdapter(
            self.dataset, mode="hold", action_horizon=4, n_obs_steps=2
        )
        observation = self._observation()
        result = adapter.infer(observation, step_id=7)
        expected = np.repeat(observation["agent_pos"][-1][None], 4, axis=0)
        np.testing.assert_array_equal(result["actions"], expected)
        self.assertFalse(result["replay"]["done"])

    def test_metadata_matches_client_contract(self):
        adapter = DatasetActionAdapter(self.dataset)
        metadata = adapter.metadata
        self.assertEqual(metadata["n_obs_steps"], 2)
        self.assertEqual(metadata["action_spec"]["shape"], [4, 6])
        self.assertEqual(
            metadata["observation_spec"]["image_front"]["policy_input_shape"],
            [2, 3, 2, 3],
        )
        self.assertEqual(metadata["replay"]["dataset_fps"], 30)

    @staticmethod
    def _observation():
        return {
            "agent_pos": np.arange(12, dtype=np.float32).reshape(2, 6),
        }


if __name__ == "__main__":
    unittest.main()
