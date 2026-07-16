import unittest
import pathlib
import tempfile

import dill
import numpy as np
from omegaconf import OmegaConf
import torch

from rl_100.serving.policy_adapter import RL100PolicyAdapter, _remove_ddp_segments
from rl_100.serving.protocol import ProtocolError


class FakePolicy(torch.nn.Module):
    def __init__(self, use_cm=False):
        super().__init__()
        self.use_cm = use_cm
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.n_action_steps = 2
        self.action_dim = 3
        self.reset_count = 0

    def predict_action(self, obs_dict, **kwargs):
        self.last_obs = obs_dict
        self.last_kwargs = kwargs
        action = torch.arange(6, device=self.anchor.device, dtype=torch.float32)
        return {"action": action.reshape(1, 2, 3)}

    def reset(self):
        self.reset_count += 1


def make_adapter():
    cfg = OmegaConf.create(
        {
            "name": "test_policy",
            "task_name": "test_task",
            "n_obs_steps": 2,
            "n_action_steps": 2,
            "shape_meta": {
                "obs": {
                    "point_cloud": {"shape": [4, 3], "type": "point_cloud"},
                    "agent_pos": {"shape": [3], "type": "low_dim"},
                },
                "action": {"shape": [3]},
            },
            "policy": {"use_cm": True},
        }
    )
    return RL100PolicyAdapter(
        FakePolicy(),
        cfg,
        device="cpu",
        weights_source="model",
        deterministic=True,
    )


class PolicyAdapterTest(unittest.TestCase):
    def test_workspace_checkpoint_prefers_ema_weights(self):
        checkpoint_policy = FakePolicy()
        checkpoint_policy.anchor.data.fill_(1.0)
        ema_policy = FakePolicy()
        ema_policy.anchor.data.fill_(2.0)
        cfg = OmegaConf.create(
            {
                "name": "checkpoint_policy",
                "task_name": "test_task",
                "n_obs_steps": 2,
                "n_action_steps": 2,
                "shape_meta": {
                    "obs": {
                        "point_cloud": {
                            "shape": [4, 3],
                            "type": "point_cloud",
                        },
                        "agent_pos": {"shape": [3], "type": "low_dim"},
                    },
                    "action": {"shape": [3]},
                },
                "policy": {
                    "_target_": "serving.test_policy_adapter.FakePolicy",
                    "use_cm": False,
                },
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = pathlib.Path(temp_dir) / "policy.ckpt"
            torch.save(
                {
                    "cfg": cfg,
                    "state_dicts": {
                        "model": checkpoint_policy.state_dict(),
                        "ema_model": ema_policy.state_dict(),
                    },
                },
                checkpoint,
                pickle_module=dill,
            )
            adapter = RL100PolicyAdapter.from_checkpoint(
                checkpoint, device="cpu", weights="auto"
            )
        self.assertEqual(adapter.metadata["weights_source"], "ema_model")
        self.assertEqual(adapter._policy.anchor.item(), 2.0)

    def test_external_training_config_is_validated(self):
        checkpoint_policy = FakePolicy()
        cfg = OmegaConf.create(
            {
                "name": "checkpoint_policy",
                "task_name": "test_task",
                "horizon": 3,
                "n_obs_steps": 2,
                "n_action_steps": 2,
                "shape_meta": {
                    "obs": {"agent_pos": {"shape": [3], "type": "low_dim"}},
                    "action": {"shape": [3]},
                },
                "policy": {
                    "_target_": "serving.test_policy_adapter.FakePolicy",
                    "use_cm": False,
                },
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = pathlib.Path(temp_dir) / "policy.ckpt"
            config = pathlib.Path(temp_dir) / "config.yaml"
            torch.save(
                {"cfg": cfg, "state_dicts": {"model": checkpoint_policy.state_dict()}},
                checkpoint,
                pickle_module=dill,
            )
            OmegaConf.save(cfg, config)
            adapter = RL100PolicyAdapter.from_checkpoint(
                checkpoint, config=config, device="cpu", weights="model"
            )
            self.assertEqual(adapter.metadata["n_obs_steps"], 2)

            mismatched_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
            mismatched_cfg.n_action_steps = 4
            OmegaConf.save(mismatched_cfg, config)
            with self.assertRaisesRegex(ValueError, "n_action_steps"):
                RL100PolicyAdapter.from_checkpoint(
                    checkpoint, config=config, device="cpu", weights="model"
                )

    def test_metadata_describes_history_and_action_chunk(self):
        metadata = make_adapter().metadata
        self.assertEqual(metadata["n_obs_steps"], 2)
        self.assertEqual(
            metadata["observation_spec"]["point_cloud"]["policy_input_shape"],
            [2, 4, 3],
        )
        self.assertEqual(metadata["action_spec"]["shape"], [2, 3])

    def test_metadata_describes_rgb_wire_format(self):
        adapter = make_adapter()
        adapter._observation_meta["camera"] = {
            "shape": [3, 480, 640],
            "type": "rgb",
        }
        camera_spec = adapter.metadata["observation_spec"]["camera"]
        self.assertEqual(camera_spec["policy_input_shape"], [2, 3, 480, 640])
        self.assertEqual(camera_spec["dtype"], "uint8")
        self.assertEqual(camera_spec["layout"], "TCHW")
        self.assertEqual(camera_spec["value_range"], [0, 255])

    def test_infer_converts_numpy_to_batched_tensor(self):
        adapter = make_adapter()
        result = adapter.infer(
            {
                "point_cloud": np.zeros((2, 4, 3), dtype=np.float32),
                "agent_pos": np.zeros((2, 3), dtype=np.float64),
            }
        )
        np.testing.assert_array_equal(
            result["actions"], np.arange(6, dtype=np.float32).reshape(2, 3)
        )
        self.assertEqual(adapter._policy.last_obs["point_cloud"].shape, (1, 2, 4, 3))
        self.assertEqual(adapter._policy.last_obs["agent_pos"].dtype, torch.float32)
        self.assertTrue(adapter._policy.last_kwargs["deterministic"])
        self.assertTrue(adapter._policy.last_kwargs["use_cm"])

    def test_infer_rejects_wrong_history_shape(self):
        adapter = make_adapter()
        with self.assertRaises(ProtocolError) as context:
            adapter.infer(
                {
                    "point_cloud": np.zeros((1, 4, 3), dtype=np.float32),
                    "agent_pos": np.zeros((2, 3), dtype=np.float32),
                }
            )
        self.assertEqual(context.exception.code, "INVALID_OBSERVATION")

    def test_infer_rejects_non_finite_observation(self):
        adapter = make_adapter()
        agent_pos = np.zeros((2, 3), dtype=np.float32)
        agent_pos[0, 0] = np.nan
        with self.assertRaises(ProtocolError):
            adapter.infer(
                {
                    "point_cloud": np.zeros((2, 4, 3), dtype=np.float32),
                    "agent_pos": agent_pos,
                }
            )

    def test_reset_delegates_to_policy(self):
        adapter = make_adapter()
        adapter.reset("episode")
        self.assertEqual(adapter._policy.reset_count, 1)

    def test_ddp_key_normalization(self):
        self.assertEqual(
            _remove_ddp_segments("module.model.module.weight"), "model.weight"
        )


if __name__ == "__main__":
    unittest.main()
