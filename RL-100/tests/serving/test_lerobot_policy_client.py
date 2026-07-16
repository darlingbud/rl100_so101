from __future__ import annotations

from collections import deque
import unittest

import numpy as np

from lerobot_policy_client import MOTORS, action_dict, observation_frame, stack_history


class LeRobotPolicyClientTest(unittest.TestCase):
    def make_raw(self, offset: float = 0.0):
        raw = {f"{motor}.pos": i + offset for i, motor in enumerate(MOTORS)}
        raw["front"] = np.zeros((480, 640, 3), dtype=np.uint8)
        raw["side"] = np.ones((480, 640, 3), dtype=np.uint8)
        return raw

    def test_observation_mapping_and_history_layout(self):
        history = deque(
            [observation_frame(self.make_raw()), observation_frame(self.make_raw(1))],
            maxlen=2,
        )
        result = stack_history(history)
        self.assertEqual(result["image_front"].shape, (2, 3, 480, 640))
        self.assertEqual(result["image_side"].shape, (2, 3, 480, 640))
        self.assertEqual(result["agent_pos"].shape, (2, 6))
        self.assertEqual(result["agent_pos"].dtype, np.float32)

    def test_action_mapping(self):
        result = action_dict(np.arange(6, dtype=np.float32))
        self.assertEqual(list(result), [f"{motor}.pos" for motor in MOTORS])
        self.assertEqual(result["gripper.pos"], 5.0)

    def test_rejects_bad_action(self):
        with self.assertRaises(ValueError):
            action_dict(np.zeros(5, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
