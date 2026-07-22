import numpy as np
import pytest

from lerobot_hitl_collect import clamp_relative_target, relative_human_target


def test_clamp_relative_target_limits_each_joint():
    current = np.asarray([0, 0, 10, -10, 5, 1], dtype=np.float32)
    target = np.asarray([10, -10, 12, -12, 5, 8], dtype=np.float32)

    result = clamp_relative_target(target, current, max_relative_target=3)

    np.testing.assert_allclose(result, [3, -3, 12, -12, 5, 4])


def test_clamp_relative_target_validates_shape():
    with pytest.raises(ValueError, match="six-dimensional"):
        clamp_relative_target(np.zeros(5), np.zeros(6), max_relative_target=3)


def test_relative_human_target_has_no_takeover_jump():
    leader_anchor = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.float32)
    follower_anchor = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32)

    at_takeover = relative_human_target(
        leader_anchor, leader_anchor, follower_anchor
    )
    after_motion = relative_human_target(
        leader_anchor + 2, leader_anchor, follower_anchor
    )

    np.testing.assert_allclose(at_takeover, follower_anchor)
    np.testing.assert_allclose(after_motion, follower_anchor + 2)
