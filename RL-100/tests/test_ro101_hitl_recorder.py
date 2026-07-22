import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from rl_100.collectors.ro101_hitl_recorder import (
    HitlEpisodeBuffer,
    HitlZarrWriter,
    discounted_return,
)


def observation(value: int):
    return {
        "agent_pos": np.full(6, value, dtype=np.float32),
        "image_front": np.full((4, 5, 3), value, dtype=np.uint8),
        "image_side": np.full((4, 5, 3), value + 1, dtype=np.uint8),
    }


def append_transition(episode, value, intervention=False):
    episode.append(
        observation=observation(value),
        next_observation=observation(value + 1),
        policy_action=np.full(6, value, dtype=np.float32),
        human_action=np.full(6, value + 1, dtype=np.float32),
        executed_action=np.full(6, value + intervention, dtype=np.float32),
        intervention=intervention,
        policy_valid=True,
        clamped=False,
        time_ns=value,
    )


def test_discounted_return():
    result = discounted_return(np.array([0.0, 0.0, 1.0]), gamma=0.5)
    np.testing.assert_allclose(result[:, 0], [0.25, 0.5, 1.0])


def test_episode_marks_terminal_reward_and_preserves_actions():
    episode = HitlEpisodeBuffer(gamma=0.5)
    append_transition(episode, 0)
    append_transition(episode, 1, intervention=True)

    data = episode.finalize(success=True)

    np.testing.assert_allclose(data["reward"][:, 0], [0.0, 1.0])
    np.testing.assert_allclose(data["return"][:, 0], [0.5, 1.0])
    np.testing.assert_array_equal(data["done"][:, 0], [False, True])
    np.testing.assert_array_equal(data["intervention"][:, 0], [False, True])
    np.testing.assert_allclose(data["action"][1], np.full(6, 2.0))
    np.testing.assert_allclose(data["next_action"][0], data["action"][1])


def test_writer_appends_episodes_and_metadata():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "hitl.zarr"
        writer = HitlZarrWriter(path, gamma=0.9)
        first = HitlEpisodeBuffer(gamma=0.9)
        append_transition(first, 0, intervention=True)
        second = HitlEpisodeBuffer(gamma=0.9)
        append_transition(second, 2)

        assert writer.append_episode(first, success=True, timeout=False) == 0
        assert writer.append_episode(second, success=False, timeout=True) == 1

        root = zarr.open_group(str(path), mode="r")
        np.testing.assert_array_equal(root["meta/episode_ends"][:], [1, 2])
        np.testing.assert_array_equal(root["meta/episode_success"][:], [True, False])
        np.testing.assert_array_equal(root["meta/episode_timeout"][:], [False, True])
        np.testing.assert_array_equal(
            root["meta/episode_intervention_steps"][:], [1, 0]
        )
        assert root["data/image_front"].shape == (2, 4, 5, 3)


def test_empty_episode_cannot_be_saved():
    with pytest.raises(RuntimeError, match="empty HITL episode"):
        HitlEpisodeBuffer().finalize(success=False)


def test_writer_rejects_existing_non_hitl_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "bc.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.create_group("data").zeros("action", shape=(1, 6), dtype=np.float32)
        root.create_group("meta").array(
            "episode_ends", np.asarray([1], dtype=np.int64)
        )

        with pytest.raises(ValueError, match="incompatible Zarr store"):
            HitlZarrWriter(path)


def test_writer_reopens_compatible_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "hitl.zarr"
        first_writer = HitlZarrWriter(path, gamma=0.9, fps=10)
        episode = HitlEpisodeBuffer(gamma=0.9)
        append_transition(episode, 0)
        first_writer.append_episode(episode, success=False, timeout=False)

        second_writer = HitlZarrWriter(path, gamma=0.9, fps=10)

        assert second_writer.n_episodes == 1
