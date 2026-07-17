"""Dataset action source compatible with the RL-100 policy wire protocol."""

from __future__ import annotations

import pathlib
import time
from typing import Any, Mapping

import numpy as np
import zarr

from rl_100.serving.protocol import MESSAGE_METADATA, PROTOCOL_VERSION, ProtocolError


class DatasetActionAdapter:
    """Returns recorded actions or a current-position hold action."""

    def __init__(
        self,
        dataset: str | pathlib.Path,
        *,
        mode: str = "replay",
        episode_index: int = 0,
        start_step: int = 0,
        action_horizon: int = 4,
        n_obs_steps: int = 2,
        loop: bool = False,
    ) -> None:
        if mode not in ("replay", "hold"):
            raise ValueError("mode must be one of: replay, hold")
        if action_horizon <= 0 or n_obs_steps <= 0:
            raise ValueError("action_horizon and n_obs_steps must be positive")

        self.dataset_path = pathlib.Path(dataset).expanduser().resolve()
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {self.dataset_path}")
        self.root = zarr.open(str(self.dataset_path), mode="r")
        try:
            self.actions = self.root["data/action"]
            self.states = self.root["data/state"]
            self.episode_ends = np.asarray(self.root["meta/episode_ends"][:], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"Dataset is missing required field: {exc}") from exc

        if self.actions.ndim != 2 or self.states.shape != self.actions.shape:
            raise ValueError("data/action and data/state must have matching [N,D] shapes")
        if len(self.episode_ends) == 0 or self.episode_ends[-1] != len(self.actions):
            raise ValueError("meta/episode_ends does not match action length")
        if not 0 <= episode_index < len(self.episode_ends):
            raise ValueError(
                f"episode_index must be in [0, {len(self.episode_ends) - 1}]"
            )

        episode_start = 0 if episode_index == 0 else int(self.episode_ends[episode_index - 1])
        episode_end = int(self.episode_ends[episode_index])
        episode_length = episode_end - episode_start
        if not 0 <= start_step < episode_length:
            raise ValueError(f"start_step must be in [0, {episode_length - 1}]")

        self.mode = mode
        self.episode_index = episode_index
        self.episode_start = episode_start
        self.episode_end = episode_end
        self.episode_length = episode_length
        self.start_step = start_step
        self.action_horizon = action_horizon
        self.n_obs_steps = n_obs_steps
        self.action_dim = int(self.actions.shape[1])
        self.loop = loop
        self._wire_episode_id: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        image_front_shape = self._image_shape("image_front")
        image_side_shape = self._image_shape("image_side")
        return {
            "message_type": MESSAGE_METADATA,
            "protocol_version": PROTOCOL_VERSION,
            "server_name": "rl100-dataset-action-server",
            "policy_name": f"dataset-{self.mode}",
            "task_name": "ro101_dataset_replay",
            "weights_source": "dataset",
            "n_obs_steps": self.n_obs_steps,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "observation_spec": {
                "image_front": self._image_spec(image_front_shape),
                "image_side": self._image_spec(image_side_shape),
                "agent_pos": {
                    "shape": [self.action_dim],
                    "policy_input_shape": [self.n_obs_steps, self.action_dim],
                    "dtype": "float32",
                    "type": "low_dim",
                },
            },
            "action_spec": {
                "shape": [self.action_horizon, self.action_dim],
                "single_action_shape": [self.action_dim],
                "dtype": "float32",
            },
            "replay": {
                "mode": self.mode,
                "dataset": str(self.dataset_path),
                "dataset_fps": int(self.root.attrs.get("fps", 0)),
                "episode_index": self.episode_index,
                "episode_length": self.episode_length,
                "start_step": self.start_step,
                "loop": self.loop,
            },
        }

    def _image_shape(self, key: str) -> tuple[int, int, int]:
        try:
            array = self.root[f"data/{key}"]
        except KeyError as exc:
            raise ValueError(f"Dataset is missing data/{key}") from exc
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(f"data/{key} must have shape [N,H,W,3]")
        return 3, int(array.shape[1]), int(array.shape[2])

    def _image_spec(self, shape: tuple[int, int, int]) -> dict[str, Any]:
        return {
            "shape": list(shape),
            "policy_input_shape": [self.n_obs_steps, *shape],
            "dtype": "uint8",
            "type": "rgb",
            "layout": "TCHW",
            "value_range": [0, 255],
        }

    def reset(self, episode_id: str | None = None) -> None:
        self._wire_episode_id = episode_id

    def infer(
        self,
        observation: Mapping[str, Any],
        *,
        step_id: int | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        if step_id is None or step_id < 0:
            raise ProtocolError("BAD_MESSAGE", "step_id must be non-negative")
        if not isinstance(observation, Mapping):
            raise ProtocolError("INVALID_OBSERVATION", "observation must be a map")
        current_state = self._current_state(observation)
        preprocess_ms = (time.monotonic() - started) * 1000

        action_started = time.monotonic()
        if self.mode == "hold":
            actions = np.repeat(current_state[None], self.action_horizon, axis=0)
            source_step = int(step_id)
            done = False
        else:
            actions, source_step, done = self._replay_chunk(int(step_id))
        action_ms = (time.monotonic() - action_started) * 1000

        return {
            "actions": np.ascontiguousarray(actions, dtype=np.float32),
            "replay": {
                "mode": self.mode,
                "episode_index": self.episode_index,
                "source_step": source_step,
                "done": done,
                "wire_episode_id": episode_id or self._wire_episode_id,
            },
            "timing": {
                "preprocess_ms": preprocess_ms,
                "policy_ms": action_ms,
                "postprocess_ms": 0.0,
            },
        }

    def _current_state(self, observation: Mapping[str, Any]) -> np.ndarray:
        if "agent_pos" not in observation:
            raise ProtocolError("INVALID_OBSERVATION", "Missing observation key: agent_pos")
        state = np.asarray(observation["agent_pos"], dtype=np.float32)
        expected_shape = (self.n_obs_steps, self.action_dim)
        if state.shape != expected_shape:
            raise ProtocolError(
                "INVALID_OBSERVATION",
                f"agent_pos shape must be {expected_shape}, received {state.shape}",
            )
        if not np.isfinite(state).all():
            raise ProtocolError("INVALID_OBSERVATION", "agent_pos contains NaN or Inf")
        return np.array(state[-1], dtype=np.float32, copy=True)

    def _replay_chunk(self, client_step: int) -> tuple[np.ndarray, int, bool]:
        relative_step = self.start_step + client_step
        available_length = self.episode_length - self.start_step
        if self.loop:
            relative_step = self.start_step + (client_step % available_length)
        done = relative_step >= self.episode_length
        if done:
            relative_step = self.episode_length - 1

        relative_indices = np.arange(relative_step, relative_step + self.action_horizon)
        if self.loop:
            relative_indices = self.start_step + (
                (relative_indices - self.start_step) % available_length
            )
        else:
            relative_indices = np.minimum(relative_indices, self.episode_length - 1)
            done = done or relative_step + self.action_horizon >= self.episode_length
        absolute_indices = self.episode_start + relative_indices
        return np.asarray(self.actions.get_orthogonal_selection((absolute_indices, slice(None)))), int(relative_step), bool(done)
