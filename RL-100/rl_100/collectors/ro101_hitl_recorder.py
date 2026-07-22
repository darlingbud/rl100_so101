from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import zarr

from rl_100.common.replay_buffer import ReplayBuffer


OBSERVATION_KEYS = ("image_front", "image_side", "agent_pos")
HITL_FORMAT = "rl100_ro101_hitl_v1"
EPISODE_META_DTYPES = {
    "episode_success": np.bool_,
    "episode_timeout": np.bool_,
    "episode_intervention_steps": np.int64,
}


def discounted_return(reward: np.ndarray, gamma: float) -> np.ndarray:
    reward = np.asarray(reward, dtype=np.float32).reshape(-1, 1)
    result = np.zeros_like(reward)
    running = 0.0
    for index in range(len(reward) - 1, -1, -1):
        running = float(reward[index, 0]) + gamma * running
        result[index, 0] = running
    return result


def _copy_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = set(OBSERVATION_KEYS) - set(observation)
    if missing:
        raise KeyError(f"Missing observation fields: {sorted(missing)}")
    return {
        key: np.asarray(observation[key]).copy()
        for key in OBSERVATION_KEYS
    }


@dataclass
class HitlEpisodeBuffer:
    gamma: float = 0.99
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def append(
        self,
        *,
        observation: dict[str, np.ndarray],
        next_observation: dict[str, np.ndarray],
        policy_action: np.ndarray,
        human_action: np.ndarray,
        executed_action: np.ndarray,
        intervention: bool,
        policy_valid: bool,
        clamped: bool,
        time_ns: int,
    ) -> None:
        actions = {
            "policy_action": np.asarray(policy_action, dtype=np.float32),
            "human_action": np.asarray(human_action, dtype=np.float32),
            "executed_action": np.asarray(executed_action, dtype=np.float32),
        }
        for name, action in actions.items():
            if action.shape != (6,) or not np.isfinite(action).all():
                raise ValueError(f"{name} must be a finite six-dimensional action")
        self.transitions.append(
            {
                "observation": _copy_observation(observation),
                "next_observation": _copy_observation(next_observation),
                **{name: value.copy() for name, value in actions.items()},
                "intervention": bool(intervention),
                "policy_valid": bool(policy_valid),
                "clamped": bool(clamped),
                "time_ns": int(time_ns),
            }
        )

    def clear(self) -> None:
        self.transitions.clear()

    def finalize(self, *, success: bool, timeout: bool = False) -> dict[str, np.ndarray]:
        if not self.transitions:
            raise RuntimeError("Cannot finalize an empty HITL episode")

        count = len(self.transitions)
        reward = np.zeros((count, 1), dtype=np.float32)
        if success:
            reward[-1, 0] = 1.0
        done = np.zeros((count, 1), dtype=np.bool_)
        done[-1, 0] = True
        timeouts = np.zeros((count, 1), dtype=np.bool_)
        timeouts[-1, 0] = bool(timeout)

        observations = [item["observation"] for item in self.transitions]
        next_observations = [item["next_observation"] for item in self.transitions]
        action = np.stack([item["executed_action"] for item in self.transitions])
        next_action = np.concatenate([action[1:], action[-1:]], axis=0)
        data = {
            "state": np.stack([item["agent_pos"] for item in observations]).astype(np.float32),
            "next_state": np.stack(
                [item["agent_pos"] for item in next_observations]
            ).astype(np.float32),
            "full_state": np.stack(
                [item["agent_pos"] for item in observations]
            ).astype(np.float32),
            "action": action.astype(np.float32),
            "next_action": next_action.astype(np.float32),
            "policy_action": np.stack(
                [item["policy_action"] for item in self.transitions]
            ).astype(np.float32),
            "human_action": np.stack(
                [item["human_action"] for item in self.transitions]
            ).astype(np.float32),
            "image_front": np.stack(
                [item["image_front"] for item in observations]
            ).astype(np.uint8),
            "image_side": np.stack(
                [item["image_side"] for item in observations]
            ).astype(np.uint8),
            "next_image_front": np.stack(
                [item["image_front"] for item in next_observations]
            ).astype(np.uint8),
            "next_image_side": np.stack(
                [item["image_side"] for item in next_observations]
            ).astype(np.uint8),
            "reward": reward,
            "return": discounted_return(reward, self.gamma),
            "done": done,
            "timeout": timeouts,
            "intervention": np.asarray(
                [[item["intervention"]] for item in self.transitions], dtype=np.bool_
            ),
            "policy_valid": np.asarray(
                [[item["policy_valid"]] for item in self.transitions], dtype=np.bool_
            ),
            "clamped": np.asarray(
                [[item["clamped"]] for item in self.transitions], dtype=np.bool_
            ),
            "time_ns": np.asarray(
                [[item["time_ns"]] for item in self.transitions], dtype=np.int64
            ),
        }
        return data


class HitlZarrWriter:
    def __init__(self, path: Path | str, *, gamma: float = 0.99, fps: float = 10.0):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        group = zarr.open_group(str(self.path), mode="a")
        existing_format = group.attrs.get("format")
        has_episodes = (
            "meta" in group
            and "episode_ends" in group["meta"]
            and len(group["meta/episode_ends"]) > 0
        )
        if has_episodes and existing_format != HITL_FORMAT:
            raise ValueError(
                f"Refusing to append HITL data to incompatible Zarr store: "
                f"{self.path} (format={existing_format!r})"
            )
        if has_episodes:
            existing_gamma = float(group.attrs.get("gamma", gamma))
            existing_fps = float(group.attrs.get("fps", fps))
            if not np.isclose(existing_gamma, gamma):
                raise ValueError(
                    f"Existing dataset gamma is {existing_gamma}, requested {gamma}"
                )
            if not np.isclose(existing_fps, fps):
                raise ValueError(
                    f"Existing dataset fps is {existing_fps}, requested {fps}"
                )
        self.buffer = ReplayBuffer.create_from_group(group)
        for key in EPISODE_META_DTYPES:
            if has_episodes and key not in self.buffer.meta:
                raise ValueError(f"Existing HITL dataset is missing meta/{key}")
            if key in self.buffer.meta and len(self.buffer.meta[key]) != self.buffer.n_episodes:
                raise ValueError(
                    f"meta/{key} has {len(self.buffer.meta[key])} entries for "
                    f"{self.buffer.n_episodes} episodes"
                )
        group.attrs.update(
            {
                "format": HITL_FORMAT,
                "gamma": float(gamma),
                "fps": float(fps),
                "image_layout": "NHWC",
                "cameras": ["front", "side"],
            }
        )
        self.gamma = float(gamma)

    @property
    def n_episodes(self) -> int:
        return self.buffer.n_episodes

    def append_episode(self, episode: HitlEpisodeBuffer, *, success: bool, timeout: bool) -> int:
        data = episode.finalize(success=success, timeout=timeout)
        image_chunks = {
            key: (1,) + value.shape[1:]
            for key, value in data.items()
            if key.startswith("image_") or key.startswith("next_image_")
        }
        previous_episodes = self.buffer.n_episodes
        try:
            self.buffer.add_episode(
                data,
                chunks=image_chunks,
                compressors=numcodecs.Blosc(
                    "zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
                ),
            )
            self._append_episode_meta("episode_success", bool(success), np.bool_)
            self._append_episode_meta("episode_timeout", bool(timeout), np.bool_)
            intervention_count = int(data["intervention"].sum())
            self._append_episode_meta(
                "episode_intervention_steps", intervention_count, np.int64
            )
        except Exception:
            if self.buffer.n_episodes > previous_episodes:
                self.buffer.drop_episode()
            for key in EPISODE_META_DTYPES:
                if key in self.buffer.meta:
                    self.buffer.meta[key].resize((previous_episodes,))
            raise
        return self.buffer.n_episodes - 1

    def _append_episode_meta(self, key: str, value: Any, dtype: Any) -> None:
        meta = self.buffer.meta
        if key not in meta:
            array = meta.zeros(
                key,
                shape=(0,),
                chunks=(64,),
                dtype=dtype,
                compressor=None,
            )
        else:
            array = meta[key]
        array.resize((len(array) + 1,))
        array[-1] = value
