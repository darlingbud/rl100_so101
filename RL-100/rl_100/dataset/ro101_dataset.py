from __future__ import annotations

import copy
from typing import Dict, Sequence

import numpy as np
import torch
import zarr

from rl_100.common.pytorch_util import dict_apply
from rl_100.common.replay_buffer import ReplayBuffer
from rl_100.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.model.common.normalizer import LinearNormalizer


class RO101Dataset(BaseDataset):
    """Disk-backed two-camera dataset for converted LeRobot RO101 data."""

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        max_sequences: int | None = None,
        validation_same_as_train: bool = False,
        image_keys: Sequence[str] = ("image_front", "image_side"),
        pre_image_norm: bool = False,
    ) -> None:
        super().__init__()
        del pre_image_norm

        group = zarr.open(zarr_path, mode="r")
        self.image_keys = tuple(image_keys)
        required_keys = ("state", "action", *self.image_keys)
        missing = [key for key in required_keys if key not in group["data"]]
        if missing:
            raise KeyError(f"Missing RO101 dataset fields: {', '.join(missing)}")

        # Keep Zarr arrays disk-backed. Copying these RGB arrays would require
        # roughly 23 GB for the current dataset.
        root = {
            "data": {key: group["data"][key] for key in required_keys},
            "meta": {"episode_ends": group["meta"]["episode_ends"]},
        }
        self.replay_buffer = ReplayBuffer(root=root)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        self.sampler = self._make_sampler(train_mask, horizon, pad_before, pad_after)
        if max_sequences is not None:
            max_sequences = int(max_sequences)
            if max_sequences < 1:
                raise ValueError("max_sequences must be at least 1")
            if len(self.sampler) > max_sequences:
                rng = np.random.default_rng(seed=seed)
                selected = rng.choice(
                    len(self.sampler), size=max_sequences, replace=False
                )
                self.sampler.indices = self.sampler.indices[np.sort(selected)]
        self.train_mask = train_mask
        self.validation_same_as_train = validation_same_as_train
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _make_sampler(self, episode_mask, horizon, pad_before, pad_after):
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=("state", "action", *self.image_keys),
            episode_mask=episode_mask,
        )

    def get_validation_dataset(self) -> "RO101Dataset":
        val_set = copy.copy(self)
        if self.validation_same_as_train:
            # The sampler is read-only during training, so sharing it gives an
            # exact train/validation comparison for overfit diagnostics.
            val_set.sampler = self.sampler
            return val_set
        val_set.sampler = self._make_sampler(
            ~self.train_mask,
            self.horizon,
            self.pad_before,
            self.pad_after,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "action": self.replay_buffer["action"],
                "agent_pos": self.replay_buffer["state"],
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer

    def get_shape_info(self, n_action_steps: int, n_obs_steps: int) -> dict:
        obs = {
            "agent_pos": (n_obs_steps, self.replay_buffer["state"].shape[-1]),
        }
        for key in self.image_keys:
            _, height, width, channels = self.replay_buffer[key].shape
            obs[key] = (n_obs_steps, channels, height, width)
        return {
            "obs": obs,
            "action": (n_action_steps, self.replay_buffer["action"].shape[-1]),
        }

    def get_length(self) -> int:
        return len(self.sampler)

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        obs = {
            "agent_pos": sample["state"].astype(np.float32),
        }
        for key in self.image_keys:
            # Preserve uint8 until the encoder resizes and normalizes the image.
            obs[key] = np.moveaxis(sample[key], -1, -3)
        data = {
            "obs": obs,
            "action": sample["action"].astype(np.float32),
        }
        return dict_apply(data, torch.from_numpy)
