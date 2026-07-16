"""Adapter from the RL-100 policy API to the lightweight wire protocol."""

from __future__ import annotations

import logging
import math
import pathlib
import threading
import time
from typing import Any, Mapping

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from rl_100.serving.protocol import MESSAGE_METADATA, PROTOCOL_VERSION, ProtocolError

logger = logging.getLogger(__name__)


def _config_value(cfg: Any, path: str, default: Any = None) -> Any:
    if OmegaConf.is_config(cfg):
        value = OmegaConf.select(cfg, path, default=default)
        return default if value is None else value
    value = cfg
    for part in path.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return default
            value = value[part]
        else:
            value = getattr(value, part, default)
    return value


def _plain_shape_meta(cfg: Any) -> dict[str, Any]:
    shape_meta = _config_value(cfg, "shape_meta")
    if shape_meta is None:
        raise ValueError("Checkpoint config does not define shape_meta")
    if OmegaConf.is_config(shape_meta):
        return OmegaConf.to_container(shape_meta, resolve=True)
    return dict(shape_meta)


def _remove_ddp_segments(key: str) -> str:
    if key.startswith("module."):
        key = key[len("module.") :]
    return key.replace(".module.", ".")


def _validate_external_config(checkpoint_cfg: Any, external_cfg: Any) -> None:
    paths = ("policy", "shape_meta", "n_obs_steps", "n_action_steps", "horizon")
    mismatches = []
    for path in paths:
        checkpoint_value = _config_value(checkpoint_cfg, path)
        external_value = _config_value(external_cfg, path)
        if OmegaConf.is_config(checkpoint_value):
            checkpoint_value = OmegaConf.to_container(checkpoint_value, resolve=True)
        if OmegaConf.is_config(external_value):
            external_value = OmegaConf.to_container(external_value, resolve=True)
        if checkpoint_value != external_value:
            mismatches.append(path)
    if mismatches:
        raise ValueError(
            "External config does not match checkpoint for: "
            + ", ".join(mismatches)
        )


class RL100PolicyAdapter:
    """Loads an RL-100 checkpoint and exposes NumPy observations and actions."""

    def __init__(
        self,
        policy: torch.nn.Module,
        cfg: Any,
        *,
        device: str | torch.device,
        weights_source: str,
        deterministic: bool = True,
        use_cm: bool | None = None,
        distill2mean: bool = False,
    ) -> None:
        self._policy = policy
        self._cfg = cfg
        self._device = torch.device(device)
        self._weights_source = weights_source
        self._deterministic = deterministic
        self._distill2mean = distill2mean
        self._lock = threading.Lock()

        configured_use_cm = bool(_config_value(cfg, "policy.use_cm", False))
        self._use_cm = configured_use_cm if use_cm is None else use_cm

        shape_meta = _plain_shape_meta(cfg)
        self._observation_meta = dict(shape_meta["obs"])
        self._action_shape = tuple(int(x) for x in shape_meta["action"]["shape"])
        self._n_obs_steps = int(_config_value(cfg, "n_obs_steps"))
        self._action_horizon = int(
            getattr(policy, "n_action_steps", _config_value(cfg, "n_action_steps"))
        )
        self._action_dim = int(
            getattr(policy, "action_dim", math.prod(self._action_shape))
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | pathlib.Path,
        *,
        config: str | pathlib.Path | None = None,
        device: str = "cuda:0",
        weights: str = "auto",
        strict: bool = True,
        deterministic: bool = True,
        use_cm: bool | None = None,
        distill2mean: bool = False,
    ) -> "RL100PolicyAdapter":
        checkpoint = pathlib.Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

        logger.info("Loading RL-100 checkpoint from %s", checkpoint)
        with checkpoint.open("rb") as checkpoint_file:
            payload = torch.load(
                checkpoint_file, pickle_module=dill, map_location="cpu"
            )
        if not isinstance(payload, dict) or "cfg" not in payload or "state_dicts" not in payload:
            raise ValueError(
                "Expected an RL-100 workspace .ckpt containing cfg and state_dicts"
            )

        checkpoint_cfg = payload["cfg"]
        if not OmegaConf.is_config(checkpoint_cfg):
            checkpoint_cfg = OmegaConf.create(checkpoint_cfg)
        cfg = checkpoint_cfg
        if config is not None:
            config = pathlib.Path(config).expanduser().resolve()
            if not config.is_file():
                raise FileNotFoundError(f"Config does not exist: {config}")
            external_cfg = OmegaConf.load(config)
            _validate_external_config(checkpoint_cfg, external_cfg)
            cfg = external_cfg
            logger.info("Using validated training config from %s", config)
        policy = hydra.utils.instantiate(cfg.policy)

        state_dicts = payload["state_dicts"]
        if weights == "auto":
            weights = "ema_model" if "ema_model" in state_dicts else "model"
        if weights not in ("model", "ema_model"):
            raise ValueError("weights must be one of: auto, model, ema_model")
        if weights not in state_dicts:
            available = ", ".join(sorted(state_dicts))
            raise KeyError(
                f"Checkpoint has no state_dict {weights!r}. Available keys: {available}"
            )

        state_dict = state_dicts[weights]
        has_dynamic_weights = any(
            _remove_ddp_segments(key).startswith(
                ("distilled_model.", "target_model.", "teacher.")
            )
            for key in state_dict
        )
        if has_dynamic_weights and not hasattr(policy, "distilled_model"):
            if not hasattr(policy, "set_target"):
                raise ValueError(
                    "Checkpoint contains distilled policy modules, but the policy "
                    "does not provide set_target()"
                )
            policy.set_target()
        try:
            policy.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            normalized = {
                _remove_ddp_segments(key): value for key, value in state_dict.items()
            }
            if list(normalized) == list(state_dict):
                raise
            logger.warning("Retrying checkpoint load after removing DDP module segments")
            policy.load_state_dict(normalized, strict=strict)

        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {target_device} was requested, but CUDA is unavailable"
            )
        policy.to(target_device)
        policy.eval()

        return cls(
            policy,
            cfg,
            device=target_device,
            weights_source=weights,
            deterministic=deterministic,
            use_cm=use_cm,
            distill2mean=distill2mean,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        observation_spec = {}
        for key, spec in self._observation_meta.items():
            raw_shape = [int(x) for x in spec["shape"]]
            observation_type = spec.get("type", "unknown")
            wire_dtype = "uint8" if observation_type == "rgb" else "float32"
            wire_spec = {
                "shape": raw_shape,
                "policy_input_shape": [self._n_obs_steps, *raw_shape],
                "dtype": wire_dtype,
                "type": observation_type,
            }
            if observation_type == "rgb":
                wire_spec.update(
                    {
                        "layout": "TCHW",
                        "value_range": [0, 255],
                    }
                )
            observation_spec[key] = wire_spec
        return {
            "message_type": MESSAGE_METADATA,
            "protocol_version": PROTOCOL_VERSION,
            "server_name": "rl100-policy-server",
            "policy_name": str(_config_value(self._cfg, "name", "rl100")),
            "task_name": str(_config_value(self._cfg, "task_name", "unknown")),
            "weights_source": self._weights_source,
            "n_obs_steps": self._n_obs_steps,
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
            "observation_spec": observation_spec,
            "action_spec": {
                "shape": [self._action_horizon, self._action_dim],
                "single_action_shape": list(self._action_shape),
                "dtype": "float32",
            },
            "inference": {
                "deterministic": self._deterministic,
                "use_cm": self._use_cm,
                "distill2mean": self._distill2mean,
            },
        }

    def reset(self, episode_id: str | None = None) -> None:
        del episode_id
        with self._lock:
            self._policy.reset()

    def _prepare_observation(
        self, observation: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        if not isinstance(observation, Mapping):
            raise ProtocolError("INVALID_OBSERVATION", "observation must be a map")
        expected_keys = set(self._observation_meta)
        missing = sorted(expected_keys - set(observation))
        if missing:
            raise ProtocolError(
                "INVALID_OBSERVATION",
                f"Missing observation keys: {', '.join(missing)}",
            )

        tensors = {}
        for key, spec in self._observation_meta.items():
            value = observation[key]
            if isinstance(value, torch.Tensor):
                raise ProtocolError(
                    "INVALID_OBSERVATION",
                    f"Observation {key!r} must be a NumPy array, not a torch.Tensor",
                )
            try:
                array = np.asarray(value)
            except (TypeError, ValueError) as exc:
                raise ProtocolError(
                    "INVALID_OBSERVATION",
                    f"Observation {key!r} cannot be converted to a NumPy array",
                ) from exc

            expected_shape = (
                self._n_obs_steps,
                *tuple(int(x) for x in spec["shape"]),
            )
            if array.shape != expected_shape:
                raise ProtocolError(
                    "INVALID_OBSERVATION",
                    f"Observation {key!r} shape must be {expected_shape}, "
                    f"received {array.shape}",
                )
            if array.dtype.kind not in ("b", "i", "u", "f"):
                raise ProtocolError(
                    "INVALID_OBSERVATION",
                    f"Observation {key!r} has unsupported dtype {array.dtype}",
                )
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ProtocolError(
                    "INVALID_OBSERVATION",
                    f"Observation {key!r} contains NaN or Inf",
                )

            # MessagePack decodes ndarray payloads from an immutable byte buffer.
            # Own the memory before passing it to torch.from_numpy.
            array = np.array(array, dtype=np.float32, order="C", copy=True)
            tensors[key] = torch.from_numpy(array).unsqueeze(0).to(self._device)
        return tensors

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        preprocess_start = time.monotonic()
        obs_tensors = self._prepare_observation(observation)
        preprocess_ms = (time.monotonic() - preprocess_start) * 1000

        with self._lock, torch.inference_mode():
            policy_start = time.monotonic()
            result = self._policy.predict_action(
                obs_tensors,
                deterministic=self._deterministic,
                use_cm=self._use_cm,
                distill2mean=self._distill2mean,
            )
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            policy_ms = (time.monotonic() - policy_start) * 1000

        postprocess_start = time.monotonic()
        if not isinstance(result, Mapping) or "action" not in result:
            raise ProtocolError(
                "INFERENCE_FAILED", "RL-100 policy result does not contain 'action'"
            )
        actions = result["action"]
        if not isinstance(actions, torch.Tensor):
            raise ProtocolError(
                "INFERENCE_FAILED", "RL-100 policy 'action' is not a torch.Tensor"
            )
        actions = actions.detach().to("cpu", dtype=torch.float32).numpy()
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ProtocolError(
                "INFERENCE_FAILED",
                f"Expected action shape [1,H,D], received {actions.shape}",
            )
        actions = np.ascontiguousarray(actions[0])
        expected_shape = (self._action_horizon, self._action_dim)
        if actions.shape != expected_shape:
            raise ProtocolError(
                "INFERENCE_FAILED",
                f"Expected action shape {expected_shape}, received {actions.shape}",
            )
        if not np.isfinite(actions).all():
            raise ProtocolError("INFERENCE_FAILED", "Policy action contains NaN or Inf")
        postprocess_ms = (time.monotonic() - postprocess_start) * 1000

        return {
            "actions": actions,
            "timing": {
                "preprocess_ms": preprocess_ms,
                "policy_ms": policy_ms,
                "postprocess_ms": postprocess_ms,
            },
        }
