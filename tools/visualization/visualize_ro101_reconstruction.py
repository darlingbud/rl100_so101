#!/usr/bin/env python3
"""Save RO101 encoder-decoder reconstruction previews from a checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "RL-100"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rl_100.serving.policy_adapter import RL100PolicyAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reconstruction_preview"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="Step relative to the start of the selected episode",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", choices=("auto", "model", "ema_model"), default="auto")
    return parser.parse_args()


def load_history(
    dataset_path: Path,
    episode_index: int,
    step: int,
    n_obs_steps: int,
    image_keys: list[str],
) -> tuple[dict[str, torch.Tensor], list[int]]:
    root = zarr.open(str(dataset_path.expanduser().resolve()), mode="r")
    episode_ends = np.asarray(root["meta/episode_ends"])
    if not 0 <= episode_index < len(episode_ends):
        raise IndexError(f"episode-index must be in [0, {len(episode_ends) - 1}]")
    episode_start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    episode_end = int(episode_ends[episode_index])
    episode_length = episode_end - episode_start
    if not 0 <= step < episode_length:
        raise IndexError(f"step must be in [0, {episode_length - 1}] for episode {episode_index}")

    absolute_step = episode_start + step
    indices = np.arange(absolute_step - n_obs_steps + 1, absolute_step + 1)
    indices = np.clip(indices, episode_start, episode_end - 1)
    observations: dict[str, torch.Tensor] = {
        "agent_pos": torch.from_numpy(
            np.asarray(root["data/state"].oindex[indices], dtype=np.float32)
        ).unsqueeze(0)
    }
    for key in image_keys:
        images = np.asarray(root[f"data/{key}"].oindex[indices], dtype=np.uint8)
        observations[key] = torch.from_numpy(np.moveaxis(images, -1, -3)).unsqueeze(0)
    return observations, indices.tolist()


def capture_reconstructions(
    policy: torch.nn.Module,
    observations: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    encoder = policy.obs_encoder
    if not getattr(encoder, "use_recon", False):
        raise RuntimeError("The checkpoint observation encoder has use_recon=false")

    device = policy.device
    observations = {key: value.to(device) for key, value in observations.items()}
    normalized = policy.normalizer.normalize(observations)
    flat = {
        key: value.reshape(-1, *value.shape[2:])
        for key, value in normalized.items()
    }

    captured: dict[str, torch.Tensor] = {}
    handles = []
    if encoder.share_rgb_model:
        ordered_outputs: list[torch.Tensor] = []

        def shared_hook(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            ordered_outputs.append(output.detach())

        handles.append(encoder.decoder.register_forward_hook(shared_hook))
    else:
        for key in encoder.rgb_keys:
            def hook(
                _module: torch.nn.Module,
                _inputs: Any,
                output: torch.Tensor,
                *,
                image_key: str = key,
            ) -> None:
                captured[image_key] = output.detach()

            handles.append(encoder.decoders[key].register_forward_hook(hook))

    try:
        with torch.inference_mode():
            _aux_loss, loss_items, _features = encoder.Recon_VIB_loss(
                flat, deterministic=True
            )
    finally:
        for handle in handles:
            handle.remove()

    if encoder.share_rgb_model:
        if len(ordered_outputs) != len(encoder.rgb_keys):
            raise RuntimeError(
                f"Expected {len(encoder.rgb_keys)} decoder outputs, got {len(ordered_outputs)}"
            )
        captured = dict(zip(encoder.rgb_keys, ordered_outputs))

    targets = {
        key: encoder._apply_spatial_transform(key, flat[key], deterministic=True).detach()
        for key in encoder.rgb_keys
    }
    scalar_losses = {
        key: float(value)
        for key, value in loss_items.items()
        if not key.startswith("_")
    }
    return targets, captured, scalar_losses


def tensor_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255).astype(np.uint8))


def labeled(image: Image.Image, label: str, label_height: int = 24) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((6, 6), label, fill="black")
    return canvas


def save_results(
    output_dir: Path,
    targets: dict[str, torch.Tensor],
    reconstructions: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"metadata": metadata, "frames": {}}
    camera_montages = []
    all_differences = []

    for key in sorted(targets):
        target_batch = targets[key].cpu()
        recon_batch = reconstructions[key].cpu().clamp(0, 1)
        batch_difference = (recon_batch - target_batch).abs()
        all_differences.append(batch_difference)
        rows = []
        camera_metrics = []
        for frame_index, (target, reconstruction) in enumerate(zip(target_batch, recon_batch)):
            difference = (reconstruction - target).abs()
            mse = float((difference.square()).mean())
            mae = float(difference.mean())
            psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
            camera_metrics.append({"frame": frame_index, "mse": mse, "mae": mae, "psnr": psnr})

            target_image = tensor_image(target)
            reconstruction_image = tensor_image(reconstruction)
            error_image = tensor_image((difference * 4.0).clamp(0, 1))
            prefix = output_dir / f"{key}_t{frame_index:02d}"
            target_image.save(prefix.with_name(prefix.name + "_target.png"))
            reconstruction_image.save(prefix.with_name(prefix.name + "_reconstruction.png"))
            error_image.save(prefix.with_name(prefix.name + "_error_x4.png"))

            tiles = [
                labeled(target_image, f"{key} t={frame_index} target"),
                labeled(reconstruction_image, f"reconstruction  PSNR={psnr:.2f}dB"),
                labeled(error_image, f"absolute error x4  MAE={mae:.4f}"),
            ]
            row = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
            x = 0
            for tile in tiles:
                row.paste(tile, (x, 0))
                x += tile.width
            rows.append(row)

        montage = Image.new(
            "RGB",
            (max(row.width for row in rows), sum(row.height for row in rows)),
            "white",
        )
        y = 0
        for row in rows:
            montage.paste(row, (0, y))
            y += row.height
        montage.save(output_dir / f"{key}_comparison.png")
        camera_montages.append((key, montage))
        metrics["frames"][key] = camera_metrics
        camera_mse = float(batch_difference.square().mean())
        metrics.setdefault("summary", {})[key] = {
            "mse": camera_mse,
            "mae": float(batch_difference.mean()),
            "psnr": float("inf") if camera_mse == 0 else -10.0 * math.log10(camera_mse),
        }

    total_difference = torch.cat([difference.flatten() for difference in all_differences])
    total_mse = float(total_difference.square().mean())
    metrics["summary"]["all_cameras"] = {
        "mse": total_mse,
        "mae": float(total_difference.mean()),
        "psnr": float("inf") if total_mse == 0 else -10.0 * math.log10(total_mse),
    }

    combined = Image.new(
        "RGB",
        (
            max(montage.width for _, montage in camera_montages),
            sum(montage.height + 24 for _, montage in camera_montages),
        ),
        "white",
    )
    y = 0
    for key, montage in camera_montages:
        section = labeled(montage, key)
        combined.paste(section, (0, y))
        y += section.height
    combined.save(output_dir / "all_cameras_comparison.png")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    adapter = RL100PolicyAdapter.from_checkpoint(
        args.checkpoint,
        config=args.config,
        device=args.device,
        weights=args.weights,
    )
    policy = adapter._policy
    image_keys = sorted(
        key
        for key, spec in adapter.metadata["observation_spec"].items()
        if spec["type"] == "rgb"
    )
    n_obs_steps = int(adapter.metadata["n_obs_steps"])
    observations, indices = load_history(
        args.dataset,
        args.episode_index,
        args.step,
        n_obs_steps,
        image_keys,
    )
    targets, reconstructions, weighted_losses = capture_reconstructions(policy, observations)
    output_dir = args.output_dir.expanduser().resolve()
    save_results(
        output_dir,
        targets,
        reconstructions,
        {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "dataset": str(args.dataset.expanduser().resolve()),
            "episode_index": args.episode_index,
            "step": args.step,
            "absolute_indices": indices,
            "n_obs_steps": n_obs_steps,
            "weights_source": adapter.metadata["weights_source"],
            "training_weighted_losses": weighted_losses,
        },
    )
    print(f"Saved reconstruction preview to {output_dir}")
    print(f"Open {output_dir / 'all_cameras_comparison.png'}")


if __name__ == "__main__":
    main()
