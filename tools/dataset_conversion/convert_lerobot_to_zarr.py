#!/usr/bin/env python3
"""Convert LeRobot v3 datasets to an RL-100 policy-training Zarr v2 store.

RGB remains uint8 NHWC at the source resolution. All episodes are concatenated;
``meta/episode_ends`` retains their boundaries. Duplicate next-image arrays are
intentionally omitted.

When launching through Conda, use ``conda run --no-capture-output`` so progress
is forwarded to the terminal immediately.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


@dataclass(frozen=True)
class Interval:
    video: Path
    src: int
    length: int
    dst: int


@dataclass
class Plan:
    root: Path
    fps: int
    height: int
    width: int
    state: np.ndarray
    action: np.ndarray
    lengths: list[int]
    local_ends: list[int]
    videos: dict[str, list[Interval]]

    @property
    def frames(self) -> int:
        return self.state.shape[0]


def enable_live_output() -> None:
    """Flush status lines immediately when the parent process does not capture them."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True, write_through=True)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--datasets", nargs="*", help="Optional child dataset names")
    p.add_argument("--cameras", nargs="+", default=["front", "side"])
    p.add_argument("--image-chunk", type=int, default=8)
    p.add_argument("--lowdim-chunk", type=int, default=2048)
    p.add_argument("--scalar-chunk", type=int, default=4096)
    p.add_argument("--write-batch", type=int, default=32)
    p.add_argument("--compression-level", type=int, default=3)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-camera time-step progress bars.",
    )
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def roots(input_path: Path, selected: list[str] | None) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if (input_path / "meta/info.json").is_file():
        found = [input_path]
    else:
        found = sorted(
            x for x in input_path.iterdir()
            if x.is_dir() and (x / "meta/info.json").is_file()
        )
    if selected:
        wanted = set(selected)
        found = [x for x in found if x.name in wanted]
        missing = wanted - {x.name for x in found}
        if missing:
            raise ValueError(f"datasets not found: {sorted(missing)}")
    if not found:
        raise ValueError(f"no LeRobot roots found under {input_path}")
    return found


def parquet_tree(root: Path, pattern: str) -> pa.Table:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(root / pattern)
    return pa.concat_tables(
        [pq.read_table(x) for x in files], promote_options="default"
    ).combine_chunks()


def matrix(table: pa.Table, key: str) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(key)
    value = np.asarray(table[key].to_pylist(), dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"invalid {key}: shape={value.shape}")
    return value


def make_plan(root: Path, cameras: list[str], dst_offset: int) -> Plan:
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{root}: only LeRobot v3.0 is supported")
    fps = int(info["fps"])
    data = parquet_tree(root, "data/**/*.parquet")
    if "index" in data.column_names:
        data = data.sort_by([("index", "ascending")])
    episodes = parquet_tree(root, "meta/episodes/**/*.parquet").sort_by(
        [("episode_index", "ascending")]
    )
    state, action = matrix(data, "observation.state"), matrix(data, "action")
    lengths = [int(x) for x in episodes["length"].to_pylist()]
    episode_ids = [int(x) for x in episodes["episode_index"].to_pylist()]
    if state.shape[0] != action.shape[0] or sum(lengths) != state.shape[0]:
        raise ValueError(f"{root}: state/action/episode length mismatch")
    if state.shape[0] != int(info["total_frames"]):
        raise ValueError(f"{root}: total_frames mismatch")

    row_eps = np.asarray(data["episode_index"].to_pylist())
    row_frames = np.asarray(data["frame_index"].to_pylist())
    cursor, local_ends, output_starts = 0, [], []
    for episode_id, length in zip(episode_ids, lengths, strict=True):
        output_starts.append(dst_offset + cursor)
        sl = slice(cursor, cursor + length)
        if not np.all(row_eps[sl] == episode_id):
            raise ValueError(f"{root}: episode {episode_id} rows are not contiguous")
        if not np.array_equal(row_frames[sl], np.arange(length)):
            raise ValueError(f"{root}: episode {episode_id} frame_index is invalid")
        cursor += length
        local_ends.append(cursor)

    resolution: tuple[int, int] | None = None
    videos: dict[str, list[Interval]] = {}
    template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    for camera in cameras:
        key = f"observation.images.{camera}"
        feature = info["features"].get(key)
        if feature is None:
            raise KeyError(f"{root}: {key}")
        height, width, channels = map(int, feature["shape"])
        if channels != 3:
            raise ValueError(f"{root}: {key} is not RGB")
        resolution = resolution or (height, width)
        if resolution != (height, width):
            raise ValueError(f"{root}: camera resolutions differ")
        prefix = f"videos/{key}"
        chunks = episodes[f"{prefix}/chunk_index"].to_pylist()
        files = episodes[f"{prefix}/file_index"].to_pylist()
        starts = episodes[f"{prefix}/from_timestamp"].to_pylist()
        ends = episodes[f"{prefix}/to_timestamp"].to_pylist()
        items = []
        for i, length in enumerate(lengths):
            src = round(float(starts[i]) * fps)
            limit = round(float(ends[i]) * fps)
            if src + length > limit:
                raise ValueError(f"{root}: {key} episode {episode_ids[i]} range is too short")
            relative = template.format(
                video_key=key,
                chunk_index=int(chunks[i]),
                file_index=int(files[i]),
            )
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            items.append(Interval(path, src, length, output_starts[i]))
        videos[camera] = items
    assert resolution is not None
    return Plan(root, fps, *resolution, state, action, lengths, local_ends, videos)


def plans(dataset_roots: list[Path], cameras: list[str]) -> list[Plan]:
    result, offset = [], 0
    for root in dataset_roots:
        item = make_plan(root, cameras, offset)
        result.append(item)
        offset += item.frames
    first = result[0]
    for item in result[1:]:
        if (item.fps, item.height, item.width) != (first.fps, first.height, first.width):
            raise ValueError("datasets have different fps or image resolutions")
        if item.state.shape[1:] != first.state.shape[1:]:
            raise ValueError("datasets have different state shapes")
        if item.action.shape[1:] != first.action.shape[1:]:
            raise ValueError("datasets have different action shapes")
    return result


def next_values(value: np.ndarray, episode_ends: np.ndarray) -> np.ndarray:
    result, start = value.copy(), 0
    for end in episode_ends:
        end = int(end)
        result[start : end - 1] = value[start + 1 : end]
        result[end - 1] = value[end - 1]
        start = end
    return result


def write_frames(
    target: Any,
    intervals: list[Interval],
    height: int,
    width: int,
    batch: int,
    progress: tqdm | None = None,
) -> None:
    grouped: dict[Path, list[Interval]] = defaultdict(list)
    for interval in intervals:
        grouped[interval.video].append(interval)
    for video, items in grouped.items():
        wanted: dict[int, int] = {}
        for item in items:
            for delta in range(item.length):
                if item.src + delta in wanted:
                    raise ValueError(f"overlapping video ranges in {video}")
                wanted[item.src + delta] = item.dst + delta
        frames: list[np.ndarray] = []
        write_start: int | None = None
        written = 0
        with av.open(str(video)) as container:
            for source_index, frame in enumerate(container.decode(video=0)):
                destination = wanted.get(source_index)
                if destination is None:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                if rgb.shape != (height, width, 3):
                    raise ValueError(f"{video}: unexpected frame shape {rgb.shape}")
                if write_start is None:
                    write_start = destination
                if destination != write_start + len(frames):
                    target[write_start : write_start + len(frames)] = np.stack(frames)
                    frames, write_start = [], destination
                frames.append(rgb)
                written += 1
                if progress is not None:
                    progress.update(1)
                if len(frames) >= batch:
                    target[write_start : write_start + len(frames)] = np.stack(frames)
                    frames, write_start = [], None
        if frames:
            assert write_start is not None
            target[write_start : write_start + len(frames)] = np.stack(frames)
        if written != len(wanted):
            raise ValueError(f"{video}: decoded {written}/{len(wanted)} requested frames")
        if progress is None:
            print(f"    {video}: {written} frames")
        else:
            progress.set_postfix_str(video.parent.parent.name, refresh=False)


def convert(args: argparse.Namespace, items: list[Plan]) -> None:
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise RuntimeError(
            "install tools/dataset_conversion/requirements-lerobot-zarr.txt"
        ) from exc
    if int(zarr.__version__.split(".")[0]) >= 3:
        raise RuntimeError(f"Zarr v2 required, found {zarr.__version__}")
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; use --overwrite")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = sum(x.frames for x in items)
    first = items[0]
    state = np.concatenate([x.state for x in items])
    action = np.concatenate([x.action for x in items])
    ends, offset = [], 0
    for item in items:
        ends.extend(offset + end for end in item.local_ends)
        offset += item.frames
    episode_ends = np.asarray(ends, dtype=np.int64)
    compressor = Blosc("zstd", args.compression_level, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(str(output), mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    lc, sc, ic = min(args.lowdim_chunk, total), min(args.scalar_chunk, total), min(args.image_chunk, total)
    for name, value in {
        "state": state,
        "action": action,
        "next_state": next_values(state, episode_ends),
        "next_action": next_values(action, episode_ends),
        "full_state": state,
    }.items():
        data.create_dataset(name, data=value, chunks=(lc, value.shape[1]), compressor=compressor)
    zeros = np.zeros((total, 1), np.float32)
    done = np.zeros((total, 1), np.bool_)
    done[episode_ends - 1] = True
    data.create_dataset("reward", data=zeros, chunks=(sc, 1), compressor=compressor)
    data.create_dataset("return", data=zeros, chunks=(sc, 1), compressor=compressor)
    data.create_dataset("done", data=done, chunks=(sc, 1), compressor=compressor)
    image_arrays = {
        camera: data.create_dataset(
            f"image_{camera}",
            shape=(total, first.height, first.width, 3),
            chunks=(ic, first.height, first.width, 3),
            dtype=np.uint8,
            compressor=compressor,
        )
        for camera in args.cameras
    }
    meta.create_dataset("episode_ends", data=episode_ends, chunks=(len(episode_ends),), compressor=compressor)
    root.attrs.update({
        "format": "rl100_policy_zarr_v1", "source_format": "lerobot_v3.0",
        "source_datasets": [x.root.name for x in items], "fps": first.fps,
        "image_layout": "NHWC", "cameras": list(args.cameras),
    })
    for camera in args.cameras:
        intervals = [v for x in items for v in x.videos[camera]]
        with tqdm(
            total=total,
            desc=f"image_{camera}",
            unit="time step",
            dynamic_ncols=True,
            disable=args.no_progress,
        ) as progress:
            write_frames(
                image_arrays[camera],
                intervals,
                first.height,
                first.width,
                args.write_batch,
                progress,
            )
    check = zarr.open_group(str(output), mode="r")
    assert int(check["meta/episode_ends"][-1]) == total
    assert all(array.shape[0] == total for _, array in check["data"].arrays())
    print(f"conversion complete: {output}")


def main() -> int:
    enable_live_output()
    args = arguments()
    for key in ("image_chunk", "lowdim_chunk", "scalar_chunk", "write_batch"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    items = plans(roots(args.input, args.datasets), args.cameras)
    total, episodes = sum(x.frames for x in items), sum(len(x.lengths) for x in items)
    raw = total * items[0].height * items[0].width * 3 * len(args.cameras) / 2**30
    for item in items:
        print(f"{item.root.name}: {len(item.lengths)} episodes, {item.frames} frames")
    print(f"total: {episodes} episodes, {total} frames, {items[0].fps} Hz")
    print(f"images: uint8 NHWC {items[0].width}x{items[0].height}; raw payload ~{raw:.2f} GiB")
    if args.inspect_only:
        print("inspection passed; no output written")
    else:
        convert(args, items)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, KeyError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
