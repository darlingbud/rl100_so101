"""Send one RO101 dataset observation to a running policy server."""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import zarr
from websockets.legacy.client import connect

from rl_100.serving.protocol import PROTOCOL_VERSION, pack_message, unpack_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument(
        "--dataset",
        default="data/DonQuihote16807.zarr",
        help="RO101 Zarr dataset used to provide a real observation",
    )
    parser.add_argument("--start", type=int, default=0)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    root = zarr.open(args.dataset, mode="r")
    async with connect(args.url, compression=None, max_size=64 * 1024 * 1024) as ws:
        metadata = unpack_message(await ws.recv())
        n_obs_steps = int(metadata["n_obs_steps"])
        stop = args.start + n_obs_steps
        observation = {
            "image_front": np.moveaxis(
                np.asarray(root["data/image_front"][args.start:stop]), -1, 1
            ),
            "image_side": np.moveaxis(
                np.asarray(root["data/image_side"][args.start:stop]), -1, 1
            ),
            "agent_pos": np.asarray(
                root["data/state"][args.start:stop], dtype=np.float32
            ),
        }
        if observation["agent_pos"].shape[0] != n_obs_steps:
            raise ValueError("Dataset slice does not contain enough observation steps")

        await ws.send(
            pack_message(
                {
                    "message_type": "infer_request",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": 0,
                    "episode_id": "server-smoke",
                    "step_id": args.start,
                    "observation": observation,
                }
            )
        )
        response = unpack_message(await ws.recv())
        if response.get("message_type") == "error":
            raise RuntimeError(f"{response['code']}: {response['message']}")
        print("metadata:", metadata)
        print("actions:", response["actions"])
        print("timing:", response["timing"])


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
