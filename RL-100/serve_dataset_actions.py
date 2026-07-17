"""Serve recorded RO101 actions through the RL-100 WebSocket protocol."""

from __future__ import annotations

import argparse
import logging

from rl_100.serving.dataset_action_adapter import DatasetActionAdapter
from rl_100.serving.websocket_server import WebSocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/DonQuihote16807.zarr")
    parser.add_argument("--mode", choices=("hold", "replay"), default="hold")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-message-mib", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    adapter = DatasetActionAdapter(
        args.dataset,
        mode=args.mode,
        episode_index=args.episode_index,
        start_step=args.start_step,
        action_horizon=args.action_horizon,
        n_obs_steps=args.n_obs_steps,
        loop=args.loop,
    )
    logging.getLogger(__name__).info("Dataset action metadata: %s", adapter.metadata)
    WebSocketPolicyServer(
        adapter,
        host=args.host,
        port=args.port,
        max_message_bytes=args.max_message_mib * 1024 * 1024,
    ).serve_forever()


if __name__ == "__main__":
    main()
