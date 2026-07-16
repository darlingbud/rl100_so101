"""Run an RL-100 checkpoint as a lightweight WebSocket policy server."""

from __future__ import annotations

import argparse
import logging

from rl_100.serving.policy_adapter import RL100PolicyAdapter
from rl_100.serving.websocket_server import WebSocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="RL-100 workspace .ckpt")
    parser.add_argument(
        "--config",
        help="Resolved config.yaml saved by the training run; checked against checkpoint",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--weights",
        choices=("auto", "model", "ema_model"),
        default="auto",
        help="State dict to load; auto prefers ema_model",
    )
    parser.add_argument("--max-message-mib", type=int, default=64)
    parser.add_argument(
        "--stochastic", action="store_true", help="Disable deterministic sampling"
    )
    cm_group = parser.add_mutually_exclusive_group()
    cm_group.add_argument("--use-cm", dest="use_cm", action="store_true")
    cm_group.add_argument("--no-use-cm", dest="use_cm", action="store_false")
    parser.set_defaults(use_cm=None)
    parser.add_argument("--distill2mean", action="store_true")
    parser.add_argument(
        "--non-strict-checkpoint",
        action="store_true",
        help="Allow missing or unexpected state-dict keys",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    policy = RL100PolicyAdapter.from_checkpoint(
        args.checkpoint,
        config=args.config,
        device=args.device,
        weights=args.weights,
        strict=not args.non_strict_checkpoint,
        deterministic=not args.stochastic,
        use_cm=args.use_cm,
        distill2mean=args.distill2mean,
    )
    logging.getLogger(__name__).info("Policy metadata: %s", policy.metadata)
    WebSocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        max_message_bytes=args.max_message_mib * 1024 * 1024,
    ).serve_forever()


if __name__ == "__main__":
    main()
