"""LeRobot SO101 client for the RL-100 WebSocket policy server.

Dry-run is the default: hardware observations are sent to the server, but returned
actions are only printed.  Use --execute and confirm interactively to move the arm.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import logging
import signal
import time
from typing import Any

import numpy as np
from websockets.legacy.client import connect

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from rl_100.serving.protocol import PROTOCOL_VERSION, pack_message, unpack_message


LOG = logging.getLogger("lerobot_policy_client")
MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://192.168.0.135:8000")
    parser.add_argument("--port", default="/dev/robot_follower")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--front-camera", type=int, default=0)
    parser.add_argument("--side-camera", type=int, default=1)
    parser.add_argument("--control-fps", type=float, default=10.0)
    parser.add_argument("--inference-fps", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="Maximum per-command joint change in LeRobot normalized units",
    )
    parser.add_argument("--execute", action="store_true", help="Actually command motors")
    parser.add_argument("--yes", action="store_true", help="Skip --execute confirmation")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run one inference request")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop cleanly after this many seconds (useful for dry-run tests)",
    )
    parser.add_argument("--episode-id", default="lerobot-live")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def make_robot(args: argparse.Namespace) -> SO101Follower:
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=args.front_camera,
            width=640,
            height=480,
            fps=30,
            fourcc=args.fourcc,
        ),
        "side": OpenCVCameraConfig(
            index_or_path=args.side_camera,
            width=640,
            height=480,
            fps=30,
            fourcc=args.fourcc,
        ),
    }
    config = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        cameras=cameras,
        use_degrees=False,
        max_relative_target=args.max_relative_target,
    )
    return SO101Follower(config)


def observation_frame(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """Map a LeRobot observation to the names/layout used during RL-100 training."""
    state = np.asarray([raw[f"{motor}.pos"] for motor in MOTORS], dtype=np.float32)
    front = np.asarray(raw["front"], dtype=np.uint8)
    side = np.asarray(raw["side"], dtype=np.uint8)
    expected_image_shape = (480, 640, 3)
    if front.shape != expected_image_shape or side.shape != expected_image_shape:
        raise ValueError(
            f"Camera shape mismatch: front={front.shape}, side={side.shape}, "
            f"expected={expected_image_shape}"
        )
    return {"image_front": front, "image_side": side, "agent_pos": state}


def stack_history(history: deque[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not history:
        raise ValueError("Observation history is empty")
    return {
        "image_front": np.moveaxis(
            np.stack([item["image_front"] for item in history]), -1, 1
        ),
        "image_side": np.moveaxis(
            np.stack([item["image_side"] for item in history]), -1, 1
        ),
        "agent_pos": np.stack([item["agent_pos"] for item in history]).astype(
            np.float32, copy=False
        ),
    }


def action_dict(action: np.ndarray) -> dict[str, float]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (len(MOTORS),) or not np.isfinite(action).all():
        raise ValueError(f"Invalid policy action shape/value: {action.shape}")
    return {f"{motor}.pos": float(value) for motor, value in zip(MOTORS, action)}


def confirm_execution(args: argparse.Namespace) -> None:
    if not args.execute:
        LOG.warning("DRY-RUN: policy actions will not be sent to the motors")
        return
    if args.yes:
        return
    answer = input(
        "The policy will move the physical arm. Clear the workspace and keep an "
        "emergency stop ready. Type MOVE to continue: "
    )
    if answer != "MOVE":
        raise SystemExit("Execution cancelled")


async def run(args: argparse.Namespace, robot: SO101Follower) -> None:
    control_period = 1.0 / args.control_fps
    inference_period = 1.0 / args.inference_fps
    stop = asyncio.Event()
    producer_done = asyncio.Event()
    hardware_lock = asyncio.Lock()
    plan_lock = asyncio.Lock()
    action_plan: deque[np.ndarray] = deque()
    executed_steps = 0
    inference_count = 0
    control_count = 0
    underruns = 0
    rtt_sum_ms = 0.0
    reported_chunk_size: int | None = None
    stats_started = time.monotonic()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    async def read_observation() -> dict[str, np.ndarray]:
        async with hardware_lock:
            raw = await asyncio.to_thread(robot.get_observation)
        return observation_frame(raw)

    async def inference_loop() -> None:
        nonlocal inference_count, rtt_sum_ms, reported_chunk_size
        request_id = 0
        async with connect(args.url, compression=None, max_size=64 * 1024 * 1024) as ws:
            metadata = unpack_message(await ws.recv())
            if metadata.get("message_type") != "metadata":
                raise RuntimeError(
                    f"Expected metadata, received {metadata.get('message_type')!r}"
                )
            n_obs_steps = int(metadata["n_obs_steps"])
            history: deque[dict[str, np.ndarray]] = deque(maxlen=n_obs_steps)
            LOG.info("Connected to policy server: %s", metadata)
            metadata_chunk_size = int(metadata["action_horizon"])
            LOG.info("Server action chunk size: %d", metadata_chunk_size)
            first = await read_observation()
            history.extend(first for _ in range(n_obs_steps))
            next_request = time.monotonic()

            while not stop.is_set():
                if request_id:
                    history.append(await read_observation())
                sent_at = time.monotonic()
                await ws.send(
                    pack_message(
                        {
                            "message_type": "infer_request",
                            "protocol_version": PROTOCOL_VERSION,
                            "request_id": request_id,
                            "episode_id": args.episode_id,
                            "step_id": executed_steps,
                            "observation": stack_history(history),
                        }
                    )
                )
                response = unpack_message(await ws.recv())
                rtt_ms = (time.monotonic() - sent_at) * 1000
                if response.get("message_type") == "error":
                    raise RuntimeError(f"{response['code']}: {response['message']}")
                if response.get("request_id") != request_id:
                    raise RuntimeError("Policy response request_id mismatch")
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != len(MOTORS):
                    raise ValueError(f"Invalid action chunk shape: {actions.shape}")
                chunk_size = len(actions)
                if reported_chunk_size != chunk_size:
                    LOG.info("Received action chunk size: %d", chunk_size)
                    reported_chunk_size = chunk_size
                if chunk_size != metadata_chunk_size:
                    LOG.warning(
                        "Returned chunk size %d differs from metadata action_horizon %d",
                        chunk_size,
                        metadata_chunk_size,
                    )
                async with plan_lock:
                    action_plan.clear()
                    action_plan.extend(actions.copy())
                inference_count += 1
                rtt_sum_ms += rtt_ms
                LOG.debug(
                    "request=%d rtt_ms=%.1f plan=%d timing=%s",
                    request_id,
                    rtt_ms,
                    chunk_size,
                    response.get("timing"),
                )
                request_id += 1
                if args.once:
                    break
                next_request += inference_period
                delay = next_request - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_request = time.monotonic()
        producer_done.set()

    async def control_loop() -> None:
        nonlocal executed_steps, control_count, underruns
        next_tick = time.monotonic()
        while not stop.is_set():
            action = None
            async with plan_lock:
                if action_plan:
                    action = action_plan.popleft()
            if action is None:
                underruns += 1
                if producer_done.is_set() and args.once:
                    stop.set()
                    break
            else:
                if args.execute:
                    async with hardware_lock:
                        await asyncio.to_thread(robot.send_action, action_dict(action))
                control_count += 1
                executed_steps += 1
                if not args.execute:
                    LOG.debug("DRY-RUN action step=%d: %s", executed_steps, action)
            next_tick += control_period
            delay = next_tick - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                LOG.warning("Control deadline missed by %.1f ms", -delay * 1000)
                next_tick = time.monotonic()

    async def stats_loop() -> None:
        nonlocal inference_count, control_count, underruns, rtt_sum_ms, stats_started
        while not stop.is_set():
            await asyncio.sleep(5)
            now = time.monotonic()
            elapsed = now - stats_started
            avg_rtt = rtt_sum_ms / inference_count if inference_count else float("nan")
            LOG.info(
                "rates: inference=%.2f Hz control=%.2f Hz avg_rtt=%.1f ms underruns=%d",
                inference_count / elapsed,
                control_count / elapsed,
                avg_rtt,
                underruns,
            )
            inference_count = control_count = underruns = 0
            rtt_sum_ms = 0.0
            stats_started = now

    async def duration_loop() -> None:
        if args.duration is None:
            await stop.wait()
            return
        await asyncio.sleep(args.duration)
        LOG.info("Test duration %.1f seconds reached", args.duration)
        stop.set()

    tasks = [
        asyncio.create_task(inference_loop(), name="policy-inference"),
        asyncio.create_task(control_loop(), name="robot-control"),
        asyncio.create_task(stats_loop(), name="rate-stats"),
        asyncio.create_task(duration_loop(), name="duration-limit"),
    ]
    try:
        done, _ = await asyncio.wait(tasks[:2], return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            if task.exception() is not None:
                raise task.exception()
        if args.once:
            await tasks[1]
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.fps is not None:
        args.control_fps = args.fps
        LOG.warning("--fps is deprecated; use --control-fps")
    if args.control_fps <= 0 or args.inference_fps <= 0:
        raise SystemExit("frequencies must be positive")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be positive")
    confirm_execution(args)
    robot = make_robot(args)
    try:
        LOG.info("Connecting SO101 on %s with cameras front=%d side=%d", args.port, args.front_camera, args.side_camera)
        robot.connect(calibrate=not args.no_calibrate)
        asyncio.run(run(args, robot))
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
