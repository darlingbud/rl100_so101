"""Collect human-intervention SO101 rollouts from an RL-100 policy server."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import select
import signal
import sys
import termios
import time
import tty
from typing import Any

import numpy as np
from websockets.asyncio.client import connect

from lerobot_policy_client import (
    MOTORS,
    action_dict,
    confirm_execution,
    make_robot,
    observation_frame,
    stack_history,
)
from rl_100.collectors.ro101_hitl_recorder import HitlEpisodeBuffer, HitlZarrWriter
from rl_100.serving.protocol import PROTOCOL_VERSION, pack_message, unpack_message


LOG = logging.getLogger("lerobot_hitl_collect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument("--port", default="/dev/follower")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--leader-port", default="/dev/leader")
    parser.add_argument("--leader-id", default="my_awesome_leader_arm")
    parser.add_argument("--front-camera", type=int, default=0)
    parser.add_argument("--side-camera", type=int, default=2)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--control-fps", type=float, default=10.0)
    parser.add_argument("--inference-fps", type=float, default=3.0)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument(
        "--leader-follow-policy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move the leader with the follower in POLICY mode",
    )
    parser.add_argument(
        "--human-control-mode",
        choices=("relative", "direct"),
        default="relative",
        help="Map leader motion relative to the takeover pose or as absolute targets",
    )
    parser.add_argument(
        "--leader-max-relative-target",
        type=float,
        default=5.0,
        help="Maximum leader joint change per control step in POLICY mode",
    )
    parser.add_argument(
        "--leader-alignment-tolerance",
        type=float,
        default=1.0,
        help="Maximum joint error before POLICY rollout starts",
    )
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def make_leader(args: argparse.Namespace):
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    return SO101Leader(
        SO101LeaderConfig(
            port=args.leader_port,
            id=args.leader_id,
            use_degrees=False,
        )
    )


def motor_array(values: dict[str, float]) -> np.ndarray:
    result = np.asarray([values[f"{motor}.pos"] for motor in MOTORS], dtype=np.float32)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError("Leader/follower action must contain six finite motor positions")
    return result


def clamp_relative_target(
    target: np.ndarray,
    current: np.ndarray,
    max_relative_target: float,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if target.shape != (6,) or current.shape != (6,):
        raise ValueError("Leader target and current position must be six-dimensional")
    return current + np.clip(
        target - current,
        -max_relative_target,
        max_relative_target,
    )


def relative_human_target(
    leader_position: np.ndarray,
    leader_anchor: np.ndarray,
    follower_anchor: np.ndarray,
) -> np.ndarray:
    arrays = [
        np.asarray(value, dtype=np.float32)
        for value in (leader_position, leader_anchor, follower_anchor)
    ]
    if any(value.shape != (6,) for value in arrays):
        raise ValueError("Relative teleoperation inputs must be six-dimensional")
    return arrays[2] + (arrays[0] - arrays[1])


class TerminalKeyReader:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.original: list[Any] | None = None

    def __enter__(self) -> "TerminalKeyReader":
        if not os.isatty(self.fd):
            raise RuntimeError("HITL collection requires an interactive terminal")
        self.original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.original is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)

    def read(self) -> list[str]:
        keys = []
        while select.select([self.fd], [], [], 0)[0]:
            keys.append(os.read(self.fd, 1).decode(errors="ignore"))
        return keys


@dataclass
class PendingTransition:
    observation: dict[str, np.ndarray]
    policy_action: np.ndarray
    human_action: np.ndarray
    executed_action: np.ndarray
    intervention: bool
    policy_valid: bool
    clamped: bool
    time_ns: int


@dataclass
class CollectionState:
    active: bool = False
    intervention: bool = False
    executed_steps: int = 0
    saved_episodes: int = 0
    session_episodes: int = 0
    leader_torque_enabled: bool = False
    leader_aligned: bool = False
    human_leader_anchor: np.ndarray | None = None
    human_follower_anchor: np.ndarray | None = None


async def collect(
    args: argparse.Namespace,
    robot: Any,
    leader: Any,
    keys: TerminalKeyReader,
) -> None:
    if args.control_fps <= 0 or args.inference_fps <= 0:
        raise ValueError("control-fps and inference-fps must be positive")
    if args.max_episode_steps < 1:
        raise ValueError("max-episode-steps must be positive")
    if args.leader_follow_policy and args.leader_max_relative_target <= 0:
        raise ValueError("leader-max-relative-target must be positive")
    if args.leader_alignment_tolerance < 0:
        raise ValueError("leader-alignment-tolerance cannot be negative")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("max-episodes must be positive")
    if not 0 < args.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")

    writer = HitlZarrWriter(args.output, gamma=args.gamma, fps=args.control_fps)
    writer.buffer.root.attrs["control_fps"] = float(args.control_fps)
    episode = HitlEpisodeBuffer(gamma=args.gamma)
    state = CollectionState(saved_episodes=writer.n_episodes)
    stop = asyncio.Event()
    force_inference = asyncio.Event()
    hardware_lock = asyncio.Lock()
    history_lock = asyncio.Lock()
    plan_lock = asyncio.Lock()
    action_plan: deque[tuple[int, int, np.ndarray]] = deque()
    pending: PendingTransition | None = None

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

    async def clear_plan_and_refresh() -> None:
        async with plan_lock:
            action_plan.clear()
        force_inference.set()

    async def disable_leader_torque() -> None:
        if state.leader_torque_enabled:
            await asyncio.to_thread(leader.disable_torque)
            state.leader_torque_enabled = False

    async def follow_policy_with_leader(
        follower_target: np.ndarray,
        leader_position: np.ndarray,
    ) -> None:
        if not args.leader_follow_policy:
            return
        leader_target = clamp_relative_target(
            follower_target,
            leader_position,
            args.leader_max_relative_target,
        )
        # Set a nearby goal before enabling torque to avoid an abrupt first move.
        await asyncio.to_thread(leader.send_feedback, action_dict(leader_target))
        if not state.leader_torque_enabled:
            await asyncio.to_thread(leader.enable_torque)
            state.leader_torque_enabled = True

    def print_controls() -> None:
        LOG.info(
            "Controls: n=start | SPACE=toggle intervention | s=success | "
            "f=failure | a=discard | q=quit"
        )

    async with connect(
        args.url,
        compression=None,
        max_size=64 * 1024 * 1024,
        ping_interval=None,
    ) as ws:
        metadata = unpack_message(await ws.recv())
        if metadata.get("message_type") != "metadata":
            raise RuntimeError("Policy server did not send metadata")
        n_obs_steps = int(metadata["n_obs_steps"])
        history: deque[dict[str, np.ndarray]] = deque(maxlen=n_obs_steps)
        LOG.info("Connected to policy server: %s", metadata)
        LOG.info("Appending HITL episodes to %s", writer.path)
        print_controls()

        async def inference_loop() -> None:
            request_id = 0
            next_request = time.monotonic()
            while not stop.is_set():
                policy_ready = (
                    state.active
                    and not state.intervention
                    and (not args.leader_follow_policy or state.leader_aligned)
                )
                if not policy_ready:
                    await asyncio.sleep(0.05)
                    next_request = time.monotonic()
                    continue

                delay = next_request - time.monotonic()
                if delay > 0 and not force_inference.is_set():
                    try:
                        await asyncio.wait_for(force_inference.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        pass
                force_inference.clear()
                async with history_lock:
                    if not history:
                        await asyncio.sleep(0)
                        continue
                    request_observation = stack_history(history)
                sent_at = time.monotonic()
                await ws.send(
                    pack_message(
                        {
                            "message_type": "infer_request",
                            "protocol_version": PROTOCOL_VERSION,
                            "request_id": request_id,
                            "episode_id": f"hitl-{state.saved_episodes}",
                            "step_id": state.executed_steps,
                            "observation": request_observation,
                        }
                    )
                )
                response = unpack_message(await ws.recv())
                if response.get("message_type") == "error":
                    raise RuntimeError(f"{response['code']}: {response['message']}")
                if response.get("request_id") != request_id:
                    raise RuntimeError("Policy response request_id mismatch")
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 6:
                    raise ValueError(f"Invalid policy action chunk: {actions.shape}")
                if state.active:
                    async with plan_lock:
                        action_plan.clear()
                        action_plan.extend(
                            (request_id, index, action.copy())
                            for index, action in enumerate(actions)
                        )
                LOG.debug(
                    "request=%d rtt=%.1fms chunk=%d",
                    request_id,
                    (time.monotonic() - sent_at) * 1000,
                    len(actions),
                )
                request_id += 1
                next_request = max(
                    next_request + 1.0 / args.inference_fps,
                    time.monotonic(),
                )

        async def finish_episode(*, success: bool, timeout: bool) -> None:
            nonlocal pending
            if pending is not None:
                raise RuntimeError("Pending transition must be closed before episode finish")
            if not episode.transitions:
                LOG.warning("Episode has no transitions; discarding")
            else:
                index = writer.append_episode(
                    episode,
                    success=success,
                    timeout=timeout,
                )
                interventions = sum(
                    item["intervention"] for item in episode.transitions
                )
                LOG.info(
                    "Saved episode %d: steps=%d success=%s timeout=%s interventions=%d",
                    index,
                    len(episode.transitions),
                    success,
                    timeout,
                    interventions,
                )
                state.saved_episodes += 1
                state.session_episodes += 1
            episode.clear()
            state.active = False
            state.intervention = False
            state.leader_aligned = False
            state.human_leader_anchor = None
            state.human_follower_anchor = None
            await disable_leader_torque()
            async with history_lock:
                history.clear()
            await clear_plan_and_refresh()
            if (
                args.max_episodes is not None
                and state.session_episodes >= args.max_episodes
            ):
                stop.set()
            else:
                LOG.info("Reset the task, then press n to start the next episode")

        async def control_loop() -> None:
            nonlocal pending
            next_tick = time.monotonic()
            while not stop.is_set():
                current = await read_observation()

                if state.active and pending is not None:
                    episode.append(
                        observation=pending.observation,
                        next_observation=current,
                        policy_action=pending.policy_action,
                        human_action=pending.human_action,
                        executed_action=pending.executed_action,
                        intervention=pending.intervention,
                        policy_valid=pending.policy_valid,
                        clamped=pending.clamped,
                        time_ns=pending.time_ns,
                    )
                    pending = None

                command_finished_episode = False
                mode_changed = False
                for key in keys.read():
                    if key.lower() == "q":
                        episode.clear()
                        pending = None
                        await disable_leader_torque()
                        stop.set()
                        break
                    if key.lower() == "n" and not state.active:
                        if not args.execute:
                            LOG.warning("Use --execute before collecting an episode")
                            continue
                        episode.clear()
                        state.active = True
                        state.intervention = False
                        state.leader_aligned = not args.leader_follow_policy
                        state.human_leader_anchor = None
                        state.human_follower_anchor = None
                        async with history_lock:
                            history.clear()
                            history.extend(current for _ in range(n_obs_steps))
                        await clear_plan_and_refresh()
                        mode_changed = True
                        if state.leader_aligned:
                            LOG.info("Episode started in POLICY mode")
                        else:
                            LOG.info("Aligning leader to follower before POLICY starts")
                    elif key == " " and state.active:
                        state.intervention = not state.intervention
                        if state.intervention:
                            await disable_leader_torque()
                            state.leader_aligned = False
                            state.human_leader_anchor = None
                            state.human_follower_anchor = current["agent_pos"].copy()
                        else:
                            state.leader_aligned = not args.leader_follow_policy
                            state.human_leader_anchor = None
                            state.human_follower_anchor = None
                        await clear_plan_and_refresh()
                        mode_changed = True
                        LOG.warning(
                            "Control mode: %s",
                            "HUMAN" if state.intervention else "POLICY",
                        )
                    elif key.lower() in ("s", "f") and state.active:
                        await finish_episode(
                            success=key.lower() == "s",
                            timeout=False,
                        )
                        command_finished_episode = True
                    elif key.lower() == "a" and state.active:
                        episode.clear()
                        pending = None
                        state.active = False
                        state.intervention = False
                        state.leader_aligned = False
                        state.human_leader_anchor = None
                        state.human_follower_anchor = None
                        await disable_leader_torque()
                        await clear_plan_and_refresh()
                        command_finished_episode = True
                        LOG.warning("Episode discarded; reset and press n")

                if stop.is_set():
                    break
                if mode_changed:
                    next_tick = time.monotonic()
                if command_finished_episode or not state.active:
                    next_tick = time.monotonic() + 1.0 / args.control_fps
                    await asyncio.sleep(1.0 / args.control_fps)
                    continue
                if len(episode.transitions) >= args.max_episode_steps:
                    await finish_episode(success=False, timeout=True)
                    continue

                async with history_lock:
                    history.append(current)
                leader_action = motor_array(await asyncio.to_thread(leader.get_action))
                human_action = leader_action
                if state.intervention and args.human_control_mode == "relative":
                    if state.human_leader_anchor is None:
                        state.human_leader_anchor = leader_action.copy()
                    if state.human_follower_anchor is None:
                        state.human_follower_anchor = current["agent_pos"].copy()
                    human_action = relative_human_target(
                        leader_action,
                        state.human_leader_anchor,
                        state.human_follower_anchor,
                    )

                if (
                    args.leader_follow_policy
                    and not state.intervention
                    and not state.leader_aligned
                ):
                    alignment_error = float(
                        np.max(np.abs(current["agent_pos"] - leader_action))
                    )
                    await follow_policy_with_leader(current["agent_pos"], leader_action)
                    if alignment_error <= args.leader_alignment_tolerance:
                        state.leader_aligned = True
                        async with history_lock:
                            history.clear()
                            history.extend(current for _ in range(n_obs_steps))
                        await clear_plan_and_refresh()
                        LOG.info(
                            "Leader aligned (max joint error %.3f); POLICY rollout started",
                            alignment_error,
                        )
                    next_tick += 1.0 / args.control_fps
                    delay = next_tick - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    else:
                        next_tick = time.monotonic()
                    continue

                planned = None
                async with plan_lock:
                    if action_plan:
                        planned = action_plan.popleft()
                policy_valid = planned is not None
                policy_action = (
                    planned[2] if planned is not None else current["agent_pos"].copy()
                )
                requested_action = human_action if state.intervention else policy_action
                async with hardware_lock:
                    sent = await asyncio.to_thread(
                        robot.send_action,
                        action_dict(requested_action),
                    )
                sent_action = motor_array(sent)
                if not state.intervention:
                    await follow_policy_with_leader(sent_action, leader_action)
                pending = PendingTransition(
                    observation=current,
                    policy_action=policy_action.copy(),
                    human_action=human_action.copy(),
                    executed_action=sent_action.copy(),
                    intervention=state.intervention,
                    policy_valid=policy_valid,
                    clamped=not np.allclose(
                        sent_action, requested_action, rtol=0.0, atol=1e-4
                    ),
                    time_ns=time.time_ns(),
                )
                state.executed_steps += 1

                next_tick += 1.0 / args.control_fps
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    LOG.warning("Control deadline missed by %.1fms", -delay * 1000)
                    next_tick = time.monotonic()

        tasks = [
            asyncio.create_task(inference_loop(), name="policy-inference"),
            asyncio.create_task(control_loop(), name="hitl-control"),
        ]
        try:
            completed, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in completed:
                if task.exception() is not None:
                    raise task.exception()
        finally:
            stop.set()
            await disable_leader_torque()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if episode.transitions or pending is not None:
                LOG.warning("Discarded unfinished episode during shutdown")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.port == args.leader_port:
        raise SystemExit("Follower and leader ports must be different")
    confirm_execution(args)
    robot = make_robot(args)
    leader = make_leader(args)
    try:
        robot.connect(calibrate=not args.no_calibrate)
        leader.connect(calibrate=not args.no_calibrate)
        with TerminalKeyReader() as keys:
            asyncio.run(collect(args, robot, leader, keys))
    finally:
        if leader.is_connected:
            try:
                leader.disable_torque()
            finally:
                leader.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
