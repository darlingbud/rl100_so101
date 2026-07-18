"""Benchmark an RL-100 policy server with synthetic observations only."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any

import numpy as np
from websockets.asyncio.client import connect

from rl_100.serving.protocol import PROTOCOL_VERSION, pack_message, unpack_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://192.168.0.135:8000")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--episode-id", default="server-frequency-benchmark")
    return parser.parse_args()


def synthetic_observation(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    for name, spec in metadata["observation_spec"].items():
        shape = tuple(int(dim) for dim in spec["policy_input_shape"])
        dtype = np.dtype(spec["dtype"])
        observation[name] = np.zeros(shape, dtype=dtype)
    return observation


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def print_metric(name: str, values: list[float]) -> None:
    print(
        f"{name:<16} mean={statistics.mean(values):8.3f} ms "
        f"p50={percentile(values, 50):8.3f} ms "
        f"p95={percentile(values, 95):8.3f} ms "
        f"min={min(values):8.3f} ms max={max(values):8.3f} ms"
    )


async def benchmark(args: argparse.Namespace) -> None:
    async with connect(
        args.url,
        compression=None,
        max_size=64 * 1024 * 1024,
        open_timeout=args.timeout,
        ping_interval=None,
    ) as websocket:
        metadata = unpack_message(await asyncio.wait_for(websocket.recv(), args.timeout))
        if metadata.get("message_type") != "metadata":
            raise RuntimeError(f"Expected metadata, received: {metadata}")

        observation = synthetic_observation(metadata)
        template = {
            "message_type": "infer_request",
            "protocol_version": PROTOCOL_VERSION,
            "episode_id": args.episode_id,
            "observation": observation,
        }
        sample_payload = pack_message({**template, "request_id": 0, "step_id": 0})

        print(f"server:              {args.url}")
        print(f"policy:              {metadata.get('policy_name')}")
        print(f"observation spec:    {metadata['observation_spec']}")
        print(f"action chunk size:   {metadata['action_horizon']}")
        print(f"request payload:     {len(sample_payload):,} bytes")
        print(f"warmup / measured:   {args.warmup} / {args.requests}")

        rtts: list[float] = []
        totals: list[float] = []
        policies: list[float] = []
        preprocess: list[float] = []
        postprocess: list[float] = []
        measured_started = 0.0
        total_requests = args.warmup + args.requests

        for request_id in range(total_requests):
            payload = pack_message(
                {**template, "request_id": request_id, "step_id": request_id}
            )
            started = time.perf_counter()
            await asyncio.wait_for(websocket.send(payload), args.timeout)
            response = unpack_message(
                await asyncio.wait_for(websocket.recv(), args.timeout)
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            if response.get("message_type") == "error":
                raise RuntimeError(f"{response['code']}: {response['message']}")
            if response.get("message_type") != "infer_response":
                raise RuntimeError(f"Unexpected response: {response}")
            if response.get("request_id") != request_id:
                raise RuntimeError("Response request_id mismatch")

            if request_id >= args.warmup:
                if not rtts:
                    measured_started = started
                timing = response["timing"]
                rtts.append(elapsed_ms)
                totals.append(float(timing["total_ms"]))
                policies.append(float(timing["policy_ms"]))
                preprocess.append(float(timing["preprocess_ms"]))
                postprocess.append(float(timing["postprocess_ms"]))

        wall_seconds = time.perf_counter() - measured_started
        throughput = args.requests / wall_seconds
        wire_mib_s = args.requests * len(sample_payload) / wall_seconds / 1024**2
        server_compute_hz = 1000 / statistics.mean(totals)
        policy_compute_hz = 1000 / statistics.mean(policies)

        print("\nResults (single persistent connection, sequential requests)")
        print(f"wall time:           {wall_seconds:.3f} s")
        print(f"end-to-end limit:    {throughput:.3f} Hz")
        print(f"server compute:      {server_compute_hz:.3f} Hz")
        print(f"policy compute:      {policy_compute_hz:.3f} Hz")
        print(f"effective upload:    {wire_mib_s:.3f} MiB/s")
        print_metric("RTT", rtts)
        print_metric("server total", totals)
        print_metric("policy", policies)
        print_metric("preprocess", preprocess)
        print_metric("postprocess", postprocess)


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.requests <= 0 or args.timeout <= 0:
        raise SystemExit("--warmup must be non-negative; --requests and --timeout must be positive")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
