"""Async WebSocket transport for the RL-100 policy adapter."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import http
import logging
import time
import traceback
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol, serve

from rl_100.serving.policy_adapter import RL100PolicyAdapter
from rl_100.serving.protocol import (
    MESSAGE_INFER_REQUEST,
    MESSAGE_INFER_RESPONSE,
    MESSAGE_RESET_REQUEST,
    MESSAGE_RESET_RESPONSE,
    PROTOCOL_VERSION,
    ProtocolError,
    error_message,
    pack_message,
    unpack_message,
    validate_envelope,
    validate_request_id,
)

logger = logging.getLogger(__name__)


class WebSocketPolicyServer:
    def __init__(
        self,
        policy: RL100PolicyAdapter,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_message_bytes: int = 64 * 1024 * 1024,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._max_message_bytes = max_message_bytes
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rl100-infer"
        )

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            logger.info("RL-100 policy server stopped")
        finally:
            self._executor.shutdown(wait=True)

    async def run(self) -> None:
        logger.info("Serving RL-100 policy on ws://%s:%d", self._host, self._port)
        async with serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_message_bytes,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            process_request=_health_check,
        ) as server:
            await server.wait_closed()

    async def _handler(
        self,
        websocket: WebSocketServerProtocol,
        path: str | None = None,
    ) -> None:
        del path
        remote = websocket.remote_address
        logger.info("Connection opened from %s", remote)
        await websocket.send(pack_message(self._policy.metadata))
        last_episode_id: str | None = None

        try:
            async for payload in websocket:
                response: dict[str, Any]
                message: dict[str, Any] | None = None
                try:
                    message = unpack_message(payload)
                    validate_envelope(message)
                    request_id = validate_request_id(message)
                    message_type = message["message_type"]

                    if message_type == MESSAGE_INFER_REQUEST:
                        episode_id = _validate_episode_id(message, request_id)
                        _validate_step_id(message, request_id)
                        if episode_id != last_episode_id:
                            await self._run_blocking(self._policy.reset, episode_id)
                            last_episode_id = episode_id
                        total_start = time.monotonic()
                        result = await self._run_blocking(
                            self._policy.infer, message.get("observation")
                        )
                        result["timing"]["total_ms"] = (
                            time.monotonic() - total_start
                        ) * 1000
                        response = {
                            "message_type": MESSAGE_INFER_RESPONSE,
                            "protocol_version": PROTOCOL_VERSION,
                            "request_id": request_id,
                            "episode_id": episode_id,
                            "server_time_ns": time.time_ns(),
                            **result,
                        }
                    elif message_type == MESSAGE_RESET_REQUEST:
                        episode_id = _validate_episode_id(message, request_id)
                        await self._run_blocking(self._policy.reset, episode_id)
                        last_episode_id = episode_id
                        response = {
                            "message_type": MESSAGE_RESET_RESPONSE,
                            "protocol_version": PROTOCOL_VERSION,
                            "request_id": request_id,
                            "episode_id": episode_id,
                        }
                    else:
                        raise ProtocolError(
                            "BAD_MESSAGE",
                            f"Unsupported message_type {message_type!r}",
                            request_id=request_id,
                        )
                except ProtocolError as exc:
                    if exc.request_id is None:
                        exc.request_id = _safe_request_id(message)
                    response = error_message(exc)
                except Exception:
                    logger.error(
                        "Unhandled request failure from %s\\n%s",
                        remote,
                        traceback.format_exc(),
                    )
                    response = error_message(
                        ProtocolError(
                            "INFERENCE_FAILED",
                            "Internal policy server error",
                            request_id=_safe_request_id(message),
                        )
                    )
                await websocket.send(pack_message(response))
        except ConnectionClosed:
            pass
        finally:
            logger.info("Connection closed from %s", remote)

    async def _run_blocking(self, function, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(function, *args)
        )


def _validate_episode_id(message: dict[str, Any], request_id: int) -> str:
    episode_id = message.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ProtocolError(
            "BAD_MESSAGE",
            "episode_id must be a non-empty string",
            request_id=request_id,
        )
    return episode_id


def _validate_step_id(message: dict[str, Any], request_id: int) -> int:
    step_id = message.get("step_id")
    if not isinstance(step_id, int) or isinstance(step_id, bool) or step_id < 0:
        raise ProtocolError(
            "BAD_MESSAGE",
            "step_id must be a non-negative integer",
            request_id=request_id,
        )
    return step_id


def _safe_request_id(message: Any) -> int | None:
    if isinstance(message, dict):
        value = message.get("request_id")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


async def _health_check(path: str, request_headers):
    del request_headers
    if path == "/healthz":
        return (
            http.HTTPStatus.OK,
            [("Content-Type", "text/plain"), ("Content-Length", "3")],
            b"OK\n",
        )
    return None
