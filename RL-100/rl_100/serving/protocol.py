"""MessagePack wire protocol shared by the RL-100 server and future clients."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import msgpack
import numpy as np

PROTOCOL_VERSION = 1
MESSAGE_METADATA = "metadata"
MESSAGE_INFER_REQUEST = "infer_request"
MESSAGE_INFER_RESPONSE = "infer_response"
MESSAGE_RESET_REQUEST = "reset_request"
MESSAGE_RESET_RESPONSE = "reset_response"
MESSAGE_ERROR = "error"


@dataclass
class ProtocolError(Exception):
    code: str
    message: str
    retryable: bool = False
    request_id: int | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise TypeError(f"Unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(order="C"),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"Cannot encode value of type {type(value).__name__}")


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if b"__ndarray__" in value:
        try:
            dtype = np.dtype(value[b"dtype"])
            shape = tuple(int(dim) for dim in value[b"shape"])
            data = value[b"data"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("BAD_MESSAGE", "Malformed NumPy array payload") from exc
        if dtype.kind in ("V", "O", "c"):
            raise ProtocolError("BAD_MESSAGE", f"Unsupported NumPy dtype: {dtype}")
        if any(dim < 0 for dim in shape):
            raise ProtocolError("BAD_MESSAGE", "NumPy shape contains a negative dimension")
        expected_size = math.prod(shape) * dtype.itemsize
        if expected_size != len(data):
            raise ProtocolError(
                "BAD_MESSAGE",
                f"NumPy payload size mismatch: expected {expected_size}, received {len(data)}",
            )
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    if b"__npgeneric__" in value:
        try:
            return np.dtype(value[b"dtype"]).type(value[b"data"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("BAD_MESSAGE", "Malformed NumPy scalar payload") from exc
    return value


def pack_message(message: Mapping[str, Any]) -> bytes:
    if not isinstance(message, Mapping):
        raise TypeError("Protocol messages must be mappings")
    return msgpack.packb(dict(message), default=_pack_numpy, use_bin_type=True)


def unpack_message(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ProtocolError("BAD_MESSAGE", "Expected a binary WebSocket frame")
    try:
        message = msgpack.unpackb(
            payload, object_hook=_unpack_numpy, raw=False, strict_map_key=False
        )
    except ProtocolError:
        raise
    except (msgpack.UnpackException, ValueError, TypeError) as exc:
        raise ProtocolError("BAD_MESSAGE", "Invalid MessagePack payload") from exc
    if not isinstance(message, dict):
        raise ProtocolError("BAD_MESSAGE", "Top-level protocol message must be a map")
    return message


def validate_envelope(
    message: Mapping[str, Any], *, expected_type: str | None = None
) -> None:
    request_id = message.get("request_id")
    version = message.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            "UNSUPPORTED_VERSION",
            f"Expected protocol_version={PROTOCOL_VERSION}, received {version!r}",
            request_id=request_id if isinstance(request_id, int) else None,
        )
    message_type = message.get("message_type")
    if not isinstance(message_type, str):
        raise ProtocolError("BAD_MESSAGE", "message_type must be a string")
    if expected_type is not None and message_type != expected_type:
        raise ProtocolError(
            "BAD_MESSAGE",
            f"message_type must be {expected_type!r}, received {message_type!r}",
            request_id=request_id if isinstance(request_id, int) else None,
        )


def validate_request_id(message: Mapping[str, Any]) -> int:
    request_id = message.get("request_id")
    if not isinstance(request_id, int) or isinstance(request_id, bool) or request_id < 0:
        raise ProtocolError("BAD_MESSAGE", "request_id must be a non-negative integer")
    return request_id


def error_message(error: ProtocolError) -> dict[str, Any]:
    return {
        "message_type": MESSAGE_ERROR,
        "protocol_version": PROTOCOL_VERSION,
        "request_id": error.request_id,
        "code": error.code,
        "message": error.message,
        "retryable": error.retryable,
    }
