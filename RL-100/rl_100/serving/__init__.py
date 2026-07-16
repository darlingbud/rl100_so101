"""Lightweight remote policy serving for RL-100."""

from typing import Any

__all__ = ["RL100PolicyAdapter", "WebSocketPolicyServer"]


def __getattr__(name: str) -> Any:
    """Keep protocol-only clients independent from the policy runtime dependencies."""
    if name == "RL100PolicyAdapter":
        from rl_100.serving.policy_adapter import RL100PolicyAdapter

        return RL100PolicyAdapter
    if name == "WebSocketPolicyServer":
        from rl_100.serving.websocket_server import WebSocketPolicyServer

        return WebSocketPolicyServer
    raise AttributeError(name)
