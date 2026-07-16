import asyncio
import socket
import unittest

import numpy as np
from websockets.legacy.client import connect
from websockets.legacy.server import serve

from rl_100.serving.protocol import PROTOCOL_VERSION, pack_message, unpack_message
from rl_100.serving.websocket_server import WebSocketPolicyServer


class FakeAdapter:
    def __init__(self):
        self.reset_ids = []

    @property
    def metadata(self):
        return {
            "message_type": "metadata",
            "protocol_version": PROTOCOL_VERSION,
            "action_horizon": 2,
            "action_dim": 3,
        }

    def reset(self, episode_id=None):
        self.reset_ids.append(episode_id)

    def infer(self, observation):
        return {
            "actions": np.asarray(observation["actions"], dtype=np.float32),
            "timing": {
                "preprocess_ms": 0.0,
                "policy_ms": 0.0,
                "postprocess_ms": 0.0,
            },
        }


class WebSocketServerTest(unittest.TestCase):
    def test_metadata_and_infer_round_trip(self):
        asyncio.run(self._round_trip())

    async def _round_trip(self):
        adapter = FakeAdapter()
        policy_server = WebSocketPolicyServer(adapter)
        port = _unused_local_port()
        try:
            async with serve(policy_server._handler, "127.0.0.1", port):
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    metadata = unpack_message(await websocket.recv())
                    self.assertEqual(metadata["message_type"], "metadata")

                    await websocket.send(
                        pack_message(
                            {
                                "message_type": "infer_request",
                                "protocol_version": PROTOCOL_VERSION,
                                "request_id": 7,
                                "episode_id": "episode-a",
                                "step_id": 12,
                                "observation": {
                                    "actions": np.arange(
                                        6, dtype=np.float32
                                    ).reshape(2, 3)
                                },
                            }
                        )
                    )
                    response = unpack_message(await websocket.recv())
                    self.assertEqual(response["message_type"], "infer_response")
                    self.assertEqual(response["request_id"], 7)
                    np.testing.assert_array_equal(
                        response["actions"],
                        np.arange(6, dtype=np.float32).reshape(2, 3),
                    )
                    self.assertIn("total_ms", response["timing"])
                    self.assertEqual(adapter.reset_ids, ["episode-a"])
        finally:
            policy_server._executor.shutdown(wait=True)

    def test_invalid_step_id_returns_structured_error(self):
        asyncio.run(self._invalid_step_id())

    async def _invalid_step_id(self):
        adapter = FakeAdapter()
        policy_server = WebSocketPolicyServer(adapter)
        port = _unused_local_port()
        try:
            async with serve(policy_server._handler, "127.0.0.1", port):
                async with connect(f"ws://127.0.0.1:{port}") as websocket:
                    await websocket.recv()
                    await websocket.send(
                        pack_message(
                            {
                                "message_type": "infer_request",
                                "protocol_version": PROTOCOL_VERSION,
                                "request_id": 9,
                                "episode_id": "episode-a",
                                "observation": {},
                            }
                        )
                    )
                    response = unpack_message(await websocket.recv())
                    self.assertEqual(response["message_type"], "error")
                    self.assertEqual(response["request_id"], 9)
                    self.assertEqual(response["code"], "BAD_MESSAGE")
        finally:
            policy_server._executor.shutdown(wait=True)


def _unused_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


if __name__ == "__main__":
    unittest.main()
