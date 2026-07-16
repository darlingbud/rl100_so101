import unittest

import msgpack
import numpy as np

from rl_100.serving.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    pack_message,
    unpack_message,
    validate_envelope,
    validate_request_id,
)


class ProtocolTest(unittest.TestCase):
    def test_numpy_round_trip(self):
        image = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
        state = np.arange(12, dtype=np.float32)[::2]
        decoded = unpack_message(
            pack_message(
                {
                    "message_type": "infer_request",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": 3,
                    "image": image,
                    "state": state,
                }
            )
        )
        np.testing.assert_array_equal(decoded["image"], image)
        np.testing.assert_array_equal(decoded["state"], state)
        self.assertTrue(decoded["state"].flags.c_contiguous)

    def test_object_array_is_rejected(self):
        with self.assertRaises(TypeError):
            pack_message({"value": np.array([object()], dtype=object)})

    def test_non_map_is_rejected(self):
        with self.assertRaises(ProtocolError) as context:
            unpack_message(msgpack.packb([1, 2, 3]))
        self.assertEqual(context.exception.code, "BAD_MESSAGE")

    def test_envelope_and_request_id(self):
        message = {
            "message_type": "infer_request",
            "protocol_version": PROTOCOL_VERSION,
            "request_id": 0,
        }
        validate_envelope(message, expected_type="infer_request")
        self.assertEqual(validate_request_id(message), 0)

    def test_unsupported_version_is_structured_error(self):
        with self.assertRaises(ProtocolError) as context:
            validate_envelope(
                {
                    "message_type": "infer_request",
                    "protocol_version": 99,
                    "request_id": 4,
                }
            )
        self.assertEqual(context.exception.code, "UNSUPPORTED_VERSION")
        self.assertEqual(context.exception.request_id, 4)


if __name__ == "__main__":
    unittest.main()
