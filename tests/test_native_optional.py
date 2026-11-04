# SPDX-License-Identifier: GPL-3.0-only
"""The optional extension can be absent without disabling Python imports."""
import unittest
from unittest import mock

from hexset.native import NativeState, NativeUnavailable, available, supported
from hexset.gym import native as native_gym
from hexset.gym.native import NativeBatch


class OptionalNativeBackend(unittest.TestCase):
    def test_missing_extension_is_detected_before_reading_or_mutating_a_game(self):
        with mock.patch("hexset.native.state.ArenaState", None):
            self.assertFalse(available())
            self.assertFalse(supported(object(), "9"))
            self.assertFalse(native_gym.available())
            with self.assertRaises(NativeUnavailable):
                NativeState(object())
        with mock.patch("hexset.gym.native._NativeRequests", None):
            with self.assertRaises(NativeUnavailable):
                NativeBatch([], space=None, pairs=(), contract="9", planned_indices=(), keep=None)

    def test_missing_request_binding_disables_lane_preflight_before_a_game_read(self):
        with mock.patch("hexset.native.state.ArenaState", object()), \
             mock.patch("hexset.gym.native._NativeRequests", None):
            self.assertTrue(available())
            self.assertFalse(native_gym.available())
            self.assertFalse(native_gym.supported(object(), "9"))


if __name__ == "__main__":
    unittest.main()
