# SPDX-License-Identifier: GPL-3.0-only
"""Optional native rules, legality and contract-9 observation state.

Importing this module succeeds without a compiler-built extension. Choose a
backend before importing a lane; a native apply failure never falls back to
another engine or repeats a Chance event.
"""
from .state import NativeState, NativeUnavailable, available, supported


def encode9(state, perspective, contract="9"):
    """An owned writable float32 observation of the perspective's facts."""
    import numpy as np
    row = np.frombuffer(state.core.encode9(perspective, contract), dtype=np.float32)
    return row


__all__ = ["NativeState", "NativeUnavailable", "available", "supported", "encode9"]
