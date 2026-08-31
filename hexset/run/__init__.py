# SPDX-License-Identifier: GPL-3.0-only
"""Run directories: the manifest is the input, and it is frozen.

Two kinds live here, and `hexset.run.manifest` explains why they must not be
confused: a frozen **run** is launchable and reproducible, a reconstructed
**record** describes a run that predates the machinery and is read-only.
"""

from . import manifest
from .manifest import (
    KIND_RECORD,
    KIND_RUN,
    Manifest,
    freeze,
    load,
    parameters,
    provenance,
    read_record,
    record,
)

__all__ = [
    "KIND_RECORD",
    "KIND_RUN",
    "Manifest",
    "freeze",
    "load",
    "manifest",
    "parameters",
    "provenance",
    "read_record",
    "record",
]
