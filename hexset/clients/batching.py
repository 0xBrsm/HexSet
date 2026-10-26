# SPDX-License-Identifier: GPL-3.0-only
"""Exact row grouping for row-invariant decision inference.

Only batch ownership coordinates are normalized. Observations, masks and
ordered compound candidates remain in the key. Groups live for one forward,
so weights and requested outputs are constant throughout their lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Coalesced", "coalesce", "coalesce_rows"]

_ROWS = ("observation", "flat", "setup")
_CANDIDATES = {"road_building": "road_building_row", "discard": "discard_row"}
_KNOWN = set(_ROWS) | set(_CANDIDATES) | set(_CANDIDATES.values()) | {"double_rows"}


@dataclass(frozen=True)
class Coalesced:
    """A compact feed and indices restoring outputs to their original order.

    `inverse` restores position rows. `candidates[name]` restores scores for
    that compound candidate block; `double_rows` restores double-turn scores.
    All indices refer to the first axis. The original feed is never modified.
    """

    feed: dict[str, np.ndarray]
    inverse: np.ndarray
    candidates: dict[str, np.ndarray]


def coalesce(feed: dict[str, np.ndarray]) -> Coalesced | None:
    """Group byte-identical complete decision inputs, retaining first order.

    Return `None` if nothing can be removed or the feed has no supported row
    ownership contract. Dtypes and trailing shapes are constant per input
    within a call; distinct fields occupy separate tuple entries in a key.
    Python's byte-key dictionary resolves hash collisions by byte comparison.

    A multirow call retains at least two rows, duplicating its representative
    if needed, to avoid a single-row kernel after compaction. This does not
    establish batch invariance for a model that couples different rows.
    """
    names = [name for name in _ROWS if name in feed]
    if not names or set(feed)-_KNOWN:
        return None
    count = len(feed[names[0]])
    if count <= 2:
        return None
    if any(len(feed[name]) != count for name in names):
        raise ValueError("decision row inputs have different lengths")
    if any(name in feed and owner not in feed for name, owner in _CANDIDATES.items()):
        return None

    # Ownership is generally sparse: ordinary decisions have no candidate
    # blocks. Build slots only for owners actually present in the feed.
    slots = {}
    for owner in (*_CANDIDATES.values(), "double_rows"):
        if owner not in feed:
            continue
        grouped = [[] for _ in range(count)]
        for index, row in enumerate(feed[owner]):
            if not 0 <= row < count:
                raise ValueError("candidate owner is outside the decision batch")
            grouped[int(row)].append(index)
        slots[owner] = grouped

    lookup, first, inverse = {}, [], np.empty(count, dtype=np.int64)
    for row in range(count):
        key = tuple(feed[name][row].tobytes() for name in names)
        for name, owner in _CANDIDATES.items():
            if owner in slots:
                indices = slots[owner][row]
                key += ((feed[name][indices].tobytes() if indices else b'')
                        if name in feed else len(indices),)
        if "double_rows" in slots:
            key += (len(slots["double_rows"][row]),)
        at = lookup.get(key)
        if at is None:
            at = len(first)
            lookup[key] = at
            first.append(row)
        inverse[row] = at
    if len(first) == count:
        return None
    if len(first) == 1:
        first.append(first[0])
    first = np.asarray(first, dtype=np.int64)
    compact = {name: feed[name][first] for name in names}
    scatter = {}
    for owner, grouped in slots.items():
        selected, owners, starts = [], [], []
        for row, original in enumerate(first):
            starts.append(len(selected))
            block = grouped[original]
            selected.extend(block)
            owners.extend([row]*len(block))
        compact[owner] = np.asarray(owners, dtype=feed[owner].dtype)
        restore = np.empty(len(feed[owner]), dtype=np.int64)
        for row, block in enumerate(grouped):
            restore[block] = np.arange(starts[inverse[row]], starts[inverse[row]]+len(block))
        name = next((n for n, o in _CANDIDATES.items() if o == owner), owner)
        scatter[name] = restore
        if name in _CANDIDATES and name in feed:
            compact[name] = feed[name][np.asarray(selected, dtype=np.int64)]
    return Coalesced(compact, inverse, scatter)


def coalesce_rows(feed: dict[str, np.ndarray]) -> Coalesced | None:
    """Group byte-identical rows of a feed whose every input carries the
    batch on its first axis (a record-contract graph's), retaining first order.

    A row's key is its bytes in every input, so two rows group only where
    the graph is given exactly the same thing. Return `None` where nothing
    can be removed, the call has two rows or fewer, or the inputs disagree
    on the row count. As `coalesce`, a multirow call keeps at least two
    rows, and grouping assumes a graph whose rows do not read one another.
    `candidates` is empty: outputs restore through `inverse` alone."""
    if not feed:
        return None
    arrays = [np.ascontiguousarray(value) for value in feed.values()]
    count = len(arrays[0])
    if count <= 2 or any(value.ndim == 0 or len(value) != count for value in arrays):
        return None
    flat = np.concatenate([value.reshape(count, -1).view(np.uint8) for value in arrays], axis=1)
    keys = np.ascontiguousarray(flat).view(np.dtype((np.void, flat.shape[1]))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    if len(first) == count:
        return None
    # `unique` orders groups by key; put them back in first-occurrence order.
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    first, inverse = first[order], rank[inverse.ravel()]
    if len(first) == 1:
        first = np.concatenate([first, first])
    return Coalesced({name: value[first] for name, value in feed.items()}, inverse, {})
