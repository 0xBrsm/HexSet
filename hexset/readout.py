# SPDX-License-Identifier: GPL-3.0-only
"""Where each policy logit comes from, and where it lands.

`ActionSpace` lays board-local actions out one slot per node, so a graph model
can emit them straight off its node embeddings instead of pooling to a fixed
vector. This module is the correspondence between the two: for each source of
embeddings it says how many logits that source's head must produce per node,
and the flat index every one of them belongs at.

It is deliberately numpy-only. The mapping is pure index arithmetic and it is
the piece that fails silently — a policy trained through a scrambled scatter
still converges, just on the wrong actions — so it is pinned by tests on the
development machine rather than discovered on the training box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import ActionSpace, ActionType

HEXES = "hexes"
VERTICES = "vertices"
EDGES = "edges"
GLOBALS = "globals"

# Which source's embedding decides each action type. Anything not named here is
# a single slot read off the global vector.
_SOURCES: dict[ActionType, str] = {
    ActionType.SETUP_SETTLEMENT: VERTICES,
    ActionType.BUILD_SETTLEMENT: VERTICES,
    ActionType.BUILD_CITY: VERTICES,
    ActionType.SETUP_ROAD: EDGES,
    ActionType.BUILD_ROAD: EDGES,
    ActionType.MOVE_ROBBER: HEXES,
    ActionType.PLAY_KNIGHT: HEXES,
}


@dataclass(frozen=True)
class Head:
    """One linear head: `width` logits per node of `source`.

    `scatter` is `(num_nodes, width)` of destination indices in the flat action
    space, so applying the head is `flat[head.scatter] = out` with `out` the
    head's `(num_nodes, width)` output.
    """

    source: str
    kinds: tuple[ActionType, ...]
    num_nodes: int
    width: int
    scatter: np.ndarray


@dataclass(frozen=True)
class Readout:
    size: int
    heads: tuple[Head, ...]

    def head(self, source: str) -> Head:
        for head in self.heads:
            if head.source == source:
                return head
        raise KeyError(source)


def _nodes(space: ActionSpace, source: str) -> int:
    return {
        HEXES: space.num_hexes,
        VERTICES: space.num_vertices,
        EDGES: space.num_edges,
        GLOBALS: 1,
    }[source]


def plan(space: ActionSpace) -> Readout:
    """The scatter for `space`, one head per source of embeddings.

    Every action type spreads its slots evenly over its source's nodes, so one
    rule covers all four heads: slot `(node, c)` of a kind sits at
    `offsets[kind] + node * span + c`, where `span` is the kind's slots per
    node. The global head has a single node, so `span` is the whole kind.
    """
    grouped: dict[str, list[ActionType]] = {}
    for kind in ActionType:
        grouped.setdefault(_SOURCES.get(kind, GLOBALS), []).append(kind)

    heads = []
    for source in (HEXES, VERTICES, EDGES, GLOBALS):
        kinds = tuple(grouped.get(source, ()))
        num_nodes = _nodes(space, source)
        spans = []
        for kind in kinds:
            span, remainder = divmod(space.sizes[kind], num_nodes)
            if remainder:
                raise ValueError(f"{kind.name} does not divide over {source}")
            spans.append(span)

        scatter = np.zeros((num_nodes, sum(spans)), dtype=np.int64)
        nodes = np.arange(num_nodes).reshape(-1, 1)
        column = 0
        for kind, span in zip(kinds, spans):
            block = space.offsets[kind] + nodes * span + np.arange(span)
            scatter[:, column : column + span] = block
            column += span

        heads.append(
            Head(
                source=source,
                kinds=kinds,
                num_nodes=num_nodes,
                width=column,
                scatter=scatter,
            )
        )
    return Readout(size=space.size, heads=tuple(heads))


def scatter_logits(readout: Readout, outputs: dict[str, np.ndarray]) -> np.ndarray:
    """Assemble a flat logit vector from each head's `(num_nodes, width)` output.

    Reference semantics for the torch version, and what the tests check against.
    """
    flat = np.zeros(readout.size, dtype=np.float32)
    for head in readout.heads:
        out = outputs[head.source]
        if out.shape != (head.num_nodes, head.width):
            raise ValueError(
                f"{head.source} head produced {out.shape}, "
                f"expected {(head.num_nodes, head.width)}"
            )
        flat[head.scatter] = out
    return flat
