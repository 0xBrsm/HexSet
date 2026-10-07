# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from dataclasses import dataclass

from .terrain import Resource
from .topology import Topology, coastal_rings

__all__ = [
    "GENERIC_RATIO",
    "SPECIFIC_RATIO",
    "BASE_TRADE_RATIO",
    "NUM_GENERIC_PORTS",
    "Port",
    "base_port_bag",
    "place_ports",
]


GENERIC_RATIO = 3
SPECIFIC_RATIO = 2
BASE_TRADE_RATIO = 4

NUM_GENERIC_PORTS = 4


@dataclass(frozen=True)
class Port:
    """A trading post on a coastal edge, usable from either of its two vertices."""

    edge: int
    vertices: tuple[int, int]
    resource: Resource | None
    ratio: int


def base_port_bag() -> list[Resource | None]:
    """The base game's port kinds for `place_ports`: `NUM_GENERIC_PORTS` generic
    ports (`None`) and one per `Resource`."""
    return [None] * NUM_GENERIC_PORTS + list(Resource)


def _spread(ring_length: int, count: int) -> list[int]:
    """Positions for `count` ports spaced as evenly as the ring allows."""
    return [round(i * ring_length / count) for i in range(count)]


def place_ports(
    topology: Topology,
    bag: list[Resource | None],
    rng: random.Random | None = None,
) -> tuple[Port, ...]:
    """Distribute ports around the longest coastline, spaced as evenly as it
    allows. On the base board's 30-edge coast that is the printed frame's
    nine harbor positions, turned so the first sits on the ring's first edge."""
    rings = coastal_rings(topology)
    if not rings:
        return ()
    ring = max(rings, key=len)

    kinds = list(bag)
    if rng is not None:
        rng.shuffle(kinds)
    if len(kinds) > len(ring):
        raise ValueError("more ports than coastal edges")

    ports = []
    for position, resource in zip(_spread(len(ring), len(kinds)), kinds):
        edge = ring[position]
        a, b = topology.edges[edge]
        ports.append(
            Port(
                edge=edge,
                vertices=(a, b),
                resource=resource,
                ratio=GENERIC_RATIO if resource is None else SPECIFIC_RATIO,
            )
        )
    return tuple(ports)
