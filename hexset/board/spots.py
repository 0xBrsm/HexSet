# SPDX-License-Identifier: GPL-3.0-only
"""Names for a board's spots: every vertex named by the numbers around it,
so that no two spots on one board share a name ("6-8-3", "11 wheat spot",
"8-5-10 at the wheat 8"). The server sends them with a table's layout
(`hexset.server.wire`), for a client to name a spot as the page does."""

from __future__ import annotations

from hexset.board.board import pips
from hexset.board.terrain import TERRAIN_RESOURCE, Resource

__all__ = ["spot_name", "spot_names"]


def spot_name(tokens) -> str:
    """A spot named by its number tokens, best first ("6-8-3"): sorted by
    pips, then by number. A hex with no token (the desert, the sea) is left
    out, and a spot with one number is that number's spot ("8 spot"), so it
    does not read as a roll."""
    numbers = [t for t in tokens if t]
    if len(numbers) == 1:
        return f"{numbers[0]} spot"
    return "-".join(str(t) for t in sorted(numbers, key=lambda t: (-pips(t), t)))


def spot_names(board) -> tuple[str, ...]:
    """Every vertex of `board` named so that no two share a name, by vertex:
    its numbers (`spot_name`, "" for a spot with none); where two spots share
    those, a one-number spot also says its resource ("11 wheat spot"); spots
    of two or more numbers still shared say a number whose hex the others
    lack ("8-5-10 at the wheat 8"); a one-number spot still shared says the
    best spot of two or more numbers one road away that tells it apart
    ("11 wheat spot by the 6-8"); and where two spots still share a name,
    one at a port says it ("at the sheep port"). Spots that none of these tells apart keep one name. Kept for
    each board read."""
    key = (board.terrain, board.tokens, board.ports, id(board.topology))
    names = _SPOT_NAMES.get(key)
    if names is None:
        if len(_SPOT_NAMES) >= 64:
            _SPOT_NAMES.clear()
        names = _SPOT_NAMES[key] = _named_spots(board)
    return names


# The names `spot_names` gave, by board.
_SPOT_NAMES: dict[tuple, tuple[str, ...]] = {}


def _named_spots(board) -> tuple[str, ...]:
    topology = board.topology
    land = []
    for v in range(len(topology.vertex_hexes)):
        land.append([(board.tokens[h], TERRAIN_RESOURCE[board.terrain[h]]) for h in topology.vertex_hexes[v]
                     if TERRAIN_RESOURCE.get(board.terrain[h]) is not None and board.tokens[h]])
    ports = {}
    for port in board.ports:
        for v in port.vertices:
            ports[v] = port.resource
    names = [spot_name([t for t, _ in hexes]) for hexes in land]

    def resolve(rename):
        seen: dict[str, list[int]] = {}
        for v, name in enumerate(names):
            if name:
                seen.setdefault(name, []).append(v)
        for group in seen.values():
            if len(group) > 1:
                for v in group:
                    names[v] = rename(v, names[v])

    def resource(r) -> str:
        return Resource(r).name.lower()

    resolve(lambda v, name: f"{land[v][0][0]} {resource(land[v][0][1])} spot" if len(land[v]) == 1 else name)

    # Spots of two or more numbers still shared: by the first number whose
    # hex the others do not share ("8-5-10 at the wheat 8").
    seen = {}
    for v, name in enumerate(names):
        if name and len(land[v]) > 1:
            seen.setdefault(name, []).append(v)
    for name, group in seen.items():
        if len(group) < 2:
            continue
        for v in group:
            ordered = sorted(land[v], key=lambda x: (-pips(x[0]), x[0]))
            pick = next(((t, r) for t, r in ordered if all((t, r) not in land[u] for u in group if u != v)), None)
            if pick is not None:
                names[v] = f"{name} at the {resource(pick[1])} {pick[0]}"

    # A one-number spot still shared: by the best spot of two or more
    # numbers one road away that no other spot of its name is by
    # ("11 wheat spot by the 6-8").
    seen: dict[str, list[int]] = {}
    for v, name in enumerate(names):
        if name and len(land[v]) == 1:
            seen.setdefault(name, []).append(v)
    for name, group in seen.items():
        if len(group) < 2:
            continue
        near = {v: [names[u] for u in sorted(topology.vertex_neighbors[v],
                                              key=lambda u: (-sum(pips(t) for t, _ in land[u]), u))
                    if len(land[u]) > 1]
                for v in group}
        for v in group:
            others = {n for u in group if u != v for n in near[u]}
            pick = next((n for n in near[v] if n not in others), None)
            if pick is not None:
                names[v] = f"{name} by the {pick}"
    resolve(lambda v, name: f"{name} at the {_port_name(ports[v])}" if v in ports else name)
    return tuple(names)


def _port_name(resource: Resource | None) -> str:
    """'3:1 port' for a generic port, 'ore port' and so on for a 2:1 one."""
    return "3:1 port" if resource is None else f"{Resource(resource).name.lower()} port"
