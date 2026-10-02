"""The served table's wire format: how the board, actions and trade
bundles look in JSON, read by the page, every API client and `api`.

`action_to_wire` renders an `Action` JSON-friendly; `wire_to_action` undoes
it, exactly, for every action `legal_actions` can produce. Resource and card
names, and every positional count on the wire, are in `RESOURCE_NAMES` and
`DEV_CARD_NAMES` order.
"""

from __future__ import annotations

import math

from hexset.actions import YEAR_OF_PLENTY_PAIRS, Action, ActionType
from hexset.board.board import Board
from hexset.board.coords import Hex
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.board.topology import Topology
from hexset.cards import DevCard
from hexset.state import MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS
from hexset.trading import Bundle

__all__ = [
    "DEV_CARD_NAMES",
    "RESOURCE_NAMES",
    "action_to_wire",
    "board_layout",
    "hex_center",
    "hex_corner",
    "round_bundle_from_wire",
    "round_offer_from_wire",
    "signed_bundle_from_wire",
    "vertex_pixels",
    "wire_to_action",
]

RESOURCE_NAMES: tuple[str, ...] = tuple(r.name.title() for r in Resource)
DEV_CARD_NAMES: tuple[str, ...] = tuple(c.name.title().replace("_", " ") for c in DevCard)



# --- Hex-to-pixel layout -----------------------------------------------------
#
# Pointy-top: a hex's six corners sit at angles 60*i - 30 degrees around its
# center, in the same i = 0..5 order as `Topology.hex_vertices`. Edge length
# equals circumradius, so every board edge measures exactly `size`.

SQRT3 = math.sqrt(3.0)


def hex_center(h: Hex, size: float) -> tuple[float, float]:
    """The pixel center of hex `h` at circumradius `size`."""
    x = size * (SQRT3 * h.q + SQRT3 / 2 * h.r)
    y = size * (1.5 * h.r)
    return (x, y)


def hex_corner(center: tuple[float, float], index: int, size: float) -> tuple[float, float]:
    """Corner `index` (0..5) of the hex at `center`."""
    angle = math.radians(60 * index - 30)
    return (center[0] + size * math.cos(angle), center[1] + size * math.sin(angle))


def vertex_pixels(topology: Topology, size: float) -> list[tuple[float, float]]:
    """One pixel position per vertex, agreeing across every hex that touches it."""
    positions: list[tuple[float, float] | None] = [None] * topology.num_vertices
    for h in range(topology.num_hexes):
        center = hex_center(topology.hexes[h], size)
        for corner_index, v in enumerate(topology.hex_vertices[h]):
            if positions[v] is None:
                positions[v] = hex_corner(center, corner_index, size)
    missing = [v for v, p in enumerate(positions) if p is None]
    if missing:
        raise AssertionError(f"vertices with no touching hex: {missing}")
    return positions  # type: ignore[return-value]


def board_layout(board: Board, size: float = 60.0) -> dict:
    """Static board geometry and contents for `/api/board`. Sent once per
    game; nothing here changes as it is played. Occupancy — vertex/edge owners,
    the robber — is in `state_view` instead."""
    topology = board.topology
    vpix = vertex_pixels(topology, size)
    hexes = []
    for h in range(topology.num_hexes):
        cx, cy = hex_center(topology.hexes[h], size)
        hexes.append(
            {
                "id": h,
                "terrain": board.terrain[h].name,
                "token": board.tokens[h] or None,
                "x": round(cx, 3),
                "y": round(cy, 3),
                # In order, so a client draws the outline from the same
                # pixels the buildings sit on.
                "vertex_ids": list(topology.hex_vertices[h]),
            }
        )
    vertices = [
        {"id": v, "x": round(x, 3), "y": round(y, 3)} for v, (x, y) in enumerate(vpix)
    ]
    edges = [
        {"id": e, "v0": a, "v1": b} for e, (a, b) in enumerate(topology.edges)
    ]
    ports = [
        {
            "edge": p.edge,
            "vertices": list(p.vertices),
            "resource": None if p.resource is None else RESOURCE_NAMES[p.resource],
            "ratio": p.ratio,
        }
        for p in board.ports
    ]
    return {
        "size": size,
        "hexes": hexes,
        "vertices": vertices,
        "edges": edges,
        "ports": ports,
        "resources": list(RESOURCE_NAMES),
        "dev_cards": list(DEV_CARD_NAMES),
        "year_of_plenty_pairs": [
            [RESOURCE_NAMES[a], RESOURCE_NAMES[b]] for a, b in YEAR_OF_PLENTY_PAIRS
        ],
        # The engine's own caps, so a client's piece display cannot drift.
        "piece_supply": {
            "road": MAX_ROADS,
            "settlement": MAX_SETTLEMENTS,
            "city": MAX_CITIES,
        },
    }


# --- Wire format for actions --------------------------------------------------


def action_to_wire(action: Action) -> dict:
    """An `Action` as its JSON wire object: `{"type", "a", "b"}`."""
    return {"type": action.type.name, "a": action.a, "b": action.b}


def wire_to_action(data: dict) -> Action:
    """A wire object back as an `Action`. `ValueError` if it is malformed."""
    try:
        kind = ActionType[str(data["type"])]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown action type {data.get('type')!r}") from exc
    try:
        return Action(type=kind, a=int(data.get("a", 0)), b=int(data.get("b", 0)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed action payload: {data!r}") from exc


# --- Wire format for the trade round (`hexset.trading`, "The trade round") ---
#
# Positional counts in `RESOURCE_NAMES` order: a round's bundles are echoed
# back verbatim by the client at every later step.


def round_bundle_from_wire(give: list, want: list) -> Bundle:
    """A signed `Bundle` for `POST .../trade/round`'s body, positive towards
    the proposer. `give`/`want` are each `NUM_RESOURCES`-long lists of
    nonnegative counts on disjoint resources, at least one a side. Nothing
    caps how many: a seat moves what it is willing to move, and the engine
    checks it holds it. `ValueError` names the first check that fails."""
    give = _int_list(give, "give")
    want = _int_list(want, "want")
    if any(n < 0 for n in give) or any(n < 0 for n in want):
        raise ValueError("give/want must be nonnegative")
    if any(g and w for g, w in zip(give, want)):
        raise ValueError("give and want must not share a resource")
    if not sum(give):
        raise ValueError("give must be at least one card")
    if not sum(want):
        raise ValueError("want must be at least one card")
    return tuple(w - g for w, g in zip(want, give))


def round_offer_from_wire(give: list, want: list, give_any: object = 0,
                          want_any: object = 0) -> tuple[Bundle, int]:
    """`POST .../trade/round`'s body as the offer it broadcasts: the signed
    `Bundle` of named cards, positive towards the proposer, and its any
    cards, signed the same way (`hexset.trading.Offer.any`). `give_any` is
    cards the proposer gives, named by whoever answers ("any card for your
    ore"), `want_any` cards it takes of the answerer's choosing ("my sheep for
    any card"); at most one of the two. A side holding any cards may name
    none of its own; without any cards the rules are `round_bundle_from_wire`'s.
    `ValueError` names the first check that fails."""
    try:
        give_any, want_any = int(give_any or 0), int(want_any or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("give_any/want_any must be integers") from exc
    if give_any < 0 or want_any < 0:
        raise ValueError("give_any/want_any must be nonnegative")
    if give_any and want_any:
        raise ValueError("any cards go on one side of an offer, not both")
    if not (give_any or want_any):
        return round_bundle_from_wire(give, want), 0
    give = _int_list(give, "give")
    want = _int_list(want, "want")
    if any(n < 0 for n in give) or any(n < 0 for n in want):
        raise ValueError("give/want must be nonnegative")
    if any(g and w for g, w in zip(give, want)):
        raise ValueError("give and want must not share a resource")
    if not give_any and not sum(give):
        raise ValueError("give must be at least one card")
    if not want_any and not sum(want):
        raise ValueError("want must be at least one card")
    return tuple(w - g for w, g in zip(want, give)), want_any - give_any


def signed_bundle_from_wire(values: list) -> Bundle:
    """A signed `Bundle` echoed straight off the wire: `.../trade/round/
    answer`'s `received` and `.../trade/round/choose`'s `bundle`. Both name an
    *exact* offer or response read out of a previous `state_view` and are
    matched verbatim, never reconstructed."""
    return tuple(_int_list(values, "bundle"))


def _int_list(values: list, name: str) -> list[int]:
    if not isinstance(values, list) or len(values) != NUM_RESOURCES:
        raise ValueError(f"{name} must be a {NUM_RESOURCES}-length list")
    try:
        return [int(n) for n in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be integers") from exc
