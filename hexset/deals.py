# SPDX-License-Identifier: GPL-3.0-only
"""Robber deals, as the seat asked: whether to promise a card for the
robber's mercy.

A seat that has to move the robber -- after its seven, or its knight -- may
ask the table for one card of a resource it names (`Ask`). A seat holding
one may promise it, for one of two deals:

* **no block** (`NO_BLOCK`): the robber goes on no hex where the payer has a
  building, so the payer is neither blocked nor robbed this move;
* **no steal** (`NO_STEAL`): the robber may stand on the payer's hexes, but
  the payer is not the one robbed.

A payer the robber spared pays once it has moved: the asked card for one of
the mover's choosing, an ordinary exchange on the mover's turn. A payer hit
anyway pays nothing.

This module is the answering side: `answer` says whether a seat promises.
Nothing here is text; a client mirroring a table where deals are made in
chat translates the ask into an `Ask` and the answer back
(`hexset.seat.Seat.deal_answer`).

**How a seat answers.** Only under `DealParams` with `answer` on, for the
`kinds` it makes, only while it holds the card, and only as the mover's
likeliest target (`targeted`): a deal buys off a hit, so a seat the mover is
not coming for has nothing to buy. A gate may answer for
itself with `deal_answer(game, seat, ask, promised, model) -> bool`;
otherwise `default_answer` prices the promise with the gate's own
`gains_many` (`promise_value`), so each bot values a deal in its own terms.
Either way the mover is read through a `RobberModel` -- `PlainMover`, or one
a runtime registers by name (`register_robber_model`, `DealParams.model`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence

from .board.board import pips
from .board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE
from .robber import allowed_targets, may_stand, occupants, victims
from .state import NO_OWNER, GameState
from .victory import public_victory_points

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .game import Game

NO_BLOCK = "no_block"
NO_STEAL = "no_steal"
KINDS = (NO_BLOCK, NO_STEAL)

#: Non-seven rolls the robber stands for, on the dice alone: a seven comes
#: one roll in six, so five others are expected before it (knights only
#: shorten it). What `PlainMover` assumes; a fitted model says its own.
DICE_STAY = 5.0


@dataclass(frozen=True)
class DealParams:
    """How one seat deals when asked: whether it promises at all
    (`answer`), which kinds of deal (`kinds`), and the registered
    `RobberModel` it reads a mover with (`model`). Off by default."""

    answer: bool = False
    kinds: tuple[str, ...] = KINDS
    model: str = "plain"

    def __post_init__(self) -> None:
        kinds = tuple(k for k in KINDS if k in self.kinds)
        unknown = sorted(set(self.kinds) - set(KINDS))
        if unknown or not kinds:
            raise ValueError(f"deal kinds are {', '.join(KINDS)}, not {list(self.kinds)}")
        object.__setattr__(self, "kinds", kinds)


@dataclass(frozen=True)
class Ask:
    """The mover's ask: one card of `resource` for a deal of `kind`."""

    mover: int
    kind: str
    resource: int


# --- the mover's robber, as another seat reads it ---------------------------


class RobberModel(Protocol):
    """A reading of where a mover puts the robber and whom it robs, from
    public facts only, and how long the robber then stands. `promised` maps
    each seat holding an accepted promise to its kind."""

    #: Non-seven rolls the robber is expected to stand where it is put.
    stay: float

    def hexes(self, state: GameState, mover: int, allowed: frozenset[int] | None,
              promised: Mapping[int, str]) -> dict[int, float]:
        """The chance of each hex the robber may go to (`allowed`, `None`
        for any it may stand on); they sum to 1."""

    def victims(self, state: GameState, mover: int, hex_index: int,
                promised: Mapping[int, str]) -> dict[int, float]:
        """The chance of each seat the mover may rob on `hex_index`; empty
        when it may rob nobody there."""


def candidate_hexes(state: GameState, allowed: frozenset[int] | None) -> list[int]:
    """The hexes the robber may go to: `allowed`, or every hex it may stand on."""
    return [h for h in range(state.board.num_hexes)
            if may_stand(state, h) and (allowed is None or h in allowed)]


def weights_on(state: GameState, hex_index: int) -> dict[int, int]:
    """Each seat's buildings on a hex, a settlement 1 and a city 2."""
    out: dict[int, int] = {}
    for v in state.board.topology.hex_vertices[hex_index]:
        owner = state.vertex_owner[v]
        if owner != NO_OWNER:
            out[owner] = out.get(owner, 0) + int(state.vertex_building[v])
    return out


def softmax(scores: Mapping[int, float]) -> dict[int, float]:
    """Chances in proportion to `exp(score)`, summing to 1; empty for no scores."""
    if not scores:
        return {}
    top = max(scores.values())
    exp = {k: math.exp(v - top) for k, v in scores.items()}
    total = sum(exp.values())
    return {k: e / total for k, e in exp.items()}


class PlainMover:
    """The plain reading of a mover: it blocks production, the leader's
    first, and keeps its promises. Each hex scores the production it blocks
    of every other seat (buildings times pips, over the five pips of the
    best number), each seat's counted in proportion to its public points
    over the leader's; the mover's own production there counts against it.
    A hex with a no-block payer on it is all but ruled out. The chance of a
    hex is the softmax of the scores. The victim is any seat with cards on
    the hex, a no-steal payer all but ruled out."""

    stay = DICE_STAY
    #: The score of a hex the mover has promised not to block.
    PROMISE = -20.0

    def hexes(self, state, mover, allowed, promised):
        vp = [public_victory_points(state, s) for s in range(state.num_players)]
        top = max([vp[s] for s in range(state.num_players) if s != mover] or [0]) or 1
        scores = {}
        for h in candidate_hexes(state, allowed):
            p = pips(state.board.tokens[h]) / 5
            on = weights_on(state, h)
            score = sum(w * p * vp[s] / top for s, w in on.items() if s != mover) - on.get(mover, 0) * p
            if any(promised.get(s) == NO_BLOCK for s in on if s != mover):
                score += self.PROMISE
            scores[h] = score
        return softmax(scores)

    def victims(self, state, mover, hex_index, promised):
        able = victims(state, hex_index, mover)
        return softmax({s: self.PROMISE if promised.get(s) == NO_STEAL else 0.0 for s in able})


_MODELS: dict[str, Callable[[], RobberModel]] = {"plain": PlainMover}
_BUILT: dict[str, RobberModel] = {}


def register_robber_model(name: str, factory: Callable[[], RobberModel]) -> None:
    """Make a `RobberModel` nameable in `DealParams.model`. `plain` is the
    engine's own and cannot be replaced; registering the same factory again
    is a no-op."""
    held = _MODELS.get(name)
    if held is not None and held is not factory:
        raise ValueError(f"robber model {name!r} is already registered")
    _MODELS[name] = factory


def robber_model(name: str) -> RobberModel:
    """The registered model `name`, built once per process."""
    if name not in _BUILT:
        if name not in _MODELS:
            raise ValueError(f"unknown robber model {name!r}; a runtime registers its own "
                             f"(`register_robber_model`)")
        _BUILT[name] = _MODELS[name]()
    return _BUILT[name]


# --- what a promise is worth -----------------------------------------------


def hit_chances(model: RobberModel, state: GameState, mover: int, allowed: frozenset[int] | None,
                promised: Mapping[int, str], seat: int) -> tuple[dict[int, float], dict[int, float]]:
    """`(chance of each hex, chance seat is robbed given that hex)` for one
    reading of the mover."""
    hexes = model.hexes(state, mover, allowed, promised)
    robbed = {h: model.victims(state, mover, h, promised).get(seat, 0.0) for h in hexes}
    return hexes, robbed


def targeted(game: "Game", seat: int, ask: Ask, promised: Sequence[int], model: RobberModel) -> bool:
    """Whether `seat` is the seat the mover is likeliest to hit, read from
    `seat`'s own view: blocked (a building on the robber's hex) for a
    no-block ask, robbed for a no-steal one, with the promises already made
    taken as kept. A tie for the likeliest counts; a seat with no chance of
    being hit is never targeted. What a deal is predicated on: a seat the
    mover is not coming for has no hit to buy off, and says nothing."""
    st = game.state(seat).state
    mover, taken = ask.mover, {s: ask.kind for s in promised}
    hexes = model.hexes(st, mover, allowed_targets(st, mover, game.robber_allowed), taken)
    if ask.kind == NO_BLOCK:
        hit = {h: occupants(st, h) for h in hexes}
    else:
        hit = {h: model.victims(st, mover, h, taken) for h in hexes}
    chance = {}
    for s in range(st.num_players):
        if s == mover or s in game.locked:
            continue
        chance[s] = sum(p * (s in hit[h] if ask.kind == NO_BLOCK else hit[h].get(s, 0.0))
                        for h, p in hexes.items())
    return chance.get(seat, 0.0) > 0 and chance[seat] >= max(chance.values())


def promise_value(game: "Game", seat: int, gate: object, ask: Ask, promised: Sequence[int],
                  model: RobberModel) -> float:
    """What promising is worth to `seat` over staying silent, in its gate's
    own units: the swap it makes if spared, times the chance it is spared
    having promised, plus what the promise saves it in robbery.

    Robbery is priced as cards: a steal is one card from the seat's hand,
    drawn in proportion to what it holds; a block is the seat's buildings on
    the hex times its pips over 36, for each of the model's `stay` rolls, in
    the hex's resource. Each card is priced at what the gate makes of giving
    it to the mover for nothing. The swap is the asked card for one of the
    four others, equally likely, since the mover chooses. Both readings of
    the mover -- with this promise and without -- take the promises already
    made as accepted."""
    from .trading import valued_many

    view = game.state(seat)
    state, mover = view.state, ask.mover
    others = {s: ask.kind for s in promised}
    allowed = allowed_targets(game._state, mover, game.robber_allowed)
    hand = list(state.hands[seat])
    received = []
    for r in range(NUM_RESOURCES):
        bundle = [0] * NUM_RESOURCES
        bundle[r] = -1
        received.append(tuple(bundle))
    swaps = [r for r in range(NUM_RESOURCES) if r != ask.resource]
    for r in swaps:
        bundle = [0] * NUM_RESOURCES
        bundle[ask.resource] = -1
        bundle[r] = 1
        received.append(tuple(bundle))
    gains = valued_many(gate, view, received, [mover] * len(received))
    loss = [-g for g in gains[:NUM_RESOURCES]]
    swap = sum(gains[NUM_RESOURCES:]) / len(swaps)
    held = sum(hand)
    steal = sum(h * l for h, l in zip(hand, loss)) / held if held else 0.0

    def cost(hexes, robbed):
        total = 0.0
        for h, p in hexes.items():
            w = weights_on(state, h).get(seat, 0)
            resource = TERRAIN_RESOURCE.get(state.board.terrain[h])
            block = 0.0 if not w or resource is None else (
                w * pips(state.board.tokens[h]) / 36 * model.stay * loss[int(resource)])
            total += p * (block + robbed[h] * steal)
        return total

    silent = hit_chances(model, state, mover, allowed, others, seat)
    sure = hit_chances(model, state, mover, allowed, {**others, seat: ask.kind}, seat)
    if ask.kind == NO_BLOCK:
        spared = sum(p for h, p in sure[0].items() if seat not in occupants(state, h))
    else:
        spared = sum(p * (1 - sure[1][h]) for h, p in sure[0].items())
    return spared * swap + cost(*silent) - cost(*sure)


def default_answer(game: "Game", seat: int, gate: object, ask: Ask, promised: Sequence[int],
                   model: RobberModel) -> bool:
    """Promise when it is worth more than silence (`promise_value`)."""
    return promise_value(game, seat, gate, ask, promised, model) > 0


def answer(game: "Game", seat: int, gate: object, params: DealParams | None, ask: Ask,
           promised: Sequence[int] = ()) -> bool:
    """Whether `seat` promises the card `ask` names, `promised` the seats
    that already have, in the order they did: never without `params` that
    answer this kind, nor without the card in hand, nor to itself, nor as a
    seat the mover is not likeliest to hit (`targeted`); then the gate's own
    `deal_answer` where it has one, else `default_answer`."""
    if (params is None or not params.answer or ask.kind not in params.kinds or seat == ask.mover
            or game.state(seat).state.hands[seat][ask.resource] <= 0):
        return False
    model = robber_model(params.model)
    if not targeted(game, seat, ask, promised, model):
        return False
    own = getattr(gate, "deal_answer", None)
    if own is not None:
        return bool(own(game, seat, ask, tuple(promised), model))
    return default_answer(game, seat, gate, ask, tuple(promised), model)


__all__ = [
    "NO_BLOCK", "NO_STEAL", "KINDS", "DICE_STAY", "DealParams", "Ask", "RobberModel", "PlainMover",
    "register_robber_model", "robber_model", "candidate_hexes", "weights_on", "softmax", "hit_chances",
    "targeted", "promise_value", "default_answer", "answer",
]
