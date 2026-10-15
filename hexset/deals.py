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

**The mover's side, at the engine's own tables.** A seat whose gate
carries `DealParams` with `ask` on (`Dealing` attaches them to any bot) asks
as the robber phase opens (`run_deal_event`) and the other seats answer in
turn (`answer`). A promise binds nobody. The mover sets aside the promises
it does not count (`void`: from a seat its own view says cannot hold the
card; `ignored`: from a seat it has blacklisted, `blacklist`, or one
under an embargo with it, `closed`), chooses which
of the rest to spare for (`accepted`), and its robber move then spares
those (`keep_promises`, `Game.robber_promised`). Once the robber has moved
and the turn is open to trading, if it spared a promiser it puts the deal's
exchange to the whole table as an ordinary offer (`settle_deals`): the asked
card for one of its own, which the spared promisers answer by honouring the
deal or not, the other seats as they answer any offer, and which it takes
from whichever answer its own trading picks. A spared promiser that does
not take the offer is `rejected`; a seat that promised void or was
rejected is blacklisted by that mover for the rest of the game. The mover's
hooks, each with a default:

* `deal_ask(game, seat) -> Ask | None` (`default_ask`): whether to ask, and
  for what;
* `deal_accept(game, seat, ask, promised) -> Sequence[int]`
  (`default_accept`): which of the counted promises to spare for;
* `deal_return(game, seat, payers, resource) -> int | None`
  (`default_return`): the card the mover offers back.

`DealParams(mover="value")` replaces all three with `ValueDealer`, which
reads them off the seat's own value head: it asks only when some seat's
deal -- the best robber move that spares it, plus its card, an exchange its
own trading would take -- beats the best robber move outright, spares for
the one promise worth most, and offers back the card it minds least.

A seat may answer through a registered `Answerer` in place of its own rule
(`register_answerer`, `DealParams.answerer`): it says whether the seat
promises and, once spared, whether it honours the deal.

A seat under a trade embargo with the mover (`Game.embargoes`, either
way) is out of its deals altogether: it is never asked or counted
(`closed`), never spared for a promise, and never promises it. A deal is
paid by an exchange, which the embargo forbids.

A game where no gate asks runs no deal and plays exactly as it would
without this module.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence

from .board.board import pips
from .board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE
from .robber import allowed_targets, may_stand, occupants, victims
from .state import NO_OWNER, GameState, copy_state
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

#: How a mover may decide its deals (`DealParams.mover`).
MOVERS = ("plain", "value")

#: The answerer `DealParams.answerer` names to answer off the seat's own
#: value head (`ValueDealer.promise`) rather than a registered `Answerer`.
VALUE_ANSWERER = "value"


@dataclass(frozen=True)
class DealParams:
    """How one seat deals: whether it promises when asked (`answer`),
    whether it asks when it moves the robber (`ask`), which kinds of deal
    either way (`kinds`), and the registered `RobberModel` it reads a mover
    with (`model`). Off on both counts by default."""

    answer: bool = False
    kinds: tuple[str, ...] = KINDS
    model: str = "plain"
    ask: bool = False
    #: How the mover decides: `"plain"` (`default_ask`, `default_accept`,
    #: `default_return`) or `"value"` (`ValueDealer`, off the seat's value head).
    mover: str = "plain"
    #: A registered `Answerer` that promises and pays for this seat in place
    #: of its own rule (`register_answerer`), or `"value"`, its own value head
    #: (`ValueDealer.promise`); `None`, its own rule.
    answerer: str | None = None
    #: The mover's chance that a spared promiser pays, which `ValueDealer`
    #: weighs each payment by where it has no promise model.
    pay: float = 1.0
    #: A registered `PromiseModel` (`register_promise_model`) that tells
    #: `ValueDealer` how likely a seat holding the card is to promise, and a
    #: spared promiser to pay; `None`, every holder promises and pays at `pay`.
    promise: str | None = None
    #: What an ask costs the mover, in its win chance: `ValueDealer` asks
    #: only where the ask is worth more.
    ask_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.mover not in MOVERS:
            raise ValueError(f"a deal mover is one of {', '.join(MOVERS)}, not {self.mover!r}")
        if not 0.0 < self.pay <= 1.0:
            raise ValueError(f"a payment chance is in (0, 1], not {self.pay}")
        if self.ask_cost < 0.0:
            raise ValueError(f"an ask's cost is not negative, not {self.ask_cost}")
        if self.answerer is not None and not self.answer:
            raise ValueError("an answerer answers only for a seat that answers (`answer`)")
        kinds = tuple(k for k in KINDS if k in self.kinds)
        unknown = sorted(set(self.kinds) - set(KINDS))
        if unknown or not kinds:
            raise ValueError(f"deal kinds are {', '.join(KINDS)}, not {list(self.kinds)}")
        object.__setattr__(self, "kinds", kinds)

    @classmethod
    def parse(cls, text: str) -> "DealParams":
        """`answer`, `ask`, the kinds (`noblock`, `nosteal`; both when none is
        named), `model=<name>`, `mover=<plain|value>`, `answerer=<name>`,
        `pay=<chance>`, `promise=<name>` and `cost=<win chance>`, comma
        separated: ``answer,ask,noblock,mover=value``.
        `none` alone is a seat that neither answers nor asks."""
        if text.strip() == "none":
            return cls()
        answer = ask = False
        kinds: list[str] = []
        model, mover, answerer, pay, promise, cost = "plain", "plain", None, 1.0, None, 0.0
        for part in filter(None, (p.strip() for p in text.split(","))):
            if part == "answer":
                answer = True
            elif part == "ask":
                ask = True
            elif part in ("noblock", "nosteal"):
                kinds.append(NO_BLOCK if part == "noblock" else NO_STEAL)
            elif part.startswith("model="):
                model = part[len("model="):]
            elif part.startswith("mover="):
                mover = part[len("mover="):]
            elif part.startswith("answerer="):
                answerer = part[len("answerer="):]
            elif part.startswith("pay="):
                pay = float(part[len("pay="):])
            elif part.startswith("promise="):
                promise = part[len("promise="):]
            elif part.startswith("cost="):
                cost = float(part[len("cost="):])
            else:
                raise ValueError(f"unknown deal setting {part!r}: answer, ask, noblock, nosteal, "
                                 "model=<name>, mover=<plain|value>, answerer=<name>, pay=<chance>, "
                                 "promise=<name>, cost=<win chance>")
        return cls(answer=answer, ask=ask, kinds=tuple(kinds) or KINDS, model=model, mover=mover,
                   answerer=answerer, pay=pay, promise=promise, ask_cost=cost)

    def spec(self) -> str:
        """The text `parse` reads back to these params."""
        parts = [p for p, on in (("answer", self.answer), ("ask", self.ask)) if on]
        if self.kinds != KINDS:
            parts += ["noblock" if k == NO_BLOCK else "nosteal" for k in self.kinds]
        if self.model != "plain":
            parts.append(f"model={self.model}")
        if self.mover != "plain":
            parts.append(f"mover={self.mover}")
        if self.answerer is not None:
            parts.append(f"answerer={self.answerer}")
        if self.pay != 1.0:
            parts.append(f"pay={self.pay:g}")
        if self.promise is not None:
            parts.append(f"promise={self.promise}")
        if self.ask_cost != 0.0:
            parts.append(f"cost={self.ask_cost:g}")
        return ",".join(parts)


#: How a seat deals unless something says otherwise: it answers both kinds
#: when asked and never asks. Answering is passive -- a promise binds only
#: when the mover spares the seat for it -- so every seat the arena or a
#: table spawns answers (`hexset.arena.spawn`); `^none` opts a seat out.
ANSWERING = DealParams(answer=True)


def deal_params_of(gate: object) -> DealParams | None:
    """The `DealParams` a seat's gate carries; `ANSWERING` for a gate that
    names none, and `None` for no gate: a seat without one does not deal."""
    if gate is None:
        return None
    return getattr(gate, "deal_params", ANSWERING)


class Dealing:
    """`bot` dealing under `params`. Every name but `deal_params`,
    `deal_answerer` (the registered `Answerer` the params name, or `None`) and
    `deal_dealer` (the `ValueDealer` a value mover decides with, or `None`)
    -- its moves, its gate, its own deal hooks -- is the bot's, read, written
    and deleted through, as `hexset.bots.TradesBy` passes its trader's.
    `seed` seeds the answerer's and the dealer's draws."""

    __slots__ = ("bot", "deal_params", "deal_answerer", "deal_dealer", "deal_seed")

    def __init__(self, bot: object, params: DealParams, seed: int = 0) -> None:
        object.__setattr__(self, "bot", bot)
        object.__setattr__(self, "deal_params", params)
        object.__setattr__(self, "deal_seed", seed)
        dealer = (ValueDealer(value_net(bot), params, random.Random(seed + 1), gate=bot)
                  if params.ask and params.mover == "value" else None)
        if params.answerer == VALUE_ANSWERER:
            answerer = dealer if dealer is not None else ValueDealer(
                value_net(bot), params, random.Random(seed + 1), gate=bot)
        else:
            answerer = None if params.answerer is None else answerer_for(params.answerer, random.Random(seed))
        object.__setattr__(self, "deal_answerer", answerer)
        object.__setattr__(self, "deal_dealer", dealer)

    def choose(self, game: "Game"):
        return self.bot.choose(game)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "bot"), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.bot, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self.bot, name)

    def __reduce__(self):
        return (Dealing, (self.bot, self.deal_params, self.deal_seed))


def deal_seed(rng: random.Random) -> int:
    """A seed for `Dealing` off `rng`'s state, without drawing from it: the
    seat's own draws stay exactly as they were."""
    return int.from_bytes(hashlib.sha256(("deals" + repr(rng.getstate())).encode()).digest()[:8], "big")


# --- answerers -----------------------------------------------------------------


class Answerer(Protocol):
    """A seat's answers to robber deals in place of its own rule: whether it
    promises (`promise`, asked only while it holds the card) and, once the
    robber has spared it, whether it honours the deal by taking the mover's
    offer (`pays`); not honouring, it answers the offer as it would any."""

    def promise(self, game: "Game", seat: int, ask: "Ask", promised: Sequence[int]) -> bool: ...

    def pays(self, game: "Game", seat: int, deal: "Deal") -> bool: ...


_ANSWERERS: dict[str, Callable[[random.Random], Answerer]] = {}


def register_answerer(name: str, factory: Callable[[random.Random], Answerer]) -> None:
    """Make `factory(rng) -> Answerer` the answerer `DealParams(answerer=name)` names."""
    _ANSWERERS[name] = factory


def answerer_for(name: str, rng: random.Random) -> Answerer:
    """The registered answerer `name`, built on `rng`."""
    if name not in _ANSWERERS:
        raise ValueError(f"no answerer is registered as {name!r} (`register_answerer`); "
                         f"registered: {sorted(_ANSWERERS) or 'none'}")
    return _ANSWERERS[name](rng)


class PromiseModel(Protocol):
    """How a mover reads the table before it asks: the chance `seat`,
    holding the card, promises `ask` (`promise_chance`), and the chance a
    seat it spared for a promise pays (`pay_chance`)."""

    def promise_chance(self, game: "Game", seat: int, ask: "Ask") -> float: ...

    def pay_chance(self, game: "Game", mover: int) -> float: ...


_PROMISE_MODELS: dict[str, PromiseModel] = {}


def register_promise_model(name: str, model: PromiseModel) -> None:
    """Make `model` the promise model `DealParams(promise=name)` names."""
    _PROMISE_MODELS[name] = model


def promise_model(name: str) -> PromiseModel:
    """The registered promise model `name`."""
    if name not in _PROMISE_MODELS:
        raise ValueError(f"no promise model is registered as {name!r} (`register_promise_model`); "
                         f"registered: {sorted(_PROMISE_MODELS) or 'none'}")
    return _PROMISE_MODELS[name]


@dataclass(frozen=True)
class Ask:
    """The mover's ask: one card of `resource` for a deal of `kind`."""

    mover: int
    kind: str
    resource: int


@dataclass
class Deal:
    """One ask and what came of it, as the game keeps it (`Game.deal_log`).
    `promised` every seat that promised, in order; of those, `void` the ones
    the mover's own view said could not hold the card and `ignored` the ones
    it had blacklisted or was under an embargo with, which it did not count; `accepted` the counted
    promises it chose to spare for. Once the robber has moved: `hex` and
    `victim`, where it went and whom it robbed; `kept` and `broken`, the
    counted promisers it spared and did not. Once the turn opens to trading:
    `offered` the card the mover offered for the asked one (`None`: no offer,
    with no spared promiser or no card to give); `taker` the seat whose
    answer it took (any seat, a promiser or not); `paid` that exchange as
    `(taker, card returned)` where the taker had promised; `rejected` the
    spared promisers that did not take the offer; `unpaid` the spared
    promisers when the mover had nothing to offer back."""

    turn: int
    mover: int
    kind: str
    resource: int
    promised: tuple[int, ...] = ()
    void: tuple[int, ...] = ()
    ignored: tuple[int, ...] = ()
    accepted: tuple[int, ...] = ()
    hex: int | None = None
    victim: int | None = None
    kept: tuple[int, ...] = ()
    broken: tuple[int, ...] = ()
    offered: int | None = None
    taker: int | None = None
    paid: tuple[tuple[int, int], ...] = ()
    rejected: tuple[int, ...] = ()
    unpaid: tuple[int, ...] = ()

    @property
    def counted(self) -> tuple[int, ...]:
        """The promises the mover counted: `promised` less `void` and `ignored`."""
        return tuple(s for s in self.promised if s not in self.void and s not in self.ignored)

    def to_json(self) -> dict:
        return {"turn": self.turn, "mover": self.mover, "kind": self.kind, "resource": self.resource,
                "promised": list(self.promised), "void": list(self.void), "ignored": list(self.ignored),
                "accepted": list(self.accepted), "hex": self.hex, "victim": self.victim,
                "kept": list(self.kept), "broken": list(self.broken), "offered": self.offered,
                "taker": self.taker, "paid": [list(p) for p in self.paid],
                "rejected": list(self.rejected), "unpaid": list(self.unpaid)}

    @classmethod
    def from_json(cls, d: Mapping) -> "Deal":
        return cls(turn=d["turn"], mover=d["mover"], kind=d["kind"], resource=d["resource"],
                   promised=tuple(d["promised"]), void=tuple(d.get("void", ())),
                   ignored=tuple(d.get("ignored", ())), accepted=tuple(d["accepted"]), hex=d["hex"],
                   victim=d["victim"], kept=tuple(d["kept"]), broken=tuple(d["broken"]),
                   offered=d.get("offered"), taker=d.get("taker"),
                   paid=tuple(tuple(p) for p in d["paid"]), rejected=tuple(d.get("rejected", ())),
                   unpaid=tuple(d.get("unpaid", ())))


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
    answer this kind, nor without the card in hand, nor to itself; then the
    seat's registered answerer where it has one (`Dealing.deal_answerer`);
    otherwise never as a seat the mover is not likeliest to hit
    (`targeted`), and then the gate's own `deal_answer` where it has one,
    else `default_answer`. Never under a trade embargo with the mover
    (`closed`): the deal could not be paid."""
    if (params is None or not params.answer or ask.kind not in params.kinds or seat == ask.mover
            or game.state(seat).state.hands[seat][ask.resource] <= 0
            or seat in closed(game, ask.mover)):
        return False
    answerer = getattr(gate, "deal_answerer", None)
    if answerer is not None:
        return bool(answerer.promise(game, seat, ask, tuple(promised)))
    model = robber_model(params.model)
    if not targeted(game, seat, ask, promised, model):
        return False
    own = getattr(gate, "deal_answer", None)
    if own is not None:
        return bool(own(game, seat, ask, tuple(promised), model))
    return default_answer(game, seat, gate, ask, tuple(promised), model)


# --- the mover's side ------------------------------------------------------


def default_ask(game: "Game", seat: int, gate: object, params: DealParams) -> Ask | None:
    """Ask for the card the gate most wants for one of its own, with the
    first of its kinds; nothing when no one-for-one swap gains."""
    from .trading import valued_many

    view = game.state(seat)
    hand = view.state.hands[seat]
    swaps = [(r, y) for r in range(NUM_RESOURCES) for y in range(NUM_RESOURCES)
             if r != y and hand[y] > 0]
    if not swaps:
        return None
    received = []
    for r, y in swaps:
        bundle = [0] * NUM_RESOURCES
        bundle[r], bundle[y] = 1, -1
        received.append(tuple(bundle))
    # Priced against each seat that might pay, the best of them counting:
    # the mover takes whichever promises it likes.
    shut = closed(game, seat)
    others = [s for s in range(game.num_players) if s != seat and s not in game.locked and s not in shut]
    if not others:
        return None
    gains = valued_many(gate, view, received * len(others), [s for s in others for _ in received])
    worth = [max(gains[i + k * len(received)] for k in range(len(others))) for i in range(len(received))]
    best = max(range(len(swaps)), key=lambda i: (worth[i], -swaps[i][0], -swaps[i][1]))
    if worth[best] <= 0:
        return None
    return Ask(seat, params.kinds[0], swaps[best][0])


def default_accept(game: "Game", seat: int, bot: object, ask: Ask, promised: Sequence[int]) -> tuple[int, ...]:
    """Every promise but from the seats the mover's own robber move would hit
    anyway, asked of the bot before any promise binds it: its move stands
    and what the others pay is a gain."""
    from .actions import victim_of

    action = bot.choose(game)
    state = game._state
    if ask.kind == NO_BLOCK:
        hit = set(occupants(state, action.a))
    else:
        victim = victim_of(game, action.b)
        hit = set() if victim is None else {victim}
    return tuple(s for s in promised if s not in hit)


def default_return(game: "Game", seat: int, gate: object, payers: Sequence[int],
                   resource: int) -> int | None:
    """The card the mover offers back for `resource`: of those it holds but
    `resource`, the one its gate prices best against the best of `payers`."""
    from .trading import valued_many

    view = game.state(seat)
    hand = view.state.hands[seat]
    choices = [y for y in range(NUM_RESOURCES) if y != resource and hand[y] > 0]
    if not choices or not payers:
        return None
    pairs = [(y, p) for y in choices for p in payers]
    gains = valued_many(gate, view, [_swap(resource, y) for y, _ in pairs], [p for _, p in pairs])
    best = {y: max(g for (z, _), g in zip(pairs, gains) if z == y) for y in choices}
    return max(choices, key=lambda y: (best[y], -y))


def blacklist(game: "Game", mover: int) -> frozenset[int]:
    """The seats whose promises `mover` no longer counts this game: every
    seat that promised it a card its view said the seat could not hold
    (`Deal.void`), or that it spared and that then did not take its offer
    (`Deal.rejected`)."""
    out: set[int] = set()
    for d in game.deal_log:
        if d.mover == mover:
            out.update(d.void)
            out.update(d.rejected)
    return frozenset(out)


def closed(game: "Game", seat: int) -> frozenset[int]:
    """The seats under a trade embargo with `seat` now, either way
    (`Game.embargoes`). A deal is paid by an exchange the embargo forbids, so
    a mover neither asks these seats nor counts their promises, and they
    promise it nothing."""
    return frozenset(b if a == seat else a for a, b in game.embargoes if seat in (a, b) and a != b)


# --- the mover's side, off a value head -------------------------------------

#: Draws from the mover's own view that `ValueDealer` reads a victim's
#: steal odds off. Cheap: no network call is made per draw.
STEAL_SAMPLES = 64


def value_net(bot: object) -> object:
    """The network inside `bot` that `ValueDealer` values with: the first of
    `bot` and what it wraps (`Dealing.bot`, `hexset.bots.TradesBy.mover`, a
    search's own gate, `hexset.clients.netbot.GatedSearch.gate`) with a
    `policy` that has a value head (`value_rows`), a `seat_at` and a
    `gains_many` in win probability (`hexset.clients.netbot.NetworkBot`).
    `ValueError` where there is none."""
    node = bot
    for _ in range(8):
        policy = _own(node, "policy")
        if policy is not None and hasattr(policy, "value_rows") and hasattr(node, "seat_at"):
            return node
        node = next((n for n in (_own(node, "bot"), _own(node, "mover"), _own(node, "gate")) if n is not None), None)
        if node is None:
            break
    raise ValueError("a value mover needs a network with a value head to deal with; "
                     f"{type(bot).__name__} has none")


def _own(node: object, name: str) -> object:
    """`node`'s own attribute `name`, never one a wrapper forwards; `None` without one."""
    try:
        return object.__getattribute__(node, name)
    except AttributeError:
        return None


@dataclass
class _Plan:
    """What `ValueDealer` worked out for one robber move: each card its
    trading wants one for one, with the card it gives for it and the seats it
    would ask (`wants`); the robber move's options, the network's own ranking
    of them (`prior`) and its pick (`pick`); for each seat considered, the
    best-ranked move that spares it (`spare_at`); the mover's win chance after
    each move valued (`value`), with the positions after it, one per card the
    steal may take, and that card's chance (`after`); each (seat, card)
    exchange that would make a deal and that its trading would take, priced
    before the move (`gain`); what sparing a seat for a card is worth over
    the pick, the exchange priced after the steal (`worth`), with the chance
    a spared promiser pays (`pay`); and the chance each seat holds each
    resource (`held`)."""

    key: tuple
    wants: dict[int, tuple[int, tuple[int, ...]]]
    options: list
    prior: list[float]
    pick: int
    spare_at: dict[int, int]
    value: dict[int, float]
    after: dict[int, list[tuple["Game", float]]]
    gain: dict[tuple[int, int], float]
    held: dict[tuple[int, int], float]
    odds: dict[int, list[float]]
    holding: dict[tuple[int, int], GameState]
    sample: GameState | None
    worth: dict[tuple[int, int], float] = field(default_factory=dict)
    pay: float = 1.0

    @property
    def best(self) -> float:
        return self.value[self.pick]

    @property
    def spare(self) -> dict[int, float]:
        return {s: self.value[i] for s, i in self.spare_at.items() if i in self.value}


class _Takes:
    """A chance source whose steal takes `resource`; everything else is `inner`'s."""

    def __init__(self, inner: object, resource: int) -> None:
        self.inner, self.resource = inner, resource

    def steal(self, hand):
        return self.resource

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


class ValueDealer:
    """The mover's side of robber deals off its own value head (`net`, found
    by `value_net`) and its own trading (`gate`), in win chance throughout.

    * **What to ask for.** Every card its trading wants, one for one, and
      the seats it would offer it to: its proposer's offers in turn
      (`candidates`, each offered one struck from the next), before any veto.
      Each (seat, card) exchange is priced before the move by the network's
      `gains_many` against every card it could give back, and counts at the
      best of those its own trading, veto and all, would take.
    * **The robber.** The network's own ranking of the robber moves picks
      the move it would make (`pick`) and, for each seat it would ask, the
      best-ranked move that spares it (`spare_at`). Only those moves are
      valued: the mover's win chance after each, averaged over the card the
      steal takes. The network reads the other seats only by hand size and
      the public record, so the steal is all the hidden cards change; its
      odds come from `STEAL_SAMPLES` draws from the mover's own view.
    * **Ask** (`ask`): a seat is worth asking for a card when sparing it and
      taking its card beats the pick, priced as its promise would be
      (`worth`): `value[spare_at[s]] + pay * gain(s, r) > value[pick]`, with
      the exchange priced where it would be made -- in the positions after
      the move that spares the seat, card by card of the steal, at the best
      card back its own trading would take there, and at nothing where it
      would take none. `pay` is its chance that a spared seat honours the
      deal: the promise model's (`DealParams.promise`), else `DealParams.pay`.
      The exchange priced before the move picks which seats are priced so;
      only a seat whose promise it would take counts towards an ask. An ask
      for a card is worth the deal it expects to make: one, with the best of
      the seats that promise, each seat by the chance it holds the card and,
      under a promise model, promises it. It asks for the card worth most,
      where that beats what an ask costs (`DealParams.ask_cost`). Blacklisted
      seats (`blacklist`) and seats under an embargo with it (`closed`) are
      not considered.
    * **Spare** (`accept`): of the counted promises, the one whose deal is
      worth most, where that beats the pick; one, since a deal is one
      exchange, priced as the ask priced it. A promiser it had not considered
      is valued then.
    * **Offer back** (`give_back`): of the cards its own trading would take
      the exchange for from a spared promiser, the one it minds least,
      priced after the steal; none such, no offer.

    * **Answer** (`promise`, as the seat's `Answerer` under
      `DealParams(answerer="value")`): asked by another mover, the seat
      promises where its win chance promising beats its win chance silent.
      Either way the mover's robber move is read off the params' robber
      model (`model`), the promises already made taken as kept, and each
      move is valued card by card of its steal, as above. Promising, the
      model reads the seat as promised too, and where the move spares it,
      the exchange is priced after the move, the asked card for each card
      the mover may give back, by the chance the mover holds it (the card
      for nothing where it holds none). Only the seat the mover is likeliest
      to hit answers (`targeted`). A spared promiser pays (`pays`).

    One plan per robber move (keyed by the turn, the seat and where the
    robber stands), kept from the ask to the choice of whom to spare."""

    def __init__(self, net: object, params: DealParams, rng: random.Random, gate: object = None) -> None:
        self.net, self.params, self.rng, self.gate = net, params, rng, gate
        self._plan: _Plan | None = None

    @staticmethod
    def _key(game: "Game", seat: int) -> tuple:
        return (game.turns, seat, game._state.robber)

    def _menu(self, game: "Game", seat: int, others: Sequence[int]) -> dict[int, tuple[int, tuple[int, ...]]]:
        """Every one-for-one exchange its trading would offer, in order, as
        {card wanted: (card given, the seats it is offered to)}: its
        proposer's `candidates` asked again with each offer struck, before any
        veto (`hexset.bots.Vetoed`, which `ask` applies seat by seat)."""
        trader = _own(self.gate, "trader") if self.gate is not None else None
        trader = trader if trader is not None else self.gate
        proposer = _own(trader, "proposer") if trader is not None else None
        proposer = proposer if proposer is not None else trader
        candidates = getattr(proposer, "candidates", None) if proposer is not None else None
        if candidates is None or not others:
            return {}
        view = game.state(seat)
        wants: dict[int, tuple[int, tuple[int, ...]]] = {}
        offered: list[tuple[int, ...]] = []
        for _ in range(4 * NUM_RESOURCES):
            menu = list(candidates(view, list(others), turn=game.turns, already_offered=tuple(offered)))
            if not menu or tuple(menu[0][1]) in offered:
                break
            bundle = tuple(menu[0][1])
            offered.append(bundle)
            got = [r for r, n in enumerate(bundle) if n > 0]
            gave = [r for r, n in enumerate(bundle) if n < 0]
            if sorted(abs(n) for n in bundle if n) == [1, 1] and got[0] not in wants:
                wants[got[0]] = (gave[0], tuple(dict.fromkeys(t for t, b in menu if tuple(b) == bundle)))
        return wants

    def _veto(self) -> object:
        """The gate vetoing its trading (`hexset.bots.Vetoed.veto`), or `None`.
        A veto other than this network would price the exchange itself, so
        then it is asked through `hexset.trading.pick` instead (`plan`)."""
        trader = _own(self.gate, "trader") if self.gate is not None else None
        return _own(trader, "veto") if trader is not None else None

    def plan(self, game: "Game", seat: int) -> _Plan:
        key = self._key(game, seat)
        if self._plan is not None and self._plan.key == key:
            return self._plan
        from .actions import ActionType, legal_actions

        view = game.state(seat)
        banned = blacklist(game, seat) | closed(game, seat)
        others = [s for s in range(game.num_players) if s != seat and s not in game.locked and s not in banned]
        self.net.seat_at(game)
        wants = self._menu(game, seat, others)
        options = [a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER]
        (prior, _), = self.net.policy.score_rows([(game, seat, tuple(options))])
        prior = [float(x) for x in prior]
        pick = max(range(len(options)), key=lambda i: (prior[i], -i))
        plan = _Plan(key, wants, options, prior, pick, {}, {}, {}, {}, {}, {}, {}, None,
                     pay=self._pay_chance(game, seat))
        hand = view.state.hands[seat]
        # Each wanted card against every card it could give, not only the one its trading names:
        # a better card back can clear the veto where the trading's own choice does not.
        pairs = [(t, r, y) for r, (give, seats) in wants.items() for t in seats
                 for y in range(NUM_RESOURCES) if y != r and hand[y] > 0]
        if pairs:
            from .trading import clears_floor

            self.net.seat_at(game)
            gains = self.net.gains_many(view, [_swap(r, y) for _, r, y in pairs], [t for t, _, _ in pairs])
            priced: dict[tuple[int, int], list[tuple[float, int]]] = {}
            for (t, r, y), g in zip(pairs, gains):
                if g > -1.0:                      # -1: cannot be covered
                    priced.setdefault((t, r), []).append((g, y))
            seats = sorted({t for t, _ in priced})
            if seats:
                self._beliefs(view, plan, others)
                for t in seats:
                    plan.spare_at[t] = self._spare_move(game, plan, t)
                self._value(game, seat, plan, {plan.pick, *[i for i in plan.spare_at.values() if i is not None]})
            # Its own trading takes an exchange it offered that seat (the proposer's rules) that clears
            # the veto's floor; the veto is this network, so `g` is the veto's own gain. Another card back
            # than the proposer's own is asked of its trading (`_takes`).
            veto = self._veto()
            for (t, r), options in priced.items():
                i = plan.spare_at.get(t)
                if i is None:
                    continue
                give = wants[r][0]
                for g, y in sorted(options, key=lambda o: (-o[0], o[1])):
                    if plan.value[i] + plan.pay * g <= plan.best:
                        break                     # the rest are worth less
                    if y == give and (veto is None or veto is self.net):
                        takes = veto is None or clears_floor(g, veto)
                    else:
                        takes = self._takes(game, seat, t, _swap(r, y), g)
                    if takes:
                        plan.gain[(t, r)] = g
                        break
            # Each pair the price before the move would ask for, priced as its promise would be.
            plan.worth.update(self._worths(game, seat, plan, list(plan.gain)))
        self._plan = plan
        return plan

    def _beliefs(self, view, plan: _Plan, others: Sequence[int]) -> None:
        """Each seat's chance of holding each resource, the steal's odds of
        each card from it, and a drawn world where it holds each."""
        samples = [view.sample(self.rng) for _ in range(STEAL_SAMPLES)]
        plan.sample = samples[0]
        for s in others:
            acc = [0.0] * NUM_RESOURCES
            for w in samples:
                hand = w.hands[s]
                total = sum(hand)
                for r in range(NUM_RESOURCES):
                    if hand[r] > 0:
                        plan.held[(s, r)] = plan.held.get((s, r), 0.0) + 1.0 / len(samples)
                        acc[r] += hand[r] / total / len(samples)
                        plan.holding.setdefault((s, r), w)
            plan.odds[s] = acc

    @staticmethod
    def _spare_move(game: "Game", plan: _Plan, s: int) -> int | None:
        """The best-ranked move that puts the robber on none of `s`'s hexes."""
        state = game._state
        keep = [i for i, o in enumerate(plan.options) if s not in occupants(state, o.a)]
        return max(keep, key=lambda i: (plan.prior[i], -i)) if keep else None

    def _value(self, game: "Game", seat: int, plan: _Plan, moves) -> None:
        """Value the robber moves `moves` not yet valued, card by card of the steal."""
        from .actions import apply, victim_of
        from .game import imagine

        moves = [i for i in sorted(moves) if i not in plan.value]
        if not moves:
            return
        rows, weights = [], []
        for i in moves:
            option = plan.options[i]
            victim = victim_of(game, option.b)
            odds = plan.odds.get(victim, ()) if victim is not None else ()
            cards = [(r, p) for r, p in enumerate(odds) if p > 0]
            for r, p in cards or [(None, 1.0)]:
                world = plan.holding.get((victim, r)) if r is not None else plan.sample
                g = imagine(game, self.rng, randomize_deck=False, seat=seat)
                if world is not None:
                    g.set_state(copy_state(world))
                if r is not None:
                    g.chance = _Takes(g.chance, r)
                apply(g, option)
                rows.append((g, seat))
                weights.append((i, p))
                plan.after.setdefault(i, []).append((g, p))
        for i in moves:
            plan.value[i] = 0.0
        for (i, w), row in zip(weights, self.net.policy.value_rows(rows)):
            plan.value[i] += w * row[seat]

    def ask(self, game: "Game", seat: int) -> "Ask | None":
        if NO_BLOCK not in self.params.kinds:
            return None
        plan = self.plan(game, seat)
        deals: dict[int, list[tuple[float, float]]] = {}      # card -> [(deal, chance it is offered)]
        for (t, r), deal in plan.worth.items():
            if deal > 0:
                chance = plan.held.get((t, r), 0.0) * self._promise_chance(game, t, Ask(seat, NO_BLOCK, r))
                deals.setdefault(r, []).append((deal, chance))
        # One deal is made: the best of the seats that promise, so each seat
        # counts only where no better one did.
        worth: dict[int, float] = {}
        for r, offers in deals.items():
            missed, total = 1.0, 0.0
            for deal, chance in sorted(offers, key=lambda o: -o[0]):
                total += missed * chance * deal
                missed *= 1.0 - chance
            worth[r] = total
        good = [r for r in worth if worth[r] > self.params.ask_cost]
        return Ask(seat, NO_BLOCK, max(good, key=lambda r: (worth[r], -r))) if good else None

    def _promise_chance(self, game: "Game", seat: int, ask: "Ask") -> float:
        """The chance `seat`, holding the card, promises `ask`: the promise
        model's (`DealParams.promise`), else certain."""
        if self.params.promise is None:
            return 1.0
        return promise_model(self.params.promise).promise_chance(game, seat, ask)

    def _pay_chance(self, game: "Game", seat: int) -> float:
        """The chance a promiser the mover `seat` spared pays it: the promise
        model's, else `DealParams.pay`."""
        if self.params.promise is None:
            return self.params.pay
        return promise_model(self.params.promise).pay_chance(game, seat)

    def accept(self, game: "Game", seat: int, ask: "Ask", promised: Sequence[int]) -> tuple[int, ...]:
        plan = self.plan(game, seat)
        if not plan.odds:                  # it had nothing to ask for: read the table now
            others = [s for s in range(game.num_players) if s != seat and s not in game.locked]
            self._beliefs(game.state(seat), plan, others)
        fresh = [s for s in promised if s not in plan.spare_at]
        for s in fresh:
            plan.spare_at[s] = self._spare_move(game, plan, s)
        self._value(game, seat, plan, {plan.pick} | {plan.spare_at[s] for s in fresh if plan.spare_at[s] is not None})
        r = ask.resource
        unpriced = [(s, r) for s in dict.fromkeys(promised) if (s, r) not in plan.worth]
        plan.worth.update(self._worths(game, seat, plan, unpriced))
        worth = {s: plan.worth[(s, r)] for s in promised}
        good = [s for s in promised if worth[s] > 0]
        return (max(good, key=lambda s: (worth[s], -s)),) if good else ()

    def _worths(self, game: "Game", seat: int, plan: _Plan, pairs) -> dict[tuple[int, int], float]:
        """What sparing each seat `s` for its card of `r`, for each `(s, r)`
        of `pairs`, is worth over the best move: the exchange priced in the
        positions after the move that spares `s`, card by card of the steal,
        at the best card back its own trading would take there, and at
        nothing where it would take none. Every position of every pair is
        priced in one batch (`_gains_across`)."""
        out: dict[tuple[int, int], float] = {}
        asks, where = [], []
        banned = blacklist(game, seat) | closed(game, seat)
        for s, r in pairs:
            i = plan.spare_at.get(s)
            if i is None or s in banned:
                out[(s, r)] = -math.inf
                continue
            out[(s, r)] = 0.0
            for g, p in plan.after[i]:
                hand = g.state(seat).state.hands[seat]
                cards = [y for y in range(NUM_RESOURCES) if y != r and hand[y] > 0]
                if cards:
                    asks.append((g, [_swap(r, y) for y in cards], [s] * len(cards)))
                    where.append(((s, r), g, p, cards))
        for (pair, g, p, cards), gains in zip(where, self._gains_across(seat, asks)):
            s, r = pair
            for y, gy in sorted(zip(cards, gains), key=lambda t: (-t[1], t[0])):
                if gy <= -1.0:            # the gate's "cannot be covered"
                    break
                if self._takes(g, seat, s, _swap(r, y), gy):
                    out[pair] += p * gy
                    break
        self.net.seat_at(game)
        for (s, r), expected in out.items():
            if expected != -math.inf:
                out[(s, r)] = plan.value[plan.spare_at[s]] + plan.pay * expected - plan.best
        return out

    def _gains_across(self, seat: int, asks) -> list[list[float]]:
        """The network's `gains_many` at each `(game, received,
        counterparties)` of `asks`: in one forward where it prices so
        (`hexset.clients.netbot.NetworkBot.gains_across`), else seated at
        each in turn."""
        across = getattr(self.net, "gains_across", None)
        if across is not None:
            return across(seat, asks)
        out = []
        for g, received, counterparties in asks:
            self.net.seat_at(g)
            out.append(self.net.gains_many(g.state(seat), received, counterparties))
        return out

    def _takes(self, g: "Game", seat: int, s: int, received: tuple[int, ...], gain: float) -> bool:
        """Whether its own trading takes `received` (signed towards it) from
        `s` at `g`, where the network gains `gain` from it. Where the veto is
        this network, `gain` is the veto's own reading, so the veto is its
        floor and only the proposer is asked; otherwise the whole trading."""
        from .trading import RESPONSE_ACCEPT, Response, clears_floor, pick

        if self.gate is None:
            return True
        answer = [Response(s, RESPONSE_ACCEPT, received)]
        veto = self._veto()
        if veto is not None and veto is self.net:
            if not clears_floor(gain, veto):
                return False
            trader = _own(self.gate, "trader")
            trader = trader if trader is not None else self.gate
            proposer = _own(trader, "proposer")
            return pick(g, proposer if proposer is not None else trader, seat, answer) is not None
        self.net.seat_at(g)               # the veto reads the position it is seated at
        return pick(g, self.gate, seat, answer) is not None

    def give_back(self, game: "Game", seat: int, payers: Sequence[int], resource: int) -> int | None:
        view = game.state(seat)
        hand = view.state.hands[seat]
        choices = [y for y in range(NUM_RESOURCES) if y != resource and hand[y] > 0]
        if not choices or not payers:
            return None
        self.net.seat_at(game)
        pairs = [(y, p) for y in choices for p in payers]
        gains = self.net.gains_many(view, [_swap(resource, y) for y, _ in pairs], [p for _, p in pairs])
        gain = dict(zip(pairs, gains))
        # Of the cards its own trading would take the exchange for from some payer, the best.
        for y in sorted(choices, key=lambda k: (-max(g for (z, _), g in zip(pairs, gains) if z == k), k)):
            if any(self._takes(game, seat, p, _swap(resource, y), gain[(y, p)]) for p in payers):
                return y
        return None

    def promise(self, game: "Game", seat: int, ask: "Ask", promised: Sequence[int]) -> bool:
        if not targeted(game, seat, ask, promised, robber_model(self.params.model)):
            return False
        return self.promise_worth(game, seat, ask, promised) > 0.0

    def pays(self, game: "Game", seat: int, deal: "Deal") -> bool:
        return True

    def promise_worth(self, game: "Game", seat: int, ask: "Ask", promised: Sequence[int]) -> float:
        """What promising `ask` is worth to `seat` over staying silent, in its win chance."""
        from .actions import ActionType, _robber_targets, victim_of
        from .game import Phase, imagine

        view = game.state(seat)
        state = view.state
        mover, r, kind = ask.mover, ask.resource, ask.kind
        if game.current_player != mover:
            return 0.0
        # The mover's robber moves read off the board, whatever the phase the ask
        # comes in (before the roll, or while seats discard), and valued from a
        # copy standing at the move: the mover's own cards are never read.
        self.net.seat_at(game)
        at = imagine(game, self.rng, randomize_deck=False, seat=seat)
        at.phase = Phase.ROBBER
        options = _robber_targets(at, ActionType.MOVE_ROBBER)
        if not options:
            return 0.0
        plan = _Plan(("answer", game.turns, seat, game._state.robber), {}, options, [], -1,
                     {}, {}, {}, {}, {}, {}, {}, None)
        self._beliefs(view, plan, [s for s in range(game.num_players) if s not in game.locked])
        model = robber_model(self.params.model)
        allowed = allowed_targets(state, mover, game.robber_allowed)
        taken = {s: kind for s in promised if s != seat}
        silent = self._move_chances(game, model, state, mover, allowed, taken, options)
        sure = self._move_chances(game, model, state, mover, allowed, {**taken, seat: kind}, options)
        self._value(at, seat, plan, {i for i, p in (*silent.items(), *sure.items()) if p > 0})

        def spares(i: int) -> bool:
            if kind == NO_BLOCK:
                return seat not in occupants(state, options[i].a)
            return victim_of(game, options[i].b) != seat

        backs = {y: plan.held.get((mover, y), 0.0) for y in range(NUM_RESOURCES) if y != r}
        total = sum(backs.values())
        bundles = ([(_swap(y, r), p / total) for y, p in backs.items() if p > 0] if total > 0
                   else [(tuple(-int(k == r) for k in range(NUM_RESOURCES)), 1.0)])
        spared = [i for i, p in sure.items() if p > 0 and spares(i)]
        asks = [(g, [b for b, _ in bundles], [mover] * len(bundles)) for i in spared for g, _ in plan.after[i]]
        gains = iter(self._gains_across(seat, asks))
        exchange = {}
        for i in spared:
            exchange[i] = 0.0
            for _, p in plan.after[i]:
                # -1 is the network's "cannot be covered": no payment is made.
                exchange[i] += p * sum(w * (g if g > -1.0 else 0.0) for (_, w), g in zip(bundles, next(gains)))
        worth = sum(p * (plan.value[i] + exchange.get(i, 0.0)) for i, p in sure.items())
        return worth - sum(p * plan.value[i] for i, p in silent.items())

    @staticmethod
    def _move_chances(game: "Game", model: "RobberModel", state: GameState, mover: int,
                      allowed: frozenset[int] | None, promised: Mapping[int, str], options) -> dict[int, float]:
        """The chance of each of `options` under one reading of the mover:
        its hex's chance, times its victim's chance there."""
        from .actions import victim_of

        hexes = model.hexes(state, mover, allowed, promised)
        robbed = {h: model.victims(state, mover, h, promised) for h in hexes}
        out = {}
        for i, o in enumerate(options):
            p = hexes.get(o.a, 0.0)
            if p > 0 and robbed[o.a]:
                p *= robbed[o.a].get(victim_of(game, o.b), 0.0)
            if p > 0:
                out[i] = p
        return out


def _swap(get: int, give: int) -> tuple[int, ...]:
    """The exchange, signed towards the mover, of one `give` for one `get`."""
    bundle = [0] * NUM_RESOURCES
    bundle[get], bundle[give] = 1, -1
    return tuple(bundle)


# --- the engine's side -----------------------------------------------------


def keep_promises(state: GameState, thief: int, allowed: frozenset[int] | None,
                  promised: Mapping[int, str]) -> tuple[frozenset[int] | None, frozenset[int]]:
    """The robber move that keeps every accepted promise: `(hexes it may go
    to, seats it may not rob)`. A no-block payer's hexes are out; a hex
    where only no-steal payers could be robbed is out, since a seat with
    cards there would have to be. Where nothing would be left the promises
    bind nothing: `(allowed, empty)`, and the payers hit are the deals
    broken."""
    if not promised:
        return allowed, frozenset()
    spared = frozenset(s for s, k in promised.items() if k == NO_STEAL)
    keep = []
    for h in candidate_hexes(state, allowed):
        on = occupants(state, h)
        if any(promised.get(s) == NO_BLOCK for s in on):
            continue
        able = victims(state, h, thief)
        if able and all(s in spared for s in able):
            continue
        keep.append(h)
    if not keep:
        return allowed, frozenset()
    return frozenset(keep), spared


def holds(state: GameState, seat: int, resource: int) -> bool:
    """Whether `seat` holds a card of `resource`, in the true hands."""
    return state.hands[seat][resource] > 0


def run_deal_event(game: "Game") -> None:
    """The mover's ask, the table's promises and the mover's choice of whom
    to spare, as the robber phase opens: called by the engine on every entry
    into `Phase.ROBBER`, and a no-op unless the mover's gate asks. Promises
    the mover does not count are set aside first (`Deal.void`, from a seat
    its view says cannot hold the card; `Deal.ignored`, from a seat it has
    blacklisted or one under an embargo with it). Its choice is `Game.robber_promised`, which its robber
    move's legality then keeps."""
    gates = game.gates
    mover = game.current_player
    if gates is None or not 0 <= mover < len(gates):
        return
    gate = gates[mover]
    params = deal_params_of(gate)
    if params is None or not params.ask:
        return
    dealer = getattr(gate, "deal_dealer", None)
    hook = dealer.ask if dealer is not None else getattr(gate, "deal_ask", None)
    ask = hook(game, mover) if hook is not None else default_ask(game, mover, gate, params)
    if ask is None:
        return
    if ask.kind not in params.kinds:
        raise ValueError(f"seat {mover} asked for a {ask.kind} deal it does not make")
    state = game._state
    promised: list[int] = []
    n = game.num_players
    for step in range(1, n):
        s = (mover + step) % n
        other = gates[s] if s < len(gates) else None
        p = deal_params_of(other)
        if s in game.locked or not holds(state, s, ask.resource):
            continue
        if answer(game, s, other, p, ask, tuple(promised)):
            promised.append(s)
    from .trading import has_room

    view = game.state(mover)
    card = tuple(int(r == ask.resource) for r in range(NUM_RESOURCES))
    banned = blacklist(game, mover) | closed(game, mover)
    deal = Deal(game.turns, mover, ask.kind, ask.resource, promised=tuple(promised),
                void=tuple(s for s in promised if not has_room(view, s, card)),
                ignored=tuple(s for s in promised if s in banned and has_room(view, s, card)))
    game.deal_log.append(deal)
    counted = deal.counted
    if counted:
        accept = dealer.accept if dealer is not None else getattr(gate, "deal_accept", None)
        taken = accept(game, mover, ask, counted) if accept is not None else \
            default_accept(game, mover, gate, ask, counted)
        deal.accepted = tuple(s for s in counted if s in set(taken))
    game.robber_promised = tuple((s, ask.kind) for s in deal.accepted)
    game.open_deal = deal


def close_deal(game: "Game", target: int, victim: int | None) -> None:
    """After the robber has moved: where it went and whom it robbed, and
    which counted promisers it spared (`kept`), to whom the deal's offer goes
    once the turn opens to trading (`settle_deals`)."""
    deal = game.open_deal
    game.open_deal = None
    game.robber_promised = ()
    if deal is None:
        return
    state = game._state
    deal.hex, deal.victim = target, victim
    on = occupants(state, target)
    kept, broken = [], []
    for s in deal.counted:
        hit = s in on if deal.kind == NO_BLOCK else s == victim
        (broken if hit else kept).append(s)
    deal.kept, deal.broken = tuple(kept), tuple(broken)
    if kept:
        game.deal_payments.append(deal)


class _Honouring:
    """A spared promiser's gate for the deal's offer: it takes the offer
    when it honours the deal (its `Answerer.pays`, or always without one)
    and otherwise answers as `gate` would. Every other name is `gate`'s.
    Its answer's kind goes into `said`, by seat."""

    __slots__ = ("gate", "game", "deal", "said")

    def __init__(self, gate: object, game: "Game", deal: Deal, said: dict) -> None:
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "game", game)
        object.__setattr__(self, "deal", deal)
        object.__setattr__(self, "said", said)

    def respond(self, view, offer):
        from .trading import RESPONSE_ACCEPT, Response
        from .trading._engine import _answer_concrete

        answerer = getattr(self.gate, "deal_answerer", None)
        if answerer is None or answerer.pays(self.game, view.perspective, self.deal):
            response = Response(view.perspective, RESPONSE_ACCEPT, offer.received)
        else:
            response = _answer_concrete(self.gate, view, offer)
        self.said[view.perspective] = getattr(response, "kind", None)
        return response

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "gate"), name)


def settle_deals(game: "Game") -> None:
    """Once the mover's turn is open to trading, each deal whose robber move
    spared a counted promiser goes to the table as an ordinary offer
    (`hexset.trading.resolve_offer`): the asked card for the one the mover
    offers back (`deal_return`). The spared promisers answer by honouring
    the deal or not (`_Honouring`), every other seat as it answers any
    offer, and the mover takes whichever answer its own trading picks, a
    promiser's or not. A spared promiser that does not take it is
    `Deal.rejected`."""
    from .game import Phase
    from .trading import Offer, resolve_offer

    if game.phase is not Phase.MAIN or not game.deal_payments:
        return
    owed, game.deal_payments = game.deal_payments, []
    gates = game.gates
    if gates is None:
        return
    for deal in owed:
        gate = gates[deal.mover]
        dealer = getattr(gate, "deal_dealer", None)
        hook = dealer.give_back if dealer is not None else getattr(gate, "deal_return", None)
        back = (hook(game, deal.mover, deal.kept, deal.resource) if hook is not None
                else default_return(game, deal.mover, gate, deal.kept, deal.resource))
        if back is None or back == deal.resource or not holds(game._state, deal.mover, back):
            deal.unpaid = deal.kept
            continue
        deal.offered = back
        able = {s for s in deal.kept if holds(game._state, s, deal.resource)}
        said: dict[int, str] = {}
        answering = list(gates)
        for s in deal.kept:
            answering[s] = _Honouring(gates[s], game, deal, said)
        trades = resolve_offer(game, answering, Offer(deal.mover, _swap(deal.resource, back)))
        taker = None
        if trades:
            t = trades[0]
            taker = t.b if t.a == deal.mover else t.a
        deal.taker = taker
        from .trading import RESPONSE_ACCEPT

        deal.paid = ((taker, back),) if taker in deal.kept else ()
        # Not taking it: no acceptance, or one its hand could not cover (the referee's pass).
        deal.rejected = tuple(s for s in deal.kept if s not in able or said.get(s) != RESPONSE_ACCEPT)


__all__ = [
    "ANSWERING",
    "NO_BLOCK", "NO_STEAL", "KINDS", "DICE_STAY", "DealParams", "Ask", "RobberModel", "PlainMover",
    "register_robber_model", "robber_model", "candidate_hexes", "weights_on", "softmax", "hit_chances",
    "targeted", "promise_value", "default_answer", "answer", "deal_params_of", "Dealing", "Deal", "default_ask",
    "default_accept", "default_return", "keep_promises", "holds", "run_deal_event", "close_deal",
    "settle_deals", "blacklist", "closed", "MOVERS", "deal_seed", "Answerer", "register_answerer", "answerer_for",
    "STEAL_SAMPLES", "value_net", "ValueDealer", "PromiseModel", "register_promise_model", "promise_model",
]
