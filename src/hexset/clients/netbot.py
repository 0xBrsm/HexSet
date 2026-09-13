# SPDX-License-Identifier: GPL-3.0-only
"""Runtime-neutral checkpoint adapters for bots, evaluation, search and trading.

Constructors accept `Checkpoint` objects. Runtime integrations own loading,
metadata and encoding; this module uses only the batched `Policy` interface.
A gate used without `choose` must be installed with `seat_at(game)` before
trading. Create separate gates for separate games and fixed seats."""

from __future__ import annotations

import copy
import importlib
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hexset.actions import Action, ActionType, apply
from hexset.board.board import pips
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.cards import DevCard
from hexset.clients.policy import Checkpoint, Policy
from hexset.clients.reachable import Hand, Purchases, Recipe, maximal, reachable_recipes
from hexset.devcards import play_year_of_plenty, spend_card
from hexset.game import Game, imagine, is_over, to_move
from hexset.mcts import Search
from hexset.actions import options_for
from hexset.clients.modelmeta import DEFAULT_GATE_PLIES, DEFAULT_TRADE_FLOOR, gate_config_of
from hexset.economy import COSTS, Purchase, pay, trade_ratios
from hexset.state import (
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    GameState,
    HiddenHand,
    city_count,
    city_upgradeable,
    copy_state,
    is_hidden,
    pile_size,
    place_road,
    place_settlement,
    road_count,
    road_placeable,
    settlement_count,
    settlement_placeable,
    upgrade_to_city,
)
from hexset.trading import exchange
from hexset.victory import update_longest_road
from hexset.view import View

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hexset.board.board import Board
    from hexset.trading import Bundle


def _check_players(game: Game, players: int) -> None:
    # true state: `num_players` is a fixed, public board property.
    num_players = game.state(0, hidden=False).num_players
    if num_players != players:
        raise ValueError(
            f"network was trained for {players} players, "
            f"not {num_players}"
        )


@dataclass(frozen=True)
class _Geometry:
    """Whether `seat` has anywhere to spend each purchase kind, and room
    under its own piece cap -- a property of the board and of `seat`'s
    pieces, not of any hand, so it is read once per ask
    (`_geometry_for`) and reused for every candidate's hypothetical hand.
    """

    road: bool
    settlement: bool
    city: bool
    dev_card: bool


def _geometry_for(state: GameState, seat: int) -> _Geometry:
    """`_Geometry` for `seat` at `state`: the same two facts
    `hexset.actions._building_actions` reads before ever asking whether a
    hand affords anything -- a legal place to put the piece, and room left
    under its cap -- plus whether the deck has a card left to draw
    (`hexset.devcards.can_buy`'s other half)."""
    topology = state.board.topology
    return _Geometry(
        road=(
            road_count(state, seat) < MAX_ROADS
            and any(road_placeable(state, seat, e) for e in range(topology.num_edges))
        ),
        settlement=(
            settlement_count(state, seat) < MAX_SETTLEMENTS
            and any(
                settlement_placeable(state, seat, v) for v in range(topology.num_vertices)
            )
        ),
        city=(
            city_count(state, seat) < MAX_CITIES
            and any(city_upgradeable(state, seat, v) for v in range(topology.num_vertices))
        ),
        dev_card=pile_size(state.deck) > 0,
    )


def _affords(hand: Sequence[int], purchase: Purchase) -> bool:
    """Arithmetic-only `hexset.economy.can_afford`, against a hand that may
    be hypothetical rather than any seat's live one."""
    return all(hand[r] >= n for r, n in enumerate(COSTS[purchase]))


def _kinds_of(hand: Sequence[int], geometry: _Geometry) -> frozenset[Purchase]:
    """Which purchase kinds `hand` alone can make right now: `geometry`
    bounds what no hand can (nowhere legal to place it, no piece left under
    the cap, no card left in the deck), and `_affords` bounds what this
    particular hand can within that. A kind absent from `geometry` never
    appears here regardless of the hand, which is what makes two hands'
    kind sets comparable at all -- `geometry` does not change candidate to
    candidate, only the hand does.

    Used only at `gate_plies > 0` now: the reachable-purchase gate
    (`hexset.clients.reachable.reachable_recipes`) replaces this filter,
    exactly rather than coarsened to four kinds, at the shipped default of
    `0` -- see `_evaluate_reachable` and the class docstring."""
    out = set()
    for purchase, available in (
        (Purchase.ROAD, geometry.road),
        (Purchase.SETTLEMENT, geometry.settlement),
        (Purchase.CITY, geometry.city),
        (Purchase.DEV_CARD, geometry.dev_card),
    ):
        if available and _affords(hand, purchase):
            out.add(purchase)
    return frozenset(out)


class _Plan:
    """One maximal reachable purchase multiset's own build in progress: a
    working copy of a position (`game`) and the spatial pieces still to
    place on it (`remaining` -- `ROAD`/`SETTLEMENT`/`CITY` only, first to
    place at index 0; `DEV_CARD` needs no placement and is already paid for
    by `NetworkBot._make_plan`). `final_values` is filled in by
    `NetworkBot._run_plans` once no piece is left to place."""

    __slots__ = ("game", "remaining", "final_values")

    def __init__(self, game: Game, remaining: list[Purchase]) -> None:
        self.game = game
        self.remaining = remaining
        self.final_values: tuple[float, ...] | None = None


def _fill_budget(candidate_lists: Sequence[Sequence[object]], budget: int) -> list[tuple[int, object]]:
    """Round-robin up to `budget` items off the front of `candidate_lists`,
    each already ranked best-first: every list gives up its own best
    candidate before any list gives up a second, so a plan with many legal
    placements never crowds out a plan with few. Returns `(list_index,
    item)` pairs in the order filled -- `NetworkBot._run_plans`'s own way
    of keeping a forward under `_MAX_ROWS` without favouring whichever plan
    happened to be built first."""
    out: list[tuple[int, object]] = []
    remaining = [list(c) for c in candidate_lists]
    while len(out) < budget and any(remaining):
        progressed = False
        for i, items in enumerate(remaining):
            if len(out) >= budget:
                break
            if items:
                out.append((i, items.pop(0)))
                progressed = True
        if not progressed:
            break
    return out


# The reachable-purchase gate's own budget (`NetworkBot._run_plans`): at
# most this many rows in either of its at most two `value_rows` calls, one
# of the three forwards a checkpoint's own continuation gate is allowed.
_MAX_ROWS = 512


@dataclass
class NetworkBot:
    """A policy answering one position at a time.

    Trades off the same value head `choose` already reads, no new
    parameters. The value head scores a candidate exchange from both
    sides at once (`_score`): the live position and the position after
    the cards move -- each read from this seat's own frame, so nothing here
    reads a hand this seat may not know, at `gate_plies == 0`. A checkpoint
    asking for a continuation (`gate_plies > 0`) rolls a mover's actions
    forward first, which is play, not a read -- see `gate_plies` below for
    why that changes what a position is built from. The head answers with
    one win probability per seat, and that vector is the whole gate:

    * `gains_many` -- this seat's own row, after minus before: its private
      gain in win probability, `hexset.trading.trade_event`'s gate and the
      round's own-side reading.
    * `estimate_many` -- the *counterparty's* row, after minus before: this
      seat's estimate of what the exchange does to that seat's chances,
      read by `hexset.trading.default_offer`/`default_respond` so a bot
      offers, and counters with, the candidate best for itself among those
      it believes the other seat gains from too. By construction
      that prices the risk of handing an opponent win probability: a
      bundle that lifts this seat's row a little and the counterparty's a
      lot is a bad offer, and it reads as one.
    * `accepts`/`accepts_many` -- `gains_many` thresholded at strictly
      positive.

    **The reachable-purchase gate (`gate_plies == 0`).** A raw value-head
    reading of two near-identical hands differs by the head's own noise --
    measured once at a won position (the winning settlement already in
    hand), where the old two-world continuation this replaced was built to
    fix exactly that: the older gate, without it, priced a trade that
    changed nothing at a small nonzero value. What a hand can buy is
    deterministic arithmetic, not a question for the head at all: a trade
    plus one development card in hand are the only influx of cards before
    this seat buys, so `hexset.clients.reachable.reachable_recipes` works
    out `R(hand)` -- every purchase multiset reachable through this turn's
    own card play, this seat's own bank trades, and every jointly payable
    combination of `ROAD`/`SETTLEMENT`/`CITY`/`DEV_CARD` -- and a candidate
    whose `R(hand + bundle)` equals `R(hand)` exactly is never scored: its
    gain and its estimate are `0.0` outright, which
    `hexset.trading.clears_floor` never clears (a gain at or below zero
    never clears), so it is never offered, accepted or countered with. A
    survivor is valued by the best position its own reachable purchases can
    build (`_evaluate_reachable`), not by its raw exchanged hand, for the
    same noise-at-the-margin reason. `gate_plies > 0` keeps the coarser
    affordability filter (`_kinds_of`) this replaces at `0` plies, below.

    `trade_floor` is the checkpoint's own, read off the file it was loaded
    from (`hexset.clients.modelmeta.gate_config`) and passed in by
    `bot_for`. A floor measured against one value head says nothing about
    another's, so it is not a constant of this module: the default below is
    what a checkpoint that declares nothing gets.

    `gate_plies` is that same checkpoint's own continuation budget
    (`hexset.clients.modelmeta.GateConfig.plies`): `0`, the default, prices
    every survivor by the reachable-purchase gate above. `N`
    rolls the mover's own greedy policy up to `N` plies forward from each
    survivor first (`_continue`) -- search on the trade decision, the same
    footing as `simulations` is for `hexset.mcts.Search`. That rollout
    *acts*: a Knight steal mid-turn resolves against whatever hand the
    position it runs on actually holds, so the position handed to it can
    never be the live table's own hands (a fact this seat may not have) --
    it must be a world this seat could plausibly believe true, sampled
    from its own belief (`View.sample`) the same way the 0.51.0 gate that
    `gate_plies` restores did, certified so a candidate's exchange can
    never go negative. `rng` is that sampling's own stream, independent of
    whatever a runtime seeds policy inference with.

    `rng` is not read at `gate_plies == 0`: nothing is sampled there.
    """

    policy: Policy
    players: int
    max_trades: int | None = None
    # This gate's clearing floor (`hexset.trading.trade_floor_of`). The
    # default is the unmeasured case: strict positivity is the whole gate.
    trade_floor: float = DEFAULT_TRADE_FLOOR
    # This gate's own continuation budget (`hexset.clients.modelmeta.GateConfig.plies`).
    # `0` is the unmeasured case: no rollout, one forward over the exchanged hand.
    gate_plies: int = DEFAULT_GATE_PLIES
    # The belief-sampled worlds `gate_plies > 0` rolls forward on
    # (`_world_rng`), own stream and salt: two bots draw different worlds,
    # and one bot asked twice about the same position draws the same one.
    # Unused, and never consulted, at `gate_plies == 0`.
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)
    _salt: int | None = field(default=None, repr=False, compare=False)
    # Which seat this bot is installed at, or `None` for a bot that answers
    # whatever seat the view names (the arena spawns one bot per seat and
    # never asks it about another). When set, the view must agree: a gate
    # wired to the wrong seat would answer for somebody else's hand, and
    # that is the one failure the mechanic must never have quietly.
    seat: int | None = None
    # The game `choose` was last handed (or `seat_at` seated), so a trade
    # event -- which runs inside the same `apply` this bot's own choice
    # already went through -- asks about the position it is actually seated
    # at. `None` only for a bot nobody has seated yet, which cannot happen
    # in play (the trade event runs after every seat has moved through
    # setup) but is the right answer for the gate regardless.
    _seated: Game | None = field(default=None, repr=False, compare=False)
    # The one ask `_score` last answered, and its answer. `gains_many` and
    # `estimate_many` are two readings of one evaluation -- the seat's own row
    # and the counterparty's -- and `hexset.trading.default_offer` and
    # `default_respond` each ask for both, back to back, over the identical
    # candidate set at the identical position. `_score` is a pure function of
    # that ask, so the second ask recomputed the whole forward to arrive at
    # numbers the first one already had: half the gate's whole cost. One
    # entry is enough because the pair is always consecutive; the key
    # carries the position, so a changed table misses rather than answering
    # stale.
    _memo: tuple[tuple, dict] | None = field(default=None, repr=False, compare=False)

    def seat_at(self, game: Game) -> None:
        """Seat this bot at `game` without asking it to move: the position
        its gate is an evaluation of. `choose` does this itself; a driver
        that installs a bot as a gate only (a searched policy, a collector's
        seat) calls it in place of poking `_seated`."""
        _check_players(game, self.players)
        self._seated = game
        self._memo = None

    def choose(self, game: Game) -> Action:
        self.seat_at(game)
        seat = to_move(game)
        return self.policy.act_rows([(game, seat, tuple(options_for(game)))])[0]

    def _perspective(self, view: View) -> int:
        if self.seat is not None and view.perspective != self.seat:
            raise ValueError(
                f"a network bot seated at {self.seat} was asked for seat {view.perspective}"
            )
        return view.perspective

    def gains_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[float]:
        """This seat's own private gain from every candidate at once:
        `V(after)[seat] - V(before)[seat]`, win probability, one forward.
        `0.0` for a candidate the affordability filter passes over, `-1.0`
        for one this seat cannot cover; nothing when nobody has seated this
        bot (no `choose` has run) or trading is switched off.
        """
        if self.max_trades == 0 or self._seated is None or not received:
            return [-1.0] * len(received)
        seat = self._perspective(view)
        before, afters, zeros = self._score(seat, view, received, counterparties)
        out = [-1.0] * len(received)
        for i in zeros:
            out[i] = 0.0
        for i, after in afters.items():
            out[i] = float(after[seat] - before[seat])
        return out

    def estimate_many(
        self, view: View, candidates: Sequence[tuple[int, Bundle]]
    ) -> list[float]:
        """This seat's estimate of each `(counterparty, bundle)` candidate's
        *counterparty*-side gain: `V(after)[them] - V(before)[them]` on the
        same two positions `gains_many` scores, from this seat's own frame
        -- its own belief about the other seat's chances, used as an
        estimate of their willingness to trade.
        """
        if self.max_trades == 0 or self._seated is None or not candidates:
            return [-1.0] * len(candidates)
        seat = self._perspective(view)
        received = [b for _, b in candidates]
        thems = [c for c, _ in candidates]
        before, afters, zeros = self._score(seat, view, received, thems)
        out = [-1.0] * len(candidates)
        for i in zeros:
            out[i] = 0.0
        for i, after in afters.items():
            them = thems[i]
            out[i] = float(after[them] - before[them])
        return out

    def accepts(self, view: View, received: Bundle, counterparty: int) -> bool:
        """`gains_many` for one candidate, thresholded at strictly positive
        -- the engine's termination argument rests on the acting seat's own
        gain strictly increasing at every step."""
        return self.gains_many(view, [received], [counterparty])[0] > 0.0

    def accepts_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[bool]:
        """`gains_many`, thresholded at strictly positive."""
        return [gain > 0.0 for gain in self.gains_many(view, received, counterparties)]

    def _score(
        self, seat: int, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> tuple[tuple[float, ...], dict[int, tuple[float, ...]], frozenset[int]]:
        """`_evaluate`, answering a repeat of the last ask from `_memo`.

        See `_memo`'s own comment for why the repeat is worth catching. The
        key is the whole of what `_evaluate` reads: the seated position
        (turn, phase, mover, free roads, and the board itself -- Road
        Building moves the board without touching a hand), the deck's size
        and this seat's own matured development cards (what
        `hexset.clients.reachable.reachable_recipes` branches on, at
        `gate_plies == 0`; the affordability filter's `DEV_CARD` geometry,
        at `gate_plies > 0`), whether a card has already been played this
        turn, the asking seat's information set (`View.signature`, every
        hand size), and the candidates asked about. A miss on any of it
        evaluates afresh. Bank stock and the board's ports are not in it:
        ports never change during a game, and bank stock going stale
        between two asks is the same unmeasured edge
        `hexset.clients.reachable.bank_conversions` already leaves alone.
        """
        game = self._seated
        assert game is not None  # callers check this first
        state = view.state
        key = (
            game.turns, game.phase, game.current_player, game.free_roads,
            seat, view.perspective, tuple(view.sizes), view.signature(),
            # The board as well as the belief: Road Building places roads
            # without spending a card, so a hand-only key would answer a
            # changed table from the previous one.
            tuple(state.edge_owner), tuple(state.vertex_building),
            pile_size(state.deck),
            game.dev_card_played, tuple(state.dev_cards[seat]),
            tuple(received), tuple(counterparties),
        )
        memo = self._memo
        if memo is not None and memo[0] == key:
            return memo[1]
        scored = self._evaluate(seat, view, received, counterparties)
        self._memo = (key, scored)
        return scored

    def _evaluate(
        self, seat: int, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> tuple[tuple[float, ...], dict[int, tuple[float, ...]], frozenset[int]]:
        """The live position's value vector, the value vector of every
        candidate that survives this checkpoint's own filter, and the set
        that does not.

        At `gate_plies == 0`, the shipped default, this is entirely
        `_evaluate_reachable`'s: the reachable-purchase gate replaces both
        the affordability filter and the plain one-hand-one-forward read
        below it -- see the class docstring and `_evaluate_reachable`'s own
        for why.

        A checkpoint asking for `gate_plies > 0` keeps the coarser
        affordability filter this replaces at `0` plies (`_kinds_of`), then
        rolls the mover's own greedy policy that many plies forward from
        each survivor (`_continue`) before a forward runs -- the trade
        decision's own search, same footing as `simulations` is for
        `hexset.mcts.Search`. That rollout *acts*, so `_after`'s hand-edited
        position (which leaves every other seat's true hand sitting right
        there, merely unread) is not safe to hand it: a Knight steal
        mid-turn would resolve against those real cards. Every world here is
        instead sampled from `seat`'s own belief (`View.sample`), certified
        so each survivor's own exchange can never go negative -- one shared
        draw: `before` and every survivor's `after` come from the same
        sampled state, so the pair stays paired the way the class docstring
        describes, and a single shared `before` (`values[0]`) prices every
        survivor exactly as the `gate_plies == 0` return does.

        A candidate `seat` cannot cover is skipped outright (`gains_many`/
        `estimate_many` answer it `-1.0`) at either `gate_plies`; every
        other candidate is classified by that ply's own filter, and only
        the surviving ones cost a position at all. When none survive,
        nothing is asked of the policy: the live position's value is never
        read by a candidate priced `0.0` outright, so there is no reason to
        forward it either.
        """
        game = self._seated
        assert game is not None  # callers check this first
        state = view.state
        hand = list(view.known[seat])

        if self.gate_plies == 0:
            return self._evaluate_reachable(seat, view, hand, received, counterparties)

        geometry = _geometry_for(state, seat)
        current = _kinds_of(hand, geometry)

        zeros: set[int] = set()
        survivors: list[tuple[int, int, Bundle]] = []  # (candidate index, them, bundle)
        for i, bundle in enumerate(received):
            if any(n + d < 0 for n, d in zip(hand, bundle)):
                continue
            candidate_hand = [n + d for n, d in zip(hand, bundle)]
            if _kinds_of(candidate_hand, geometry) == current:
                zeros.add(i)
                continue
            survivors.append((i, counterparties[i], bundle))

        if not survivors:
            return (), {}, frozenset(zeros)

        # `gate_plies > 0`: one belief world, sampled from this seat's own
        # information set and certified to cover every survivor's bundle at
        # once (`View.certify` combines repeat counterparties by taking the
        # per-resource max), so no survivor's exchange can go negative. The
        # rollout below plays on copies of this same sampled state --
        # never the live table's, and never a different draw per survivor.
        certified = View(
            view.state, view.ledger, seat,
            certify=[(them, [max(0, n) for n in bundle]) for _, them, bundle in survivors],
        )
        rng = self._world_rng(view, seat)
        sampled = certified.sample(rng)
        before = imagine(game, rng, randomize_deck=True)
        before.set_state(sampled)

        afters: dict[int, Game] = {}
        for i, them, bundle in survivors:
            after = imagine(game, rng, randomize_deck=True)
            exchanged = copy_state(sampled)
            hands_before = [h[:] for h in exchanged.hands]
            exchange(exchanged, seat, them, bundle)
            after.set_state(exchanged)
            after.ledger.apply_hand_diff(hands_before, exchanged.hands)
            afters[i] = after

        worlds = [before] + list(afters.values())
        values = self._continue(to_move(game), seat, worlds, self.gate_plies)
        afters_values = {i: values[row] for row, i in enumerate(afters, start=1)}
        return values[0], afters_values, frozenset(zeros)

    def _evaluate_reachable(
        self, seat: int, view: View, hand: Sequence[int],
        received: Sequence[Bundle], counterparties: Sequence[int],
    ) -> tuple[tuple[float, ...], dict[int, tuple[float, ...]], frozenset[int]]:
        """`_evaluate`'s `gate_plies == 0` path: the reachable-purchase gate.

        Filters every coverable candidate by comparing `R(hand)` to
        `R(hand + bundle)` (`hexset.clients.reachable.reachable_recipes`,
        compared as the set of keys) -- a bundle that changes neither is
        priced `0.0` outright, exactly as `_kinds_of` did for `gate_plies >
        0`, only exact rather than coarsened to four kinds (a bundle that
        swaps which two roads are affordable for one road and one
        settlement now survives, where the old kind set could not tell
        them apart).

        Every survivor is then valued by the best position its own
        reachable purchases can build, not by its raw exchanged hand: for
        the baseline hand and each survivor, every *maximal* reachable
        purchase multiset (`hexset.clients.reachable.maximal` -- affording
        the bigger one affords the smaller for free, so only the biggest
        are worth building) becomes a `_Plan` (`_make_plan`) and is played
        out and valued (`_run_plans`). A candidate's worth is the best of
        its own rows' `values[seat]`; the baseline's likewise -- so
        `gains_many` reads as the change in the *best reachable position*,
        not in the raw hand the affordability filter used to read
        directly, which is the resolution problem the class docstring
        describes. `estimate_many` reads `values[them]` off that same
        winning row, exactly as `gains_many`'s `_evaluate` always has.
        """
        base_hand: Hand = tuple(hand)
        ctx = self._reach_context(seat, view)
        base_recipes = reachable_recipes(base_hand, **ctx)
        base_R = frozenset(base_recipes)

        zeros: set[int] = set()
        survivors: list[tuple[int, int, Bundle, dict[Purchases, Recipe]]] = []
        recipe_cache: dict[Hand, dict[Purchases, Recipe]] = {base_hand: base_recipes}
        for i, bundle in enumerate(received):
            if any(n + d < 0 for n, d in zip(hand, bundle)):
                continue
            candidate_hand: Hand = tuple(n + d for n, d in zip(hand, bundle))
            recipes = recipe_cache.get(candidate_hand)
            if recipes is None:
                recipes = reachable_recipes(candidate_hand, **ctx)
                recipe_cache[candidate_hand] = recipes
            if frozenset(recipes) == base_R:
                zeros.add(i)
                continue
            survivors.append((i, counterparties[i], bundle, recipes))

        if not survivors:
            return (), {}, frozenset(zeros)

        baseline_seed = self._seeded(seat, base_hand, view)
        owners: dict[int | None, list[_Plan]] = {
            None: [
                self._make_plan(baseline_seed, seat, m, base_recipes[m])
                for m in maximal(base_R)
            ]
        }
        for i, them, bundle, recipes in survivors:
            seed = self._after(seat, them, hand, bundle, view)
            owners[i] = [
                self._make_plan(seed, seat, m, recipes[m]) for m in maximal(frozenset(recipes))
            ]

        self._run_plans(seat, owners)

        baseline_row = max(
            (plan.final_values for plan in owners[None]), key=lambda values: values[seat]
        )
        afters_values = {
            i: max((plan.final_values for plan in owners[i]), key=lambda values: values[seat])
            for i, _, _, _ in survivors
        }
        return baseline_row, afters_values, frozenset(zeros)

    def _reach_context(self, seat: int, view: View) -> dict:
        """The reachable-purchase gate's inputs that do not vary candidate
        to candidate: this seat's own matured development cards and
        whether one has already been played this turn, the bank, every
        other seat's known row, this seat's own best bank-trade ratios
        (`hexset.economy.trade_ratios`), the piece room still open to it,
        and the placements legal for it right now. Computed once per ask;
        only `hand` itself varies across the baseline and every candidate
        in `_evaluate_reachable`."""
        game = self._seated
        assert game is not None  # callers check this first
        state = view.state
        topology = state.board.topology
        room = (
            MAX_ROADS - road_count(state, seat),
            MAX_SETTLEMENTS - settlement_count(state, seat),
            MAX_CITIES - city_count(state, seat),
        )
        legal = (
            sum(1 for e in range(topology.num_edges) if road_placeable(state, seat, e)),
            sum(1 for v in range(topology.num_vertices) if settlement_placeable(state, seat, v)),
            sum(1 for v in range(topology.num_vertices) if city_upgradeable(state, seat, v)),
        )
        return dict(
            dev_cards_held=tuple(state.dev_cards[seat]),
            dev_card_played=game.dev_card_played,
            bank=tuple(state.bank),
            known_others=[
                tuple(view.known[other]) for other in range(state.num_players) if other != seat
            ],
            ratios=trade_ratios(state, seat),
            room=room,
            legal=legal,
            deck_size=pile_size(state.deck),
        )

    def _seeded(self, seat: int, hand: Hand, view: View) -> Game:
        """The baseline's own starting point: a copy of the seated game
        with `seat`'s hand set to `hand` and nothing else moved -- no
        counterparty, no ledger change beyond the seat's own row. What the
        baseline's own maximal purchase multisets are built from in
        `_evaluate_reachable`; a survivor's are built from `_after`
        instead, which also moves the counterparty's known row."""
        game = self._seated
        assert game is not None  # callers check this first
        state = copy_state(view.state)
        state.hands[seat] = list(hand)
        seeded = copy.copy(game)
        seeded.set_state(state)
        seeded.ledger = view.ledger.copy()
        seeded.ledger.seats[seat].known = list(hand)
        seeded.ledger.seats[seat].unknown = 0
        return seeded

    def _copy_position(self, game: Game) -> Game:
        """A copy of `game` -- copied state and ledger both -- safe to
        mutate for one plan's own build without disturbing `game` or any
        other plan started from it."""
        new = copy.copy(game)
        new.set_state(copy_state(game.state(game.current_player, hidden=False)))
        new.ledger = game.ledger.copy()
        return new

    def _make_plan(self, seed: Game, seat: int, purchases: Purchases, recipe: Recipe) -> _Plan:
        """One maximal multiset's own working copy: `seed` (the baseline's
        `_seeded` position, or a survivor's `_after`) with `recipe`'s
        development-card branch and bank trades applied
        (`_apply_branch`), then paid for (`_pay_purchases`) -- everything
        before a piece needs a place put on it."""
        plan_game = self._copy_position(seed)
        self._apply_branch(plan_game, seat, recipe)
        self._pay_purchases(plan_game, seat, purchases)
        remaining = (
            [Purchase.ROAD] * purchases[0]
            + [Purchase.SETTLEMENT] * purchases[1]
            + [Purchase.CITY] * purchases[2]
        )
        return _Plan(plan_game, remaining)

    def _apply_branch(self, plan_game: Game, seat: int, recipe: Recipe) -> None:
        """Mutate `plan_game`'s own copied state (and ledger) to reflect
        `recipe`'s development-card branch, if any, then the bank trades
        between the branched hand and `recipe.hand` -- everything
        `reachable_recipes` found reachable before any purchase.

        Honest by construction: Year of Plenty
        (`hexset.devcards.play_year_of_plenty`) moves only the bank and
        this seat's own hand, both already this seat's own to know;
        Monopoly (`_apply_known_monopoly`) moves only what the ledger
        already certifies as known, never a true hand this seat cannot
        read.

        The bank trades themselves are applied as one net vector rather
        than replayed step by step: a bank trade only ever moves a
        resource between a hand and the bank, so the aggregate delta from
        the (possibly branched) hand to `recipe.hand` is exactly what the
        bank gave up or took back, regardless of which trades produced it.
        """
        state = plan_game.state(seat, hidden=False)
        if recipe.branch == "year_of_plenty":
            a, b = recipe.param
            play_year_of_plenty(state, seat, [Resource(a), Resource(b)])
            plan_game.dev_card_played = True
        elif recipe.branch == "monopoly":
            self._apply_known_monopoly(plan_game, seat, recipe.param)
            plan_game.dev_card_played = True

        branch_hand = state.hands[seat]
        delta = [v - h for v, h in zip(recipe.hand, branch_hand)]
        state.hands[seat] = list(recipe.hand)
        for r in range(NUM_RESOURCES):
            state.bank[r] = max(0, state.bank[r] - delta[r])
        plan_game.ledger.seats[seat].known = list(recipe.hand)
        plan_game.ledger.seats[seat].unknown = 0

    def _apply_known_monopoly(self, plan_game: Game, seat: int, resource: int) -> None:
        """Monopoly, honestly: take only what this seat's ledger already
        certifies each other seat holds of `resource` -- the same sum
        `hexset.clients.reachable.branch_hands` used to decide this branch
        exists at all -- never a true hand this seat cannot read. A
        concrete opponent hand is reduced by that known amount; a hidden
        one only shrinks by that many cards, since a `HiddenHand` carries
        no composition to remove from -- the same known-safe pattern
        `_after` uses for a counterparty's row.
        """
        state = plan_game.state(seat, hidden=False)
        ledger = plan_game.ledger
        spend_card(state, seat, DevCard.MONOPOLY)
        total = 0
        for other in range(state.num_players):
            if other == seat:
                continue
            amount = ledger.seats[other].known[resource]
            if amount <= 0:
                continue
            total += amount
            ledger.seats[other].known[resource] = 0
            if is_hidden(state.hands[other]):
                state.hands[other] = HiddenHand(max(0, len(state.hands[other]) - amount))
            else:
                state.hands[other][resource] = max(0, state.hands[other][resource] - amount)
        state.hands[seat][resource] += total

    def _pay_purchases(self, plan_game: Game, seat: int, purchases: Purchases) -> None:
        """Pay for every piece in `purchases` -- resource cost only,
        `hexset.economy.pay`. A development card is paid for and left
        there: `pay` never touches the deck, so nothing is drawn into a
        hypothetical and the position's value is read with the deck
        exactly as it was, the same as a real buy leaves it until the turn
        ends."""
        state = plan_game.state(seat, hidden=False)
        for purchase, count in zip(
            (Purchase.ROAD, Purchase.SETTLEMENT, Purchase.CITY, Purchase.DEV_CARD), purchases
        ):
            for _ in range(count):
                pay(state, seat, purchase)

    def _legal_for(self, state: GameState, seat: int, piece: Purchase) -> list[int]:
        """Every edge or vertex `piece` could be placed on right now,
        freshly read off `state` -- unlike the reachable gate's own
        `legal` count (`_reach_context`), asked again after every piece a
        plan places, so a road that just opened a vertex is seen by the
        next placement."""
        topology = state.board.topology
        if piece is Purchase.ROAD:
            return [e for e in range(topology.num_edges) if road_placeable(state, seat, e)]
        if piece is Purchase.SETTLEMENT:
            return [
                v for v in range(topology.num_vertices) if settlement_placeable(state, seat, v)
            ]
        if piece is Purchase.CITY:
            return [v for v in range(topology.num_vertices) if city_upgradeable(state, seat, v)]
        raise ValueError(f"{piece!r} needs no placement")

    def _place_piece(self, plan_game: Game, seat: int, piece: Purchase, spot: int) -> None:
        state = plan_game.state(seat, hidden=False)
        if piece is Purchase.ROAD:
            place_road(state, seat, spot)
        elif piece is Purchase.SETTLEMENT:
            place_settlement(state, seat, spot)
        elif piece is Purchase.CITY:
            upgrade_to_city(state, seat, spot)
        else:
            raise ValueError(f"{piece!r} needs no placement")
        update_longest_road(state)

    def _vertex_pips(self, state: GameState, vertex: int) -> int:
        topology = state.board.topology
        return sum(pips(state.board.tokens[h]) for h in topology.vertex_hexes[vertex])

    def _placement_pips(self, state: GameState, piece: Purchase, spot: int) -> int:
        """The static ranking a placement is picked by once the forward
        budget runs out: total production pips of a vertex's own adjacent
        hexes, or the better of an edge's two endpoints -- cheap,
        board-only, and blind to anything the value head would have read.
        """
        if piece is Purchase.ROAD:
            a, b = state.board.topology.edges[spot]
            return max(self._vertex_pips(state, a), self._vertex_pips(state, b))
        return self._vertex_pips(state, spot)

    def _run_plans(self, seat: int, owners: dict[int | None, list[_Plan]]) -> None:
        """Fill in every plan's `final_values`, in at most `_MAX_ROWS` rows
        a `value_rows` call, ordinarily across just two such calls (of the
        three a checkpoint's own continuation gate is allowed).

        Round one prices every plan with no piece left to place outright
        (a multiset that was only ever `DEV_CARD`s, or empty) and the
        first placement of every plan that still has one, in one forward
        batched across the baseline and every survivor together -- "value
        the first piece's placements in a forward, keep the best" from the
        class docstring. Whichever plan is still short a piece after that
        places every one it has left by the static ranking
        (`_placement_pips`) rather than a further forward -- "place the
        next" -- and a closing forward reads the finished position, in
        `_MAX_ROWS`-sized chunks on the rare hand with enough incomparable
        maximal multisets to need more than one. Rows beyond `_MAX_ROWS` in
        the first forward are dropped round-robin across plans
        (`_fill_budget`), never favouring whichever plan happened to be
        built first.
        """
        plans = [p for group in owners.values() for p in group]

        rows: list[tuple[Game, int]] = []
        finished: list[tuple[_Plan, int]] = []
        choosing: list[tuple[_Plan, Purchase, list[int]]] = []
        for plan in plans:
            if not plan.remaining:
                finished.append((plan, len(rows)))
                rows.append((plan.game, seat))
            else:
                piece = plan.remaining[0]
                plan_state = plan.game.state(seat, hidden=False)
                legal = self._legal_for(plan_state, seat, piece)
                ranked = sorted(
                    legal,
                    key=lambda spot, piece=piece, plan_state=plan_state: -self._placement_pips(
                        plan_state, piece, spot
                    ),
                )
                choosing.append((plan, piece, ranked))

        budget = _MAX_ROWS - len(rows)
        choice_rows: dict[int, list[int]] = {}
        choice_games: dict[int, list[Game]] = {}
        if budget > 0 and choosing:
            candidate_lists = [
                [(plan, piece, spot) for spot in ranked] for plan, piece, ranked in choosing
            ]
            for _, (plan, piece, spot) in _fill_budget(candidate_lists, budget):
                candidate_game = self._copy_position(plan.game)
                self._place_piece(candidate_game, seat, piece, spot)
                choice_rows.setdefault(id(plan), []).append(len(rows))
                choice_games.setdefault(id(plan), []).append(candidate_game)
                rows.append((candidate_game, seat))

        values = self.policy.value_rows(rows) if rows else []

        for plan, row in finished:
            plan.final_values = values[row]

        pending: list[_Plan] = []
        for plan, piece, ranked in choosing:
            indices = choice_rows.get(id(plan))
            if not indices:
                # The budget never reached this plan at all: it falls
                # through to the closing round with every piece it started
                # with still unplaced.
                pending.append(plan)
                continue
            games = choice_games[id(plan)]
            best = max(range(len(indices)), key=lambda k: values[indices[k]][seat])
            plan.game = games[best]
            plan.remaining = plan.remaining[1:]
            if plan.remaining:
                pending.append(plan)
            else:
                plan.final_values = values[indices[best]]

        if not pending:
            return

        close_rows: list[tuple[Game, int]] = []
        for plan in pending:
            state = plan.game.state(seat, hidden=False)
            for piece in plan.remaining:
                legal = self._legal_for(state, seat, piece)
                if not legal:
                    # Should not happen -- `reachable_recipes`' own `legal`
                    # count already guaranteed a spot for this piece kind
                    # -- but a plan that cannot place what it paid for is
                    # left as it stands rather than crashing the ask.
                    continue
                best_spot = max(legal, key=lambda spot: self._placement_pips(state, piece, spot))
                self._place_piece(plan.game, seat, piece, best_spot)
            plan.remaining = []
            close_rows.append((plan.game, seat))

        # `close_rows` holds one row per still-pending plan -- the number
        # of maximal multisets across the baseline and every survivor,
        # never the number of placements any one of them has, so unlike
        # the first round it is not bounded by `_MAX_ROWS` from the
        # placements alone: a hand rich enough to reach many *incomparable*
        # maximal multisets (typically through many different bank-trade
        # chains) can still leave more pending plans than one forward
        # should take. Chunked at `_MAX_ROWS` a call, so no closing forward
        # ever exceeds it either; two such calls plus round one is exactly
        # the three-forward budget. A plan count deep enough to spill past
        # a third forward is not expected from any real hand -- every
        # fixture and self-play game this was checked against needed one --
        # so the last chunk is still read rather than left without a value.
        for start in range(0, len(close_rows), _MAX_ROWS):
            chunk_rows = close_rows[start : start + _MAX_ROWS]
            chunk_plans = pending[start : start + _MAX_ROWS]
            chunk_values = self.policy.value_rows(chunk_rows)
            for plan, row_values in zip(chunk_plans, chunk_values):
                plan.final_values = row_values

    def _world_rng(self, view: View, seat: int) -> random.Random:
        """The draw behind `gate_plies > 0`'s belief-sampled worlds: seeded
        by the information set (`View.signature`, every hand size, the
        perspective) alone, salted by this bot's own `rng` once at
        construction -- no candidate in the key, unlike the 0.51.0 gate's
        own `_world_rng`, so every survivor scored at the same ask and the
        shared `before` all draw from one stream, and two asks about the
        same position agree while two bots draw different worlds."""
        if self._salt is None:
            self._salt = self.rng.getrandbits(64)
        key = (self._salt, view.perspective, tuple(view.sizes), view.signature())
        return random.Random(hash(key))

    def _continue(
        self, mover: int, seat: int, worlds: Sequence[Game], plies: int
    ) -> list[tuple[float, ...]]:
        """Each world's value vector, from `seat`'s frame, after `mover`'s
        best play from it, `plies` deep -- the trade gate's own search,
        run only when a checkpoint's `gate_plies` asks for it.

        Restored from the 0.51.0 continuation gate (`git show
        6593b7d:src/hexset/clients/netbot.py`), verbatim in how it plays:
        in lockstep across all worlds, the policy picks `mover`'s next
        action wherever it is still `mover`'s turn (`act_rows`, one batched
        forward a ply) -- the bot's own policy standing in for whoever
        moves, acting on the hand the world gives that seat; an `END_TURN`
        pick stops that world where it stands, a finished world stops as
        its winner, and `plies` bounds the rest. What is left is valued in
        one forward; a finished world is the one-hot winner, board-seat
        order, exactly as `LeafEvaluator.terminal` reads it. The worlds
        handed in here are always belief-sampled copies (`_evaluate`),
        never the live table's own state -- see its docstring for why.

        Written for the g4 game of 2026-09-08 19:47Z, round 19: a won
        position (the winning settlement already in hand) where a raw
        value-head reading of two near-identical hands differed by the
        head's own noise and priced a trade that changed nothing. The
        affordability filter now catches that particular case for free;
        this only runs at all when a checkpoint asks for a continuation on
        top of it.
        """
        live = [i for i, g in enumerate(worlds) if not is_over(g) and to_move(g) == mover]
        for _ in range(plies):
            if not live:
                break
            rows = [(worlds[i], mover, tuple(options_for(worlds[i]))) for i in live]
            chosen = self.policy.act_rows(rows)
            still: list[int] = []
            for i, action in zip(live, chosen):
                if action.type is ActionType.END_TURN:
                    continue
                apply(worlds[i], action)
                g = worlds[i]
                if not is_over(g) and to_move(g) == mover:
                    still.append(i)
            live = still
        out: list[tuple[float, ...] | None] = [None] * len(worlds)
        pending = []
        for i, g in enumerate(worlds):
            if is_over(g):
                out[i] = tuple(1.0 if s == g.won_by else 0.0 for s in range(self.players))
            else:
                pending.append(i)
        if pending:
            values = self.policy.value_rows([(worlds[i], seat) for i in pending])
            for i, v in zip(pending, values):
                out[i] = tuple(v)
        return out  # type: ignore[return-value]

    def _after(
        self, seat: int, them: int, hand: Sequence[int], bundle: Bundle, view: View
    ) -> Game:
        """The position `bundle` leaves, read from `seat`'s own frame: a
        shallow copy of the seated game over a copied state *and* a copied
        ledger, so nothing here mutates the live table. `seat`'s hand
        becomes `hand` plus the bundle, exactly -- it is this seat's own,
        known regardless.

        `them`'s known row is raised first to cover what the bundle says
        they give (`certified`: an offer is evidence of the cards behind
        it, the same certification a sampled belief world used to rest on)
        and only then moved by the bundle -- never the true hand, which
        this seat may not read, and never a flooring guess: the
        certification guarantees the subtraction cannot go negative. The
        residual uncertainty (`certified.unknown[them]`) is untouched by a
        trade that only moves named, certified cards.

        A bare `copy.copy(game)` shares the live ledger, and `View` reads
        `them`'s known cards from the ledger, not from `state.hands` -- so
        a copy that only rewrote the state answered every ask about `them`
        from the position *before* the trade, with only `them`'s hand
        *size* having moved: precisely the stale-known-against-a-new-size
        mismatch the affordability filter exists to keep away from the
        head, reintroduced one layer down. The ledger is copied and its
        `them` row rewritten to match.
        """
        game = self._seated
        assert game is not None  # callers check this first
        state = copy_state(view.state)
        state.hands[seat] = [n + d for n, d in zip(hand, bundle)]

        certified = View(
            view.state, view.ledger, seat,
            certify=[(them, [max(0, n) for n in bundle])],
        )
        their_known = [k - d for k, d in zip(certified.known[them], bundle)]

        # Composition beyond `known` is never read for a seat that is not
        # the perspective (`View` reads `state.hands[them]` for its size
        # alone), so a `HiddenHand` need only carry the right size, and a
        # concrete hand -- `View.from_game` hands the true state through,
        # so this is the usual case -- is the true hand moved by the bundle,
        # as a cleared trade would leave it: the size is public, the
        # composition stays unread. Replacing it with the known row would
        # shrink `them` by every card this seat cannot name and price the
        # counterparty poorer on every candidate, trade or no trade.
        if is_hidden(state.hands[them]):
            state.hands[them] = HiddenHand(view.sizes[them] - sum(bundle))
        else:
            state.hands[them] = [max(0, n - d) for n, d in zip(state.hands[them], bundle)]

        ledger = view.ledger.copy()
        ledger.seats[them].known = their_known
        ledger.seats[them].unknown = certified.unknown[them]
        ledger.seats[seat].known = list(state.hands[seat])
        ledger.seats[seat].unknown = 0

        after = copy.copy(game)
        after.set_state(state)
        after.ledger = ledger
        return after


@dataclass
class LeafEvaluator:
    """A whole wave of `hexset.mcts` leaves in one forward."""

    policy: Policy
    pad_to: int | None = None

    def __post_init__(self) -> None:
        if self.pad_to is not None and self.pad_to < 1:
            raise ValueError("pad_to must be positive")

    def evaluate(self, leaves):
        if not leaves:
            return []
        count = len(leaves)
        padded = list(leaves)
        if self.pad_to is not None and count < self.pad_to:
            padded.extend([leaves[-1]] * (self.pad_to - count))
        rows = [(leaf.game, leaf.seat, leaf.options) for leaf in padded]
        return self.policy.score_rows(rows)[:count]

    def terminal(self, game: Game) -> Sequence[float]:
        """`hexset.mcts.Evaluator.terminal`: the one-hot winner, board-seat
        order.

        A contract-6 value head is trained on a win probability, whether it
        came from PPO or from distillation, which was ported to the same
        target. So every non-terminal leaf in a wave is scored on that scale,
        and returning `terminal_relative_points` here would back a points
        margin up the tree alongside them.

        Raises if `game` has not finished: `Search` only calls this on a
        terminal node, so a caller passing an unfinished game has a bug of its
        own.
        """
        if not is_over(game):
            raise ValueError("terminal() called on a game that has not finished")
        players = game.state(0, hidden=False).num_players
        winner = game.won_by
        return tuple(1.0 if seat == winner else 0.0 for seat in range(players))


class GatedSearch(Search):
    """Policy/value-guided MCTS with the checkpoint's own trade gate.

    `choose` binds the gate to the live game before searching. Trade
    acceptance and counterparty estimates delegate to `NetworkBot`.
    """

    def __init__(self, evaluator, gate: NetworkBot, **kwargs) -> None:
        super().__init__(evaluator, **kwargs)
        self.gate = gate
        self.trade_floor = gate.trade_floor
        self.gate_plies = gate.gate_plies

    def choose(self, game: Game) -> Action:
        self.gate.seat_at(game)
        return super().choose(game)

    def accepts(self, view: View, received: Bundle, counterparty: int) -> bool:
        return self.gate.accepts(view, received, counterparty)

    def accepts_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[bool]:
        return self.gate.accepts_many(view, received, counterparties)

    def gains_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[float]:
        return self.gate.gains_many(view, received, counterparties)

    def estimate_many(self, view: View, candidates: Sequence[tuple[int, Bundle]]) -> list[float]:
        return self.gate.estimate_many(view, candidates)


def bot_for(
    checkpoint: Checkpoint, *, max_trades: int | None = None,
    rng: random.Random | None = None,
) -> NetworkBot:
    """`checkpoint`, playing one position at a time.

    `max_trades` of `None` means the trade switch the checkpoint recorded
    training under -- the default that measures a policy on the game it
    learned. Pass `0` to disable this bot's trading. `rng` seeds the
    belief-sampled worlds a `gate_plies > 0` checkpoint's continuation rolls
    forward on (`NetworkBot._world_rng`); a checkpoint asking for none never
    reads it.

    The gate's floor and its continuation budget both come from the
    checkpoint too, for the same reason: they describe the exported model,
    not this adapter. A checkpoint that declares none of it is read at the
    unmeasured defaults.
    """
    gate = gate_config_of(checkpoint)
    return NetworkBot(
        policy=checkpoint.policy,
        players=checkpoint.players,
        max_trades=checkpoint.max_trades if max_trades is None else max_trades,
        trade_floor=gate.trade_floor,
        gate_plies=gate.plies,
        rng=random.Random() if rng is None else rng,
    )


def searcher_for(
    checkpoint: Checkpoint,
    *,
    simulations: int = 128,
    wave: int = 16,
    max_trades: int | None = None,
    inference_batch: int | None = None,
    rng: random.Random | None = None,
) -> GatedSearch:
    """`checkpoint` as a batched PUCT search, trading through its own
    value-head gate (`GatedSearch`).

    A supplied `rng` seeds both the search itself (`hexset.mcts.Search`)
    and, independently, the trade gate's own belief-sampling stream
    (`gate_rng`, derived from `rng` so the whole entrant is reproducible
    from one seed) -- a checkpoint whose `gate_plies` is `0` never reads
    the latter.
    """
    gate_rng = None if rng is None else random.Random(rng.getrandbits(128))
    budget = checkpoint.max_trades if max_trades is None else max_trades
    return GatedSearch(
        LeafEvaluator(
            policy=checkpoint.policy,
            pad_to=inference_batch,
        ),
        bot_for(checkpoint, max_trades=budget, rng=gate_rng),
        simulations=simulations,
        wave=wave,
        max_trades=budget,
        rng=rng,
    )


def _checkpoint_path(weights: object, what: str) -> str:
    """An `Entrant.weights` read as a checkpoint path.

    `weights` is fitted evaluation coefficients for a handcrafted entrant and
    a path for a network-backed one; a lineup that mixes the two up is
    refused here rather than inside a loader, where the error would name a
    numpy array.
    """
    if isinstance(weights, str):
        return weights
    raise ValueError(f"{what}'s weights is a checkpoint path")


def register_entrants(loader) -> None:
    """Register network and MCTS arena factories for a checkpoint loader.

    ``loader(path, topology)`` returns a Checkpoint. Call this explicitly in
    each process that will spawn entrants; imports do not choose a runtime.
    """
    from hexset.arena import register_entrant_kind

    def spawn_network(entrant, board: Board, rng) -> NetworkBot:
        return bot_for(
            loader(_checkpoint_path(entrant.weights, "network"), board.topology),
            max_trades=entrant.max_trades,
            rng=rng,
        )

    def spawn_mcts(entrant, board: Board, rng) -> GatedSearch:
        return searcher_for(
            loader(_checkpoint_path(entrant.weights, "mcts"), board.topology),
            simulations=entrant.simulations, wave=entrant.wave,
            max_trades=entrant.max_trades, rng=rng,
        )

    register_entrant_kind("network", spawn_network)
    register_entrant_kind("mcts", spawn_mcts)


def load_runtime(module: str) -> None:
    """Import `module` for its side effect: a `register_entrants` call.

    A runtime that can open a checkpoint needs torch, or onnxruntime, or
    something else this distribution does not depend on, so hexset cannot
    import one by name of its own -- and a driver that wrapped a hexset CLI
    just to get its own import in first was carrying a whole module to say
    one line. Naming the module on the command line is that line, moved to
    where the entrant is resolved.

    Module scope on purpose, and taking a string rather than a callable:
    `hexset.arena.compete` hands this to a spawned worker as its
    `worker_initializer`, where a closure or a bound loader would not
    survive the pickle.
    """
    importlib.import_module(module)
