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
from hexset.cards import DevCard, ROAD_BUILDING_ROADS
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


# The reachable-purchase gate builds exactly one row per side -- the
# baseline and each survivor -- so one ask almost never approaches this;
# it only chunks `value_rows` if it somehow does (`NetworkBot._evaluate_reachable`).
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
        `R(hand + bundle)` (`_recipes_with_road_building`, compared as the
        set of keys) -- a bundle that changes neither is priced `0.0`
        outright, exactly as `_kinds_of` did for `gate_plies > 0`, only
        exact rather than coarsened to four kinds.

        Every survivor is then valued by exactly one built position, not
        the maximum over every placement of every maximal multiset: a
        maximum over many noisy value-head rows drifts up with how many
        options a hand happens to have, which is its own version of the
        resolution problem the class docstring describes. `_choose_multiset`
        picks one maximal reachable multiset by a fixed rule (most
        settlements and cities, then dev cards, then roads, then total
        pieces, ties broken by the tuple itself), and `_build_row` places
        it by arithmetic alone (`_place_deterministic` -- no value read
        chooses a placement). The baseline and every survivor go into one
        `value_rows` forward together; `gains_many` reads the change in
        `values[seat]` between two single, deterministic rows, and
        `estimate_many` reads `values[them]` off the same pair.
        """
        ctx = self._reach_context(seat, view)
        rb_context = self._road_building_context(seat, view, ctx)
        base_hand: Hand = tuple(hand)
        base_recipes = self._recipes_with_road_building(base_hand, ctx, rb_context)
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
                recipes = self._recipes_with_road_building(candidate_hand, ctx, rb_context)
                recipe_cache[candidate_hand] = recipes
            if frozenset(recipes) == base_R:
                zeros.add(i)
                continue
            survivors.append((i, counterparties[i], bundle, recipes))

        if not survivors:
            return (), {}, frozenset(zeros)

        baseline_seed = self._seeded(seat, base_hand, view)
        baseline_recipe = base_recipes[self._choose_multiset(base_recipes)]
        rows: list[tuple[Game, int]] = [
            (self._build_row(baseline_seed, seat, baseline_recipe), seat)
        ]
        row_index: dict[int, int] = {}
        for i, them, bundle, recipes in survivors:
            seed = self._after(seat, them, hand, bundle, view)
            recipe = recipes[self._choose_multiset(recipes)]
            rows.append((self._build_row(seed, seat, recipe), seat))
            row_index[i] = len(rows) - 1

        values: list[tuple[float, ...]] = []
        for start in range(0, len(rows), _MAX_ROWS):
            values.extend(self.policy.value_rows(rows[start : start + _MAX_ROWS]))

        baseline_values = values[0]
        afters_values = {i: values[row_index[i]] for i, _, _, _ in survivors}
        return baseline_values, afters_values, frozenset(zeros)

    def _recipes_with_road_building(
        self, hand: Hand, ctx: dict, rb_context: tuple | None
    ) -> dict[Purchases, Recipe]:
        """`reachable_recipes(hand, **ctx)`, plus whatever a Road Building
        branch reaches that the ordinary hand-only search does not.

        Road Building spends no resource and touches no other seat's row,
        so it never changes `hand` itself -- only the board (`rb_context`,
        `_road_building_context`: the room and legal counts this seat would
        have after its two free roads). A second `reachable_recipes` call
        against that room/legal, `dev_card_played` forced true (Road
        Building already spent this turn's one card), finds whatever a
        corner the free roads opened newly affords; a multiset the plain
        search already reaches keeps its plain recipe, since a hand that
        can buy it either way owes nothing to the free roads for it.
        """
        recipes = reachable_recipes(hand, **ctx)
        if rb_context is not None:
            edges, room, legal = rb_context
            rb_ctx = dict(ctx, dev_card_played=True, room=room, legal=legal)
            for multiset, recipe in reachable_recipes(hand, **rb_ctx).items():
                if multiset not in recipes:
                    recipes[multiset] = Recipe("road_building", edges, recipe.hand, multiset)
        return recipes

    def _road_building_context(
        self, seat: int, view: View, ctx: dict
    ) -> tuple[tuple[int, ...], tuple[int, int, int], tuple[int, int, int]] | None:
        """If `seat` may still play Road Building this turn, the edges its
        two free roads would land on (`_far_endpoint_pips`, the same
        expansion rule a bought road places by) and the room/legal this
        seat would have afterwards -- `None` if the card is not held,
        already played this turn, or has nowhere to place even one road.

        Placed on a throwaway copy of the *true* board (`view.state`:
        occupancy is public regardless of any hand), never the live game.
        """
        if ctx["dev_card_played"] or ctx["dev_cards_held"][DevCard.ROAD_BUILDING] <= 0:
            return None
        state = copy_state(view.state)
        placed: list[int] = []
        room_left = ctx["room"][0]
        for _ in range(ROAD_BUILDING_ROADS):
            if room_left <= 0:
                break
            legal = self._legal_for(state, seat, Purchase.ROAD)
            if not legal:
                break
            edge = max(legal, key=lambda e: self._far_endpoint_pips(state, seat, e))
            place_road(state, seat, edge)
            placed.append(edge)
            room_left -= 1
        if not placed:
            return None
        topology = state.board.topology
        legal = (
            sum(1 for e in range(topology.num_edges) if road_placeable(state, seat, e)),
            sum(1 for v in range(topology.num_vertices) if settlement_placeable(state, seat, v)),
            sum(1 for v in range(topology.num_vertices) if city_upgradeable(state, seat, v)),
        )
        room = (room_left, ctx["room"][1], ctx["room"][2])
        return tuple(placed), room, legal

    def _choose_multiset(self, recipes: dict[Purchases, Recipe]) -> Purchases:
        """The one maximal reachable multiset a plan is built from: most
        settlements and cities first (each is a victory point), then most
        development cards, then most roads, then more total pieces, ties
        broken by the multiset's own tuple order -- fixed and arithmetic,
        so the baseline and every survivor are compared on the same
        footing rather than on however many placements each happens to
        have (see `_evaluate_reachable`)."""

        def key(multiset: Purchases) -> tuple:
            nr, ns, nc, nd = multiset
            return (-(ns + nc), -nd, -nr, -(nr + ns + nc + nd), multiset)

        return min(maximal(frozenset(recipes)), key=key)

    def _build_row(self, seed: Game, seat: int, recipe: Recipe) -> Game:
        """The one position `recipe` builds from `seed`: its development-card
        branch and bank trades applied (`_apply_branch`), paid for
        (`_pay_purchases`), and every piece placed by arithmetic alone
        (`_place_deterministic`) -- no value read chooses among placements.
        """
        plan_game = self._copy_position(seed)
        self._apply_branch(plan_game, seat, recipe)
        self._pay_purchases(plan_game, seat, recipe.purchases)
        self._place_deterministic(plan_game, seat, recipe.purchases)
        return plan_game

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
        read; Road Building places the two free roads
        `_road_building_context` already chose (`recipe.param`, edges),
        moving no card at all.

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
        elif recipe.branch == "road_building":
            for edge in recipe.param:
                place_road(state, seat, edge)
            update_longest_road(state)
            spend_card(state, seat, DevCard.ROAD_BUILDING)
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
        return sum(
            pips(state.board.tokens[h]) for h in state.board.topology.vertex_hexes[vertex]
        )

    def _far_endpoint_pips(self, state: GameState, seat: int, edge: int) -> int:
        """The expansion ranking a bought or free road is placed by: the
        pip value of whichever endpoint is *not* already this seat's own --
        the direction a road actually reaches new production in, rather
        than the endpoint it is anchored from. Both endpoints count if
        neither is owned yet (the seat's very first road)."""
        a, b = state.board.topology.edges[edge]
        far = [v for v in (a, b) if state.vertex_owner[v] != seat] or [a, b]
        return max(self._vertex_pips(state, v) for v in far)

    def _place_deterministic(self, plan_game: Game, seat: int, purchases: Purchases) -> None:
        """Place every piece in `purchases` by arithmetic alone -- no
        value read chooses among placements: a settlement or city at the
        legal vertex with the highest production pip sum
        (`_vertex_pips`), a road at the legal edge whose far endpoint has
        the highest (`_far_endpoint_pips`, expansion), one piece at a time
        so a second road or settlement sees whatever the first opened.
        """
        nr, ns, nc, _ = purchases
        state = plan_game.state(seat, hidden=False)
        for _ in range(nr):
            legal = self._legal_for(state, seat, Purchase.ROAD)
            if not legal:
                break
            edge = max(legal, key=lambda e: self._far_endpoint_pips(state, seat, e))
            self._place_piece(plan_game, seat, Purchase.ROAD, edge)
        for _ in range(ns):
            legal = self._legal_for(state, seat, Purchase.SETTLEMENT)
            if not legal:
                break
            vertex = max(legal, key=lambda v: self._vertex_pips(state, v))
            self._place_piece(plan_game, seat, Purchase.SETTLEMENT, vertex)
        for _ in range(nc):
            legal = self._legal_for(state, seat, Purchase.CITY)
            if not legal:
                break
            vertex = max(legal, key=lambda v: self._vertex_pips(state, v))
            self._place_piece(plan_game, seat, Purchase.CITY, vertex)

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
