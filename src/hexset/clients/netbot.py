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
from hexset.clients.policy import Checkpoint, Policy
from hexset.game import Game, imagine, is_over, to_move
from hexset.mcts import Search
from hexset.actions import options_for
from hexset.clients.modelmeta import DEFAULT_GATE_PLIES, DEFAULT_TRADE_FLOOR, gate_config_of
from hexset.economy import COSTS, Purchase
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
    road_count,
    road_placeable,
    settlement_count,
    settlement_placeable,
)
from hexset.trading import exchange
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
    candidate, only the hand does."""
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

    **The affordability filter.** A raw value-head reading of two
    near-identical hands differs by the head's own noise -- measured once
    at a won position (the winning settlement already in hand), where the
    old two-world continuation this replaced was built to fix exactly that:
    the older gate, without it, priced a trade that changed nothing at a
    small nonzero value. Rather than asking the head to resolve a
    difference below its own resolution, `_kinds_of` asks a cheaper
    question first: does this trade change what the seat can buy? A hand
    and a candidate's hand affording exactly the same set of `ROAD`,
    `SETTLEMENT`, `CITY` and `DEV_CARD` (`hexset.actions._building_actions`,
    `hexset.devcards.can_buy` decide the same "afford, and somewhere legal
    to spend it" question for the current player's real turn; this is that
    same test's arithmetic half, run against a hypothetical hand) is a
    residual-hand difference the head cannot resolve above its own noise --
    "does this let me buy something I could not" is a deduction, and the
    filter makes it with arithmetic rather than asking the head. Such a
    candidate is never scored: its gain and its estimate are `0.0` outright,
    which `hexset.trading.clears_floor` never clears (a gain at or below
    zero never clears), so it is never offered, accepted or countered with.
    Only a candidate that adds a kind or loses one reaches the forward. The
    known simplification: a count change within a kind (two roads bought
    instead of one) does not survive the filter.

    `trade_floor` is the checkpoint's own, read off the file it was loaded
    from (`hexset.clients.modelmeta.gate_config`) and passed in by
    `bot_for`. A floor measured against one value head says nothing about
    another's, so it is not a constant of this module: the default below is
    what a checkpoint that declares nothing gets.

    `gate_plies` is that same checkpoint's own continuation budget
    (`hexset.clients.modelmeta.GateConfig.plies`): `0`, the default, prices
    every survivor in the one forward `_evaluate` already builds, exactly
    as `hand` plus the bundle, this seat's own frame, and nothing more. `N`
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
        (the affordability filter's `DEV_CARD` geometry), the asking seat's
        information set (`View.signature`, every hand size), and the
        candidates asked about. A miss on any of it evaluates afresh.
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
        candidate that survives the affordability filter, and the set that
        does not.

        At `gate_plies == 0`, the shipped default, each candidate's
        post-trade position is read from `seat`'s own frame -- its own
        hand exactly (`hand` plus the bundle) and the counterparty's known
        lower bound moved the same way (`view.known[them]`), never the true
        hand, which this seat may not read (`_after`) -- and read in one
        forward, exactly as it stands. No world is sampled: see the class
        docstring for why a paired reading at all, and why the filter below
        is what makes most of those readings unnecessary.

        A checkpoint asking for `gate_plies > 0` rolls the mover's own
        greedy policy that many plies forward from each survivor first
        (`_continue`) before that forward runs -- the trade decision's own
        search, same footing as `simulations` is for `hexset.mcts.Search`.
        That rollout *acts*, so `_after`'s hand-edited position (which
        leaves every other seat's true hand sitting right there, merely
        unread) is not safe to hand it: a Knight steal mid-turn would
        resolve against those real cards. Every world here is instead
        sampled from `seat`'s own belief (`View.sample`), certified so each
        survivor's own exchange can never go negative -- one shared draw:
        `before` and every survivor's `after` come from the same sampled
        state, so the pair stays paired the way the class docstring
        describes, and a single shared `before` (`values[0]`) prices every
        survivor exactly as the `gate_plies == 0` return does.

        A candidate `seat` cannot cover is skipped outright (`gains_many`/
        `estimate_many` answer it `-1.0`); every other candidate is
        classified by the affordability filter, and only the surviving ones
        cost a position at all. When none survive, nothing is asked of the
        policy: the live position's value is never read by a candidate
        priced `0.0` outright, so there is no reason to forward it either.
        """
        game = self._seated
        assert game is not None  # callers check this first
        state = view.state
        hand = list(view.known[seat])
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

        if self.gate_plies == 0:
            afters = {
                i: self._after(seat, them, hand, bundle, view)
                for i, them, bundle in survivors
            }
            rows: list[tuple[Game, int]] = [(game, seat)] + [
                (after, seat) for after in afters.values()
            ]
            values = self.policy.value_rows(rows)
            afters_values = {i: values[row] for row, i in enumerate(afters, start=1)}
            return values[0], afters_values, frozenset(zeros)

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
