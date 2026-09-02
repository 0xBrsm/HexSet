# SPDX-License-Identifier: GPL-3.0-only
"""The offer-protocol adapter: a valuation, protocol-free, then a thin
protocol adapter over it.

The valuation (marginal values, `deficit` and `surplus`, `candidate_bundles`,
`score_proposal`, `accept_rule`, `counter_of`, `rank_partners`) is what is
fitted or tested for strength. The adapter (`propose_actions`, and the
`ACCEPT_TRADE` gate `search.Heximax._options_in` calls through `accept_rule`)
is mechanical, untuned, and expected to be rewritten whenever the engine's
trading protocol changes. `_TradeMixin` supplies these as methods on
`Heximax` (`search.Heximax(_TradeMixin)`); everything here reads `self` the
same way whether `self` is defined in this file or in `search.py` --
`self.omniscient`, `self._rank`, `self.evaluator` and the dataclass fields
declared on `Heximax` all resolve through the concrete class's attributes at
runtime, the same attribute lookup as before the split.

NOTE: this layer is scheduled to be replaced by the one-event valuation
mechanic (HexSet/HexNet trading design §8) -- keep it intact, do not redesign.
"""

from __future__ import annotations

from typing import Sequence

from hexset.actions import Action, ActionType
from hexset.board.terrain import NUM_RESOURCES
from ..search2 import STANCES
from hexset.economy import BANK_TRADE_RATIO, trade_ratios
from hexset.game import Game
from hexset.ledger import PublicLedger
from hexset.state import GameState, copy_state
from hexset.trading import Bundle, Offer, can_propose

# How many of `candidate_bundles`' candidates `propose_actions` runs the
# partner-aware `score_proposal` on, after the cheap `deficit`/`surplus`
# pre-filter. `score_proposal` is the adapter's dominant cost (a handful of
# `bundle_delta` calls per candidate, each two evaluations); this bound is
# what keeps that cost off the leaf budget's books. See `propose_actions`.
PROPOSE_SHORTLIST = 5


class _TradeMixin:
    """`Heximax`'s trade methods, mixed into the dataclass defined in
    `search.py`. Not a dataclass itself -- it carries no fields, only
    behaviour, so mixing it in changes nothing about `Heximax`'s generated
    `__init__`/`__repr__`/`__eq__`."""

    # --- trade --------------------------------------------------------------

    def _vector(self, state: GameState, ledger: PublicLedger, knower: int) -> list[float]:
        """The per-seat vector for `state`, read through `knower`'s own belief
        (its hand exact, everyone else's `expected_hand`)."""
        belief = self.evaluator.belief_for(state, ledger, knower)
        return self.evaluator.evaluate(state, knower, belief)

    def _read_row(
        self, state: GameState, ledger: PublicLedger, knower: int, target: int, rank, *,
        vector: list[float] | None = None,
    ) -> float:
        """`target`'s row of `knower`'s vector, under `rank` -- how the
        valuation ever estimates "what would someone else do" without
        reading that someone's true hand, see `_partner_delta`. `vector`,
        when given, skips rebuilding it for a caller that already has it."""
        if vector is None:
            vector = self._vector(state, ledger, knower)
        return rank(vector, target)

    def _read(
        self, state: GameState, ledger: PublicLedger, seat: int, *,
        vector: list[float] | None = None,
    ) -> float:
        """`seat`'s own row of the vector, under the bot's configured stance
        (`_read_row` with `knower == target == seat`) -- the common case
        every valuation method that isn't comparing seats uses; the others
        call `_partner_delta` directly with an explicit `rank`."""
        return self._read_row(state, ledger, seat, seat, self._rank, vector=vector)

    def marginal_gain(
        self, game: Game, seat: int, resource: int, *,
        before_vector: list[float] | None = None,
    ) -> float:
        """Eval(hand + one `resource`) - Eval(hand), from `seat`'s own reading."""
        before = self._read(game.state, game.ledger, seat, vector=before_vector)
        state = copy_state(game.state)
        state.hands[seat][resource] += 1
        if state.bank[resource] > 0:
            state.bank[resource] -= 1
        return self._read(state, game.ledger, seat) - before

    def marginal_loss(
        self, game: Game, seat: int, resource: int, *,
        before_vector: list[float] | None = None,
    ) -> float:
        """Eval(hand) - Eval(hand less one `resource`); zero when none is held."""
        if game.state.hands[seat][resource] < 1:
            return 0.0
        before = self._read(game.state, game.ledger, seat, vector=before_vector)
        state = copy_state(game.state)
        state.hands[seat][resource] -= 1
        state.bank[resource] += 1
        ledger = game.ledger.copy()
        ledger.spend(seat, resource, 1)
        return before - self._read(state, ledger, seat)

    def deficit(
        self, game: Game, seat: int, *, before_vector: list[float] | None = None,
    ) -> dict[int, float]:
        """Marginal gain of receiving one more of each resource, `seat`'s own
        reading. Every resource is included, even ones already held: a
        second wheat can still be worth more than nothing."""
        return {
            r: self.marginal_gain(game, seat, r, before_vector=before_vector)
            for r in range(NUM_RESOURCES)
        }

    def surplus(
        self, game: Game, seat: int, *, before_vector: list[float] | None = None,
    ) -> dict[int, float]:
        """Marginal loss of giving up one of each resource `seat` actually
        holds. Resources not held are left out rather than scored zero:
        `marginal_loss` reads 0.0 for an empty resource because there is
        nothing to give, not because giving it away would cost nothing, and
        keeping it in would make an absent resource look like the cheapest
        one to part with."""
        return {
            r: self.marginal_loss(game, seat, r, before_vector=before_vector)
            for r in range(NUM_RESOURCES)
            if game.state.hands[seat][r] > 0
        }

    def _partner_delta(
        self, game: Game, knower: int, target: int, give: Sequence[int], want: Sequence[int],
        counterparty: int, rank, *, before_vector: list[float] | None = None,
    ) -> float:
        """`target`'s row of the vector if `target` gave `give` and got `want`
        from `counterparty`, read entirely through `knower`'s own belief.

        `bundle_delta` is the case `target == knower == seat`, where the
        trader's own hand is legitimately exact; this generalisation
        estimates what a DIFFERENT seat would make of a trade from
        `knower`'s belief alone (`score_proposal`'s `willing`,
        `rank_partners` and the search's gate on a simulated opponent's
        `ACCEPT_TRADE` all read someone else's row this way). Invariants:
        the ledger is updated from `give`/`want` directly, never by diffing
        `state.hands`, so a third party's hidden composition cannot leak
        through a clamp-at-zero; and `state.hands` moves exactly for
        `knower`, folded into one total for anyone else, because only
        `knower`'s own row is ever read verbatim. `before_vector`, when
        given, is `_vector(game.state, game.ledger, knower)` -- the "before"
        side does not depend on `give`/`want`/`counterparty`, so a caller
        making several of these against one unmutated game computes it once.
        """
        before = self._read_row(
            game.state, game.ledger, knower, target, rank, vector=before_vector
        )
        state = copy_state(game.state)
        exact = self.omniscient
        self._move_hand(state, knower, target, gains=want, losses=give, exact=exact)
        self._move_hand(state, knower, counterparty, gains=give, losses=want, exact=exact)
        ledger = game.ledger.copy()
        for r in range(NUM_RESOURCES):
            if give[r]:
                ledger.spend(target, r, give[r])
                ledger.receive(counterparty, r, give[r])
            if want[r]:
                ledger.spend(counterparty, r, want[r])
                ledger.receive(target, r, want[r])
        return self._read_row(state, ledger, knower, target, rank) - before

    @staticmethod
    def _move_hand(
        state: GameState, knower: int, seat: int, *,
        gains: Sequence[int], losses: Sequence[int], exact: bool = False,
    ) -> None:
        """`seat`'s hand in `state` after gaining `gains` and losing `losses`.

        Exact, per resource, when `seat == knower` -- its own hand is read
        verbatim, so it must reflect the trade precisely. Otherwise only the
        total moves (folded into one resource slot): see `_partner_delta`.

        `exact` forces the per-resource move for every seat, and
        `_partner_delta` passes `self.omniscient`. The fold is not an
        approximation the honest bot tolerates but an exact identity for it:
        a non-knower's hand reaches the honest evaluation only through
        `Belief.expected_hand`, which reads `known`/`unknown`/`pool` off the
        ledger and the bank, and the one thing it takes from `state.hands` is
        the *size* (`Belief.sizes`) -- which the fold preserves while a
        per-resource move, clamped at zero when the seat cannot cover
        `losses`, would not. Under omniscience that reasoning is void: `known`
        *is* `state.hands`, every row is scored on the real cards, and folding
        would price an all-one-resource fiction whose `progress`, `diversity`
        and `scarce` terms are nothing like the position's -- it overstated a
        one-for-one trade's cost to the partner by ~10x, so `willing` almost
        never fired and `accept_rule`, reading a counterparty it had just
        impoverished under `relative`, cleared far too easily.
        """
        hand = state.hands[seat]
        if seat == knower or exact:
            for r in range(len(hand)):
                hand[r] += gains[r] - losses[r]
                if hand[r] < 0:
                    hand[r] = 0
        else:
            net = sum(gains) - sum(losses)
            state.hands[seat] = [max(0, sum(hand) + net)] + [0] * (len(hand) - 1)

    def bundle_delta(
        self, game: Game, seat: int, give: Sequence[int], want: Sequence[int],
        counterparty: int, *, before_vector: list[float] | None = None,
    ) -> float:
        """How much `seat` gains by giving `give` for `want` with `counterparty`.

        Read from `seat`'s own information under the stance, so under
        `relative` it depends on who the counterparty is: the ledger
        certifies `counterparty` as having received `give` and spent `want`
        regardless of what it truly held, and what it can do with the
        result is part of the reading. `seat` is assumed to hold `give`
        (true for every candidate `candidate_bundles` emits, `can_propose`
        checked); its own hand is clamped at zero rather than driven
        negative if not. `before_vector`: see `_partner_delta`.
        """
        return self._partner_delta(
            game, seat, seat, give, want, counterparty, self._rank, before_vector=before_vector
        )

    def candidate_bundles(
        self, game: Game, seat: int, *, max_side: int = 2,
    ) -> list[tuple[Bundle, Bundle]]:
        """Bundles built from `deficit` x `surplus`'s resources, 1-2 cards a side.

        The give side is drawn from resources `seat` holds (a proposer can
        only offer what it has); the want side from every resource, since
        wanting one needs only a table that might supply it
        (`score_proposal`'s `p_holds` prices that later). A 2-for-1 of a
        single surplus resource is added whenever a port makes it cheaper
        than the bank, independent of `max_side`. Every returned pair is
        `can_propose` against the current state, so every candidate is legal
        to emit as-is. The resource sets are built directly rather than
        through `deficit`/`surplus`, whose `evaluate()`-backed marginal
        reads this method never uses; `propose_actions` calls those itself.
        """
        state = game.state
        hand = state.hands[seat]
        deficits = list(range(NUM_RESOURCES))
        surpluses = [r for r in range(NUM_RESOURCES) if hand[r] > 0]

        give_options = _bundle_options(surpluses, max_side, cap=lambda r: hand[r])
        ratios = trade_ratios(state, seat)
        for r in surpluses:
            if ratios[r] < BANK_TRADE_RATIO and hand[r] >= 2:
                ported = _one_hot(r, 2)
                if ported not in give_options:
                    give_options.append(ported)

        want_options = _bundle_options(deficits, max_side)

        # `_bundle_options` never repeats a bundle, so the pairs are distinct
        # by construction and no seen-set is needed; `can_propose` is
        # `well_formed` and `holds`, so calling it alone is the whole test.
        return [
            (give, want)
            for give in give_options
            for want in want_options
            if can_propose(state, Offer(proposer=seat, give=give, want=want))
        ]

    def score_proposal(
        self, game: Game, seat: int, give: Sequence[int], want: Sequence[int], *,
        before_vector: list[float] | None = None,
    ) -> float:
        """`dEval_me(after, best counterparty) x sum_opp p_holds(opp, want) * willing(opp)`.

        `dEval_me` is `bundle_delta` maximised over which opponent ends up on
        the other side of the trade, so this asks who the trade would be
        best made *with*, not merely whether to make it at all. `willing(opp)`
        reads `opp`'s row of the vector on `opp`'s *expected* hand under
        `seat`'s own belief (`_partner_delta`, never `opp`'s true hand), the
        same evaluator applied from that seat's point of view -- hence
        `self._rank` rather than a fixed stance. It is a crisp 0/1 test
        rather than a smoothed one: there is no fitted slope to smooth it
        with yet. `before_vector`: see `_partner_delta`.
        """
        opponents = [p for p in range(game.state.num_players) if p != seat]
        if not opponents:
            return 0.0
        if before_vector is None:
            before_vector = self._vector(game.state, game.ledger, seat)
        delta_me = max(
            self.bundle_delta(game, seat, give, want, opp, before_vector=before_vector)
            for opp in opponents
        )
        belief = self.evaluator.belief_from_game(game, seat)
        weight = 0.0
        for opp in opponents:
            chance = belief.p_holds(opp, want)
            if chance <= 0:
                continue
            theirs = self._partner_delta(
                game, seat, opp, want, give, seat, self._rank, before_vector=before_vector
            )
            if theirs > 0:
                weight += chance
        return delta_me * weight

    def accept_rule(
        self, game: Game, seat: int, offer: Offer, margin: float, *, knower: int | None = None,
    ) -> bool:
        """Accept iff taking `offer` clears `margin`.

        Taking it means `seat` gives `offer.want` and receives `offer.give`
        with `offer.proposer` as the counterparty. `margin` is a parameter
        rather than always `self.accept_margin` so the rule can be probed at
        any threshold. `knower` is whose belief the read is anchored to; it
        defaults to `seat` itself (the bot's own real decision, hand exact),
        but `_options_in` passes the search's root seat instead when this
        gates a simulated opponent deeper in the tree, so that opponent's
        hand stays read as `knower`'s `expected_hand`, never the sampled
        world's truth.
        """
        return self._partner_delta(
            game,
            seat if knower is None else knower,
            seat,
            offer.want,
            offer.give,
            offer.proposer,
            self._rank,
        ) > margin

    def counter_of(self, game: Game, seat: int, offer: Offer) -> tuple[Bundle, Bundle] | None:
        """The candidate nearest `offer.want`, among my own surplus-built bundles.

        Distance is taxicab on the `want` side: the candidate I would ask
        for that most resembles what the table has already shown interest
        in. `None` when I have no candidate at all. Today's engine has no
        counter action, so nothing calls this yet; it is unit-tested only.
        """
        candidates = self.candidate_bundles(game, seat)
        if not candidates:
            return None

        def distance(pair: tuple[Bundle, Bundle]) -> int:
            _, want = pair
            return sum(abs(a - b) for a, b in zip(want, offer.want))

        return min(candidates, key=distance)

    def rank_partners(
        self, game: Game, seat: int, give: Sequence[int], want: Sequence[int], *,
        before_vector: list[float] | None = None,
    ) -> tuple[int, ...]:
        """Opponents ranked for `Action.ask`, first to whoever it helps least.

        `p_holds(opp, want) * (-bundle_delta_them)` under `paranoid`
        regardless of the bot's own configured stance -- `paranoid` is the
        only stance that can tell opponents apart, which is what ordering an
        ask needs. `p_holds` uses the belief, never `state.hands[opp]`;
        "how much would it help them" is `opp`'s row read through `seat`'s
        own belief (`_partner_delta`, `opp`'s *expected* hand, never its true
        one). `before_vector`: see `_partner_delta` -- still the vector read
        under `seat`'s own belief; only the read-out differs by stance.
        """
        belief = self.evaluator.belief_from_game(game, seat)
        paranoid = STANCES["paranoid"]
        opponents = [p for p in range(game.state.num_players) if p != seat]
        if before_vector is None:
            before_vector = self._vector(game.state, game.ledger, seat)

        def score(opp: int) -> float:
            chance = belief.p_holds(opp, want)
            theirs = self._partner_delta(
                game, seat, opp, want, give, seat, paranoid, before_vector=before_vector
            )
            return chance * -theirs

        return tuple(sorted(opponents, key=score, reverse=True))

    # -- the adapter: everything above is protocol-free valuation; this is
    # where it meets today's protocol, at two touch points called from
    # `search` above: `_root_options` calls `propose_actions` (entry);
    # `_options_in` calls `accept_rule` (exit, a plain predicate reused as a
    # gate). A later protocol changes this method and that call site only.

    def propose_actions(self, game: Game, seat: int) -> list[Action]:
        """The adapter: the top `propose_top_n` scored proposals.

        Replaces the engine's one-for-one `PROPOSE_TRADE` sample
        (`actions._offer_actions`) among the root options -- mechanical and
        untuned, rewritten whenever the protocol changes. Candidates and
        their scores are read off the true game, not per determinized world:
        an offer's legality depends only on what the proposer holds, which
        is always exact, and the valuation already reads opponents through
        the honest belief. The `k` worlds still decide what a proposal is
        *worth* once it is a root option. Candidates at or below
        `propose_margin` are dropped, and `score_proposal` -- a
        `bundle_delta`/`_partner_delta` per opponent, twice over -- is the
        module's single largest cost, so the cheap `deficit`/`surplus`
        shortlist the field to `PROPOSE_SHORTLIST` first.
        """
        # Every "before" read below shares this one (game.state, game.ledger,
        # seat) triple, computed once rather than per candidate.
        before_vector = self._vector(game.state, game.ledger, seat)
        candidates = self.candidate_bundles(game, seat)
        if not candidates:
            return []
        deficit = self.deficit(game, seat, before_vector=before_vector)
        surplus = self.surplus(game, seat, before_vector=before_vector)

        def cheap_score(pair: tuple[Bundle, Bundle]) -> float:
            give, want = pair
            gain = sum(want[r] * deficit.get(r, 0.0) for r in range(NUM_RESOURCES))
            loss = sum(give[r] * surplus.get(r, 0.0) for r in range(NUM_RESOURCES))
            return gain - loss

        shortlist = sorted(candidates, key=cheap_score, reverse=True)[:PROPOSE_SHORTLIST]
        scored = [
            (self.score_proposal(game, seat, give, want, before_vector=before_vector), give, want)
            for give, want in shortlist
        ]
        scored = [s for s in scored if s[0] > self.propose_margin]
        scored.sort(key=lambda s: -s[0])
        return [
            Action(
                ActionType.PROPOSE_TRADE,
                give=give,
                want=want,
                ask=self.rank_partners(game, seat, give, want, before_vector=before_vector),
            )
            for _, give, want in scored[: self.propose_top_n]
        ]



def _one_hot(resource: int, n: int) -> Bundle:
    """`n` cards of one `resource`, as a `Bundle`."""
    out = [0] * NUM_RESOURCES
    out[resource] = n
    return tuple(out)


def _two_hot(r1: int, r2: int) -> Bundle:
    """One card each of two distinct resources, as a `Bundle`."""
    out = [0] * NUM_RESOURCES
    out[r1] += 1
    out[r2] += 1
    return tuple(out)


def _bundle_options(resources: list[int], max_side: int, *, cap=None) -> list[Bundle]:
    """One-hot bundles of each resource up to `max_side` (or `cap(r)` if
    given), plus every two-hot pair once `max_side >= 2`."""
    size = (lambda r: min(max_side, cap(r))) if cap else (lambda r: max_side)
    options = [_one_hot(r, n) for r in resources for n in range(1, size(r) + 1)]
    if max_side >= 2:
        options.extend(
            _two_hot(r1, r2) for i, r1 in enumerate(resources) for r2 in resources[i + 1 :]
        )
    return options
