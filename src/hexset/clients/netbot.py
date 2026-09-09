# SPDX-License-Identifier: GPL-3.0-only
"""Runtime-neutral checkpoint adapters for bots, evaluation, search and trading.

Constructors accept `Checkpoint` objects. Runtime integrations own loading,
metadata and encoding; this module uses only the batched `Policy` interface.
A gate used without `choose` must be installed with `seat_at(game)` before
trading. Create separate gates for separate games and fixed seats."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hexset.actions import Action, ActionSpace, ActionType, apply
from hexset.clients.policy import Checkpoint, Policy
from hexset.game import Game, imagine, is_over, to_move
from hexset.mcts import Search
from hexset.actions import options_for
from hexset.state import copy_state
from hexset.trading import NETWORK_GATE_ROWS, exchange
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


@dataclass
class NetworkBot:
    """A policy answering one position at a time.

    Trades off the same value head `choose` already reads, no new
    parameters. The value head scores a candidate exchange from both
    sides at once (`_score`): the live position and the position after
    the cards move -- *both* hands, and the counterparty's ledger row, as a
    real clearing would leave them -- each read from this seat's own
    frame, so nothing here reads a hand this seat may not know. The head
    answers with one win probability per seat, and that vector is the
    whole gate:

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

    `trade_floor` is `0.0`: this gate's resolution has not been measured
    (heximax's has, `hexset.bots.heximax.HEXIMAX_TRADE_FLOOR`); its gains
    are in win probability, so a paired-chance measurement would replace it.
    """

    policy: Policy
    # Carried for a caller that builds a bot by hand and for symmetry with
    # `LeafEvaluator`; nothing here indexes it any more, because encoding a
    # position is the runtime's own business (`hexset.clients.policy`).
    space: ActionSpace
    players: int
    max_trades: int | None = None
    # This gate's clearing floor (`hexset.trading.trade_floor_of`): `accepts`
    # is a strict value-head comparison and `gains_many` reads +1/-1 off it,
    # so there is no resolution for a floor to express; strict positivity is
    # the whole gate.
    trade_floor: float = 0.0
    # Which seat this bot is installed at, or `None` for a bot that answers
    # whatever seat the view names (the arena spawns one bot per seat and
    # never asks it about another). When set, the view must agree: a gate
    # wired to the wrong seat would answer for somebody else's hand, and
    # that is the one failure the mechanic must never have quietly.
    seat: int | None = None
    # Draws for the gate's imagined continuations (`_continue`): the deck a
    # rolled-out position buys from is reshuffled so the gate never reads the
    # real deck order, and a knight's steal draws here rather than from the
    # live table's chance.
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)
    # Drawn once from `rng` when first needed: what makes this bot's worlds its
    # own while keeping each candidate's world a pure function of the ask.
    _salt: int | None = field(default=None, repr=False, compare=False)
    # The game `choose` was last handed (or `seat_at` seated), so a trade
    # event -- which runs inside the same `apply` this bot's own choice
    # already went through -- asks about the position it is actually seated
    # at. `None` only for a bot nobody has seated yet, which cannot happen
    # in play (the trade event runs after every seat has moved through
    # setup) but is the right answer for the gate regardless.
    _seated: Game | None = field(default=None, repr=False, compare=False)

    def seat_at(self, game: Game) -> None:
        """Seat this bot at `game` without asking it to move: the position
        its gate is an evaluation of. `choose` does this itself; a driver
        that installs a bot as a gate only (a searched policy, a collector's
        seat) calls it in place of poking `_seated`."""
        _check_players(game, self.players)
        self._seated = game

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
        `-1.0` for a candidate this seat cannot cover and for any past the
        scored set (`_score`'s cap); nothing when nobody has seated this bot
        (no `choose` has run) or trading is switched off.
        """
        if self.max_trades == 0 or self._seated is None or not received:
            return [-1.0] * len(received)
        seat = self._perspective(view)
        pairs = self._score(seat, view, received, counterparties)
        out = [-1.0] * len(received)
        for i, (before, after) in pairs.items():
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
        pairs = self._score(seat, view, received, thems)
        out = [-1.0] * len(candidates)
        for i, (before, after) in pairs.items():
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
    ) -> dict[int, tuple[tuple[float, ...], tuple[float, ...]]]:
        """For each scored candidate, the value vector of the position without
        the trade and with it, both after the mover's best play from there,
        both read from `seat`'s frame -- a paired reading, so a trade that
        changes nothing a seat can do prices at exactly zero.

        **Continuations, not hands.** Two raw value estimates of nearly
        identical hands differ by the head's noise, and a gate that compared
        them offered noise: at a won position (the winning settlement in
        hand) it priced a trade at +0.006 that changed nothing. So every
        position is valued after the *mover's* own best play from it
        (`_continue`): the policy's greedy actions through the rest of the
        turn, a finished game reading as its one-hot winner. The mover is
        whoever is to move -- this seat when it is the actor asking about
        its own offer, the actor when this seat is asked to respond -- so a
        responder prices "what does the actor do with these cards" and a
        counter into a seat that wins next action reads as that win for the
        actor and zero for everyone else. The g4 game of 2026-09-08 19:47Z,
        round 19, is the case this was written for.

        **Honest worlds.** The rollout needs hands to act on, and this seat
        may not read another's. Each candidate is scored in one world drawn
        from this seat's own belief (`View.sample`), with the counterparty
        certified to hold what the candidate says it gives (an offer is
        evidence of the cards behind it); the same world, exchanged and not,
        is what makes the pair paired. The draw is seeded by the position and
        the candidate (`_world_rng`), so `gains_many` and `estimate_many`
        asked about the same candidate at the same position read the same
        world -- a round's own gain and its estimate of the other side are
        then one judgement, not two draws -- and the gate is a pure function
        of what it was asked. Worlds are `imagine`d copies -- their own
        state, ledger and a fresh chance with the deck reshuffled -- so
        nothing here touches the live table or reads its deck.

        Every candidate two cards or fewer a side (`_is_small`) is scored;
        the rest fill whatever is left of `NETWORK_GATE_ROWS` in the order
        the engine enumerated them -- the stated cost bound on this gate's
        own evaluation. A candidate `seat` cannot cover is never scored.
        """
        game = self._seated
        assert game is not None  # callers check this first
        hand = list(view.known[seat])
        small = [i for i, bundle in enumerate(received) if _is_small(bundle)]
        rest = [i for i in range(len(received)) if not _is_small(received[i])]
        order = (small + rest)[: max(NETWORK_GATE_ROWS, len(small))]
        mover = to_move(game)

        worlds: list[Game] = []
        scored: dict[int, int] = {}
        for i in order:
            bundle = received[i]
            if any(n + d < 0 for n, d in zip(hand, bundle)):
                continue
            them = counterparties[i]
            certified = View(
                view.state, view.ledger, seat,
                certify=[(them, [max(0, n) for n in bundle])],
            )
            rng = self._world_rng(view, them, bundle)
            sampled = certified.sample(rng)
            before = imagine(game, rng, randomize_deck=True)
            before.set_state(sampled)
            after = imagine(game, rng, randomize_deck=True)
            state = copy_state(sampled)
            hands_before = [h[:] for h in state.hands]
            exchange(state, seat, them, bundle)
            after.set_state(state)
            after.ledger.apply_hand_diff(hands_before, state.hands)
            scored[i] = len(worlds)
            worlds.extend((before, after))
        values = self._continue(mover, seat, worlds)
        return {i: (values[row], values[row + 1]) for i, row in scored.items()}

    def _world_rng(self, view: View, them: int, bundle: Bundle) -> random.Random:
        """The draw behind one candidate's world: seeded by the information
        set (`View.signature`, every hand size, the perspective), the
        counterparty and the bundle, salted by this bot's own `rng` once at
        construction, so two asks about the same candidate at the same
        position agree and two bots draw different worlds."""
        if self._salt is None:
            self._salt = self.rng.getrandbits(64)
        key = (self._salt, view.perspective, tuple(view.sizes), view.signature(), them, tuple(bundle))
        return random.Random(hash(key))

    def _continue(
        self, mover: int, seat: int, worlds: Sequence[Game]
    ) -> list[tuple[float, ...]]:
        """Each world's value vector, from `seat`'s frame, after `mover`'s
        best play from it.

        The worlds are already imagined copies (`_score`), safe to play on.
        In lockstep across all of them, the policy picks `mover`'s next
        action wherever it is still `mover`'s turn (`act_rows`, one batched
        forward a ply) -- the bot's own policy standing in for whoever moves,
        acting on the hand the world gives that seat; an `END_TURN` pick
        stops that world where it stands, a finished world stops as its
        winner, and `CONTINUATION_PLIES` bounds the rest. What is left is
        valued in one forward; a finished world is the one-hot winner,
        board-seat order, exactly as `LeafEvaluator.terminal` reads it.
        """
        live = [i for i, g in enumerate(worlds) if not is_over(g) and to_move(g) == mover]
        for _ in range(CONTINUATION_PLIES):
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


# How far a gate's imagined continuation may run: enough for a whole turn's
# worth of builds and buys (a rich hand rarely takes more than a handful of
# actions), bounded so a policy that never picks END_TURN cannot spin.
CONTINUATION_PLIES = 8


def _is_small(bundle: Bundle) -> bool:
    """Both sides of `bundle` move at most two cards -- always scored,
    uncapped, because two-for-one and two-for-two are the overwhelming
    majority of what a table trades."""
    give = sum(-n for n in bundle if n < 0)
    take = sum(n for n in bundle if n > 0)
    return give <= 2 and take <= 2


@dataclass
class NetworkEvaluator:
    """A checkpoint value head exposed as per-seat game evaluation."""

    policy: Policy
    players: int
    max_trades: int | None = None

    def evaluate_game(self, game: Game, seat: int) -> list[float]:
        _check_players(game, self.players)
        return list(self.policy.value_rows([(game, seat)])[0])


@dataclass
class LeafEvaluator:
    """A whole wave of `hexset.mcts` leaves in one forward."""

    policy: Policy
    # The action space declared by the checkpoint.
    space: ActionSpace
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
    learned. Pass `0` to disable this bot's trading. `rng` controls imagined
    trade continuations; policy action sampling remains the runtime's job.
    """
    return NetworkBot(
        policy=checkpoint.policy,
        space=checkpoint.space,
        players=checkpoint.players,
        max_trades=checkpoint.max_trades if max_trades is None else max_trades,
        rng=random.Random() if rng is None else rng,
    )


def evaluator_for(
    checkpoint: Checkpoint, *, max_trades: int | None = None
) -> NetworkEvaluator:
    """Expose a checkpoint value head for position evaluation."""
    return NetworkEvaluator(
        policy=checkpoint.policy,
        players=checkpoint.players,
        max_trades=checkpoint.max_trades if max_trades is None else max_trades,
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

    A supplied `rng` seeds both search and a separate trade-gate stream.
    The policy runtime must seed any stochastic inference of its own.
    """
    gate_rng = None if rng is None else random.Random(rng.getrandbits(128))
    budget = checkpoint.max_trades if max_trades is None else max_trades
    return GatedSearch(
        LeafEvaluator(
            policy=checkpoint.policy,
            space=checkpoint.space,
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
        bot = bot_for(loader(_checkpoint_path(entrant.weights, "network"), board.topology),
                      max_trades=entrant.max_trades, rng=rng)
        return bot

    def spawn_mcts(entrant, board: Board, rng) -> GatedSearch:
        return searcher_for(
            loader(_checkpoint_path(entrant.weights, "mcts"), board.topology),
            simulations=entrant.simulations, wave=entrant.wave,
            max_trades=entrant.max_trades, rng=rng,
        )

    register_entrant_kind("network", spawn_network)
    register_entrant_kind("mcts", spawn_mcts)
