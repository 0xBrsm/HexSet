"""The checkpoint runtime, driven by a policy that is not a checkpoint.

`hexset.clients.netbot` is the bot, the leaf evaluation, the search and the
trade gate; `hexset.clients.policy.Policy` is the whole of what a runtime has
to provide for those to work. These tests hold the second claim to account:
a stub `Policy` of twenty lines -- no onnxruntime, no torch, no model file --
drives `choose`, `gains_many`, `estimate_many`, a real `trade_event` and a
`GatedSearch` end to end, and the ONNX policy is put through the same
runtime-neutral checks at the bottom.

The stub reads the true hands (`game.state(seat, hidden=False)`) rather than
an information set. A real runtime never does, and never could -- it sees
what its encoder was given -- but a closed-form value is what makes the gate's
arithmetic checkable here, and the point under test is the seam, not the
honesty of the numbers behind it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
from conftest import step_randomly

from hexset.actions import ActionSpace, apply, build_space
from hexset.board.board import random_base_board
from hexset.clients.netbot import (
    bot_for,
    evaluator_for,
    register_entrants,
    searcher_for,
)
from hexset.game import Phase, run_trade_event, start, to_move
from hexset.server.rules import options_for
from hexset.trading import _candidates, valued_many

# Five distinct weights, as in `fixtures/build_stub.py --valued`, but read
# through a per-seat rotation: seat *s* prices resource *r* at
# `WEIGHTS[(r + s) % 5]`. A single set of weights makes every exchange
# exactly zero-sum, so no trade could ever clear both gates and there would
# be no trade event to test; rotating them gives the seats different tastes,
# which is the whole reason a table trades.
WEIGHTS = (0.006, -0.011, 0.004, 0.013, -0.008)

PLAYERS = 4


@dataclass
class HandValuePolicy:
    """A `Policy` with no network behind it: value is a fixed linear read of
    each seat's hand, the prior is uniform over the row's options, and the
    action is the lowest-indexed legal one.

    Deterministic and in closed form, so a test can say what the gate must
    answer rather than only that it answered something.
    """

    space: ActionSpace

    def act_rows(self, rows):
        return [min(options, key=self.space.index) for _, _, options in rows]

    def value_rows(self, rows):
        return [self._value(game) for game, _ in rows]

    def score_rows(self, rows):
        return [
            ([1.0 / len(options)] * len(options), self._value(game))
            for game, _, options in rows
        ]

    def _value(self, game):
        """One number per seat, board-seat order -- what a value head emits."""
        hands = game.state(0, hidden=False).hands
        return tuple(value_of_hand(hand, seat) for seat, hand in enumerate(hands))


def value_of_hand(hand, seat: int) -> float:
    return sum(WEIGHTS[(r + seat) % len(WEIGHTS)] * n for r, n in enumerate(hand))


@dataclass(frozen=True)
class StubCheckpoint:
    """`hexset.clients.policy.Checkpoint` over `HandValuePolicy`."""

    policy: HandValuePolicy
    space: ActionSpace
    players: int = PLAYERS
    max_trades: int | None = None


def stub_checkpoint(board) -> StubCheckpoint:
    space = build_space(
        board.topology.num_vertices,
        board.topology.num_edges,
        board.topology.num_hexes,
        PLAYERS,
    )
    return StubCheckpoint(policy=HandValuePolicy(space=space), space=space)


@pytest.fixture
def board():
    return random_base_board(random.Random(0))


def seated(bot, board, steps: int = 40):
    """A game stepped into play with `bot` having chosen at least once --
    every gate answers about the position `choose` last handed it, so a bot
    nobody has asked to move refuses everything by construction."""
    game = start(board, PLAYERS, random.Random(2))
    for _ in range(steps):
        step_randomly(game, random.Random(2))
    bot.choose(game)
    return game


def check_gate_is_self_consistent(bot, game):
    """What every runtime's gate must satisfy, whatever its numbers are.

    Run against the stub below and against the ONNX policy at the bottom of
    this file: the two share no code but this module and the module under
    test.
    """
    seat = to_move(game)
    view = game.state(seat)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset()))
    received = [b for _, b in candidates]
    thems = [c for c, _ in candidates]

    gains = bot.gains_many(view, received, thems)
    assert len(gains) == len(candidates)
    # `valued_many` is what the clearing house and the trade round read.
    assert valued_many(bot, view, received, thems) == gains
    assert bot.accepts_many(view, received, thems) == [g > 0.0 for g in gains]
    # One at a time agrees with the batch, for a handful of candidates the
    # batch actually scored (one past `NETWORK_GATE_ROWS` reads `-1.0` in the
    # batch and is scored for real when asked alone, which is the cap's
    # documented behaviour, not a disagreement).
    spot = [(i, r, c) for i, (r, c) in enumerate(zip(received, thems)) if gains[i] != -1.0]
    assert spot, "no candidate was scored at all"
    assert [bot.accepts(view, r, c) for _, r, c in spot[:8]] == [
        gains[i] > 0.0 for i, _, _ in spot[:8]
    ]
    assert len(bot.estimate_many(view, candidates)) == len(candidates)

    # A bundle this seat cannot cover is never scored, and the live game is
    # left exactly as `choose` left it.
    hand = list(view.known[seat])
    short = tuple(-(hand[r] + 1) if r == 0 else (1 if r == 1 else 0) for r in range(len(hand)))
    assert bot.gains_many(view, [short], [thems[0]]) == [-1.0]
    assert game.state(seat, hidden=False).hands[seat] == hand
    return gains


def test_a_runtime_free_policy_drives_the_bot_and_its_gate(board):
    """No checkpoint anywhere: `choose` plays the policy's action, and the
    gate's two readings are the value head's own arithmetic on the position
    a real clearing would leave -- this seat's row for `gains_many`, the
    counterparty's for `estimate_many`."""
    bot = bot_for(stub_checkpoint(board))
    game = seated(bot, board)
    seat = to_move(game)

    assert bot.choose(game) in options_for(game)
    gains = check_gate_is_self_consistent(bot, game)

    view = game.state(seat)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset()))
    estimates = bot.estimate_many(view, candidates)
    hands = game.state(seat, hidden=False).hands
    for i, (them, bundle) in enumerate(candidates):
        if gains[i] == -1.0:
            continue
        # Both hands move: this seat's row is the exchange one way, the
        # counterparty's the other, each priced in that seat's own weights.
        mine = [n + d for n, d in zip(hands[seat], bundle)]
        theirs = [n - d for n, d in zip(hands[them], bundle)]
        assert gains[i] == pytest.approx(
            value_of_hand(mine, seat) - value_of_hand(hands[seat], seat)
        )
        assert estimates[i] == pytest.approx(
            value_of_hand(theirs, them) - value_of_hand(hands[them], them)
        )
    # Not vacuous: the gate says yes to something.
    assert any(g > 0.0 for g in gains)


def test_a_trade_event_clears_through_the_runtime_free_gate(board):
    """The engine's own trade event, with a stub-policy bot on every seat:
    `hexset.trading` asks the gates, both sides gain, and the cards move."""
    checkpoint = stub_checkpoint(board)
    bots = [bot_for(checkpoint) for _ in range(PLAYERS)]
    game = start(board, PLAYERS, random.Random(2))
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        seat = to_move(game)
        apply(game, bots[seat].choose(game))

    # Seat 0 holds only what seat 1 prices highly and vice versa, so an
    # exchange exists that both gates price above their floor.
    state = game.state(0, hidden=False)
    state.hands[0] = [0, 0, 0, 0, 3]
    state.hands[1] = [0, 0, 0, 3, 0]
    state.hands[2] = [0, 0, 0, 0, 0]
    state.hands[3] = [0, 0, 0, 0, 0]
    game.phase = Phase.MAIN
    game.current_player = 0
    game.trade_event_turn = -1
    # Every seat's bot chose through setup above, so every gate is seated at
    # this game already -- exactly as a driver leaves them.
    game.gates = tuple(bots)
    before = [hand[:] for hand in state.hands]

    run_trade_event(game)

    assert game.trades, "no exchange cleared both stub gates"
    for trade in game.trades:
        assert trade.gain_a > 0.0 and trade.gain_b > 0.0
    assert state.hands[0] != before[0]
    assert value_of_hand(state.hands[0], 0) > value_of_hand(before[0], 0)


def test_a_searched_runtime_free_policy_plays_and_gates_like_the_plain_bot(board):
    """`GatedSearch` is `hexset.mcts.Search` over the same policy with the
    plain bot's gate: it plays legally through the leaf evaluation, and its
    four gate methods answer exactly what the plain bot answers."""
    checkpoint = stub_checkpoint(board)
    search = searcher_for(checkpoint, simulations=8, wave=4, rng=random.Random(0))
    plain = bot_for(checkpoint)
    game = start(board, PLAYERS, random.Random(2))
    for _ in range(20):
        assert search.choose(game) in options_for(game)
        step_randomly(game, random.Random(2))

    for _ in range(20):
        step_randomly(game, random.Random(2))
    seat = to_move(game)
    search.choose(game)
    plain.choose(game)
    view = game.state(seat)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset()))
    received = [b for _, b in candidates]
    thems = [c for c, _ in candidates]

    assert search.trade_floor == plain.trade_floor == 0.0
    assert search.gains_many(view, received, thems) == plain.gains_many(view, received, thems)
    assert search.accepts_many(view, received, thems) == plain.accepts_many(
        view, received, thems
    )
    assert search.estimate_many(view, candidates) == plain.estimate_many(view, candidates)

    # The same policy as a leaf evaluation for the handcrafted search, one
    # vector per seat in board-seat order.
    vector = evaluator_for(checkpoint).evaluate_game(game, seat)
    assert len(vector) == PLAYERS


def test_a_runtime_registers_the_arena_entrant_kinds_in_one_call(board, arena_registry):
    """`register_entrants(loader)` is the whole of what a runtime does to
    make "network"/"mcts" entrants spawnable: no runtime carries its own
    spawn functions any more."""
    from hexset.arena import Entrant, spawn
    from hexset.clients.netbot import GatedSearch, NetworkBot

    checkpoint = stub_checkpoint(board)
    register_entrants(lambda path, topology: checkpoint)

    plain = spawn(Entrant("stub", kind="network", weights="a-path"), board, random.Random(0))
    searched = spawn(
        Entrant("stub", kind="mcts", weights="a-path", simulations=4, wave=2),
        board,
        random.Random(0),
    )
    assert isinstance(plain, NetworkBot)
    assert isinstance(searched, GatedSearch)

    game = start(board, PLAYERS, random.Random(2))
    assert plain.choose(game) in options_for(game)
    assert searched.choose(game) in options_for(game)

    with pytest.raises(ValueError, match="checkpoint path"):
        spawn(Entrant("stub", kind="network", weights=[1.0]), board, random.Random(0))


@pytest.fixture
def arena_registry():
    """`hexset.arena`'s registries are process-global, so a test that
    registers into them puts them back."""
    from hexset import arena

    kinds = dict(arena._ENTRANT_KIND_FACTORIES)
    evaluators = dict(arena._EVALUATOR_PROVIDERS)
    loader = arena._CHECKPOINT_LOADER
    leaf = arena._LEAF_EVALUATOR_FACTORY
    yield
    arena._ENTRANT_KIND_FACTORIES.clear()
    arena._ENTRANT_KIND_FACTORIES.update(kinds)
    arena._EVALUATOR_PROVIDERS.clear()
    arena._EVALUATOR_PROVIDERS.update(evaluators)
    arena._CHECKPOINT_LOADER = loader
    arena._LEAF_EVALUATOR_FACTORY = leaf


def test_the_onnx_policy_satisfies_the_same_protocol():
    """The point of the protocol: `V2Policy` is a `Policy` and nothing else,
    and the same `netbot` bot it is handed to passes the same gate checks the
    stub policy passes above. Nothing here mentions onnxruntime beyond
    loading the file."""
    pytest.importorskip("onnxruntime", reason="the ONNX policy needs onnxruntime")
    from pathlib import Path

    from hexset.clients.netbot import bot_for as netbot_bot_for
    from hexset.clients.onnxbot import _load_cached, load

    fixture = Path(__file__).parent / "fixtures" / "stub-contract6-valued.onnx"
    onnx_board = random_base_board(random.Random(0))
    try:
        checkpoint = load(str(fixture), onnx_board.topology)
        bot = netbot_bot_for(checkpoint)
        game = seated(bot, onnx_board)
        assert bot.choose(game) in options_for(game)
        gains = check_gate_is_self_consistent(bot, game)
        assert any(g != 0.0 for g in gains), "the valued stub is not a constant head"
    finally:
        _load_cached.cache_clear()
