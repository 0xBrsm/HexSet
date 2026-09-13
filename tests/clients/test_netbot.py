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

from hexset.actions import ActionSpace, ActionType, apply, build_space
from hexset.board.board import random_base_board
from hexset.clients.netbot import (
    bot_for,
    register_entrants,
    searcher_for,
)
from hexset.game import Phase, run_trade_event, start, to_move
from hexset.actions import options_for
from hexset.clients.modelmeta import DEFAULT_GATE_PLIES, DEFAULT_TRADE_FLOOR
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
    trade_floor: float = DEFAULT_TRADE_FLOOR
    gate_plies: int = DEFAULT_GATE_PLIES


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
    # One at a time agrees with the batch, for every candidate the seat can
    # cover (a candidate it cannot reads `-1.0` in both and is excluded here).
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


def test_the_gate_settings_are_the_checkpoints_own_not_the_adapters(board):
    """A floor measured against one value head says nothing about another's,
    so it rides on the checkpoint (`hexset.clients.modelmeta.gate_config`
    reads it off the file), and `bot_for` must carry it over rather than
    seat every checkpoint alike at this module's default."""
    default = bot_for(stub_checkpoint(board))
    assert default.trade_floor == DEFAULT_TRADE_FLOOR

    from dataclasses import replace

    declared = replace(stub_checkpoint(board), trade_floor=0.0197)
    bot = bot_for(declared)
    assert bot.trade_floor == 0.0197
    # And the search built over the same checkpoint reads the same floor.
    assert searcher_for(declared, simulations=2, wave=2).trade_floor == 0.0197


def test_a_checkpoint_predating_this_key_still_seats_and_trades(board):
    """The key is read by name, not required by inheritance: a loader in
    another repo that has never heard of it still spawns a bot, at the
    behaviour it had before it existed. This is the training repo's own
    checkpoint class, which `bot_for` must keep accepting."""
    @dataclass(frozen=True)
    class OlderCheckpoint:
        policy: HandValuePolicy
        space: ActionSpace
        players: int = PLAYERS
        max_trades: int | None = None

    space = stub_checkpoint(board).space
    bot = bot_for(OlderCheckpoint(policy=HandValuePolicy(space=space), space=space))
    assert bot.trade_floor == DEFAULT_TRADE_FLOOR
    check_gate_is_self_consistent(bot, seated(bot, board))


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
    state = game.state(seat, hidden=False)
    hands = state.hands
    from hexset.clients.netbot import _geometry_for, _kinds_of

    geometry = _geometry_for(state, seat)
    current_kinds = _kinds_of(hands[seat], geometry)
    for i, (them, bundle) in enumerate(candidates):
        if gains[i] == -1.0:
            continue
        # Both hands move: this seat's row is the exchange one way, the
        # counterparty's the other, each priced in that seat's own weights.
        mine = [n + d for n, d in zip(hands[seat], bundle)]
        theirs = [n - d for n, d in zip(hands[them], bundle)]
        if _kinds_of(mine, geometry) == current_kinds:
            # The affordability filter never lets this one reach the head.
            assert gains[i] == 0.0
            assert estimates[i] == 0.0
            continue
        assert gains[i] == pytest.approx(
            value_of_hand(mine, seat) - value_of_hand(hands[seat], seat)
        )
        assert estimates[i] == pytest.approx(
            value_of_hand(theirs, them) - value_of_hand(hands[them], them)
        )
    # Not vacuous: the gate actually engages with at least one candidate --
    # whether the affordability filter prices it at `0.0` or the head reads
    # it for real, rather than refusing everything as uncoverable.
    assert any(g != -1.0 for g in gains)


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
    # exchange exists that both gates price above their floor -- and it is
    # built to survive the affordability filter too: seat 0 is one wheat
    # short of a city (it already holds the ore), seat 1 already affords one
    # and gives up the very wheat that pays for it, so the trade both sides
    # want also changes what each can build.
    state = game.state(0, hidden=False)
    state.hands[0] = [0, 1, 0, 1, 3]
    state.hands[1] = [0, 0, 0, 2, 3]
    state.hands[2] = [0, 0, 0, 0, 0]
    state.hands[3] = [0, 0, 0, 0, 0]
    state.deck = []  # dev cards off the table: only the city kind is in play
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


def test_the_after_position_moves_the_counterpartys_ledger_row_too(board):
    """`_after` copies `game`, not just its state: a bare `copy.copy(game)`
    shares the live ledger, and `View` reads a seat that is not the
    perspective from the *ledger*, not from `state.hands` -- so a copy that
    only rewrote the state answered every ask about the counterparty from
    the position before the trade, with only their hand's *size* having
    moved. That mismatch is exactly what the reachable-purchase gate's
    filter exists to keep away from the head, reintroduced one layer down
    if the ledger is never rewritten to match.

    Called directly rather than through `gains_many`: `_after` only builds
    the *seed* the reachable-purchase gate's plans start from now (this
    same hand also unlocks a reachable city, so a live ask would go on to
    place one), and this test is about `_after` itself, not what is built
    on top of it.
    """
    from hexset.clients.netbot import NetworkBot
    from hexset.ledger import PublicLedger
    from hexset.view import View

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = start(board, PLAYERS, random.Random(2))
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, bot.choose(game))
    state = game.state(0, hidden=False)
    # Same kind-changing exchange as the trade-event test above: seat 0 is
    # one wheat short of a city, seat 1 already affords one and gives up
    # the wheat that pays for it -- guaranteed to survive the gate's filter
    # and reach the head.
    state.hands[0] = [0, 1, 0, 1, 3]
    state.hands[1] = [0, 0, 0, 2, 3]
    state.deck = []
    # Every hand is public knowledge here, so the belief is exact and the
    # expected numbers below are exact too.
    ledger = PublicLedger.new(state.num_players)
    ledger.apply_hand_diff([[0] * 5 for _ in state.hands], state.hands)
    game.ledger = ledger
    game.phase = Phase.MAIN
    game.current_player = 0
    bot.seat_at(game)
    view = game.state(0)
    bundle = (0, -1, 0, 1, 0)  # seat 0 gives a brick, receives a wheat

    after_game = bot._after(0, 1, list(view.known[0]), bundle, view)

    after_view = View.from_game(after_game, 0)
    assert after_view.known[0] == [0, 0, 0, 2, 3]  # seat 0's own hand, exact
    assert after_view.known[1] == [0, 1, 0, 1, 3]  # seat 1's known row moved too
    assert after_view.sizes[1] == view.sizes[1] - sum(bundle) == 5
    assert all(k >= 0 for k in after_view.known[1])  # never below zero
    assert sum(after_view.known[1]) + after_view.unknown[1] == after_view.sizes[1]

    # The live game -- its state and its ledger both -- is untouched.
    assert game.state(0, hidden=False).hands[1] == [0, 0, 0, 2, 3]
    assert game.ledger.seats[1].known == [0, 0, 0, 2, 3]


def test_the_after_position_keeps_the_cards_this_seat_cannot_name(board, monkeypatch):
    """`View.from_game` hands the true state through, so `_after` sees the
    counterparty's concrete hand. It must move that hand by the bundle,
    not replace it with the known row: the size is public, and a
    counterparty shrunk by every card this seat cannot name would price
    poorer on every candidate, trade or no trade, and the two-sided test
    would refuse every offer."""
    from hexset.clients.netbot import NetworkBot
    from hexset.ledger import PublicLedger
    from hexset.view import View

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = start(board, PLAYERS, random.Random(2))
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, bot.choose(game))
    state = game.state(0, hidden=False)
    state.hands[0] = [0, 1, 0, 1, 3]
    state.hands[1] = [0, 0, 0, 2, 3]
    state.deck = []
    ledger = PublicLedger.new(state.num_players)
    ledger.apply_hand_diff([[0] * 5 for _ in state.hands], state.hands)
    # Seat 0 can name only one wheat and one ore of seat 1's five cards.
    ledger.seats[1].known = [0, 0, 0, 1, 1]
    ledger.seats[1].unknown = 3
    game.ledger = ledger
    game.phase = Phase.MAIN
    game.current_player = 0
    bot.seat_at(game)
    view = game.state(0)
    assert view.unknown[1] == 3 and view.sizes[1] == 5
    bundle = (0, -1, 0, 1, 0)  # seat 0 gives a brick, receives a wheat

    captured = []
    original = HandValuePolicy.value_rows

    def capture(self, rows):
        captured.extend(rows)
        return original(self, rows)

    monkeypatch.setattr(HandValuePolicy, "value_rows", capture)

    bot.gains_many(view, [bundle], [1])
    after_game, _ = captured[1]
    after_view = View.from_game(after_game, 0)
    assert after_view.sizes[1] == 5  # one card out, one card in
    assert after_view.known[1] == [0, 1, 0, 0, 1]
    assert after_view.unknown[1] == 3
    assert sum(after_view.known[1]) + after_view.unknown[1] == after_view.sizes[1]
    assert game.ledger.seats[1].known == [0, 0, 0, 1, 1]  # live ledger untouched
    assert game.ledger is not after_game.ledger


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

    # The gate is a pure function of the position and the ask, with no
    # random stream of its own any more, so the searched and plain bots over
    # the same checkpoint answer identically outright.
    assert search.gains_many(view, received, thems) == plain.gains_many(view, received, thems)
    assert search.accepts_many(view, received, thems) == plain.accepts_many(view, received, thems)
    assert search.estimate_many(view, candidates) == plain.estimate_many(view, candidates)



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


def test_a_bot_seated_at_one_seat_refuses_to_answer_for_another(board):
    """`seat` is the guard a driver installs one gate per seat with: a gate
    wired to the wrong seat would answer for somebody else's hand, and that
    is the one failure the mechanic must never have quietly."""
    bot = bot_for(stub_checkpoint(board))
    bot.seat = 1
    game = start(board, PLAYERS, random.Random(2))
    for _ in range(40):
        step_randomly(game, random.Random(2))
    bot.seat_at(game)
    candidates = list(_candidates(game.state(0, hidden=False), 0, frozenset()))
    received = [b for _, b in candidates][:3]
    thems = [c for c, _ in candidates][:3]
    with pytest.raises(ValueError, match="seated at 1"):
        bot.gains_many(game.state(0), received, thems)
    with pytest.raises(ValueError, match="seated at 1"):
        bot.estimate_many(game.state(0), list(zip(thems, received)))
    # Its own seat is answered, and `seat_at` alone -- no `choose` -- is enough
    # to seat the gate.
    own = list(_candidates(game.state(1, hidden=False), 1, frozenset()))
    if own:
        gains = bot.gains_many(game.state(1), [b for _, b in own], [c for c, _ in own])
        assert len(gains) == len(own)


@pytest.fixture
def arena_registry():
    """`hexset.arena`'s registries are process-global, so a test that
    registers into them puts them back."""
    from hexset import arena

    kinds = dict(arena._ENTRANT_KIND_FACTORIES)
    yield
    arena._ENTRANT_KIND_FACTORIES.clear()
    arena._ENTRANT_KIND_FACTORIES.update(kinds)


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
        check_gate_is_self_consistent(bot, game)

        # The valued stub is not a constant head: a hand mutation the
        # affordability filter has no reason to zero out at this particular
        # position still moves the raw value read directly (independent of
        # the trade gate, which may legitimately price every coverable
        # candidate here at zero if none of them change a kind).
        import copy as copy_module

        from hexset.state import copy_state

        seat = to_move(game)
        state = game.state(seat, hidden=False)
        richer = copy_state(state)
        richer.hands[seat] = [n + 3 for n in richer.hands[seat]]
        richer_game = copy_module.copy(game)
        richer_game.set_state(richer)
        before_value = bot.policy.value_rows([(game, seat)])[0]
        after_value = bot.policy.value_rows([(richer_game, seat)])[0]
        assert before_value != after_value, "the valued stub is not a constant head"
    finally:
        _load_cached.cache_clear()


# --- the affordability filter: the gate prices what a trade lets a seat buy ---


def _position_with_a_settlement_in_hand(board):
    """A MAIN position where seat 0 may build a settlement and holds exactly
    the cards for one (1 wood, 1 brick, 1 sheep, 1 wheat); seat 1 holds
    plenty of everything so any counter-bundle is coverable."""
    for seed in range(2, 60):
        game = start(board, PLAYERS, random.Random(seed))
        for _ in range(80):
            if game.phase is Phase.MAIN and to_move(game) == 0 and any(
                a.type is ActionType.BUILD_SETTLEMENT for a in options_for(game)
            ):
                state = game.state(0, hidden=False)
                state.hands[0] = [1, 1, 1, 1, 0]
                state.hands[1] = [4, 4, 4, 4, 4]
                return game
            step_randomly(game, random.Random(seed))
    raise AssertionError("no position with a settlement spot came up")


def test_a_trade_that_changes_no_affordable_kind_prices_at_zero(board, monkeypatch):
    """A trade that neither adds nor removes anything the seat could not
    already buy is priced at exactly `0.0` in both readings, without the
    candidate ever reaching the head. This is the defect the affordability
    filter replaces the continuation rollout with: two raw readings of
    near-identical hands differ by the head's own noise, and noise near
    zero used to read as a real, if tiny, trade."""
    from hexset.clients.netbot import NetworkBot
    from hexset.trading import default_offer

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    state.deck = []  # dev cards off the table: an ore alone cannot unlock a kind
    bot.seat_at(game)
    view = game.state(0)
    unchanged = (0, 0, 0, 0, 1)  # +1 ore: the road and the settlement were and
                                 # stay affordable, the city and a dev card stay out of reach

    calls: list[int] = []
    original = HandValuePolicy.value_rows

    def counted(self, rows):
        calls.append(len(rows))
        return original(self, rows)

    monkeypatch.setattr(HandValuePolicy, "value_rows", counted)

    assert bot.gains_many(view, [unchanged], [1]) == [0.0]
    assert bot.estimate_many(view, [(1, unchanged)]) == [0.0]
    assert not calls, "a kind-preserving candidate must never reach the head"
    assert default_offer(bot, view, [(1, unchanged)]) is None


def test_a_trade_that_changes_an_affordable_kind_reaches_the_head(board, monkeypatch):
    """Making a new kind buyable -- or taking one away -- is a real
    deduction the reachable-purchase gate's filter cannot make on its own,
    so it is the one case that still costs a forward, and the head's own
    reading of the one position each side builds stands.

    Seat 0 holds nothing that could ever afford a `ROAD` or `SETTLEMENT`
    (no wood, brick or sheep at all) and the deck is emptied, so the only
    dimension either hand's reachable set can move on is `CITY` -- one
    wheat short of it before the bundle, exactly affording it after. The
    baseline builds nothing (nothing is reachable) and the candidate builds
    its one city at whichever of seat 0's two settlements has the higher
    pip sum -- one row each, however many legal vertices there were.
    """
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    state.hands[0] = [0, 0, 0, 1, 3]
    state.deck = []
    bot.seat_at(game)
    view = game.state(0)
    unlocks_city = (0, 0, 0, 1, 0)  # +1 wheat: the city is now affordable

    calls: list[int] = []
    original = HandValuePolicy.value_rows

    def counted(self, rows):
        calls.append(len(rows))
        return original(self, rows)

    monkeypatch.setattr(HandValuePolicy, "value_rows", counted)

    gain = bot.gains_many(view, [unlocks_city], [1])[0]
    # One forward, one row per side: the unbuilt baseline and the one built city.
    assert calls == [2]
    assert gain == pytest.approx(
        value_of_hand([0, 0, 0, 0, 0], 0) - value_of_hand([0, 0, 0, 1, 3], 0)
    )


def test_gate_plies_zero_reads_the_survivor_in_one_forward(board, monkeypatch):
    """The shipped default: no continuation, one forward over the best
    position the reachable-purchase gate finds for the baseline and for
    the one survivor -- same fixture as the test above."""
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS, gate_plies=0)
    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    state.hands[0] = [0, 0, 0, 1, 3]
    state.deck = []
    bot.seat_at(game)
    view = game.state(0)
    unlocks_city = (0, 0, 0, 1, 0)  # +1 wheat: the city is now affordable

    value_calls: list[int] = []
    original_value = HandValuePolicy.value_rows

    def counted_value(self, rows):
        value_calls.append(len(rows))
        return original_value(self, rows)

    monkeypatch.setattr(HandValuePolicy, "value_rows", counted_value)

    act_calls: list[int] = []
    original_act = HandValuePolicy.act_rows

    def counted_act(self, rows):
        act_calls.append(len(rows))
        return original_act(self, rows)

    monkeypatch.setattr(HandValuePolicy, "act_rows", counted_act)

    bot.gains_many(view, [unlocks_city], [1])
    assert value_calls == [2]  # the unbuilt baseline, plus the one built city
    assert not act_calls, "gate_plies=0 must never roll a continuation"


def test_gate_plies_rolls_a_continuation_without_touching_the_live_table(board, monkeypatch):
    """A checkpoint asking for `gate_plies > 0` rolls the mover's own greedy
    policy forward from the survivor before it is valued (`_continue`) --
    at least one `act_rows` call -- and does so on copies: the live game's
    state, ledger and chance are exactly what they were before the ask."""
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS, gate_plies=8)
    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    state.hands[0] = [1, 1, 1, 0, 0]
    bot.seat_at(game)
    view = game.state(0)
    unlocks_settlement = (0, 0, 0, 1, 0)  # +1 wheat: the settlement is now affordable

    before_hands = [hand[:] for hand in game.state(0, hidden=False).hands]
    before_known = [list(seat.known) for seat in game.ledger.seats]
    before_unknown = [seat.unknown for seat in game.ledger.seats]
    before_chance = game.chance

    act_calls: list[int] = []
    original_act = HandValuePolicy.act_rows

    def counted_act(self, rows):
        act_calls.append(len(rows))
        return original_act(self, rows)

    monkeypatch.setattr(HandValuePolicy, "act_rows", counted_act)

    bot.gains_many(view, [unlocks_settlement], [1])
    assert act_calls, "gate_plies > 0 must roll at least one ply"

    after_state = game.state(0, hidden=False)
    assert after_state.hands == before_hands
    assert [list(seat.known) for seat in game.ledger.seats] == before_known
    assert [seat.unknown for seat in game.ledger.seats] == before_unknown
    assert game.chance is before_chance


def test_gate_plies_rolls_on_a_belief_world_not_the_live_tables_true_hands(board, monkeypatch):
    """The rollout ACTS on the positions it is handed -- a Knight steal
    mid-turn would resolve against whatever hand the world it runs on
    actually holds -- so `gate_plies > 0` must never roll forward on the
    live table's own true hands, only a world sampled from the asking
    seat's belief (`View.sample`), certified so a survivor's own exchange
    cannot go negative.

    Seat 0 asks the gate as a *responder* while seat 1 (the actor) is to
    move -- exactly the shape the class docstring's `mover` distinction
    describes -- and seat 0's ledger can name only one of seat 1's five
    cards; the bundle's certification only pins the wheat it names, never
    the 3 ore seat 1 truly holds. `View.sample` is stubbed to hand back a
    hand for seat 1 nothing like its true one; every world the rollout
    hands to `act_rows` must carry that stand-in, never the true hand."""
    from hexset.clients.netbot import NetworkBot
    from hexset.ledger import PublicLedger
    from hexset.view import View

    space = stub_checkpoint(board).space
    bot = NetworkBot(
        policy=HandValuePolicy(space), players=PLAYERS, gate_plies=8, rng=random.Random(7)
    )
    # Reuses the position that guarantees a *reachable* settlement spot
    # (`options_for` already found one legal there), same as the
    # affordability-filter tests above.
    game = _position_with_a_settlement_in_hand(board)
    true_hand_1 = [0, 0, 3, 2, 3]  # 3 ore this ledger never certified
    state = game.state(0, hidden=False)
    state.hands[0] = [1, 1, 1, 0, 0]  # one wheat short of a settlement
    state.hands[1] = true_hand_1
    state.deck = []
    ledger = PublicLedger.new(state.num_players)
    ledger.apply_hand_diff([[0] * 5 for _ in state.hands], state.hands)
    ledger.seats[1].known = [0, 0, 0, 1, 1]  # names one wheat, one ore -- never the other two ore
    ledger.seats[1].unknown = 6
    game.ledger = ledger
    game.phase = Phase.MAIN
    game.current_player = 1  # seat 1, the counterparty, is the mover
    bot.seat_at(game)
    view = game.state(0)
    unlocks_settlement = (0, 0, 0, 1, 0)  # +1 wheat: the settlement is now affordable

    original_sample = View.sample

    def fake_sample(self, rng):
        sampled = original_sample(self, rng)
        sampled.hands[1] = [9, 9, 9, 9, 9]  # unlike true_hand_1 in every way
        return sampled

    monkeypatch.setattr(View, "sample", fake_sample)

    act_rows_calls = []
    original_act = HandValuePolicy.act_rows

    def capture_act(self, rows):
        act_rows_calls.extend(rows)
        return original_act(self, rows)

    monkeypatch.setattr(HandValuePolicy, "act_rows", capture_act)

    bot.gains_many(view, [unlocks_settlement], [1])
    assert act_rows_calls, "gate_plies > 0 must roll at least one ply"
    for world, mover, _ in act_rows_calls:
        assert mover == 1
        assert world.state(1, hidden=False).hands[1] != true_hand_1

    # The live game itself was never read into the rollout.
    assert game.state(0, hidden=False).hands[1] == true_hand_1


def test_a_won_position_offers_nothing_it_could_not_already_buy(board):
    """The g4 defect this replaces the rollout for, found 2026-09-08: at a
    won position (the winning settlement already in hand and placeable), a
    raw value-head reading of two near-identical hands differs by the
    head's own noise, so the old gate priced a trade that changed nothing
    about what the seat could do at a small nonzero value. Nothing here
    changes what seat 0 can buy, so nothing reaches the head, and
    `default_offer` confirms nothing is offered."""
    from hexset.clients.netbot import NetworkBot
    from hexset.trading import default_offer

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    state.deck = []  # dev cards off the table for the same reason as above
    bot.seat_at(game)
    view = game.state(0)
    unchanged = (0, 0, 0, 0, 1)  # +1 ore, same reasoning as the kind-preserving case above

    assert bot.gains_many(view, [unchanged], [1]) == [0.0]
    assert bot.estimate_many(view, [(1, unchanged)]) == [0.0]
    assert default_offer(bot, view, [(1, unchanged)]) is None


def test_the_gate_is_a_pure_function_of_the_ask(board):
    """Asked twice about the same candidates at the same position, a gate
    answers exactly the same both times -- there is no random stream of its
    own left to disturb it."""
    from hexset.clients.netbot import NetworkBot
    from hexset.trading import _candidates

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = seated(bot, board)
    seat = to_move(game)
    view = game.state(seat)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset()))[:12]
    received = [b for _, b in candidates]
    thems = [c for c, _ in candidates]
    first = bot.gains_many(view, received, thems)
    assert bot.gains_many(view, received, thems) == first
    assert bot.estimate_many(view, candidates) == bot.estimate_many(view, candidates)


def test_a_repeated_ask_is_answered_without_re_evaluating(board, monkeypatch):
    """`gains_many` and `estimate_many` are two readings of one evaluation.

    `hexset.trading.default_offer` and `default_respond` each ask for both,
    back to back, over the identical candidates at the identical position --
    so the second ask must serve itself from the first's forward rather than
    rebuilding it. The numbers must be exactly what an unmemoised gate
    answers, which is what the second half checks.
    """
    from hexset.clients.netbot import NetworkBot
    from hexset.trading import _candidates

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = seated(bot, board)
    seat = to_move(game)
    view = game.state(seat)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset()))[:12]
    received = [b for _, b in candidates]
    thems = [c for c, _ in candidates]

    calls: list[int] = []
    original = NetworkBot._evaluate

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NetworkBot, "_evaluate", counted)

    gains = bot.gains_many(view, received, thems)
    estimates = bot.estimate_many(view, candidates)
    assert len(calls) == 1  # one evaluation served both readings

    bot._memo = None
    assert bot.gains_many(view, received, thems) == gains
    bot._memo = None
    assert bot.estimate_many(view, candidates) == estimates
    assert len(calls) == 3


def test_the_memo_misses_when_the_board_moves_without_a_hand(board, monkeypatch):
    """Road Building places roads without spending a card, so a key built
    from hands alone would answer the previous table. The board is in the
    key, and a road moves it."""
    from hexset.clients.netbot import NetworkBot
    from hexset.state import road_placeable
    from hexset.trading import _candidates

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = seated(bot, board)
    seat = to_move(game)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset()))[:8]
    received = [b for _, b in candidates]
    thems = [c for c, _ in candidates]

    calls: list[int] = []
    original = NetworkBot._evaluate

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NetworkBot, "_evaluate", counted)

    bot.gains_many(game.state(seat), received, thems)
    assert len(calls) == 1
    bot.gains_many(game.state(seat), received, thems)
    assert len(calls) == 1  # nothing moved: served from the memo

    state = game._state
    edge = next(
        e for e in range(state.board.topology.num_edges) if road_placeable(state, seat, e)
    )
    state.edge_owner[edge] = seat  # a free road, no card spent

    bot.gains_many(game.state(seat), received, thems)
    assert len(calls) == 2  # the board moved: evaluated afresh


def test_a_policy_duels_through_compete_batched(board):
    """`PolicyPolicy` is the whole adapter between a runtime and an evaluation.

    The stub is a `Policy` and nothing more -- no bot, no gate, no arena
    entrant -- and that is all `hexset.bench.versus` needs to seat it on
    both sides of a duel. The seat given the checkpoint as well is gated by
    `bot_for`'s own trade gate, priced on the *run's* budget rather than the
    checkpoint's, which is the reason `gate` takes a third argument.
    """
    from hexset.bench.versus import PolicyPolicy, _budgeted, compete_batched

    checkpoint = stub_checkpoint(board)
    bare = PolicyPolicy(checkpoint.policy)
    gated = PolicyPolicy(checkpoint.policy, checkpoint)

    game = start(board, PLAYERS, random.Random(2))
    assert bare.gate(game, 0, 0) is None
    assert gated.gate(game, 0, 0).max_trades == 0
    # The seater is what actually hands the budget over.
    assert _budgeted(gated.gate, 0)(game, 0).max_trades == 0
    def plain_gate(game, seat):
        return "seated"

    assert _budgeted(plain_gate, 0) is plain_gate  # two-argument: untouched

    verdict = compete_batched(
        {0: gated, 1: bare},
        4,
        players=PLAYERS,
        seed=5,
        lanes=2,
        action_cap=400,
        max_trades=0,
    )
    assert verdict.games == 4
    assert 0 <= verdict.wins <= 4


def test_a_policy_policy_gate_is_seated_where_it_is_installed(board):
    """`PolicyPolicy.gate` hands `compete_batched` a NetworkBot seated at the
    game it will be asked about, at that seat -- unseated, the bot priced
    every candidate at -1.0 and a network duellist silently never traded."""
    from hexset.bench.versus import PolicyPolicy
    from hexset.clients.netbot import NetworkBot

    checkpoint = stub_checkpoint(board)
    game = start(board, PLAYERS, random.Random(2))
    gate = PolicyPolicy(checkpoint.policy, checkpoint).gate(game, 2, 3)
    assert isinstance(gate, NetworkBot)
    assert gate._seated is game and gate.seat == 2 and gate.max_trades == 3


def test_a_survivor_is_exactly_one_row_chosen_by_a_fixed_ordering(board, monkeypatch):
    """The reachable-purchase gate builds exactly one row per side, chosen
    deterministically -- no maximum over every placement of every maximal
    multiset, which is what let a candidate's price drift up with how many
    options it happened to have.

    Seat 0 can afford exactly two roads (no sheep or wheat to spare, so no
    bank trade stretches the hand into anything else); a spare wheat also
    lets it afford one road *and* a settlement jointly, so the survivor's
    reachable set carries two incomparable maximal multisets -- `2 roads`
    and `1 road, 1 settlement`. `_choose_multiset` picks the settlement (a
    victory point) over the second road, and only one `value_rows` call,
    two rows, ever runs -- one per side, not one per placement.
    """
    from hexset.clients.netbot import NetworkBot
    from hexset.clients.reachable import reachable_recipes

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    state.hands[0] = [2, 2, 1, 0, 0]
    state.deck = []
    bot.seat_at(game)
    view = game.state(0)
    unlocks_settlement = (0, 0, 0, 1, 0)  # +1 wheat: a road-and-settlement combo joins the roads

    calls: list[int] = []
    original = HandValuePolicy.value_rows

    def counted(self, rows):
        calls.append(len(rows))
        return original(self, rows)

    monkeypatch.setattr(HandValuePolicy, "value_rows", counted)

    gain = bot.gains_many(view, [unlocks_settlement], [1])[0]
    assert gain != -1.0  # the bundle was covered and reached the gate
    assert calls == [2]  # one forward, one row per side -- no per-placement fan-out

    ctx = bot._reach_context(0, view)
    candidate_hand = (2, 2, 1, 1, 0)
    recipes = reachable_recipes(candidate_hand, **ctx)
    assert (1, 1, 0, 0) in recipes and (2, 0, 0, 0) in recipes
    assert bot._choose_multiset(recipes) == (1, 1, 0, 0)  # the settlement, not the second road


def test_a_road_building_branch_places_two_roads_before_a_settlement(board):
    """A seat holding a playable Road Building card reaches a settlement
    corner its hand alone never could: two free roads placed by the same
    expansion rule a bought road uses (`_far_endpoint_pips`), opening a
    vertex `settlement_placeable` would have refused before them, and only
    then is the settlement paid for and placed.

    `_road_building_context` is run for real against this fixture's board
    (`edges` below are genuinely legal placements for seat 0 here), but
    `ctx["legal"]` is hand-lowered to 0 legal settlement corners: this
    fixture's own guarantee is that *a* spot is already legal, which is
    exactly the ordinary case the merge in `_recipes_with_road_building`
    only adds *new* multisets to -- forcing the "before" count to 0 isolates
    the one case worth a dedicated test, a corner only the free roads
    reach.
    """
    from hexset.cards import DevCard
    from hexset.clients.netbot import NetworkBot
    from hexset.clients.reachable import reachable_recipes
    from hexset.state import road_count

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    # Exactly a settlement's worth of resources, and one matured Road
    # Building card -- no dev card played yet this turn.
    state.hands[0] = [1, 1, 1, 1, 0]
    state.dev_cards[0] = [0, 0, 1, 0, 0]  # one matured Road Building
    state.deck = []
    game.dev_card_played = False
    bot.seat_at(game)
    view = game.state(0)

    ctx = bot._reach_context(0, view)
    rb_context = bot._road_building_context(0, view, ctx)
    assert rb_context is not None
    edges, room, legal = rb_context
    assert 1 <= len(edges) <= 2

    no_settlement_yet = dict(ctx, legal=(ctx["legal"][0], 0, ctx["legal"][2]))
    hand = tuple(view.known[0])
    without_road_building = reachable_recipes(hand, **no_settlement_yet)
    assert not any(m[1] >= 1 for m in without_road_building), "the hand-lowered legal count still let a settlement through"

    recipes = bot._recipes_with_road_building(hand, no_settlement_yet, rb_context)
    settlement_multisets = [m for m, r in recipes.items() if r.branch == "road_building" and m[1] >= 1]
    assert settlement_multisets, "no settlement reached through the Road Building branch"
    recipe = recipes[settlement_multisets[0]]
    assert recipe.param == edges

    seed = bot._seeded(0, hand, view)
    built = bot._build_row(seed, 0, recipe)
    built_state = built.state(0, hidden=False)
    # The two free roads and the bought settlement all actually landed:
    # the card is spent, the turn's one-card rule is set, and the seat owns
    # two more roads and one more settlement than it started with.
    assert built_state.dev_cards[0][DevCard.ROAD_BUILDING] == 0
    assert built.dev_card_played is True
    assert road_count(built_state, 0) == road_count(state, 0) + len(edges) + recipe.purchases[0]
    from hexset.state import settlement_count

    assert settlement_count(built_state, 0) == settlement_count(state, 0) + recipe.purchases[1]


def test_the_monopoly_branch_never_reads_a_true_hand_it_cannot_name(board):
    """Built with `bot._build_row`, directly: seat 1's *true* hand holds far
    more wheat than seat 0's ledger can name, so a Monopoly branch that
    read the true hand would credit seat 0 with all of it. The built
    position must credit only the known amount, leave seat 1's excess
    exactly where it was, and never touch the live game or its ledger at
    all -- `_evaluate_reachable` never mutates what it was handed, only
    copies of it.
    """
    from hexset.clients.netbot import NetworkBot
    from hexset.ledger import PublicLedger

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = start(board, PLAYERS, random.Random(2))
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, bot.choose(game))
    state = game.state(0, hidden=False)
    # One wheat short of a city; seat 1's true hand holds five wheat, but
    # seat 0's ledger can only name one of them.
    state.hands[0] = [0, 0, 0, 1, 3]
    state.hands[1] = [0, 0, 0, 5, 0]
    state.hands[2] = [0, 0, 0, 0, 0]
    state.hands[3] = [0, 0, 0, 0, 0]
    state.dev_cards[0] = [0, 0, 0, 0, 1]  # one matured Monopoly
    state.deck = []
    ledger = PublicLedger.new(state.num_players)
    ledger.apply_hand_diff([[0] * 5 for _ in state.hands], state.hands)
    ledger.seats[1].known = [0, 0, 0, 1, 0]  # only one of the five wheat named
    ledger.seats[1].unknown = 4
    game.ledger = ledger
    game.phase = Phase.MAIN
    game.current_player = 0
    game.dev_card_played = False
    bot.seat_at(game)
    view = game.state(0)
    true_hand_1 = state.hands[1][:]
    true_ledger_1_known = list(ledger.seats[1].known)

    from hexset.clients.reachable import reachable_recipes

    hand = tuple(view.known[0])
    ctx = bot._reach_context(0, view)
    recipes = reachable_recipes(hand, **ctx)
    recipe = recipes[(0, 0, 1, 0)]
    assert recipe.branch == "monopoly"
    # Credited with the one known wheat, not the four seat 1 truly has
    # beyond it.
    assert recipe.hand == (0, 0, 0, 2, 3)

    seed = bot._seeded(0, hand, view)
    built = bot._build_row(seed, 0, recipe)
    built_state = built.state(0, hidden=False)
    assert built_state.hands[0] == [0, 0, 0, 0, 0]  # paid the city off in full
    # Seat 1 loses exactly the known wheat -- five minus one -- never more.
    assert built_state.hands[1] == [0, 0, 0, 4, 0]

    # The live game -- its state and its ledger both -- is untouched.
    assert game.state(0, hidden=False).hands[1] == true_hand_1
    assert game.ledger.seats[1].known == true_ledger_1_known
