"""The checkpoint runtime, driven by a policy that is not a checkpoint: a stub
`Policy` drives `choose`, the gate, a real `trade_event` and a `GatedSearch`
end to end. The stub values each seat by the cards the row's own seat can
name (`View.known`), so the gate's arithmetic is checkable in closed form
and the stub reads nothing a real runtime could not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

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
from hexset.trading import ENUMERATION_CARDS, UNLIMITED, TradeParams, valued_many
from hexset.trading._engine import _candidates
from hexset.view import View
from traders import FRAGMENTED
from helpers import give

# Read through a per-seat rotation: seat *s* prices resource *r* at
# `WEIGHTS[(r + s) % 5]`. One shared set makes every exchange zero-sum,
# so no trade could clear both gates; rotating gives the seats tastes.
WEIGHTS = (0.006, -0.011, 0.004, 0.013, -0.008)

PLAYERS = 4


@dataclass
class HandValuePolicy:
    """Value is a fixed linear read of the cards the row's seat can name in
    each seat's hand -- its own whole hand, another's `View.known` row -- the
    prior uniform, the action the lowest-indexed legal one.
    """

    space: ActionSpace

    def act_rows(self, rows):
        return [min(options, key=self.space.index) for _, _, options in rows]

    def value_rows(self, rows):
        return [self._value(game, seat) for game, seat in rows]

    def score_rows(self, rows):
        return [
            ([1.0 / len(options)] * len(options), self._value(game, seat))
            for game, seat, options in rows
        ]

    def _value(self, game, seat):
        known = View.from_game(game, seat).known
        return tuple(value_of_hand(hand, s) for s, hand in enumerate(known))


def value_of_hand(hand, seat: int) -> float:
    return sum(WEIGHTS[(r + seat) % len(WEIGHTS)] * n for r, n in enumerate(hand))


def their_estimate(view, seat: int, them: int, bundle) -> float:
    """What the gate's `estimate_many` is in closed form under this stub:
    `them`'s nameable cards after the exchange -- the known row raised to
    cover what the bundle says they give, then moved by it -- valued less
    the row before."""
    certified = View(view.state, view.ledger, seat, certify=[(them, [max(0, n) for n in bundle])])
    after = [k - d for k, d in zip(certified.known[them], bundle)]
    return value_of_hand(after, them) - value_of_hand(view.known[them], them)


@dataclass(frozen=True)
class StubCheckpoint:

    policy: HandValuePolicy
    space: ActionSpace
    players: int = PLAYERS
    trade_floor: float = UNLIMITED.trade_floor
    gate_plies: int = UNLIMITED.gate_plies
    # A whole parameter set, for a checkpoint that asks to bargain by more
    # than a floor. `None` leaves the two loose attributes above to answer.
    trade_params: TradeParams | None = None


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
    game = start(board, PLAYERS, random.Random(2))
    for _ in range(steps):
        step_randomly(game, random.Random(2))
    bot.choose(game)
    return game


def check_gate_is_self_consistent(bot, game):
    seat = to_move(game)
    view = game.state(seat)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset(), ENUMERATION_CARDS))
    received = [b for _, b in candidates]
    thems = [c for c, _ in candidates]

    gains = bot.gains_many(view, received, thems)
    assert len(gains) == len(candidates)
    assert valued_many(bot, view, received, thems) == gains
    assert bot.accepts_many(view, received, thems) == [g > 0.0 for g in gains]
    spot = [(i, r, c) for i, (r, c) in enumerate(zip(received, thems)) if gains[i] != -1.0]
    assert spot, "no candidate was scored at all"
    assert [bot.accepts(view, r, c) for _, r, c in spot[:8]] == [
        gains[i] > 0.0 for i, _, _ in spot[:8]
    ]
    assert len(bot.estimate_many(view, candidates)) == len(candidates)

    hand = list(view.known[seat])
    short = tuple(-(hand[r] + 1) if r == 0 else (1 if r == 1 else 0) for r in range(len(hand)))
    assert bot.gains_many(view, [short], [thems[0]]) == [-1.0]
    assert game.state(seat, hidden=False).hands[seat] == hand
    return gains


def test_a_checkpoint_that_says_nothing_installs_no_protocol_of_its_own(board):
    """The adapter gained a negotiation protocol; a checkpoint that does not
    ask for one is still answered by the engine's own defaults."""
    bot = bot_for(stub_checkpoint(board))
    assert bot.trade == UNLIMITED
    for hook in ("candidates", "offer", "respond", "consent_gain"):
        assert getattr(bot, hook, None) is None, hook
    # No declared budget is no budget: it offers while it has an offer left.
    assert bot.trade_offer_budget == -1


def test_a_searching_entrant_bargains_exactly_like_its_one_forward_twin(board):
    """`GatedSearch` is what a table seats, so it carries the checkpoint's
    parameters rather than leaving them behind on the `NetworkBot`."""
    declared = replace(stub_checkpoint(board), trade_params=FRAGMENTED)
    searcher = searcher_for(declared, simulations=2, wave=2)
    assert searcher.trade == bot_for(declared).trade
    assert searcher.trade_offer_budget == 2
    assert searcher.candidates is not None


def test_a_checkpoints_card_cap_binds_its_value_head(board):
    """The cap is a refusal, not a preference: it is applied before the head
    is consulted and whoever is asking."""
    declared = replace(stub_checkpoint(board), trade_params=FRAGMENTED)
    bot = bot_for(declared)
    game = start(board, PLAYERS, random.Random(2))
    game.phase = Phase.MAIN
    state = game.state(0, hidden=False)  # true state: stocking a fixed hand
    for _ in range(3):
        give(state, 0, 0)
    give(state, 1, 1)
    bot.seat_at(game)
    view = game.state(0)
    two_out = (-2, 1, 0, 0, 0)
    three_out = (-3, 1, 0, 0, 0)
    assert bot.consent_gain(view, two_out, 1, role="actor") != -1.0
    assert bot.consent_gain(view, three_out, 1, role="actor") == -1.0
    assert not bot.accepts(view, three_out, 1)


def test_a_runtime_free_policy_drives_the_bot_and_its_gate(board):
    bot = bot_for(stub_checkpoint(board))
    game = seated(bot, board)
    seat = to_move(game)

    assert bot.choose(game) in options_for(game)
    gains = check_gate_is_self_consistent(bot, game)

    view = game.state(seat)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset(), ENUMERATION_CARDS))
    estimates = bot.estimate_many(view, candidates)
    hand = view.known[seat]
    for i, (them, bundle) in enumerate(candidates):
        if gains[i] == -1.0:
            continue
        mine = [n + d for n, d in zip(hand, bundle)]
        assert gains[i] == pytest.approx(
            value_of_hand(mine, seat) - value_of_hand(hand, seat)
        )
        assert estimates[i] == pytest.approx(their_estimate(view, seat, them, bundle))
    assert any(g != -1.0 for g in gains)


def test_a_trade_event_clears_through_the_runtime_free_gate(board):
    checkpoint = stub_checkpoint(board)
    bots = [bot_for(checkpoint) for _ in range(PLAYERS)]
    game = start(board, PLAYERS, random.Random(2))
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        seat = to_move(game)
        apply(game, bots[seat].choose(game))

    # Seat 0 is one wheat short of a city; seat 1 affords one and gives up
    state = game.state(0, hidden=False)
    state.hands[0] = [0, 1, 0, 1, 3]
    state.hands[1] = [0, 0, 0, 2, 3]
    state.hands[2] = [0, 0, 0, 0, 0]
    state.hands[3] = [0, 0, 0, 0, 0]
    state.deck = []  # dev cards off the table: only the city kind is in play
    game.phase = Phase.MAIN
    game.current_player = 0
    game.trade_event_turn = -1
    game.gates = tuple(bots)
    before = [hand[:] for hand in state.hands]

    run_trade_event(game)

    assert game.trades, "no exchange cleared both stub gates"
    for trade in game.trades:
        assert trade.gain_a > 0.0 and trade.gain_b > 0.0
    assert state.hands[0] != before[0]
    assert value_of_hand(state.hands[0], 0) > value_of_hand(before[0], 0)


def test_the_after_position_keeps_the_cards_this_seat_cannot_name(board, monkeypatch):
    """`_after` moves the counterparty's hand by the bundle rather than replacing
    it with the ledger's known row: shrunk to what this seat can name, it
    would price poorer on every candidate.
    """
    from hexset.clients.netbot import NetworkBot
    from hexset.ledger import PublicLedger, SeatLedger

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
    # Seat 0 can name only one wheat and one ore of seat 1's five cards,
    # and bounds none of them.
    ledger.seats[1] = SeatLedger(known=[0, 0, 0, 1, 1], unknown=3)
    game.ledger = ledger
    game.phase = Phase.MAIN
    game.current_player = 0
    bot.seat_at(game)
    view = game.state(0)
    assert view.unknown[1] == 3 and view.sizes[1] == 5
    bundle = (0, -1, 0, 1, 0)

    captured = []
    original = HandValuePolicy.value_rows

    def capture(self, rows):
        captured.extend(rows)
        return original(self, rows)

    monkeypatch.setattr(HandValuePolicy, "value_rows", capture)

    bot.gains_many(view, [bundle], [1])
    after_game, _ = captured[1]
    after_view = View.from_game(after_game, 0)
    assert after_view.sizes[1] == 5
    assert after_view.known[1] == [0, 1, 0, 0, 1]
    assert after_view.unknown[1] == 3
    assert sum(after_view.known[1]) + after_view.unknown[1] == after_view.sizes[1]
    assert game.ledger.seats[1].known == [0, 0, 0, 1, 1]  # live ledger untouched
    assert game.ledger is not after_game.ledger


def test_a_searched_runtime_free_policy_plays_and_gates_like_the_plain_bot(board):
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
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset(), ENUMERATION_CARDS))
    received = [b for _, b in candidates]
    thems = [c for c, _ in candidates]

    assert search.trade_floor == plain.trade_floor == 0.0

    assert search.gains_many(view, received, thems) == plain.gains_many(view, received, thems)
    assert search.accepts_many(view, received, thems) == plain.accepts_many(view, received, thems)
    assert search.estimate_many(view, candidates) == plain.estimate_many(view, candidates)


def test_a_runtime_registers_the_arena_entrant_kinds_in_one_call(board, arena_registry):
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
    bot = bot_for(stub_checkpoint(board))
    bot.seat = 1
    game = start(board, PLAYERS, random.Random(2))
    for _ in range(40):
        step_randomly(game, random.Random(2))
    bot.seat_at(game)
    candidates = list(_candidates(game.state(0, hidden=False), 0, frozenset(), ENUMERATION_CARDS))
    received = [b for _, b in candidates][:3]
    thems = [c for c, _ in candidates][:3]
    with pytest.raises(ValueError, match="seated at 1"):
        bot.gains_many(game.state(0), received, thems)
    with pytest.raises(ValueError, match="seated at 1"):
        bot.estimate_many(game.state(0), list(zip(thems, received)))
    own = list(_candidates(game.state(1, hidden=False), 1, frozenset(), ENUMERATION_CARDS))
    if own:
        gains = bot.gains_many(game.state(1), [b for _, b in own], [c for c, _ in own])
        assert len(gains) == len(own)


def test_a_checkpoint_that_names_a_trader_is_seated_with_it_unless_the_entrant_names_one(
    board, arena_registry
):
    """The trader a checkpoint declares answers its trades wherever a lineup
    seats it; an entrant's own `~trader` wins, and the seat is wrapped once."""
    from hexset.arena import entrant_from_name, spawn
    from hexset.bots import RandomBot, TradesBy
    from hexset.clients.netbot import NetworkBot
    from traders import ScarcityTrader

    checkpoint = _Traded(stub_checkpoint(board), "test-trader")
    register_entrants(lambda path, topology: checkpoint)

    seat = spawn(entrant_from_name("network:/tmp/x.onnx"), board, random.Random(1))
    assert isinstance(seat, TradesBy)
    assert isinstance(seat.mover, NetworkBot) and isinstance(seat.trader, ScarcityTrader)

    named = spawn(entrant_from_name("network:/tmp/x.onnx~random"), board, random.Random(1))
    assert isinstance(named, TradesBy) and isinstance(named.trader, RandomBot)
    assert isinstance(named.mover, NetworkBot), "one wrap: the entrant's trader replaces the file's"


def test_an_entrant_whose_checkpoint_names_an_unbuildable_trader_fails_at_spawn(
    board, arena_registry
):
    from hexset.arena import entrant_from_name, spawn

    checkpoint = _Traded(stub_checkpoint(board), "no-such-bot")
    register_entrants(lambda path, topology: checkpoint)

    for spec in ("network:/tmp/x.onnx", "mcts:/tmp/x.onnx"):
        with pytest.raises(ValueError) as caught:
            spawn(entrant_from_name(spec), board, random.Random(1))
        message = str(caught.value)
        assert "/tmp/x.onnx" in message and "`trader`" in message and "--runtime" in message
    # The entrant's own trader replaces the file's, so nothing is resolved.
    assert spawn(entrant_from_name("network:/tmp/x.onnx~random"), board, random.Random(1))


def test_a_rollout_seat_journals_every_decision_it_made(board, arena_registry):
    """The game's journal line carries the decisions the rollout seat logged,
    found under the trader it is dealt with, and reads back unchanged."""
    import json

    from hexset.arena import (
        MAX_ACTIONS, _journal_entry, _outcome_from_entry, _play_one, entrant_from_name,
    )
    from hexset.rules import STANDARD_GAME

    register_entrants(lambda path, topology: _Traded(stub_checkpoint(board), "random"))
    lineup = tuple(entrant_from_name(name) for name in (
        "rollout:/tmp/x.onnx@2:top=2:horizon=1", "random", "random", "random",
    ))
    outcome = _play_one((lineup, 0, 7, MAX_ACTIONS, False, False, "round", STANDARD_GAME, 3))
    log, *others = outcome.searches
    assert log and not any(others)
    searched = [d for d in log if d["worlds"]]
    assert searched and all(
        len(d["outcomes"]) == 2 and all(len(row) == len(d["candidates"]) for row in d["outcomes"])
        for d in searched
    )
    entry = json.loads(json.dumps(_journal_entry(outcome)))
    assert _outcome_from_entry(entry).searches == outcome.searches


def test_search_logs_finds_the_searchers_own_log_through_the_seats_wrappers(board, arena_registry):
    """`hexset.arena.search_logs` hands back the rollout searcher's own
    `decision_log`, through the trader the seat is dealt with, so a decision
    made after the call is in it."""
    from hexset.arena import entrant_from_name, search_logs, spawn
    from hexset.clients.netbot import GatedRollouts

    register_entrants(lambda path, topology: stub_checkpoint(board))
    seat = spawn(entrant_from_name("rollout:/tmp/x.onnx@2:top=2:horizon=1~random"), board, random.Random(3))
    assert not isinstance(seat, GatedRollouts)   # dealt inside its trader
    logs = search_logs(seat)
    assert len(logs) == 1 and logs[0] == []
    seated(seat, board)
    assert logs[0] and logs[0] is search_logs(seat)[0]
    assert search_logs(bot_for(stub_checkpoint(board))) == []


@dataclass(frozen=True)
class _Traded:
    """A stub checkpoint that also names a trader."""

    base: StubCheckpoint
    trader: str

    def __getattr__(self, name):
        return getattr(self.base, name)


@pytest.fixture
def arena_registry():
    """`hexset.arena`'s registries are process-global."""
    from hexset import arena

    yield
    arena.unregister_entrant_kind("network")
    arena.unregister_entrant_kind("mcts")
    arena.unregister_entrant_kind("rollout")


# --- the gate: one forward over every coverable hand ---


def _position_with_a_settlement_in_hand(board):
    """Seat 0 holds exactly the cards for a settlement; seat 1 holds plenty, so
    any counter-bundle is coverable.
    """
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


def test_every_coverable_candidate_reaches_the_head_in_one_forward(board, monkeypatch):
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    state.hands[0] = [1, 1, 1, 0, 0]  # one wheat short of the settlement
    state.deck = []
    bot.seat_at(game)
    view = game.state(0)
    unchanged = (0, 0, 0, 0, 1)  # +1 ore: still one wheat short of anything
    unlocks_settlement = (0, 0, 0, 1, 0)  # +1 wheat: the settlement is now affordable

    calls: list[int] = []
    original = HandValuePolicy.value_rows

    def counted(self, rows):
        calls.append(len(rows))
        return original(self, rows)

    monkeypatch.setattr(HandValuePolicy, "value_rows", counted)

    gains = bot.gains_many(view, [unchanged, unlocks_settlement], [1, 1])
    assert calls == [3]
    assert gains[0] == pytest.approx(
        value_of_hand([1, 1, 1, 0, 1], 0) - value_of_hand([1, 1, 1, 0, 0], 0)
    )
    assert gains[1] == pytest.approx(
        value_of_hand([1, 1, 1, 1, 0], 0) - value_of_hand([1, 1, 1, 0, 0], 0)
    )
    assert bot.estimate_many(view, [(1, unchanged), (1, unlocks_settlement)]) == pytest.approx(
        [their_estimate(view, 0, 1, unchanged), their_estimate(view, 0, 1, unlocks_settlement)]
    )
    assert calls == [3], "the paired estimate is answered from the memo, not a second forward"


def test_an_ask_the_counterparty_cannot_cover_is_never_valued(board, monkeypatch):
    """The offer menu asks for any cards, so it can name a seat whose public
    hand cannot hold them. There is no position after that exchange: the
    gate prices it `-1.0` without building one, and still values the rest.
    Asked with a rollout horizon, so the refusal comes before any rollout."""
    from hexset.clients.netbot import NetworkBot
    from hexset.ledger import SeatLedger

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS,
                     trade=TradeParams(gate_plies=2))
    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    state.hands[0] = [1, 1, 1, 0, 0]
    state.hands[2] = [0, 0, 0, 0, 0]
    game.ledger.seats[2] = SeatLedger()
    bot.seat_at(game)
    view = game.state(0)
    wheat_for_wood = (-1, 0, 0, 1, 0)
    assert view.sizes[2] == 0

    rows: list[int] = []
    original = HandValuePolicy.value_rows

    def counted(self, batch):
        rows.append(len(batch))
        return original(self, batch)

    monkeypatch.setattr(HandValuePolicy, "value_rows", counted)

    assert bot.gains_many(view, [wheat_for_wood], [2]) == [-1.0]
    assert bot.estimate_many(view, [(2, wheat_for_wood)]) == [-1.0]
    assert not rows, "nothing to value, so nothing reaches the head"
    gains = bot.gains_many(view, [wheat_for_wood, wheat_for_wood], [2, 1])
    assert gains[0] == -1.0 and gains[1] != -1.0


def test_gate_plies_rolls_a_continuation_without_touching_the_live_table(board, monkeypatch):
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS, trade=TradeParams(gate_plies=8))
    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    state.hands[0] = [1, 1, 1, 0, 0]
    bot.seat_at(game)
    view = game.state(0)
    unlocks_settlement = (0, 0, 0, 1, 0)

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
    """The rollout ACTS on the positions it is handed -- a Knight steal resolves
    against whatever hand that world holds -- so it must roll only on a world
    sampled from the asking seat's belief. `View.sample` is stubbed to hand
    back a seat-1 hand nothing like the true one.
    """
    from hexset.clients.netbot import NetworkBot
    from hexset.ledger import PublicLedger, SeatLedger
    from hexset.state import BANK_PER_RESOURCE

    space = stub_checkpoint(board).space
    bot = NetworkBot(
        policy=HandValuePolicy(space), players=PLAYERS,
        trade=TradeParams(gate_plies=8), rng=random.Random(7)
    )
    game = _position_with_a_settlement_in_hand(board)
    true_hand_1 = [0, 0, 3, 2, 3]  # 3 ore this ledger never certified
    state = game.state(0, hidden=False)
    state.hands[0] = [1, 1, 1, 0, 0]  # one wheat short of a settlement
    state.hands[1] = true_hand_1
    state.bank[:] = [BANK_PER_RESOURCE - sum(h[r] for h in state.hands) for r in range(5)]
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
    unlocks_settlement = (0, 0, 0, 1, 0)

    original_sample = View.sample

    def fake_sample(self, rng):
        sampled = original_sample(self, rng)
        sampled.hands[1] = [9, 9, 9, 9, 9]
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

    assert game.state(0, hidden=False).hands[1] == true_hand_1


def _off_turn_position(board):
    """Seat 0 answering on seat 1's turn, with a ledger naming one wheat and
    one ore of seat 1's eight cards and every card of seat 0's own."""
    from hexset.ledger import PublicLedger, SeatLedger

    game = _position_with_a_settlement_in_hand(board)
    state = game.state(0, hidden=False)
    state.hands[0] = [1, 1, 1, 0, 0]
    state.hands[1] = [0, 0, 3, 2, 3]
    state.deck = []
    ledger = PublicLedger.new(state.num_players)
    ledger.apply_hand_diff([[0] * 5 for _ in state.hands], state.hands)
    ledger.seats[1] = SeatLedger(known=[0, 0, 0, 1, 1], unknown=6)
    game.ledger = ledger
    game.current_player = 1
    return game


def _captured_value_rows(monkeypatch):
    captured: list = []
    original = HandValuePolicy.value_rows

    def capture(self, rows):
        captured.extend(rows)
        return original(self, rows)

    monkeypatch.setattr(HandValuePolicy, "value_rows", capture)
    return captured


def test_on_another_seats_turn_the_gate_reads_from_the_movers_frame(board, monkeypatch):
    """Off-turn, every position reaches the head from the seat to move's
    frame, on a world sampled from this seat's belief rather than the true
    hands, in one forward, and the gain is this seat's entry of the vector."""
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS, rng=random.Random(7),
                     trade=TradeParams(gate_own_frame=False))
    game = _off_turn_position(board)
    true_hand_1 = list(game.state(0, hidden=False).hands[1])
    bot.seat_at(game)
    view = game.state(0)
    gets_wheat = (0, 0, 0, 1, 0)

    original_sample = View.sample

    def fake_sample(self, rng):
        sampled = original_sample(self, rng)
        sampled.hands[1] = [9, 9, 9, 9, 9]
        return sampled

    monkeypatch.setattr(View, "sample", fake_sample)
    captured = _captured_value_rows(monkeypatch)

    gains = bot.gains_many(view, [gets_wheat], [1])
    assert len(captured) == 2
    for world, frame in captured:
        assert frame == 1
        assert world.state(1, hidden=False).hands[1] != true_hand_1
    # Seat 1's frame names all of seat 0's cards, so the stub's read of seat
    # 0 is its whole hand moved by the bundle.
    assert gains == [pytest.approx(
        value_of_hand([1, 1, 1, 1, 0], 0) - value_of_hand([1, 1, 1, 0, 0], 0)
    )]
    assert game.state(0, hidden=False).hands[1] == true_hand_1


def test_from_the_movers_frame_the_gate_still_reads_its_own_whole_hand(board, monkeypatch):
    """Where the public record cannot name all of this seat's cards (a steal
    took one into `unknown`), the mover's frame still shows them all: every
    world it reads has this seat's ledger row equal to its hand, before and
    after the exchange, and the gain is the whole hand's."""
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS, rng=random.Random(7),
                     trade=TradeParams(gate_own_frame=False))
    game = _off_turn_position(board)
    game.ledger.seats[0].known = [1, 0, 0, 0, 0]
    game.ledger.seats[0].unknown = 2
    bot.seat_at(game)
    captured = _captured_value_rows(monkeypatch)
    gets_wheat = (0, 0, 0, 1, 0)

    gains = bot.gains_many(game.state(0), [gets_wheat], [1])
    assert [frame for _, frame in captured] == [1, 1]
    for world, _ in captured:
        row = world.ledger.seats[0]
        hand = list(world.state(0, hidden=False).hands[0])
        assert row.known == hand and row.most == hand and row.unknown == 0
    assert captured[1][0].ledger.seats[0].known == [1, 1, 1, 1, 0]
    assert gains == [pytest.approx(
        value_of_hand([1, 1, 1, 1, 0], 0) - value_of_hand([1, 1, 1, 0, 0], 0)
    )]
    assert game.ledger.seats[0].known == [1, 0, 0, 0, 0]       # the live ledger is untouched
    assert game.ledger.seats[0].unknown == 2


def _frame_scaled_values(monkeypatch):
    """`HandValuePolicy.value_rows` replaced by a read that depends on the
    frame: every entry is (frame + 1) x seat 0's card count / 100, so a gain
    averaged over frames is that mean factor times the card change. The
    frames asked for are recorded, row by row."""
    frames: list[int] = []

    def read(self, rows):
        out = []
        for world, frame in rows:
            frames.append(frame)
            cards = sum(world.state(0, hidden=False).hands[0])
            out.append(tuple((frame + 1) * cards / 100 for _ in range(PLAYERS)))
        return out

    monkeypatch.setattr(HandValuePolicy, "value_rows", read)
    return frames


@pytest.mark.parametrize("choice, frames_asked, factor", [
    ("one", [1, 1], 2.0),
    ("pair", [0, 1, 0, 1], 1.5),
    ("all", [0, 1, 2, 3, 0, 1, 2, 3], 2.5),
])
def test_gate_frames_averages_this_seats_entry_over_the_frames_off_turn(board, monkeypatch, choice, frames_asked, factor):
    """On seat 1's turn, `gate_frames` reads the before and after world from
    the mover's frame (`one`), this seat's and the mover's (`pair`) or every
    seat's (`all`), in one forward, and the gain is the mean over them."""
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    frames = _frame_scaled_values(monkeypatch)
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS, rng=random.Random(7),
                     trade=TradeParams(gate_own_frame=False, gate_frames=choice))
    game = _off_turn_position(board)
    bot.seat_at(game)
    gains = bot.gains_many(game.state(0), [(0, 0, 0, 1, 0)], [1])
    assert frames == frames_asked
    assert gains == [pytest.approx(factor * 1 / 100)]


def test_gate_frames_reads_this_seats_own_turn_from_its_own_frame_alone(board, monkeypatch):
    """A position this seat is to move in has one right frame, its own:
    `gate_frames` leaves it there."""
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    frames = _frame_scaled_values(monkeypatch)
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS, trade=TradeParams(gate_frames="all"))
    game = _off_turn_position(board)
    game.current_player = 0
    bot.seat_at(game)
    bot.gains_many(game.state(0), [(0, 0, 0, 1, 0)], [1])
    assert frames == [0, 0]


def test_an_off_turn_read_imagines_this_seats_own_rows_and_no_other_seats(board, monkeypatch):
    """What another seat saw stolen stays out of every world this seat values
    -- the mover's frame reads the table's rows there -- and what this seat
    saw goes in."""
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    captured = _captured_value_rows(monkeypatch)
    gets_wheat = (0, 0, 0, 1, 0)

    def read(mover_saw: bool):
        game = _off_turn_position(board)
        ledger = game.ledger
        ledger.private = [None] * PLAYERS
        mine = ledger.knowledge(0)
        mine.seats[1].known = [0, 0, 1, 1, 1]
        mine.seats[1].unknown = 5
        ledger.private[0] = mine
        if mover_saw:
            theirs = ledger.knowledge(1)
            theirs.seats[2].known = list(game.state(0, hidden=False).hands[2])
            theirs.seats[2].unknown = 0
            ledger.private[1] = theirs
        bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS, rng=random.Random(7),
                         trade=TradeParams(gate_own_frame=False))
        bot.seat_at(game)
        captured.clear()
        gains = bot.gains_many(game.state(0), [gets_wheat], [1])
        assert captured and all(frame == 1 for _, frame in captured)
        for world, _ in captured:
            assert world.ledger.for_seat(1) is world.ledger
        # The world before the exchange; after it, seat 1's wheat has moved.
        assert captured[0][0].ledger.for_seat(0).seats[1].known == [0, 0, 1, 1, 1]
        return gains, [world.ledger.seats for world, _ in captured]

    assert read(True) == read(False)


def test_gate_own_frame_reads_off_turn_from_this_seats_frame(board, monkeypatch):
    """`gate_own_frame` keeps the one-forward read from this seat's own frame
    on every turn, over `_after`'s copy of the live view, with no world
    sampled; on its own turn the gate reads that way regardless."""
    from hexset.clients.netbot import NetworkBot

    space = stub_checkpoint(board).space
    captured = _captured_value_rows(monkeypatch)

    def no_sample(self, rng):
        raise AssertionError("an own-frame read samples no world")

    monkeypatch.setattr(View, "sample", no_sample)
    gets_wheat = (0, 0, 0, 1, 0)

    own = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS,
                     trade=TradeParams(gate_own_frame=True))
    game = _off_turn_position(board)
    own.seat_at(game)
    own.gains_many(game.state(0), [gets_wheat], [1])
    assert [frame for _, frame in captured] == [0, 0]
    assert captured[0][0] is game

    captured.clear()
    mover = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game.current_player = 0
    mover.seat_at(game)
    mover.gains_many(game.state(0), [gets_wheat], [1])
    assert [frame for _, frame in captured] == [0, 0]


def test_the_memo_misses_when_the_board_moves_without_a_hand(board, monkeypatch):
    """Road Building places roads without spending a card, so a memo key built
    from hands alone would answer for the previous table.
    """
    from hexset.clients.netbot import NetworkBot
    from hexset.state import road_placeable
    from hexset.trading import ENUMERATION_CARDS
    from hexset.trading._engine import _candidates

    space = stub_checkpoint(board).space
    bot = NetworkBot(policy=HandValuePolicy(space), players=PLAYERS)
    game = seated(bot, board)
    seat = to_move(game)
    candidates = list(_candidates(game.state(seat, hidden=False), seat, frozenset(), ENUMERATION_CARDS))[:8]
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
    assert len(calls) == 1

    state = game._state
    edge = next(
        e for e in range(state.board.topology.num_edges) if road_placeable(state, seat, e)
    )
    state.edge_owner[edge] = seat  # a free road, no card spent

    bot.gains_many(game.state(seat), received, thems)
    assert len(calls) == 2


def test_a_policy_duels_through_compete_batched(board, monkeypatch):
    """`compete_batched` seats a `PolicyPolicy`'s gate, so each seat given the
    checkpoint prices trades through its own `bot_for` gate and a bare seat
    has none. Two games cut at 30 actions: the stub never spends a card, so
    its hands only grow and an unlimited gate prices every bundle of them
    each round -- 400 actions a game took over a minute and proved no more.
    """
    from hexset.bench.versus import PolicyPolicy, compete_batched
    from hexset.clients.netbot import NetworkBot

    checkpoint = stub_checkpoint(board)
    bare = PolicyPolicy(checkpoint.policy)
    gated = PolicyPolicy(checkpoint.policy, checkpoint)
    assert bare.gate(start(board, PLAYERS, random.Random(2)), 0) is None

    installed: list[tuple[int, int]] = []
    tables = []  # held, so no later table can reuse a finished one's id

    def installing(game, seat):
        tables.append(game)
        installed.append((id(game), seat))
        return PolicyPolicy.gate(gated, game, seat)

    gated.gate = installing
    priced: list[tuple[int, int]] = []
    original = NetworkBot.gains_many

    def counted(self, view, received, counterparties):
        priced.append((id(self._seated), self.seat))
        return original(self, view, received, counterparties)

    monkeypatch.setattr(NetworkBot, "gains_many", counted)

    verdict = compete_batched(
        {0: gated, 1: bare}, 2, players=PLAYERS, seed=5, lanes=1, action_cap=30
    )
    assert verdict.games == 2 and verdict.truncated == 2
    assert len(installed) == len(set(installed)) == 4, "two gated seats a game"
    assert priced, "no gated seat priced a trade"
    assert set(priced) <= set(installed)


def test_plain_onnx_web_spawn_matches_arena_seeded_trade_worlds(board, monkeypatch):
    """`spawn` used to drop its `rng` on the branch that builds a plain
    checkpoint, so the web path sampled unseeded worlds while the arena path,
    which builds through `bot_for`, sampled seeded ones on the same seed."""
    # `hexset.clients.onnxbot` imports onnxruntime at module scope.
    pytest.importorskip("onnxruntime", reason="the ONNX runtime is not installed")
    from types import SimpleNamespace

    from hexset.clients import onnxbot

    loaded = SimpleNamespace(
        **vars(stub_checkpoint(board)), search=SimpleNamespace(searches=())
    )
    monkeypatch.setattr(onnxbot, "load", lambda *args, **kwargs: loaded)

    arena = bot_for(loaded, rng=random.Random(41))
    web = onnxbot.spawn("stub.onnx", board, rng=random.Random(41))
    other_seed = onnxbot.spawn("stub.onnx", board, rng=random.Random(42))

    game = seated(arena, board)
    seat = to_move(game)
    view = game.state(seat)
    expected = [arena._world_rng(view, seat).random() for _ in range(5)]
    assert [web._world_rng(view, seat).random() for _ in range(5)] == expected
    assert [other_seed._world_rng(view, seat).random() for _ in range(5)] != expected



def test_a_rollout_file_spawns_as_the_arenas_rollout_entrant_of_its_settings(
    board, monkeypatch, arena_registry
):
    """A file whose metadata asks for rollouts (`search = rollout`) is seated
    as the arena's `rollout:` entrant of the same settings: the same bot,
    choosing the same moves on the same seed."""
    pytest.importorskip("onnxruntime", reason="the ONNX runtime is not installed")
    from types import SimpleNamespace

    from hexset.arena import entrant_from_name, spawn
    from hexset.clients import onnxbot
    from hexset.clients._modelmeta import RolloutConfig, SearchConfig
    from hexset.clients.netbot import GatedRollouts

    checkpoint = stub_checkpoint(board)
    loaded = SimpleNamespace(
        **vars(checkpoint),
        search=SearchConfig(decisive=0.97, rollout=RolloutConfig(worlds=2, top=2, horizon=1)),
    )
    monkeypatch.setattr(onnxbot, "load", lambda *args, **kwargs: loaded)
    register_entrants(lambda path, topology: checkpoint)

    served = onnxbot.spawn("stub.onnx", board, rng=random.Random(5))
    entrant = spawn(
        entrant_from_name("rollout:/tmp/x.onnx@2:top=2:decisive=0.97:horizon=1"),
        board, random.Random(5),
    )
    assert isinstance(served, GatedRollouts) and isinstance(entrant, GatedRollouts)
    assert (served.worlds, served.top, served.decisive, served.horizon) == (2, 2, 0.97, 1)
    assert not served.full_setup and not entrant.full_setup
    opening_loaded = SimpleNamespace(**vars(checkpoint), search=SearchConfig(
        decisive=0.97, rollout=RolloutConfig(worlds=2, top=2, horizon=1, full_setup=True)))
    monkeypatch.setattr(onnxbot, "load", lambda *args, **kwargs: opening_loaded)
    assert onnxbot.spawn("stub.onnx", board, rng=random.Random(5)).full_setup
    assert spawn(entrant_from_name("rollout:/tmp/x.onnx@2:top=2:decisive=0.97:horizon=1:setup=full"),
                 board, random.Random(5)).full_setup
    monkeypatch.setattr(onnxbot, "load", lambda *args, **kwargs: loaded)

    game = start(board, PLAYERS, random.Random(2))
    walk = random.Random(2)
    chosen = searched = 0
    while chosen < 30:
        if to_move(game) == 0:
            move = served.choose(game)
            assert entrant.choose(game) == move
            searched += bool(served.decision_log[-1].worlds)
            chosen += 1
            apply(game, move)
        else:
            step_randomly(game, walk)
    assert searched, "no decision was played out"


# --- own_calibration in the gate: gains on the calibrated scale -------------------------------------------


@dataclass
class ChancePolicy(HandValuePolicy):
    """Win chances: seat 0's grows with its own card count from 5%, the rest
    share what is left equally. Enough to watch a gain change scale."""

    def _value(self, game, seat):
        own = sum(View.from_game(game, seat).known[0])
        p0 = min(0.05 + 0.01 * own, 0.9)
        return (p0,) + ((1 - p0) / (PLAYERS - 1),) * (PLAYERS - 1)


def test_a_calibrated_gate_prices_gains_on_the_calibrated_scale(board):
    from hexset.clients.netbot import NetworkBot
    from hexset.mcts import calibrate_own

    cal = (0.32, 0.81)
    game = _position_with_a_settlement_in_hand(board)
    assert to_move(game) == 0
    space = stub_checkpoint(board).space
    gains = {}
    for name, c in (("raw", None), ("calibrated", cal)):
        bot = NetworkBot(policy=ChancePolicy(space), players=PLAYERS, rng=random.Random(7), own_calibration=c)
        bot.seat_at(game)
        view = game.state(0)
        gets_two_gives_one = (1, 1, -1, 0, 0)
        gains[name] = bot.gains_many(view, [gets_two_gives_one], [1])[0]
        across = bot.gains_across(0, [(game, [gets_two_gives_one], [1])])[0][0]
        assert across == pytest.approx(gains[name])
    own = sum(game.state(0).known[0])
    before = ChancePolicy(space)._value(game, 0)
    p = lambda cards: (min(0.05 + 0.01 * cards, 0.9),) + ((1 - min(0.05 + 0.01 * cards, 0.9)) / 3,) * 3
    assert gains["raw"] == pytest.approx(p(own + 1)[0] - before[0])
    assert gains["calibrated"] == pytest.approx(
        calibrate_own(p(own + 1), 0, cal)[0] - calibrate_own(before, 0, cal)[0])
    assert gains["calibrated"] > gains["raw"] > 0      # behind, a gain counts for more
