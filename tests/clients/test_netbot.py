"""The checkpoint runtime, driven by a policy that is not a checkpoint: a stub
`Policy` drives `choose`, the gate, a real `trade_event` and a `GatedSearch`
end to end. The stub reads the true hands, which no real runtime could, so
that the gate's arithmetic is checkable in closed form.
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
from traders import FRAGMENTED
from helpers import give

# Read through a per-seat rotation: seat *s* prices resource *r* at
# `WEIGHTS[(r + s) % 5]`. One shared set makes every exchange zero-sum,
# so no trade could clear both gates; rotating gives the seats tastes.
WEIGHTS = (0.006, -0.011, 0.004, 0.013, -0.008)

PLAYERS = 4


@dataclass
class HandValuePolicy:
    """Value is a fixed linear read of each seat's hand, the prior uniform, the
    action the lowest-indexed legal one.
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
        hands = game.state(0, hidden=False).hands
        return tuple(value_of_hand(hand, seat) for seat, hand in enumerate(hands))


def value_of_hand(hand, seat: int) -> float:
    return sum(WEIGHTS[(r + seat) % len(WEIGHTS)] * n for r, n in enumerate(hand))


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
    state = game.state(seat, hidden=False)
    hands = state.hands
    for i, (them, bundle) in enumerate(candidates):
        if gains[i] == -1.0:
            continue
        mine = [n + d for n, d in zip(hands[seat], bundle)]
        theirs = [n - d for n, d in zip(hands[them], bundle)]
        assert gains[i] == pytest.approx(
            value_of_hand(mine, seat) - value_of_hand(hands[seat], seat)
        )
        assert estimates[i] == pytest.approx(
            value_of_hand(theirs, them) - value_of_hand(hands[them], them)
        )
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
        [value_of_hand([4, 4, 4, 4, 3], 1) - value_of_hand([4, 4, 4, 4, 4], 1),
         value_of_hand([4, 4, 4, 3, 4], 1) - value_of_hand([4, 4, 4, 4, 4], 1)]
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
    from hexset.ledger import PublicLedger
    from hexset.view import View

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
