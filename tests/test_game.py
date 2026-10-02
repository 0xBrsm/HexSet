# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest
from helpers import clear_hand, give, independent_vertices, mini_board, victim_on

from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.board.topology import coastal_rings
from hexset.cards import DevCard
from hexset.economy import COSTS, Purchase, expected_total, total_in_play
from hexset.game import (
    Phase,
    build_city,
    build_road,
    build_settlement,
    end_turn,
    legal_initial_roads,
    move_robber_to,
    place_initial_road,
    place_initial_settlement,
    play_knight_card,
    play_monopoly_card,
    players_owing_discards,
    roll_dice,
    run_trade_event,
    start,
    submit_discard,
    trade_with_bank,
)
from hexset.state import NO_OWNER, Building, can_place_settlement
from hexset.victory import WINNING_POINTS, update_longest_road, victory_points


def a_game(players: int = 3, seed: int = 0):
    return start(random_base_board(random.Random(seed)), players, random.Random(seed))


def free_vertex(game):
    return next(
        v
        for v in range(game._state.board.topology.num_vertices)
        if can_place_settlement(game._state, game.current_player, v, connected=False)
    )


def run_setup(game):
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        if game.phase is Phase.SETUP_SETTLEMENT:
            place_initial_settlement(game, free_vertex(game))
        else:
            place_initial_road(game, legal_initial_roads(game)[0])
    return game


def fund(state, player, purchase):
    for resource, count in enumerate(COSTS[purchase]):
        give(state, player, resource, count)


def test_setup_uses_snake_order():
    game = a_game(players=3)
    assert game.setup_queue == [0, 1, 2, 2, 1, 0]


def test_only_the_first_round_is_unpaid():
    game = a_game(players=2)
    place_initial_settlement(game, free_vertex(game))
    assert game._state.hands[0] == [0] * 5

    run_setup(game)
    assert any(sum(hand) > 0 for hand in game._state.hands)
    assert total_in_play(game._state) == expected_total()


def test_opening_road_must_touch_the_new_settlement():
    game = a_game()
    place_initial_settlement(game, free_vertex(game))
    illegal = next(
        e
        for e in range(game._state.board.topology.num_edges)
        if e not in legal_initial_roads(game)
    )
    with pytest.raises(ValueError):
        place_initial_road(game, illegal)


def test_actions_are_rejected_in_the_wrong_phase():
    game = a_game()
    with pytest.raises(ValueError):
        roll_dice(game)
    with pytest.raises(ValueError):
        end_turn(game)


def test_a_big_hand_must_discard_on_seven():
    game = run_setup(a_game())
    clear_hand(game._state, 0)
    for resource in Resource:
        give(game._state, 0, resource, 2)
    assert sum(game._state.hands[0]) == 10

    game.last_roll = 7
    game.phase = Phase.ROLL
    game.rng = random.Random(1)
    while roll_dice(game) != 7:
        game.phase = Phase.ROLL

    assert game.phase is Phase.DISCARD
    assert 0 in players_owing_discards(game)
    assert game.discard_quota[0] == 5

    submit_discard(game, 0, [1, 1, 1, 1, 1])
    assert sum(game._state.hands[0]) == 5
    assert game.discard_quota[0] == 0


def test_building_costs_resources_and_advances_the_road():
    game = run_setup(a_game())
    game.phase = Phase.MAIN
    clear_hand(game._state, 0)
    fund(game._state, 0, Purchase.ROAD)
    topology = game._state.board.topology
    mine = game._state.edge_owner.index(0)
    junction = topology.edges[mine][0]
    edge = next(
        e
        for e in topology.vertex_edges[junction]
        if game._state.edge_owner[e] == NO_OWNER
    )

    build_road(game, edge)

    assert game._state.edge_owner[edge] == 0
    assert game._state.hands[0] == [0] * 5
    assert total_in_play(game._state) == expected_total()


def test_ending_a_turn_matures_cards_and_passes_play():
    game = run_setup(a_game(players=3))
    game.phase = Phase.MAIN
    game._state.new_dev_cards[0][DevCard.MONOPOLY] = 1

    end_turn(game)

    assert game._state.dev_cards[0][DevCard.MONOPOLY] == 1
    assert game._state.new_dev_cards[0][DevCard.MONOPOLY] == 0
    assert game.current_player == 1
    assert game.phase is Phase.ROLL


def test_the_card_allowance_resets_each_turn():
    game = run_setup(a_game(players=2))
    game.phase = Phase.MAIN
    game._state.dev_cards[0][DevCard.MONOPOLY] = 1
    play_monopoly_card(game, Resource.ORE)
    assert game.dev_card_played

    end_turn(game)
    assert not game.dev_card_played


def test_reaching_ten_points_ends_the_game():
    game = run_setup(a_game(players=2))
    game.phase = Phase.MAIN
    fund(game._state, 0, Purchase.CITY)

    game._state.dev_cards[0][DevCard.VICTORY_POINT] = 7
    settlement = game._state.vertex_owner.index(0)

    build_city(game, settlement)

    assert game.phase is Phase.GAME_OVER
    assert game.won_by == 0


def test_a_knight_that_wins_the_game_ends_it_at_once_with_no_robber_move():
    """Rulebook order: spend, update Largest Army, check the win, then the robber."""
    game = run_setup(a_game(players=2))
    game.phase = Phase.MAIN
    game._state.dev_cards[0][DevCard.VICTORY_POINT] = 7
    game._state.dev_cards[0][DevCard.KNIGHT] = 1
    game._state.knights_played[0] = 2

    play_knight_card(game)

    assert game.phase is Phase.GAME_OVER
    assert game.won_by == 0
    assert game._state.largest_army_holder == 0


def _spy_on_trade_event(monkeypatch):
    """Patched on the names `run_trade_event` calls, which `game.py` imported directly."""
    import hexset.game as gamemod

    calls: list[Phase] = []
    real_event = gamemod.trade_event
    real_round = gamemod.trade_round

    def spy_event(game, gate):
        calls.append(game.phase)
        return real_event(game, gate)

    def spy_round(game, gates, already_offered=None, offers_made=None, counter_steps=1):
        calls.append(game.phase)
        return real_round(
            game, gates, already_offered=already_offered, offers_made=offers_made,
            counter_steps=counter_steps,
        )

    monkeypatch.setattr(gamemod, "trade_event", spy_event)
    monkeypatch.setattr(gamemod, "trade_round", spy_round)
    return calls


from hexset.trading import TradeParams


class _Gate:
    """The minimum a seated gate needs to be asked anything. `budget` is the
    offers it declares in its own `trade_params`; leaving it off is a gate
    that declares no limit, which is the default now that the table sets
    none."""

    def __init__(self, budget=None):
        if budget is not None:
            self.trade_params = TradeParams(max_offers=budget)


def _seated(game, budget=None):
    """The minimum for `run_trade_event` not to short-circuit on `gates is None`."""
    n = game._state.num_players
    game.gates = tuple(_Gate(budget) for _ in range(n))
    return game


@pytest.mark.parametrize(
    "setup, act",
    [
        (
            lambda g: fund(g._state, 0, Purchase.CITY),
            lambda g: build_city(g, g._state.vertex_owner.index(0)),
        ),
        (
            lambda g: g._state.dev_cards[0].__setitem__(DevCard.MONOPOLY, 1),
            lambda g: play_monopoly_card(g, Resource.ORE),
        ),
    ],
    ids=["build_city", "monopoly"],
)
def test_a_main_action_does_not_run_the_trade_event_again(monkeypatch, setup, act):
    game = _seated(run_setup(a_game(players=3)))
    game.phase = Phase.MAIN
    setup(game)
    calls = _spy_on_trade_event(monkeypatch)

    act(game)

    assert calls == []


def test_pending_offers_survive_a_main_action_later_in_the_same_turn():
    from hexset.trading import Trade, one_for_one

    game = _seated(run_setup(a_game(players=3)))
    game.phase = Phase.MAIN
    clear_hand(game._state, 0)
    give(game._state, 0, Resource.WOOD, 4)
    game.pending.append(Trade(1, 0, one_for_one(Resource.WOOD, Resource.ORE)))

    trade_with_bank(game, Resource.WOOD, Resource.ORE)

    assert game.pending == [Trade(1, 0, one_for_one(Resource.WOOD, Resource.ORE))]


def test_trade_event_runs_once_a_turn_even_when_a_knight_re_enters_main(monkeypatch):
    """`Game.trade_event_turn` is what keeps it to one."""
    from hexset.game import enter_main

    game = _seated(run_setup(a_game(players=3)))
    game._state.dev_cards[0][DevCard.KNIGHT] = 1
    target = (game._state.robber + 1) % game._state.board.num_hexes
    calls = _spy_on_trade_event(monkeypatch)

    enter_main(game)
    assert calls == [Phase.MAIN]
    play_knight_card(game)
    move_robber_to(game, target, victim_on(game, target))
    assert game.phase is Phase.MAIN
    assert calls == [Phase.MAIN], "no second event in the same turn"

    end_turn(game)
    game.phase = Phase.MAIN
    enter_main(game)
    assert calls == [Phase.MAIN, Phase.MAIN], "the next turn gets its own"


def test_trade_event_never_runs_during_discard_resolution(monkeypatch):
    game = _seated(run_setup(a_game()))
    clear_hand(game._state, 0)
    for resource in Resource:
        give(game._state, 0, resource, 2)
    game.last_roll = 7
    game.phase = Phase.ROLL
    game.rng = random.Random(1)
    calls = _spy_on_trade_event(monkeypatch)

    while roll_dice(game) != 7:
        game.phase = Phase.ROLL
        calls.clear()
    assert game.phase is Phase.DISCARD
    assert calls == []

    submit_discard(game, 0, [1, 1, 1, 1, 1])
    assert game.phase is Phase.ROBBER
    assert calls == []

    target = (game._state.robber + 1) % game._state.board.num_hexes
    move_robber_to(game, target, victim_on(game, target))
    assert game.phase is Phase.MAIN
    assert calls == [Phase.MAIN]
    legal_actions(game)
    assert calls == [Phase.MAIN]


def test_preroll_roads_resolve_before_dice_without_spending_or_trading():
    game = run_setup(a_game(players=2))
    game._state.dev_cards[0][DevCard.ROAD_BUILDING] = 1
    hands = [hand[:] for hand in game._state.hands]
    bank = game._state.bank[:]
    rng_state = game.rng.getstate()
    apply(game, Action(ActionType.PLAY_ROAD_BUILDING))
    for remaining in (2, 1):
        options = legal_actions(game)
        assert options and all(a.type is ActionType.BUILD_ROAD for a in options)
        with pytest.raises(ValueError, match="free roads"):
            roll_dice(game)
        assert game.rng.getstate() == rng_state
        apply(game, options[0])
        assert game.free_roads == remaining - 1
        assert game.phase is Phase.ROLL
    assert game._state.hands == hands
    assert game._state.bank == bank
    assert game.trades == []
    assert legal_actions(game) == [Action(ActionType.ROLL)]
    roll_dice(game, roll=6)
    assert game.phase is Phase.MAIN


def test_unplaceable_preroll_roads_do_not_block_the_roll_or_carry_over():
    game = run_setup(a_game(players=2))
    game._state.dev_cards[0][DevCard.ROAD_BUILDING] = 1
    game._state.edge_owner[:] = [1] * len(game._state.edge_owner)
    apply(game, Action(ActionType.PLAY_ROAD_BUILDING))
    assert legal_actions(game) == [Action(ActionType.ROLL)]
    roll_dice(game, roll=6)
    assert game.phase is Phase.MAIN
    assert game.free_roads == 0


def test_only_one_card_total_across_roll_and_main():
    game = run_setup(a_game(players=2))
    game._state.dev_cards[0][DevCard.KNIGHT] = 1
    game._state.dev_cards[0][DevCard.MONOPOLY] = 1

    play_knight_card(game)
    target = (game._state.robber + 1) % game._state.board.num_hexes
    move_robber_to(game, target, victim_on(game, target))
    assert game.phase is Phase.ROLL
    roll_dice(game, roll=6)

    with pytest.raises(ValueError):
        play_monopoly_card(game, Resource.ORE)


def test_legal_actions_before_the_roll_offer_every_playable_card():
    game = run_setup(a_game(players=2))
    game._state.dev_cards[0][DevCard.ROAD_BUILDING] = 1
    game._state.dev_cards[0][DevCard.MONOPOLY] = 1
    game._state.dev_cards[0][DevCard.YEAR_OF_PLENTY] = 1

    kinds = {a.type for a in legal_actions(game)}

    assert ActionType.ROLL in kinds
    assert ActionType.PLAY_ROAD_BUILDING in kinds
    assert ActionType.PLAY_MONOPOLY in kinds
    assert ActionType.PLAY_YEAR_OF_PLENTY in kinds
    assert ActionType.BUILD_ROAD not in kinds
    assert ActionType.BUY_DEV_CARD not in kinds
    assert ActionType.BANK_TRADE not in kinds


def test_a_seat_that_crosses_ten_off_turn_wins_at_the_start_of_its_own_turn():
    game = start(mini_board(), 3, random.Random(0))
    state = game._state
    topology = state.board.topology
    game.phase = Phase.MAIN
    game.current_player = 0

    ring = coastal_rings(topology)[0]
    p2_path = ring[0:5]
    for e in p2_path:
        state.edge_owner[e] = 2
    update_longest_road(state)
    assert state.longest_road_holder == 2

    shared = set(topology.edges[p2_path[1]]) & set(topology.edges[p2_path[2]])
    break_vertex = shared.pop()

    p1_path = ring[10:16]
    for e in p1_path:
        state.edge_owner[e] = 1
    spots = [
        v
        for v in independent_vertices(state.board, 8)
        if v not in topology.vertex_neighbors[break_vertex] and v != break_vertex
    ]
    for v in spots[:4]:
        state.vertex_owner[v] = 1
        state.vertex_building[v] = Building.CITY
    state.vertex_owner[spots[4]] = 1
    state.vertex_building[spots[4]] = Building.SETTLEMENT
    assert victory_points(state, 1) == 9

    third_edge = next(e for e in topology.vertex_edges[break_vertex] if e not in p2_path)
    state.edge_owner[third_edge] = 0
    update_longest_road(state)
    fund(state, 0, Purchase.SETTLEMENT)

    build_settlement(game, break_vertex)

    assert state.longest_road_holder == 1
    assert victory_points(state, 1) >= WINNING_POINTS
    assert game.won_by is None
    assert game.phase is Phase.MAIN

    end_turn(game)

    assert game.current_player == 1
    assert game.won_by == 1
    assert game.phase is Phase.GAME_OVER


#


def _fake_rounds(monkeypatch, results, offers=None):
    """`offers` is how many distinct offers the actor has; running out is the
    signal `_run_trade_rounds` stops on, not a round that nobody took.
    """
    import hexset.game as gamemod

    calls = {"n": 0}
    queue = list(results)
    left = len(results) if offers is None else offers

    def fake(game, gates, already_offered=None, offers_made=None, counter_steps=1):
        calls["n"] += 1
        if calls["n"] <= left:
            if already_offered is not None:
                already_offered.add((calls["n"],))
            if offers_made is not None:
                offers_made.append((calls["n"],))
        return queue.pop(0) if queue else []

    monkeypatch.setattr(gamemod, "trade_round", fake)
    return calls


def _in_main_with_gates(monkeypatch, budget=None):
    game = _seated(
        start(random_base_board(random.Random(0)), 4, random.Random(0)), budget
    )
    game.phase = Phase.MAIN
    game.turns = 1
    game.trade_event_turn = -1
    return game


def test_a_gate_declaring_no_budget_offers_until_it_runs_out(monkeypatch):
    """The table has no say in how many offers a turn holds: a gate that
    declares no limit keeps going while it has an offer it has not made."""
    game = _in_main_with_gates(monkeypatch)
    assert game.trade_mode == "round"
    calls = _fake_rounds(monkeypatch, [], offers=3)
    run_trade_event(game)
    assert calls["n"] == 4


def test_a_gate_at_zero_offers_is_never_asked_under_either_mechanism(monkeypatch):
    import hexset.game as gamemod

    for mode in ("round", "auto"):
        game = _in_main_with_gates(monkeypatch, budget=0)
        game.trade_mode = mode
        rounds = _fake_rounds(monkeypatch, [])
        cleared = {"n": 0}
        monkeypatch.setattr(
            gamemod, "trade_event", lambda game, gate: cleared.__setitem__("n", 1) or []
        )
        run_trade_event(game)
        assert rounds["n"] == 0 and cleared["n"] == 0


def test_an_externally_driven_game_runs_no_mechanism_at_all(monkeypatch):
    import hexset.game as gamemod

    for _ in range(2):
        game = _in_main_with_gates(monkeypatch)
        game.trade_mode = "external"
        rounds = _fake_rounds(monkeypatch, [["a"]])
        cleared = {"n": 0}
        monkeypatch.setattr(
            gamemod, "trade_event", lambda game, gate: cleared.__setitem__("n", 1) or []
        )
        run_trade_event(game)
        assert rounds["n"] == 0 and cleared["n"] == 0


def test_a_table_rule_narrows_where_the_robber_may_go_for_one_move():
    """A "friendly robber" rule (and any host rule like it) forbids
    hexes the rulebook allows, and says only which remain. `Game.robber_allowed`
    is that list for the move at hand: `legal_actions` offers nothing else,
    `move_robber_to` refuses anything else, a search's copy sees the same
    rule, and the rule is gone once the move is made."""
    from hexset.actions import ActionType, legal_actions
    from hexset.game import imagine

    game = run_setup(a_game())
    game.phase = Phase.ROBBER
    everywhere = {a.a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER}
    assert len(everywhere) == game._state.board.num_hexes - 1

    allowed = frozenset(sorted(everywhere)[:3])
    game.robber_allowed = allowed
    assert {a.a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER} == allowed
    copy = imagine(game, random.Random(1))
    assert copy.robber_allowed == allowed
    assert {a.a for a in legal_actions(copy) if a.type is ActionType.MOVE_ROBBER} == allowed

    forbidden = next(h for h in everywhere if h not in allowed)
    with pytest.raises(ValueError, match="does not allow"):
        move_robber_to(game, forbidden)
    move_robber_to(game, min(allowed), victim_on(game, min(allowed)))
    assert game._state.robber == min(allowed)
    assert game.robber_allowed is None              # the rule was for that move alone
    assert game.phase is Phase.MAIN


def test_setup_hands_the_first_turn_to_a_seat_that_is_still_playing():
    """A seat retired before it ever placed cannot take the first real turn.
    `lock_seat` moves the snake off `current_player`, but a seat locked while
    some other seat was on the clock was never pointed at -- and the handoff
    out of setup reads the head of the queue, which is exactly that seat when
    the retired seats sit at the front of the order."""
    from hexset.game import lock_seat

    game = a_game(players=4)
    for seat in (0, 1):
        lock_seat(game, seat)
    run_setup(game)

    assert game.phase is Phase.ROLL
    assert game.current_player not in game.locked
    from hexset.game import to_move

    assert to_move(game) == 2
