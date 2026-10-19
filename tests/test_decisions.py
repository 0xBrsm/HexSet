"""Compound decisions (`hexset.decisions`): each the complete atomic sequences, and nothing else."""
from __future__ import annotations

import copy
import random

import numpy as np

from hexset import decisions as D
from hexset.actions import Action, ActionType, apply, legal_actions, space_for
from hexset.board.board import random_base_board
from hexset.contract9 import incidences
from hexset.game import Phase, is_over, players_owing_discards, start, to_move


def _sequences(game, seat, done):
    """Every complete atomic sequence of the decision at hand: every path of
    legal moves by `seat` until `done(game, path)`."""
    out = set()

    def walk(g, path):
        if done(g, path):
            out.add(tuple(path))
            return
        asking = seat if g.phase is Phase.DISCARD else None
        for action in legal_actions(g, asking):
            h = copy.deepcopy(g)
            apply(h, action, asking)
            walk(h, path + [action])

    walk(game, [])
    return out


def _decider(game):
    return players_owing_discards(game)[0] if game.phase is Phase.DISCARD else to_move(game)


def _play(seed, until):
    """A game played on random decisions until `until(game)` holds."""
    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    for _ in range(3000):
        if is_over(game) or until(game):
            return game
        seat = _decider(game)
        D.apply_option(game, rng.choice(D.options(game, seat)), seat if game.phase is Phase.DISCARD else None)
    return None


def test_a_setup_turn_is_a_settlement_with_its_road():
    game = _play(1, lambda g: g.phase is Phase.SETUP_SETTLEMENT and g.setup_step == 2)
    seat = to_move(game)
    options = D.options(game, seat)
    assert all(o.kind is D.CompoundKind.SETUP for o in options)
    want = _sequences(game, seat, lambda g, path: len(path) == 2)
    assert {D.key(D.Compound(D.CompoundKind.SETUP, s)) for s in want} == {D.key(o) for o in options}


def test_the_snake_seat_places_both_settlements_in_one_decision():
    game = _play(2, lambda g: D.is_double_turn(g))
    assert game.setup_step == 3 and game.setup_queue[3] == game.setup_queue[4]
    options = D.options(game)
    assert all(o.kind is D.CompoundKind.SETUP_DOUBLE and len(o.parts) == 4 for o in options)
    decision = D.tensors(game, to_move(game), space_for(game))
    ok = D.double_pairs(game._state.board.topology)
    assert decision.double and int(ok[np.ix_(decision.setup, decision.setup)].sum()) == len(options)
    # Each pair is legal played out, and the second settlement is the paying one.
    for option in random.Random(0).sample(options, 40):
        trial = copy.deepcopy(game)
        D.apply_option(trial, option)
        assert trial._state.vertex_owner[option.parts[2].a] == game.current_player


def test_a_discard_is_every_hand_the_seat_can_keep():
    found = 0
    for seed in range(40):
        game = _play(seed, lambda g: g.phase is Phase.DISCARD)
        if game is None or game.phase is not Phase.DISCARD:
            continue
        seat = _decider(game)
        quota = game.discard_quota[seat]
        options = D.options(game, seat)
        want = _sequences(game, seat, lambda g, path, q=quota: len(path) == q)
        assert {D.key(D.Compound(D.CompoundKind.DISCARD, s)) for s in want} == {D.key(o) for o in options}
        found += 1
        if found == 3:
            return
    assert found


def test_road_building_is_the_card_with_every_road_set_it_can_place():
    found = 0
    for seed in range(200):
        game = _play(seed, lambda g: g.phase in (Phase.ROLL, Phase.MAIN) and g.free_roads == 0 and any(
            a.type is ActionType.PLAY_ROAD_BUILDING for a in legal_actions(g)))
        if game is None or is_over(game):
            continue
        seat = to_move(game)
        played = copy.deepcopy(game)
        apply(played, Action(ActionType.PLAY_ROAD_BUILDING))
        want = _sequences(played, seat, lambda g, path: g.free_roads == 0 or g.phase is Phase.GAME_OVER
                          or not any(a.type is ActionType.BUILD_ROAD for a in legal_actions(g)))
        got = {D.key(o)[1] for o in D.options(game, seat) if isinstance(o, D.Compound)}
        assert {frozenset(a.a for a in s) for s in want} == got
        assert not any(isinstance(o, Action) and o.type is ActionType.PLAY_ROAD_BUILDING for o in D.options(game, seat))
        found += 1
        if found == 2:
            return
    assert found


def _road_sets_edge_by_edge(game):
    """`decisions._road_building` with each second road found by asking
    `road_placeable` of every edge in turn, as it once did."""
    from hexset.state import MAX_ROADS, NO_OWNER, placeable_roads, road_count, road_placeable
    from hexset.victory import LONGEST_ROAD_VP, victory_points

    state, player = game._state, game.current_player
    first = placeable_roads(state, player)
    if MAX_ROADS - road_count(state, player) < 2:
        return [(e,) for e in first]
    near = victory_points(state, player) + LONGEST_ROAD_VP >= state.rules.winning_points
    out, seen = [], set()
    for e1 in first:
        if near and D._wins_with(game, e1):
            out.append((e1,))
            continue
        state.edge_owner[e1] = player
        try:
            second = [e for e in range(len(state.edge_owner)) if road_placeable(state, player, e)]
        finally:
            state.edge_owner[e1] = NO_OWNER
        if not second:
            out.append((e1,))
        for e2 in second:
            if frozenset((e1, e2)) not in seen:
                seen.add(frozenset((e1, e2)))
                out.append((e1, e2))
    return out


def test_the_cards_road_sets_are_the_edge_by_edge_scans_in_its_order():
    checked = 0
    for seed in range(40):
        rng = random.Random(seed)
        game = start(random_base_board(rng), 4, rng)
        for step in range(600):
            if is_over(game):
                break
            if game.phase in (Phase.ROLL, Phase.MAIN) and game.free_roads == 0 and step % 5 == 0:
                assert D._road_building(game) == _road_sets_edge_by_edge(game)
                checked += 1
            seat = _decider(game)
            D.apply_option(game, rng.choice(D.options(game, seat)), seat if game.phase is Phase.DISCARD else None)
    assert checked > 500


def test_every_option_locates_in_its_tensors_and_applies_as_its_parts():
    rng = random.Random(7)
    for seed in range(4):
        game = start(random_base_board(random.Random(seed)), 4, random.Random(seed))
        space = space_for(game)
        pairs = incidences(game._state.board.topology)
        for _ in range(400):
            if is_over(game):
                break
            seat = _decider(game)
            options = D.options(game, seat)
            decision = D.tensors(game, seat, space)
            seen = set()
            for option in options if len(options) < 300 else rng.sample(options, 200):
                kind, index = D.locate(decision, option, game, seat, space)
                assert (kind, index) not in seen
                seen.add((kind, index))
                if kind is D.DecisionKind.DOUBLE:
                    assert 0 <= index < len(pairs) ** 2
            choice = rng.choice(options)
            asking = seat if game.phase is Phase.DISCARD else None
            atomic = copy.deepcopy(game)
            for part in (choice.parts if isinstance(choice, D.Compound) else (choice,)):
                apply(atomic, part, asking)
                if is_over(atomic):
                    break
            D.apply_option(game, choice, asking)
            assert game._state.hands == atomic._state.hands and game.phase is atomic.phase
            assert game._state.edge_owner == atomic._state.edge_owner


def test_robber_victims_are_seat_relative_in_the_flat_mask():
    for seed in range(60):
        game = _play(seed, lambda g: g.phase is Phase.ROBBER)
        if game is None or game.phase is not Phase.ROBBER:
            continue
        seat = to_move(game)
        space = space_for(game)
        decision = D.tensors(game, seat, space)
        for action in legal_actions(game, seat):
            relative = action.b if action.b >= 4 else (action.b - seat) % 4
            assert decision.flat[space.index(Action(action.type, action.a, relative))]
        return


def _won_by_playing(game, edge):
    """The card played and its first road on `edge`, on a copy: whether that ends the game."""
    trial = copy.deepcopy(game)
    trial.gates = None
    apply(trial, Action(ActionType.PLAY_ROAD_BUILDING))
    apply(trial, Action(ActionType.BUILD_ROAD, edge))
    return trial.phase is Phase.GAME_OVER


def test_a_first_free_road_wins_exactly_when_the_engine_says_it_does():
    """Read off the position, without playing it: every placeable first road
    of seats set two points short, at whatever road lengths the game left."""
    from hexset.devcards import DevCard
    from hexset.state import placeable_roads
    from hexset.victory import victory_points

    checked = wins = 0
    for seed in range(40):
        game = _play(seed, lambda g: g.phase is Phase.MAIN and g.turns > 30 + seed and g.free_roads == 0)
        if game is None or is_over(game):
            continue
        state = game._state
        player = game.current_player
        short = state.rules.winning_points - 2 - victory_points(state, player)
        if short < 0:
            continue
        state.dev_cards[player][DevCard.VICTORY_POINT] += short
        state.dev_cards[player][DevCard.ROAD_BUILDING] += 1
        game.dev_card_played = False
        for edge in placeable_roads(state, player):
            assert D._wins_with(game, edge) == _won_by_playing(game, edge), (seed, edge)
            checked += 1
            wins += D._wins_with(game, edge)
    assert checked and wins


# --- where a run of options ends (`closes`) ------------------------------------------


def _at(seed, phase, turns=8):
    """A position past the setup in `phase`, from random play."""
    return _play(seed, lambda g: g.phase is phase and g.turns >= turns)


def _taking(game, kind, seat=None, double_turn=True):
    """Apply the first legal option of `kind` (an `ActionType`, or a
    `CompoundKind`) on a copy: the copy and whether the run closes with it."""
    game = copy.deepcopy(game)
    seat = _decider(game) if seat is None else seat
    found = [o for o in D.options(game, seat, double_turn=double_turn)
             if (o.kind if isinstance(o, D.Compound) else o.type) is kind]
    assert found, f"no {kind!r} here"
    before = game.trade_event_turn
    D.apply_option(game, found[0], seat if game.phase is Phase.DISCARD else None)
    return game, D.closes(game, found[0], seat, before)


def _holding(game, seat, card):
    from hexset.cards import DevCard

    game._state.dev_cards[seat][DevCard[card]] += 1
    return game


def _rich(game, seat):
    from hexset.board.terrain import Resource
    from helpers import deal

    for resource in Resource:
        deal(game, seat, resource, 4)
    return game


def test_the_roll_closes_a_run():
    game = _at(3, Phase.ROLL)
    assert _taking(game, ActionType.ROLL)[1]


def test_the_turn_handed_on_closes_a_run():
    game = _at(3, Phase.MAIN)
    assert _taking(game, ActionType.END_TURN)[1]


def test_a_dev_card_bought_closes_a_run():
    game = _at(3, Phase.MAIN)
    assert _taking(_rich(game, to_move(game)), ActionType.BUY_DEV_CARD)[1]


def test_builds_and_bank_trades_run_on():
    game = _at(3, Phase.MAIN)
    _rich(game, to_move(game))
    for kind in (ActionType.BUILD_ROAD, ActionType.BANK_TRADE):
        after, closed = _taking(game, kind)
        assert not closed and not is_over(after), kind.name
    built, closed = _taking(game, ActionType.BUILD_ROAD)
    # A run of builds: road, then road again from where the first left it.
    assert not closed and not _taking(built, ActionType.BUILD_ROAD)[1]


def test_a_knight_runs_on_to_its_robber_and_the_steal_closes():
    game = _at(3, Phase.MAIN)
    seat = to_move(game)
    knighted, closed = _taking(_holding(game, seat, "KNIGHT"), ActionType.PLAY_KNIGHT)
    assert not closed and knighted.phase is Phase.ROBBER and to_move(knighted) == seat
    assert _taking(knighted, ActionType.MOVE_ROBBER)[1]


def test_a_monopoly_shows_what_it_takes_and_year_of_plenty_does_not():
    game = _at(3, Phase.MAIN)
    seat = to_move(game)
    assert _taking(_holding(copy.deepcopy(game), seat, "MONOPOLY"), ActionType.PLAY_MONOPOLY)[1]
    assert not _taking(_holding(copy.deepcopy(game), seat, "YEAR_OF_PLENTY"), ActionType.PLAY_YEAR_OF_PLENTY)[1]


def test_road_building_runs_on():
    game = _at(3, Phase.MAIN)
    after, closed = _taking(_holding(game, to_move(game), "ROAD_BUILDING"), D.CompoundKind.ROAD_BUILDING)
    assert not closed or is_over(after)


def test_a_discard_closes_once_the_seat_has_given_its_quota():
    game = _play(5, lambda g: g.phase is Phase.DISCARD and max(g.discard_quota) >= 2)
    seat = next(s for s, owed in enumerate(game.discard_quota) if owed >= 2)
    assert _taking(game, D.CompoundKind.DISCARD, seat)[1]
    # A move at a time: one card leaves the run open, the last closes it.
    game = copy.deepcopy(game)
    owed = game.discard_quota[seat]
    for k in range(owed):
        action = legal_actions(game, seat)[0]
        apply(game, action, seat)
        assert D.closes(game, action, seat, game.trade_event_turn) == (k == owed - 1)


def test_the_snake_seats_two_setup_turns_are_one_run():
    game = _play(2, lambda g: D.is_double_turn(g))
    seat = to_move(game)
    first, closed = _taking(game, D.CompoundKind.SETUP, double_turn=False)
    assert not closed and to_move(first) == seat and first.phase is Phase.SETUP_SETTLEMENT
    assert not D.is_double_turn(first)
    assert _taking(first, D.CompoundKind.SETUP, double_turn=False)[1]
    other = _play(2, lambda g: g.phase is Phase.SETUP_SETTLEMENT and g.setup_step == 1)
    assert _taking(other, D.CompoundKind.SETUP)[1]


def test_a_trade_event_the_seat_may_offer_in_closes_its_run():
    import importlib

    from hexset.game import run_trade_event

    timing = importlib.import_module("test_trade_timing")
    for budget, closes_on_yes in ((None, True), (0, False)):
        actor = timing.Asker([False, False, True])
        if budget is not None:
            actor.max_offers = budget
        game = timing.a_table(actor)
        run_trade_event(game)                                  # entering MAIN: no
        for answer in (False, closes_on_yes):                  # a bank trade: no, then yes
            before = game.trade_event_turn
            trade = timing.a_bank_trade(game)
            apply(game, trade)
            assert D.closes(game, trade, 0, before) == answer
        assert game.trade_event_turn == game.turns
