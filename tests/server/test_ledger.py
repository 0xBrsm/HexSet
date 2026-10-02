from __future__ import annotations

import random

from hexset.actions import build_space
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.game import UNSTRUCTURED_TURN_CAP, Phase, move_robber_to, start, to_move

from hexset.onnx_record import record_from_game
from hexset.actions import options_for


def _rig_one_card_steal(num_players: int, thief: int, victim: int, resource: int):
    board = random_base_board(random.Random(0))
    game = start(board, num_players, random.Random(1),
                 turn_cap=UNSTRUCTURED_TURN_CAP)
    game.phase = Phase.ROBBER
    for hand in game._state.hands:
        hand[:] = [0] * NUM_RESOURCES
    game._state.hands[victim][resource] = 1
    # A robber move to hex 1 must rob a seat with a card there.
    state = game._state
    corner = state.board.topology.hex_vertices[1][0]
    state.vertex_owner[corner] = victim
    state.vertex_building[corner] = 1
    return game


def test_a_steal_is_identity_independent_in_the_record():
    """Two worlds, identical except which resource the victim secretly holds: a
    ledger that moved a specific `known[r]` on a steal would leak the identity
    through *which entry visibly dropped*.
    """
    thief, victim, bystander = 0, 1, 2
    world_brick = _rig_one_card_steal(3, thief, victim, resource=0)
    world_wood = _rig_one_card_steal(3, thief, victim, resource=1)

    for game in (world_brick, world_wood):
        move_robber_to(game, target=1, victim=victim)
        assert game.phase is Phase.MAIN
        assert sum(game._state.hands[thief]) == 1
        assert sum(game._state.hands[victim]) == 0

    def record_for(game, seat):
        space = build_space(
            game._state.board.topology.num_vertices,
            game._state.board.topology.num_edges,
            game._state.board.topology.num_hexes,
            game._state.num_players,
        )
        options = tuple(options_for(game))
        return record_from_game(game, seat, space, options)

    # `action_mask` is the *mover's* own options, and the mover is the thief,
    # whose hand did receive a different card in each world.
    # one on move is ever served a record, so this differs legitimately.
    MOVERS_OWN = {"action_mask"}

    bystander_brick = record_for(world_brick, bystander)
    bystander_wood = record_for(world_wood, bystander)
    assert bystander_brick.keys() == bystander_wood.keys()
    for key in bystander_brick:
        if key in MOVERS_OWN:
            continue
        assert (bystander_brick[key] == bystander_wood[key]).all(), key

    thief_brick = record_for(world_brick, thief)
    thief_wood = record_for(world_wood, thief)
    assert thief_brick.keys() == thief_wood.keys()
    differing = {
        key
        for key in thief_brick
        if not (thief_brick[key] == thief_wood[key]).all()
    }
    assert differing <= {"own_hand"} | MOVERS_OWN, differing
    assert "own_hand" in differing


def test_undo_restores_the_ledger_with_the_state():
    """A ledger left ahead of the state certifies cards the undone action spent,
    and both `known[r] <= hand[r]` and `sum(known) + unknown == hand size` are
    floors the record states to every seat, which `PublicLedger.spend`'s clamp
    hides rather than raises. A bank trade, because it moves the ledger in
    both directions at once.
    """
    from hexset.actions import ActionType, legal_actions
    from hexset.server._webplay import GameSession
    from hexset.server.wire import action_to_wire

    board = random_base_board(random.Random(0))
    game = start(board, 4, random.Random(1), turn_cap=UNSTRUCTURED_TURN_CAP)
    game.phase = Phase.MAIN
    game.current_player = 0
    game._state.hands[0] = [8, 0, 0, 0, 0]
    # The ledger must start in sync for its invariant to mean anything.
    game.ledger.seats[0].known = [8, 0, 0, 0, 0]
    game.ledger.seats[0].unknown = 0

    session = GameSession(game=game, claimed_seats={0})
    before_known = list(game.ledger.seats[0].known)
    before_unknown = game.ledger.seats[0].unknown

    trade = next(
        a
        for a in legal_actions(game)
        if a.type is ActionType.BANK_TRADE and a.a == 0 and a.b == 4
    )
    session.submit(0, action_to_wire(trade))
    assert game.ledger.seats[0].known != before_known

    session.undo_last_build(0)

    assert game._state.hands[0] == [8, 0, 0, 0, 0]
    assert game.ledger.seats[0].known == before_known
    assert game.ledger.seats[0].unknown == before_unknown
    for seat in range(game._state.num_players):
        row = game.ledger.seats[seat]
        hand = game._state.hands[seat]
        assert sum(row.known) + row.unknown == sum(hand), seat
        assert all(row.known[r] <= hand[r] for r in range(NUM_RESOURCES)), seat


def test_undo_keeps_the_ledger_honest_across_a_played_game():
    """Setup placements are excluded: undoing every one leaves a random mover
    placing and undoing forever, and they cost no resources anyway.
    """
    from hexset.actions import ActionType
    from hexset.server._webplay import _UNDOABLE_BUILDS, GameSession
    from hexset.server.wire import action_to_wire

    PAID = _UNDOABLE_BUILDS - {ActionType.SETUP_SETTLEMENT, ActionType.SETUP_ROAD}

    board = random_base_board(random.Random(0))
    rng = random.Random(0)
    game = start(board, 4, rng, turn_cap=UNSTRUCTURED_TURN_CAP)
    session = GameSession(game=game, claimed_seats=set(range(4)))

    undone = 0
    for _ in range(600):
        if game.won_by is not None:
            break
        seat = to_move(game)
        session.claimed_seats.add(seat)
        action = rng.choice(options_for(game))
        session.submit(seat, action_to_wire(action))
        if action.type in PAID and session._undo is not None:
            session.undo_last_build(seat)
            undone += 1
            for s in range(game._state.num_players):
                row = game.ledger.seats[s]
                hand = game._state.hands[s]
                assert sum(row.known) + row.unknown == sum(hand), (s, undone, action)
                assert all(
                    row.known[r] <= hand[r] for r in range(NUM_RESOURCES)
                ), (s, undone, action)
    assert undone > 0, "the playout never reached a paid undoable action"
