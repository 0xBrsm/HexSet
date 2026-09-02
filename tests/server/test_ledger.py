from __future__ import annotations

import random

from hexset.actions import build_space, within_offer_budget
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.game import Phase, move_robber_to, start, to_move

from hexset.onnx_record import record_from_game
from hexset.server.rules import options_for

from conftest import step_randomly


def _rig_one_card_steal(num_players: int, thief: int, victim: int, resource: int):
    """A game parked in `Phase.ROBBER` with `victim` holding exactly one card
    of `resource` and nobody else holding anything -- so `steal` is forced
    to take that one card, deterministically, whichever resource it is."""
    board = random_base_board(random.Random(0))
    game = start(board, num_players, random.Random(1))
    game.phase = Phase.ROBBER
    for hand in game.state.hands:
        hand[:] = [0] * NUM_RESOURCES
    game.state.hands[victim][resource] = 1
    return game


def test_a_steal_is_identity_independent_in_the_record():
    """Two worlds, identical except which single resource the victim secretly
    holds. A bystander's record must not tell them apart, and the thief's own
    record may only differ in `own_hand` -- the one place the truth is allowed
    to show, since it is the thief's own hand. This is the regression test for
    a ledger that reads the true stolen resource and only sometimes moves a
    specific `known[r]`, which leaks the identity through *which entry
    visibly dropped* (see `hexset.ledger`'s module docstring)."""
    thief, victim, bystander = 0, 1, 2
    world_brick = _rig_one_card_steal(3, thief, victim, resource=0)
    world_wood = _rig_one_card_steal(3, thief, victim, resource=1)

    for game in (world_brick, world_wood):
        move_robber_to(game, target=1, victim=victim)
        assert game.phase is Phase.MAIN
        assert sum(game.state.hands[thief]) == 1
        assert sum(game.state.hands[victim]) == 0

    def record_for(game, seat):
        space = build_space(
            game.state.board.topology.num_vertices,
            game.state.board.topology.num_edges,
            game.state.board.topology.num_hexes,
            game.state.num_players,
        )
        options = tuple(within_offer_budget(game, options_for(game), None))
        return record_from_game(game, seat, space, options)

    # The two mask fields describe the *mover's* own options, and the mover
    # here is the thief. Since this branch serves one honest mask to every
    # seat (`hexset.server.rules`), the offer half of it is a function of the
    # mover's own hand -- which really did receive a different card in each
    # world -- so they differ, legitimately, and only for the thief: no seat
    # but the one on move is ever served a record at all
    # (`api.Tables.record` 409s everyone else), and `state_view` gives a
    # seat that is not on move an empty `legal_actions`.
    MOVERS_OWN = {"action_mask", "pair_mask"}

    # The bystander sees nothing that distinguishes the two worlds.
    bystander_brick = record_for(world_brick, bystander)
    bystander_wood = record_for(world_wood, bystander)
    assert bystander_brick.keys() == bystander_wood.keys()
    for key in bystander_brick:
        if key in MOVERS_OWN:
            continue
        assert (bystander_brick[key] == bystander_wood[key]).all(), key

    # The thief's own record differs only where the truth is theirs to know:
    # their own hand, and the options that hand affords them. Everything else
    # -- including the victim's ledger row -- must still agree.
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


def test_the_ledger_invariant_holds_over_a_random_playout():
    """`sum(known) + unknown == true hand size` and `known[r] <= true[r]` for
    every seat, at every step of a real game -- the two properties
    `PublicLedger.spend`/`.steal` are proved (not merely tested) to
    preserve, provided the ledger started in sync (see their docstrings)."""
    board = random_base_board(random.Random(0))
    rng = random.Random(2)
    game = start(board, 3, rng)

    for _ in range(200):
        if game.won_by is not None:
            break
        step_randomly(game, rng)
        for seat, hand in enumerate(game.state.hands):
            seat_ledger = game.ledger.seats[seat]
            assert sum(seat_ledger.known) + seat_ledger.unknown == sum(hand)
            for resource, true_count in enumerate(hand):
                assert seat_ledger.known[resource] <= true_count


def test_undo_restores_the_ledger_with_the_state():
    """PR #2 defect 2, reproduced at the size the PI's review measured it.

    `_UndoPoint` snapshotted `game.state` and not `game.ledger`, so after an
    undone `BANK_TRADE` the ledger still certified the cards the trade had
    spent and received: a `known[r]` above the seat's true count, and
    `sum(known) + unknown` no longer the hand size. Both are *floors the
    record states to every seat* -- once one is wrong, every later
    `/api/record` and `state_view` carries a falsehood, and `PublicLedger.
    spend`'s clamp hides it rather than raising.

    Four-to-one at the bank rather than a build, because a bank trade moves
    the ledger in both directions at once: four of one resource certified
    away, one of another certified in.
    """
    from hexset.actions import ActionType, legal_actions
    from hexset.server.webplay import GameSession, action_to_wire

    board = random_base_board(random.Random(0))
    game = start(board, 4, random.Random(1))
    game.phase = Phase.MAIN
    game.current_player = 0
    game.state.hands[0] = [8, 0, 0, 0, 0]
    # Certify the hand, the way a distribution would have: the ledger has to
    # start in sync for its invariant to mean anything (see `PublicLedger`).
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
    assert game.ledger.seats[0].known != before_known  # the trade was recorded

    session.undo_last_build(0)

    assert game.state.hands[0] == [8, 0, 0, 0, 0]
    assert game.ledger.seats[0].known == before_known
    assert game.ledger.seats[0].unknown == before_unknown
    # The two invariants, restated directly rather than trusted from equality:
    # a floor that exceeds the truth is the failure mode that reaches the wire.
    for seat in range(game.state.num_players):
        row = game.ledger.seats[seat]
        hand = game.state.hands[seat]
        assert sum(row.known) + row.unknown == sum(hand), seat
        assert all(row.known[r] <= hand[r] for r in range(NUM_RESOURCES)), seat


def test_undo_keeps_the_ledger_honest_across_a_played_game():
    """The same defect, found the way the review found it: play until a paid
    build or bank trade shows up, undo it, and check the invariant -- the
    check `test_the_ledger_invariant_holds_over_a_random_playout` cannot
    make, because it never undoes anything.

    Setup placements are excluded from the undo, and not because they are
    uninteresting: undoing every one of them leaves a random mover picking a
    fresh settlement and undoing that too, forever, and the game never leaves
    setup. They cost no resources anyway, so the ledger has nothing to say
    about them; the paid actions are the ones that move it.
    """
    from hexset.actions import ActionType
    from hexset.server.webplay import _UNDOABLE_BUILDS, GameSession, action_to_wire

    PAID = _UNDOABLE_BUILDS - {ActionType.SETUP_SETTLEMENT, ActionType.SETUP_ROAD}

    board = random_base_board(random.Random(0))
    rng = random.Random(0)
    game = start(board, 4, rng)
    session = GameSession(game=game, claimed_seats=set(range(4)))

    undone = 0
    for _ in range(3000):
        if game.won_by is not None:
            break
        seat = to_move(game)
        session.claimed_seats.add(seat)
        action = rng.choice(options_for(game))
        session.submit(seat, action_to_wire(action))
        if action.type in PAID and session._undo is not None:
            session.undo_last_build(seat)
            undone += 1
            for s in range(game.state.num_players):
                row = game.ledger.seats[s]
                hand = game.state.hands[s]
                assert sum(row.known) + row.unknown == sum(hand), (s, undone, action)
                assert all(
                    row.known[r] <= hand[r] for r in range(NUM_RESOURCES)
                ), (s, undone, action)
    assert undone > 0, "the playout never reached a paid undoable action"
