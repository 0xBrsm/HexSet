"""A host's refused moves at a position (`Game.barred`, `Seat.bar`): left out of
`legal_actions` so a bot asked again gives its next best, for that position only."""

import random
import unittest

from hexset.actions import apply, legal_actions
from hexset.game import imagine, start
from hexset.board.board import base_board


def _setup_game(seed: int = 3):
    return start(base_board("random", random.Random(seed)), 2, random.Random(seed + 1))


class Barred(unittest.TestCase):
    def test_a_barred_move_is_left_out_until_the_next_action(self):
        game = _setup_game()                 # the first settlement: one move per open corner
        moves = legal_actions(game)
        refused = moves[0]
        game.barred = frozenset({refused})
        self.assertNotIn(refused, legal_actions(game))
        self.assertEqual(len(legal_actions(game)), len(moves) - 1)
        self.assertNotIn(refused, legal_actions(imagine(game, random.Random(0))))   # a search's root copy
        apply(game, next(a for a in legal_actions(game)))
        self.assertEqual(game.barred, frozenset())

    def test_a_seat_bars_moves_on_its_own_game(self):
        from hexset.seat import Seat
        board = base_board("random", random.Random(5))
        first_legal = type("FirstLegal", (), {"choose": lambda self, game: legal_actions(game)[0]})()
        seat = Seat.sit(board, 2, 0, first_legal)
        first = legal_actions(seat.game)
        seat.bar(first[:1])
        self.assertEqual(seat.choose(), first[1])


def _discarding():
    """A four-seat game whose seat 0 owes a discard of 5 from two of each resource."""
    from hexset.game import Phase, roll_dice

    game = start(base_board("random", random.Random(3)), 4, random.Random(4))
    while game.phase is not Phase.ROLL:
        apply(game, legal_actions(game)[0])
    game._state.hands[0] = [2] * 5
    roll_dice(game, roll=7)
    assert game.phase is Phase.DISCARD and game.discard_quota[0] == 5
    return game


class BarredWholeDecisions(unittest.TestCase):
    """A whole decision opened by a barred move is left out of `options` and
    `tensors`, as `legal_actions` leaves the move out, so a graph that chooses
    whole decisions never opens one with a move the host refused."""

    def test_a_barred_setup_settlement_opens_no_setup_turn(self):
        from hexset import decisions as D
        from hexset.actions import space_for

        game = _setup_game()
        refused = legal_actions(game)[0]
        before = D.options(game, double_turn=False)
        game.barred = frozenset({refused})
        after = D.options(game, double_turn=False)
        self.assertTrue(after)
        self.assertEqual(after, [o for o in before if o.parts[0] != refused])
        pairs = D.incidences(game._state.board.topology)
        setup = D.tensors(game, 0, space_for(game), double_turn=False).setup
        self.assertFalse(any(setup[i] for i, (v, _) in enumerate(pairs) if v == refused.a))
        self.assertEqual(int(setup.sum()), len(after))

    def test_a_barred_first_card_opens_no_discard(self):
        from hexset import decisions as D
        from hexset.actions import space_for

        game = _discarding()
        refused = legal_actions(game, 0)[0]
        before = D.options(game, 0)
        game.barred = frozenset({refused})
        after = D.options(game, 0)
        self.assertTrue(after)
        self.assertEqual(after, [o for o in before if o.parts[0] != refused])
        self.assertEqual(len(D.tensors(game, 0, space_for(game)).discard), len(after))

    def _asked_again(self, game, seat):
        """The graph's move at `game`, then its move there once that one is barred."""
        try:
            from hexset.clients.onnxbot import DecisionPolicy, load
        except ImportError:
            self.skipTest("hexset.clients.onnxbot needs onnxruntime installed")
        from pathlib import Path

        stub = Path(__file__).parent / "clients" / "fixtures" / "stub-contract9.onnx"
        loaded = load(str(stub), game._state.board.topology).policy
        policy = DecisionPolicy(loaded.session, loaded.space, loaded.topology, loaded.players)
        (first,) = policy.act_rows([(game, seat, tuple(legal_actions(game, seat)))])
        game.barred = frozenset({first})
        options = tuple(legal_actions(game, seat))
        (second,) = policy.act_rows([(game, seat, options)])
        self.assertIn(second, options)
        self.assertNotEqual(second, first)
        # A decision taken afresh, not the next part of the plan the refused move opened.
        fresh = DecisionPolicy(loaded.session, loaded.space, loaded.topology, loaded.players)
        self.assertEqual(fresh.act_rows([(game, seat, options)]), [second])

    def test_a_decision_graph_asked_again_after_a_barred_setup_turn_gives_its_next_best(self):
        """The setup mask leaves the barred settlement out, so the graph cannot
        open its decision with it again (`Plans` would raise)."""
        self._asked_again(start(base_board("random", random.Random(3)), 4, random.Random(4)), 0)

    def test_a_decision_graph_asked_again_after_a_barred_discard_gives_its_next_best(self):
        self._asked_again(_discarding(), 0)

if __name__ == "__main__":
    unittest.main()
