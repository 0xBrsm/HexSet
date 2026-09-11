import random
from types import SimpleNamespace

from hexset.actions import Action, ActionType
from hexset.board.board import random_base_board
from hexset.bots.heximax import HonestEvaluator
from hexset.bots.heximax.bank_quiescence import BankTradeQuiescence
from hexset.game import Phase, start


class TraceExtension(BankTradeQuiescence):
    def __post_init__(self):
        super().__post_init__()
        self.trace = []

    def _plain_child(self, game, action):
        self.trace.append(("child", action.type))
        return game.child

    def _options_in(self, game, knower):
        self.trace.append(("options", knower))
        return game.options

    def _best_of(self, game, options, depth, mover, knower, ply):
        self.trace.append(("best", depth, ply, len(options)))
        return [1.0] * game.num_players

    def _leaf(self, game, knower):
        self.trace.append(("leaf", knower))
        self._spent += 1
        return [0.0] * game.num_players


def make_game():
    game = start(random_base_board(random.Random(3)), 2, random.Random(3))
    game.phase = Phase.MAIN
    return game


def test_bank_trade_extension_adds_one_post_trade_action_and_resets():
    game = make_game()
    child = SimpleNamespace(num_players=2, current_player=0, options=[Action(ActionType.BUILD_ROAD)])
    game.child = child
    bot = TraceExtension(HonestEvaluator(game._state.board), max_nodes=600, width=6)
    result = bot._after(game, Action(ActionType.BANK_TRADE, 0, 1), 1, 0, 2)
    assert result == [1.0, 1.0]
    assert bot.trace == [("child", ActionType.BANK_TRADE), ("options", 0), ("best", 1, 3, 1)]
    assert bot._bank_quiescence_used is False
    bot.trace.clear()
    bot._after(game, Action(ActionType.BANK_TRADE, 0, 1), 1, 0, 2)
    assert ("best", 1, 3, 1) in bot.trace


def test_bank_trade_extension_preserves_budget_and_ordinary_leaf():
    game = make_game()
    child = SimpleNamespace(num_players=2, current_player=0, options=[Action(ActionType.BANK_TRADE, 0, 1)])
    game.child = child
    bot = TraceExtension(HonestEvaluator(game._state.board), max_nodes=1, width=6)
    bot._after(game, Action(ActionType.BANK_TRADE, 0, 1), 1, 0, 0)
    assert bot.nodes == 0  # fake best-of does not consume leaves
    # A non-bank horizon action follows the ordinary leaf path.
    bot._after(game, Action(ActionType.BUILD_ROAD, 0), 1, 0, 0)
    assert bot.nodes == 1


def test_opt_in_presets_spawn_subclass_without_changing_baseline():
    import hexset.bots.heximax.bank_quiescence  # registration side effect
    from hexset.arena import PRESETS, spawn
    from hexset.bots.heximax.search import Heximax
    from hexset.bots.heximax import heximax
    board_game = make_game()
    extended = spawn(PRESETS["heximax-bankext"], board_game._state.board, random.Random(1))
    wide = spawn(PRESETS["heximax-bankext-wide"], board_game._state.board, random.Random(1))
    baseline = heximax(board_game._state.board, random.Random(1), mode="notrade", max_nodes=600, width=6)
    assert isinstance(extended, BankTradeQuiescence)
    assert isinstance(wide, BankTradeQuiescence)
    assert isinstance(baseline, Heximax) and type(baseline) is Heximax
    assert (extended.max_nodes, extended.width) == (600, 6)
    assert (wide.max_nodes, wide.width) == (2400, 12)


def test_actual_engine_two_bank_trades_then_purchase_conserve_hand_and_bank():
    from hexset.actions import apply, legal_actions
    from hexset.economy import Purchase
    game = make_game()
    state = game._state
    # Give seat 0 exactly enough for two 4:1 bank trades, then a dev card.
    for r, n in enumerate(state.hands[0]):
        state.bank[r] += n
        state.hands[0][r] = 0
    hand = [8, 0, 0, 1, 1]
    for r, n in enumerate(hand):
        state.bank[r] -= n
        state.hands[0][r] = n
    before_total = sum(state.hands[0]) + sum(state.bank)
    trade = next(a for a in legal_actions(game) if a.type.name == "BANK_TRADE" and a.a == 0 and a.b == 2)
    apply(game, trade)
    trade2 = next(a for a in legal_actions(game) if a.type.name == "BANK_TRADE" and a.a == 0 and a.b == 2)
    apply(game, trade2)
    assert state.hands[0] == [0, 0, 2, 1, 1]
    assert sum(state.hands[0]) + sum(state.bank) == before_total
    assert any(a.type.name == "BUY_DEV_CARD" for a in legal_actions(game))
    buy = next(a for a in legal_actions(game) if a.type.name == "BUY_DEV_CARD")
    apply(game, buy)
    assert sum(state.hands[0]) + sum(state.bank) + len(state.deck) + sum(state.dev_cards[0]) == before_total + len(state.deck) + sum(state.dev_cards[0])


def test_public_ledger_whitelist_masks_unknown_results_and_preserves_bounds():
    from hexset.catanatron.public_ledger import PublicResourceLedger, observe_public_action
    book = PublicResourceLedger.new(2)
    book.initialise([[2, 0, 0, 0, 0], [0, 0, 0, 0, 2]], object())
    observe_public_action(book, seat=0, action_type="BUILD_ROAD", value=object(), hand_sizes=[1, 2])
    assert book.ledger.seats[0].known == [0, 0, 0, 0, 0]
    assert book.ledger.seats[0].unknown == 0
    observe_public_action(book, seat=1, action_type="MOVE_ROBBER", value={"stolen": "ORE"}, hand_sizes=[1, 1])
    assert book.ledger.seats[1].known == [0, 0, 0, 0, 0]
    assert book.ledger.seats[1].unknown == 1


def test_injected_public_ledger_changes_view_knowledge():
    from hexset.catanatron.public_ledger import PublicResourceLedger
    from hexset.view import View
    from hexset.ledger import SeatLedger, PublicLedger
    game = make_game()
    game._state.hands[1] = [2, 0, 0, 0, 0]
    book = PublicResourceLedger(PublicLedger([SeatLedger(unknown=0), SeatLedger(known=[1,0,0,0,0], unknown=1)]))
    view = View(game._state, book.snapshot(), 0)
    assert view.known[1][0] == 1 and view.unknown[1] == 1
