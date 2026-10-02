# SPDX-License-Identifier: GPL-3.0-only
"""The gate parameters themselves: what they refuse, how they are read off a
checkpoint's metadata, which protocol hooks they ask for, and how they meet the
table's own card cap.

The point of the object is that one gate's limits live in one place and are the
same object whether a search bot or a value head is behind them, so most of
what is worth testing here is refusals and equivalences rather than play.
"""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.game import Phase, start
from hexset.trading import (
    UNLIMITED,
    ENUMERATION_CARDS,
    TradeParams,
    TradeProtocol,
    bundle,
    hooks_for,
    install,
    params_of,
)
from traders import FRAGMENTED
from helpers import give

WOOD, BRICK, SHEEP, WHEAT, ORE = (int(r) for r in Resource)


def a_game(players: int = 4):
    rng = random.Random(0)
    game = start(random_base_board(rng), players, rng)
    game.phase = Phase.MAIN
    game.current_player = 0
    return game


class Eager:
    """A valuer that wants everything: enough to answer the protocol, without
    a model behind it."""

    def gains_many(self, view, received, counterparties):
        return [1.0] * len(received)

    def estimate_many(self, view, candidates):
        return [1.0] * len(candidates)


def a_gate(params: TradeParams) -> Eager:
    gate = Eager()
    gate.trade_floor = params.trade_floor
    gate.trade_params = params
    gate.protocol = install(gate, TradeProtocol(gate, params, seed=1))
    return gate


# -- what a parameter set refuses -------------------------------------------


@pytest.mark.parametrize("changes", [{"trade_floor": -0.1}, {"max_fragments": 3}])
def test_an_incoherent_parameter_set_is_refused_at_construction(changes):
    with pytest.raises(ValueError):
        replace(UNLIMITED, **changes)


def test_a_fragment_cannot_be_wider_than_what_the_gate_will_move():
    """A plan whose pieces the gate would itself refuse never reaches the
    table, so the contradiction is caught where it is written."""
    with pytest.raises(ValueError, match="widest side"):
        replace(FRAGMENTED, fragment_cards=3, max_give_cards=2)


def test_a_gate_declaring_nothing_has_no_limits_of_its_own():
    """Not "the standard three": a gate that declares no cap is bound only by
    the table it sits at, so a looser table is not silently narrowed back to
    three by a default nobody chose."""
    assert UNLIMITED == TradeParams()
    assert UNLIMITED.max_give_cards is None
    assert UNLIMITED.max_offers is None and UNLIMITED.fragment_cards is None
    assert UNLIMITED.responder_card_risk == 0.0 and not UNLIMITED.fragment_trades
    # No declared limit: it keeps offering while it has an offer it has not
    # already made this turn, which is what `-1` means to the driver.
    assert UNLIMITED.trade_offer_budget == -1
    assert UNLIMITED.trades


# -- which hooks a parameter set asks for -----------------------------------


def test_a_gate_that_constrains_nothing_installs_no_hooks_at_all():
    """The whole back-compatibility argument: a neutral parameter set is
    answered by the engine's own defaults, so carrying parameters is not a
    behaviour change in disguise."""
    assert hooks_for(UNLIMITED) == frozenset()
    gate = a_gate(UNLIMITED)
    for hook in ("candidates", "offer", "respond", "respond_any", "pick", "consent_gain"):
        assert getattr(gate, hook, None) is None, hook


def test_planning_asks_for_every_hook():
    assert hooks_for(FRAGMENTED) == frozenset(
        {"candidates", "offer", "respond", "respond_any", "pick", "consent_gain",
         "allow_repeated_offer", "trade_round_finished"}
    )


# -- the caps bind in both directions ---------------------------------------


def test_the_give_cap_refuses_an_exchange_whoever_asks():
    game = a_game()
    for _ in range(3):
        give(game._state, 0, WOOD)
    gate = a_gate(replace(UNLIMITED, max_give_cards=1))
    view = game.state(0)
    one_out = bundle(brick=1, wood=-1)
    two_out = bundle(brick=1, wood=-2)
    assert gate.consent_gain(view, one_out, 1, role="actor") == 1.0
    assert gate.consent_gain(view, two_out, 1, role="actor") == -1.0
    assert gate.consent_gain(view, two_out, 1, role="responder") == -1.0


# -- the engine enumerates at the gate's own width -------------------------


def test_enumeration_follows_the_gates_own_caps_not_a_table_rule():
    """There is no table rule to follow. A gate that declares a narrow give
    side is enumerated narrow; one that declares nothing is enumerated at
    `ENUMERATION_CARDS`, which is a search bound and refuses nothing."""
    def width(gate):
        return params_of(gate).enumeration_cards

    assert width(a_gate(UNLIMITED)) == ENUMERATION_CARDS
    assert width(a_gate(replace(UNLIMITED, max_give_cards=1))) == ENUMERATION_CARDS
    wide = a_gate(replace(UNLIMITED, max_give_cards=5))
    assert width(wide) == 5, "a gate that will move five is offered five"


# -- reading a checkpoint's metadata ----------------------------------------


def test_a_checkpoint_that_says_nothing_bargains_the_way_it_always_did():
    assert TradeParams.from_meta({}) == UNLIMITED
    assert TradeParams.from_meta({"unrelated": "17"}) == UNLIMITED


def test_a_checkpoint_can_ask_for_the_fragmented_policy_by_name():
    assert TradeParams.from_meta(FRAGMENTED.as_meta()) == FRAGMENTED


def test_an_explicit_zero_is_honoured_where_zero_is_meaningful():
    """`gate_plies=0` is a real setting -- one forward over the exchanged
    hand -- so it must not be mistaken for an absent key."""
    base = replace(UNLIMITED, gate_plies=8)
    assert TradeParams.from_meta({"gate_plies": "0"}, base=base).gate_plies == 0
    assert TradeParams.from_meta({}, base=base).gate_plies == 8


def test_a_gate_written_against_the_old_attributes_still_describes_itself():
    class Old:
        trade_floor = 0.0197
        gate_plies = 4

    read = params_of(Old())
    assert read.trade_floor == pytest.approx(0.0197)
    assert read.gate_plies == 4
    assert read.max_offers is None and not read.fragment_trades


def test_a_fragment_is_no_wider_than_the_gate_will_move_when_it_declares_no_width():
    """`fragment_cards=None` is read at `max_give_cards`, so a gate that
    parts with two never plans a three-card fragment."""
    two = replace(FRAGMENTED, fragment_cards=None, max_give_cards=2)
    assert two.fragment_width == 2
    assert TradeProtocol(Eager(), two).fragment_cap() == 2
    assert replace(two, max_give_cards=None).fragment_width == ENUMERATION_CARDS
    assert replace(two, fragment_cards=1).fragment_width == 1


def test_install_keeps_the_protocol_so_a_retune_reinstalls_it():
    """A gate need not keep its protocol itself: `retune` finds the one
    `install` bound, and a change that drops every limit takes every hook
    back."""
    from hexset.trading import DeclaredTrade, retune

    class Mine(DeclaredTrade, Eager):
        def __init__(self, trade):
            self.trade = trade
            install(self, TradeProtocol(self, trade, seed=0))

    gate = Mine(replace(UNLIMITED, max_give_cards=2))
    assert gate.respond.__self__.params.max_give_cards == 2
    retune(gate, max_give_cards=None)
    assert getattr(gate, "respond", None) is None


def test_a_planned_target_breaks_a_tie_towards_the_smaller_exchange():
    from hexset.trading import choose_initial

    scored = [(1, (1, -2, 0, 0, 0), 1.0, 1.0), (1, (1, -1, 0, 0, 0), 1.0, 1.0)]
    chosen = choose_initial(scored, cutoff=0.0, draw_key=(0, 0, 0, 0), max_cards=2, max_fragments=1)
    assert chosen.bundle == (1, -1, 0, 0, 0)


def test_a_plain_gates_loose_offer_budget_is_its_own():
    class Loose:
        trade_floor = 0.0
        trade_offer_budget = 2

    assert params_of(Loose()).max_offers == 2
    Loose.trade_offer_budget = -1
    assert params_of(Loose()).max_offers is None
