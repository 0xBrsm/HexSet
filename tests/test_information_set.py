# SPDX-License-Identifier: GPL-3.0-only
"""The information-set gate: no bot the arena can seat may read a card it
cannot see.

Two probes, run against every seatable bot at a main-phase, a discard and a
trade-response decision:

  * the permutation probe -- redeal the *other* seats' hidden resource and
    development cards among themselves, composition-preserving, so the acting
    seat's `View` is bit-identical, and require the decision not to move;
  * the `observed_by` probe -- replace those piles with `state.Hidden` counts,
    so any identity read raises `state.HiddenRead`.

A composition-preserving redeal is exactly the move a correct information-set
policy cannot detect, so a bot that moves has read through its own belief.
`test_the_probe_can_fail` is the control: the same probe against a tree rooted
on the true state, which is the defect this file was written for.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pytest

from hexset.actions import ActionSpace, build_space, options_for
from hexset.board.board import random_base_board
from hexset.devcards import dev_count
from hexset.economy import hand_size
from hexset.game import Phase, imagine, is_over, start, to_move
from hexset.mcts import Leaf, Search
from hexset.play import step_randomly
from hexset.state import HiddenRead, copy_state, observed_by
from hexset.trading import ENUMERATION_CARDS, UNLIMITED, TradeParams
from hexset.trading._engine import _belief_candidates
from hexset.view import View

PLAYERS = 4
SIMULATIONS = 16
WAVE = 4
# The sharp probe's budget: enough descents per edge that, rooted on the
# truth, its most-visited move follows the hands it reads (the control,
# `test_the_probe_can_fail`); a wave's virtual losses even out a smaller one.
SHARP = dict(simulations=256, wave=8)


# --------------------------------------------------------------------------
# positions


def positions(seed=5, wanted=4, players=PLAYERS, phase=Phase.MAIN, devs_needed=1):
    """Decisions where the acting seat has several real options, several
    untyped opponent cards, and (by default) opponents holding dev cards --
    everything the probe needs something to permute."""
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    found = []
    for _ in range(3000):
        if is_over(game) or len(found) == wanted:
            break
        seat = to_move(game)
        if game.phase is phase and len(options_for(game)) > 3:
            view = View.from_game(game, seat)
            # true state: the probe is a test of what a bot may read, so it
            # reads the truth to decide whether there is anything to permute.
            state = game.state(0, hidden=False)
            devs = sum(dev_count(state, s) for s in range(players) if s != seat)
            if sum(view.unknown) >= 4 and devs >= devs_needed:
                found.append((imagine(game, random.Random(1), randomize_deck=False), seat))
        step_randomly(game, rng)
    assert len(found) == wanted, f"only {len(found)} usable positions"
    return found


def permuted(game, seat, rng):
    """`game` with every other seat's hidden cards redealt among them, each
    hand's size and each seat's dev-card count preserved."""
    twin = imagine(game, random.Random(1), randomize_deck=False)
    state = copy_state(game.state(0, hidden=False))
    seats = [s for s in range(state.num_players) if s != seat]

    def redeal(piles):
        pool = [c for s in seats for c, n in enumerate(piles[s]) for _ in range(n)]
        rng.shuffle(pool)
        cursor = 0
        for s in seats:
            size = sum(piles[s])
            fresh = [0] * len(piles[s])
            for card in pool[cursor : cursor + size]:
                fresh[card] += 1
            cursor += size
            piles[s] = fresh

    redeal(state.hands)
    redeal(state.dev_cards)
    twin.set_state(state)
    return twin


# --------------------------------------------------------------------------
# stubs


@dataclass
class PublicPolicy:
    """A `Policy` that reads nothing hidden: each seat is worth its hand size
    and its dev-card count, both public, so a one-forward bot built on it is
    honest by construction and any movement under the probe is the caller's.
    """

    space: ActionSpace

    def act_rows(self, rows):
        return [min(options, key=self.space.index) for _, _, options in rows]

    def value_rows(self, rows):
        return [self._value(game) for game, _ in rows]

    def score_rows(self, rows):
        return [
            (np.full(len(options), 1.0 / len(options)), self._value(game))
            for game, _, options in rows
        ]

    def _value(self, game):
        # true state: sizes and counts only, which `View` reports identically.
        state = game.state(0, hidden=False)
        return tuple(
            0.01 * hand_size(state, seat) + 0.02 * dev_count(state, seat)
            for seat in range(state.num_players)
        )


@dataclass(frozen=True)
class StubCheckpoint:
    policy: PublicPolicy
    space: ActionSpace
    players: int = PLAYERS
    trade_floor: float = UNLIMITED.trade_floor
    gate_plies: int = UNLIMITED.gate_plies
    trade_params: TradeParams | None = None


def stub_checkpoint(board) -> StubCheckpoint:
    space = build_space(
        board.topology.num_vertices,
        board.topology.num_edges,
        board.topology.num_hexes,
        PLAYERS,
    )
    return StubCheckpoint(policy=PublicPolicy(space=space), space=space)


class HandReader:
    """An evaluator that reads the *composition* of every hand in the leaf it
    is given. Honest inside a determinized tree, where those hands are a
    sample; a direct read of the truth in a tree rooted on it. That is what
    makes it the control's instrument."""

    def evaluate(self, leaves):
        return [(np.full(len(leaf.options), 1.0 / len(leaf.options)), self._value(leaf.game))
                for leaf in leaves]

    def terminal(self, game):
        return self._value(game)

    @staticmethod
    def _value(game):
        # true state: of a hypothetical leaf, which in a determinized tree is
        # a sampled world.
        state = game.state(0, hidden=False)
        return tuple(
            sum((-1) ** r * (r + 1) * n for r, n in enumerate(state.hands[seat])) / 50.0
            for seat in range(state.num_players)
        )


# --------------------------------------------------------------------------
# the bots the arena can seat


def mcts_bot(board):
    """The arena's `kind="mcts"` entrant: `GatedSearch` over a checkpoint,
    exactly as `netbot.spawn_mcts` builds it, on a stub policy so the gate is
    cheap and its arithmetic is not the subject."""
    from hexset.clients.netbot import searcher_for

    return searcher_for(
        stub_checkpoint(board),
        simulations=SIMULATIONS,
        wave=WAVE,
        rng=random.Random(0),
    )


def network_bot(board):
    """The one-forward entrant from the same checkpoint."""
    from hexset.clients.netbot import bot_for

    return bot_for(stub_checkpoint(board), rng=random.Random(0))


def catanatron_bot(board):
    # `catanatron.game`, not `catanatron`: `tests/catanatron/` has no
    # `__init__.py`, and pytest puts `tests/` first on `sys.path`, so with the
    # real package absent the bare name resolves to that directory as a
    # namespace package and the guard passes. Every sibling guard in
    # `tests/catanatron/` already probes the submodule; this one did not.
    pytest.importorskip(
        "catanatron.game", reason="the catanatron extra is not installed"
    )
    from hexset.catanatron.bot import CatanatronBot, alpha_beta

    try:
        player = alpha_beta(2)
    except AttributeError as error:  # pragma: no cover - depends on the pin
        pytest.skip(f"the installed catanatron is not the pinned revision: {error}")
    # `worlds>0` is the information-set read; `worlds=0`, the entrant's own
    # default, is the omniscient reference baseline and is not gated here.
    return CatanatronBot(player, rng=random.Random(0), worlds=4)


def mcts_tree(board):
    """The same tree over `HandReader`, which reads the composition of every
    hand in the leaf it is handed. Nothing but the determinized root keeps this
    arm honest, so it is the sharp probe for the tree itself -- `mcts` above
    can only move if the checkpoint's own priors and values move with it."""
    return Search(HandReader(), **SHARP, rng=random.Random(0))


BOTS = {
    "mcts": mcts_bot,
    "mcts-tree": mcts_tree,
    "network": network_bot,
    "catanatron": catanatron_bot,
}


def build(name, board):
    bot = BOTS[name](board)
    if name == "catanatron":
        # The pinned API is only exercised once a decision is asked for.
        try:
            bot.choose(imagine_free(board))
        except AttributeError as error:  # pragma: no cover - depends on the pin
            pytest.skip(f"the installed catanatron is not the pinned revision: {error}")
    return bot


def imagine_free(board):
    return start(board, PLAYERS, random.Random(3))


# --------------------------------------------------------------------------
# the gate


# Every seatable bot at a main-phase decision; the discard probe on the sharp
# tree only, which keeps the gate fast.
_CHOICE_PROBES = [
    pytest.param(name, Phase.MAIN, 1, id=f"main-{name}") for name in sorted(BOTS)
] + [
    pytest.param(name, Phase.DISCARD, 0, id=f"discard-{name}")
    for name in ("mcts-tree",)
]


@pytest.mark.parametrize("name,phase,devs", _CHOICE_PROBES)
def test_permuting_hidden_cards_does_not_move_the_choice(name, phase, devs):
    board_of = {}
    moved = []
    for truth, seat in positions(phase=phase, wanted=3, devs_needed=devs):
        board = truth.state(0, hidden=False).board
        board_of[id(board)] = board
        twin = permuted(truth, seat, random.Random(11))
        assert View.from_game(truth, seat) == View.from_game(twin, seat)
        here = build(name, board).choose(truth)
        there = build(name, board).choose(twin)
        if here != there:
            moved.append((seat, here, there))
    assert not moved, (
        f"{name}: {len(moved)} {phase.name} decisions moved when only the other "
        f"seats' hidden cards were permuted: {moved}"
    )


@pytest.mark.parametrize("name", sorted(BOTS))
def test_permuting_hidden_cards_does_not_move_a_trade_response(name):
    moved = []
    for truth, seat in positions(wanted=2):
        board = truth.state(0, hidden=False).board
        here, there = build(name, board), build(name, board)
        if not hasattr(here, "accepts"):
            pytest.skip(f"{name} declines every exchange and has no responder gate")
        twin = permuted(truth, seat, random.Random(11))
        others = [s for s in range(truth.num_players) if s != seat]
        candidates = [
            (them, bundle)
            for them in others
            for bundle in _belief_candidates(
                truth.state(seat), seat, them, ENUMERATION_CARDS
            )
        ][:20]
        for them, bundle in candidates:
            if here.accepts(truth.state(seat), bundle, them) != there.accepts(
                twin.state(seat), bundle, them
            ):
                moved.append((seat, them, bundle))
    assert not moved, f"{name}: a trade response moved with the hidden cards: {moved}"


@pytest.mark.parametrize("name", sorted(BOTS))
def test_the_bot_plays_on_an_observed_state(name):
    """The fast half: any read of a pile's identity raises, no permutation
    needed."""
    truth, seat = positions(wanted=1)[0]
    board = truth.state(0, hidden=False).board
    seen = imagine(truth, random.Random(1), randomize_deck=False)
    seen.set_state(observed_by(seen.state(seat, hidden=False), seat))
    try:
        build(name, board).choose(seen)
    except HiddenRead as error:
        pytest.fail(f"{name} read a hidden pile: {error!r}")


def test_the_probe_can_fail():
    """The control. A tree rooted on the true state resolves steals, dev draws
    and opponents' options against cards its seat cannot see, and the probe
    catches it; the same tree over determinized roots does not move."""
    omniscient, determinized = [], []
    for truth, seat in positions(wanted=4):
        twin = permuted(truth, seat, random.Random(11))
        for hidden, moved in ((False, omniscient), (True, determinized)):
            here = Search(HandReader(), **SHARP, hidden=hidden, rng=random.Random(0)).choose(truth)
            there = Search(HandReader(), **SHARP, hidden=hidden, rng=random.Random(0)).choose(twin)
            if here != there:
                moved.append(seat)
    assert omniscient, (
        "the omniscient tree answered every probe identically: this gate cannot "
        "detect the defect it was written for"
    )
    assert not determinized, f"the determinized tree moved at {determinized}"


def test_a_searched_leaf_is_valued_in_a_sampled_world():
    """The leak's other end: whatever an evaluator reads off a leaf, that leaf
    belongs to a world drawn from the mover's belief, so its value cannot be a
    function of the true hands."""
    truth, seat = positions(wanted=1)[0]
    twin = permuted(truth, seat, random.Random(11))
    reader = HandReader()
    values = []
    for game in (truth, twin):
        search = Search(reader, simulations=SIMULATIONS, wave=WAVE, rng=random.Random(0))
        root, _, _ = search.run(game)
        (_, value), = reader.evaluate([Leaf(root.game, root.mover, root.options)])
        values.append(value)
    assert values[0] == values[1]
