# SPDX-License-Identifier: GPL-3.0-only
"""The honesty permutation test (`docs/gym-design.md` §4, a merge
requirement, not optional): redeal every opponent's hidden hand and shuffle
the unrevealed deck in the *true* state, composition-preserving, and assert
`HexSetAEC.observe`'s observation does not move -- adapted from
`tests/catanatron/test_catanatron_information_set.py`'s audit, for the
native engine instead of a foreign one, driven through `Game.set_state`
(`docs/gym-design.md`'s own instruction) rather than the private `_state`
field.

The control (`test_the_audit_can_fail`) moves something the perspective seat
*can* see -- its own hand -- and asserts the observation does move, so a test
that permutes nothing reachable cannot pass by accident.

**No skip.** This test carried one, for a genuine leak in
`encoding._offer_parts`: its "who has answered" block re-derived responder
eligibility from the *live* true state rather than from a fact frozen at
proposal time, so from a standing offer's proposer the observation moved
under a counterfactual redeal. The offer block is gone with the offer
protocol (`hexset.trading`): a seat trades by answering a private gate, and
nothing about that gate rides in the observation at all any more -- so the
leak is closed by deletion, not by patching, and the exemption with it.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from hexset.state import copy_state  # noqa: E402

from hexset.gym.aec import HexSetAEC, agent_name  # noqa: E402

from gym_helpers import play_randomly  # noqa: E402


def _redeal(hands: list[list[int]], seats: list[int], rng: random.Random) -> bool:
    """Pool every card the given seats hold and re-deal it among them,
    preserving each seat's hand *size* -- the one thing about an opponent's
    hand `hexset.encoding.encode` ever reads (a count; composition comes only
    from `hexset.ledger`'s own bookkeeping, untouched here). Returns whether
    anything actually changed (a redeal can permute to itself)."""
    pool: list[int] = []
    totals: dict[int, int] = {}
    before = {s: hands[s][:] for s in seats}
    for s in seats:
        totals[s] = sum(hands[s])
        pool.extend(r for r, count in enumerate(hands[s]) for _ in range(count))
    rng.shuffle(pool)
    cursor = 0
    for s in seats:
        dealt = pool[cursor : cursor + totals[s]]
        cursor += totals[s]
        new_hand = [0] * len(hands[s])
        for r in dealt:
            new_hand[r] += 1
        hands[s] = new_hand
    return any(hands[s] != before[s] for s in seats)


def _play_to_tick(seed: int, ticks: int) -> HexSetAEC:
    env = HexSetAEC()
    env.reset(seed=seed)
    play_randomly(env, seed=seed * 7919 + 3, max_steps=ticks)
    return env


def _identical(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return all(np.array_equal(a[key], b[key]) for key in ("hexes", "vertices", "edges", "globals"))


def _first_difference(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> str:
    for key in ("hexes", "vertices", "edges", "globals"):
        if not np.array_equal(a[key], b[key]):
            where = np.argwhere(a[key] != b[key])
            index = tuple(where[0])
            return f"{key}{list(index)}: {a[key][index]} -> {b[key][index]}"
    return "no difference"


@pytest.mark.parametrize("perspective", range(2))
@pytest.mark.parametrize("seed", range(2))
def test_opponent_hands_and_deck_do_not_reach_the_observation(seed, perspective):
    env = _play_to_tick(seed, ticks=80)
    if not env.agents:
        pytest.skip("episode ended before reaching a stable mid-game position")

    game = env._game

    agent = agent_name(perspective)
    before = env.observe(agent)["observation"]

    state = copy_state(game.state(0, hidden=False))
    opponents = [s for s in range(env.num_players) if s != perspective]
    rng = random.Random(seed * 104729 + perspective)

    moved = _redeal(state.hands, opponents, rng)
    deck_before = list(state.deck)
    rng.shuffle(state.deck)
    moved |= list(state.deck) != deck_before
    assert sorted(state.deck) == sorted(deck_before), "the shuffle must not change deck composition"

    game.set_state(state)

    if not moved:
        pytest.skip("nothing hidden was actually permutable in this position")

    after = env.observe(agent)["observation"]
    assert _identical(before, after), (
        f"hidden state reached seat {perspective}'s observation: {_first_difference(before, after)}"
    )


@pytest.mark.parametrize("seed", range(2))
def test_the_audit_can_fail(seed):
    """The control: move the perspective seat's own hand, which it may see.
    Without this, a test permuting nothing reachable would pass forever."""
    env = _play_to_tick(seed, ticks=80)
    if not env.agents:
        pytest.skip("episode ended before reaching a stable mid-game position")

    perspective = 0
    agent = agent_name(perspective)
    before = env.observe(agent)["observation"]

    game = env._game
    state = copy_state(game.state(0, hidden=False))
    state.hands[perspective] = [n + 1 for n in state.hands[perspective]]
    game.set_state(state)

    after = env.observe(agent)["observation"]
    assert not _identical(before, after), (
        "the seat's own hand changed and the observation did not -- "
        "the audit above cannot detect anything"
    )
