# SPDX-License-Identifier: GPL-3.0-only
"""Vote over sampled information-set worlds, evaluating each distinct key once.

The cache lasts one decision. ``worlds`` bounds sampling work; it is not a
quota of distinct worlds. The key must cover everything the callback
observes, and a stochastic chooser answers once per key, so this is not an
ensemble of independent stochastic searches. Callbacks get isolated
hypothetical games and may mutate them, but must not retain one as the live
seat a trade gate reads. No RNG here consumes the real game's stream.
"""

from __future__ import annotations

import math
import operator
import random
from collections import Counter
from collections.abc import Callable, Hashable

import numpy as np

from hexset.actions import Action
from hexset.game import Game, imagine, to_move
from hexset.view import HoldReading
from hexset.mcts import visit_policy
from hexset.state import GameState

__all__ = [
    "WorldKey",
    "Chooser",
    "holdings_signature",
    "world_signature",
    "distinct_worlds",
    "determinized",
]


WorldKey = Callable[[GameState, int], Hashable]
Chooser = Callable[[Game], Action | None]


def holdings_signature(state: GameState, perspective: int) -> tuple:
    """Opponent resource and development holdings; only valid within a root.
    Ignores the sampled deck, so one representative deck is retained per
    holding -- use when the chooser cannot observe the deck."""
    return tuple(
        (tuple(state.hands[seat]), tuple(state.dev_cards[seat]),
         tuple(state.new_dev_cards[seat]))
        for seat in range(state.num_players) if seat != perspective
    )


def world_signature(state: GameState, perspective: int) -> tuple:
    """Every field varied by ``View.sample``, including development deck
    order. Public state and the perspective's own holdings are omitted, so
    this is not a key for caching across decisions."""
    return holdings_signature(state, perspective), tuple(state.deck)


def distinct_worlds(
    belief, rng: random.Random, draws: int, world_key: WorldKey = world_signature,
    hold: HoldReading | None = None,
) -> list[tuple[float, GameState]]:
    """`draws` samples from `belief`, folded to the distinct worlds among
    them, each with its share of the draws. `draws` caps distinct worlds, it
    is not a quota; `world_key` is the one rule for "the same world"; `hold`
    is the caller's reading of the development cards dealt (`View.sample`),
    `None` to deal every unseen card alike."""
    draws = operator.index(draws)
    if draws < 1:
        raise ValueError("at least one draw is needed")
    seat = belief.perspective
    found: dict[Hashable, tuple[GameState, int]] = {}
    for _ in range(draws):
        # A belief with no development cards to read takes no `hold`.
        sampled = belief.sample(rng) if hold is None else belief.sample(rng, hold)
        key = world_key(sampled, seat)
        state, count = found.get(key, (sampled, 0))
        found[key] = (state, count + 1)
    return [(count / draws, state) for state, count in found.values()]


def determinized(
    choose: Chooser, worlds: int, rng: random.Random | None = None, *,
    temperature: float = 0.0, select: str = "argmax",
    world_key: WorldKey = world_signature,
) -> Chooser:
    """Wrap a model/search callback in a frequency-weighted world vote.

    ``worlds=0`` returns ``choose`` unchanged and draws no randomness;
    ``None`` answers abstain, and all-abstain returns ``None``. Temperature
    1 weights actions by vote share, 0 splits mass between tied winners;
    ``argmax`` breaks ties by action order, ``sample`` draws with ``rng``."""
    worlds = operator.index(worlds)
    if worlds < 0:
        raise ValueError("worlds must be nonnegative")
    if select not in ("argmax", "sample"):
        raise ValueError(f"select must be 'argmax' or 'sample', got {select!r}")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if worlds == 0:
        return choose
    rng = rng if rng is not None else random.Random(0)

    def pick(game: Game) -> Action | None:
        seat = to_move(game)
        belief = game.state(seat)
        # One seed per decision, independent of cache misses and of chance
        # draws made inside the callback.
        search_rng = random.Random(rng.getrandbits(128))
        votes: Counter[Action] = Counter()
        for share, sampled in distinct_worlds(belief, rng, worlds, world_key):
            world = imagine(game, search_rng, randomize_deck=False)
            world.set_state(sampled)
            action = choose(world)
            if action is not None:
                votes[action] += share
        if not votes:
            return None
        actions = sorted(votes)
        # Normalize first: a tiny temperature would otherwise overflow
        # `visit_policy`'s count**(1/temperature).
        counts = np.array([votes[action] for action in actions], dtype=float)
        policy = visit_policy(counts / counts.max(), temperature)
        if select == "sample":
            return rng.choices(actions, weights=policy)[0]
        return actions[int(np.argmax(policy))]

    return pick
