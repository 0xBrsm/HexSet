# SPDX-License-Identifier: GPL-3.0-only
"""Vote over sampled information-set worlds, evaluating each distinct key once.

The cache lasts one decision: public state, own holdings and the mover stay
fixed. Every draw contributes a vote, including duplicates. ``worlds`` bounds
sampling work as well as chooser calls; it is not a quota of distinct worlds.

The default key includes opponent holdings and the sampled development deck.
``holdings_signature`` is an explicit alternative for models that do not read
the development deck. It retains the first deck drawn for those holdings. A stochastic
chooser likewise supplies one answer per key, not fresh exploration on each
duplicate. This is not an exact replacement for an ensemble of independent
stochastic searches. Choose a key covering everything the callback observes.

Callbacks receive isolated hypothetical games and may mutate them. They should
not retain those games as the live seat used by a trade gate. Sampling and vote
selection use the supplied RNG; hypothetical chance uses a separate stream.
Neither consumes the real game's chance stream.
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
from hexset.mcts import visit_policy
from hexset.state import GameState

WorldKey = Callable[[GameState, int], Hashable]
Chooser = Callable[[Game], Action | None]


def holdings_signature(state: GameState, perspective: int) -> tuple:
    """Opponent resource and development holdings; only valid within a root.

    Ignores the sampled deck, including its composition and order. Use when
    the chooser cannot observe it, or intentionally wants one representative
    deck per holding.
    """
    return tuple(
        (tuple(state.hands[seat]), tuple(state.dev_cards[seat]),
         tuple(state.new_dev_cards[seat]))
        for seat in range(state.num_players) if seat != perspective
    )


def world_signature(state: GameState, perspective: int) -> tuple:
    """Every field varied by ``View.sample``, including development deck order.

    Public state and the perspective's own holdings are constant within a
    decision, so are omitted. This is not a key for caching across decisions.
    """
    return holdings_signature(state, perspective), tuple(state.deck)


def determinized(
    choose: Chooser, worlds: int, rng: random.Random | None = None, *,
    temperature: float = 0.0, select: str = "argmax",
    world_key: WorldKey = world_signature,
) -> Chooser:
    """Wrap a model/search callback in a frequency-weighted world vote.

    Zero returns ``choose`` unchanged, without drawing randomness. A positive
    count samples exactly that many worlds. ``None`` answers are cached but
    abstain; all abstentions return ``None``. Temperature 1 weights actions by
    vote share; 0 splits mass between tied winners. ``argmax`` breaks ties by
    action order; ``sample`` draws from that distribution with ``rng``.
    """
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
        # One seed per decision, independent of the number of cache misses and
        # of any chance draws made inside the callback.
        search_rng = random.Random(rng.getrandbits(128))
        answered: dict[Hashable, Action | None] = {}
        votes: Counter[Action] = Counter()
        for _ in range(worlds):
            sampled = belief.sample(rng)
            key = world_key(sampled, seat)
            if key not in answered:
                world = imagine(game, search_rng, randomize_deck=False)
                world.set_state(sampled)
                answered[key] = choose(world)
            action = answered[key]
            if action is not None:
                votes[action] += 1
        if not votes:
            return None
        actions = sorted(votes)
        # Normalize before exponentiation so even tiny positive temperatures
        # cannot overflow visit_policy's count**(1/temperature).
        counts = np.array([votes[action] for action in actions], dtype=float)
        policy = visit_policy(counts / counts.max(), temperature)
        if select == "sample":
            return rng.choices(actions, weights=policy)[0]
        return actions[int(np.argmax(policy))]

    return pick
