# SPDX-License-Identifier: GPL-3.0-only
"""A catanatron `Player` backed by any dev-catan `Bot` entrant.

Registered via catanatron's own extension point (`--code`, see `register.py`)
rather than by forking catanatron's source -- `register_cli_player` is the
documented mechanism for exactly this.

One thing needs handling beyond a straight translate-decide-translate-back:

`to_catanatron` can fail for reasons documented in `actions.py`'s
test suite -- real, confirmed differences between the two engines' rule sets
(a piece cap dev-catan doesn't enforce, a stale flag in catanatron itself) --
which should not crash a benchmark run over a rare position. Falling back to
a uniform-random choice from catanatron's own `playable_actions` keeps the
game playable; `fallbacks` counts how often it happens so the rate is visible
rather than silently absorbed.
"""

from __future__ import annotations

import random
from dataclasses import replace

from hexset.arena import entrant_from_name, spawn

import hexset.bots  # noqa: F401 -- registers the bot presets ("heximax"/
# "heximax-omni"/"heximax-notrade" among them) with `hexset.arena.PRESETS`
# before `entrant_from_name`, below, ever looks one up. Neither this module
# nor `hexset.catanatron.duel` imported it before, so a worker process asking
# for `DC:heximax-notrade` got a bare `KeyError` on the name -- every other
# entry point into an entrant by name (`hexset.bench.duel`, `hexset.server`)
# imports `hexset.bots` itself or imports something that does; this bridge
# was the one that didn't. A module-level import, so it runs once whichever
# way a worker process comes to exist -- forked (inherits the parent's
# already-imported module) or re-exec'd via `_ensure_pythonhashseed_zero`
# (a fresh interpreter re-runs this import from scratch either way).

from catanatron.models.player import Color, Player

from .actions import to_catanatron
from .board import translate_board
from .state import translate

# Stable per-seat integer, independent of Python's (per-process) string hash
# randomization -- `list(Color)` is a fixed, code-defined order (RED, BLUE,
# ORANGE, WHITE), so this is the same across every process and every run.
_SEAT_INDEX = {color: i for i, color in enumerate(Color)}


class DevCatanPlayer(Player):
    """`--players=DC:<entrant>`, e.g. `DC:search2-notrade` or `DC:network:<path>`.

    catanatron's own CLI splits `--players` on every `:` and passes each piece
    as a separate positional argument, so an entrant spec that itself contains
    a colon -- `network:<path>` and `mcts:<path>@N` both do -- arrives here in
    parts rather than whole. Rejoining them is this class's job, not
    catanatron's: `parse_cli_string` has no way to know which colons are
    structural and which belong to the payload.

    Player-to-player trading is forced off (`max_trades=0`) regardless of
    what the entrant spec would otherwise use: catanatron never generates
    `OFFER_TRADE` as a playable action (see `state.py`), so there is nothing
    for a proposal to resolve to.
    """

    def __init__(self, color, *entrant_parts: str):
        super().__init__(color)
        self.entrant_spec = ":".join(entrant_parts) if entrant_parts else "search2-notrade"
        self.fallbacks = 0
        self.decisions = 0
        self._mapping = None
        self._bot = None
        self._rng = None

    def reset_state(self) -> None:
        self._mapping = None
        self._bot = None
        self._rng = None

    def decide(self, game, playable_actions):
        self.decisions += 1
        if self._mapping is None:
            self._mapping = translate_board(game.state.board.map)

        if self._rng is None:
            # Seed deterministically from catanatron's own per-game seed
            # (`Game.seed`, set once in `Game.__init__` -- either the caller's
            # `--seed`-derived value or, if none was given, a value drawn from
            # the (already-seeded) global `random` module) plus this player's
            # seat, so that the same `--seed` reproduces this bot's belief
            # sampling and steal/draw resolution exactly, seat for seat, run
            # for run -- not just catanatron's own dice/deck RNG, which was
            # already reproducible because it lives on the global `random`
            # module that `duel._play_chunk` seeds once per shard.
            #
            # Deliberately avoids `hash()` on anything but ints: `hash(str)`
            # is randomized per-process (`PYTHONHASHSEED`) unless disabled,
            # which would silently reintroduce the same irreproducibility
            # this fix exists to remove, just moved one layer down.
            derived = (game.seed * len(_SEAT_INDEX) + _SEAT_INDEX[self.color]) & 0xFFFFFFFF
            self._rng = random.Random(derived)

        our_game, seats = translate(game, self._mapping, self._rng)

        if self._bot is None:
            entrant = replace(entrant_from_name(self.entrant_spec), max_trades=0)
            self._bot = spawn(entrant, self._mapping.board, self._rng)

        action = self._bot.choose(our_game)

        try:
            their_action = to_catanatron(
                action, our_game, self._mapping, seats, playable_actions
            )
        except ValueError:
            self.fallbacks += 1
            return self._rng.choice(playable_actions)

        return their_action
