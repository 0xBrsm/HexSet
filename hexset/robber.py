# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from .board.terrain import Resource, Terrain
from .chance import UNSEEN, Chance
from .economy import hand_size
from .state import NO_OWNER, OFF_BOARD, GameState, is_hidden, pile_size
from .victory import public_victory_points

__all__ = [
    "FRIENDLY_ROBBER_POINTS",
    "occupants",
    "may_stand",
    "victims",
    "shielded",
    "friendly_hexes",
    "allowed_targets",
    "move_robber",
    "check_target",
    "steal",
    "discard_count",
    "discard",
]


# A friendly-robber table shields a seat at or below this many *public* points
# (victory-point cards stay hidden, so they do not lift the shield).
FRIENDLY_ROBBER_POINTS = 2


def occupants(state: GameState, hex_index: int) -> tuple[int, ...]:
    """Distinct players with a building on the given hex; nobody for
    `OFF_BOARD`, so a caller may ask who sits under the robber before it has
    been placed."""
    if hex_index == OFF_BOARD:
        return ()
    topology = state.board.topology
    found = {
        state.vertex_owner[v]
        for v in topology.hex_vertices[hex_index]
        if state.vertex_owner[v] != NO_OWNER
    }
    return tuple(sorted(found))


def victims(state: GameState, hex_index: int, thief: int) -> tuple[int, ...]:
    """Players the thief may steal from: present on the hex, and holding cards."""
    return tuple(
        p
        for p in occupants(state, hex_index)
        if p != thief and hand_size(state, p) > 0
    )


def shielded(state: GameState, thief: int) -> frozenset[int]:
    """Seats a friendly-robber table protects from `thief`: every other seat
    whose public points are at or below `FRIENDLY_ROBBER_POINTS`. The thief is
    never shielded from itself -- its own hexes stay available."""
    return frozenset(
        p
        for p in range(state.num_players)
        if p != thief and public_victory_points(state, p) <= FRIENDLY_ROBBER_POINTS
    )


def may_stand(state: GameState, hex_index: int) -> bool:
    """Whether the robber may stand on `hex_index`: any terrain hex but the
    one it stands on now. Sea is not terrain; the robber never goes there."""
    return hex_index != state.robber and state.board.terrain[hex_index] is not Terrain.SEA


def friendly_hexes(state: GameState, thief: int) -> frozenset[int] | None:
    """The hexes `thief` may take under the friendly-robber rule, or `None`
    for "the rule restricts nothing here" -- which is also the answer when it
    would leave no legal hex at all, since the phase still has to resolve."""
    protect = shielded(state, thief)
    if not protect:
        return None
    allowed = frozenset(
        h
        for h in range(state.board.num_hexes)
        if may_stand(state, h) and not protect.intersection(occupants(state, h))
    )
    return allowed or None


def allowed_targets(
    state: GameState, thief: int, host: frozenset[int] | None
) -> frozenset[int] | None:
    """The hexes this move may take: a host's own rule wins where it gave one
    (it is mirroring a real table and has already applied that table's rules),
    otherwise the game type's. `None` means the rulebook's "anywhere but where
    it stands"."""
    if host is not None:
        return host
    if state.rules.friendly_robber:
        return friendly_hexes(state, thief)
    return None


def move_robber(state: GameState, target: int) -> None:
    """Put the robber on hex `target`, stealing nothing. Raises `ValueError`
    when there is no such hex, it is sea, or the robber already stands there."""
    check_target(state, target)
    state.robber = target


def check_target(state: GameState, target: int) -> None:
    """Raise `ValueError` unless the robber may move to hex `target`
    (`may_stand`); writes nothing."""
    if not 0 <= target < state.board.num_hexes:
        raise ValueError(f"no such hex: {target}")
    if target == state.robber:
        raise ValueError("the robber must move to a different hex")
    if state.board.terrain[target] is Terrain.SEA:
        raise ValueError(f"hex {target} is sea, where the robber cannot stand")


def steal(
    state: GameState, thief: int, victim: int, chance: Chance
) -> Resource | int | None:
    """Take one card at random, so the chance of each resource follows the
    hand. `chance.steal` decides which; an empty hand consumes no event.

    On an observed state a hidden hand is either side of it: the host says
    which card moved when this seat was party to the steal, and
    `chance.UNSEEN` when it was not -- then only the two counts move, and
    `UNSEEN` is what this returns. The identity is private to the two seats
    either way, so no public flow is recorded (`ledger.PublicLedger.steal`)."""
    hand = state.hands[victim]
    taker = state.hands[thief]
    if is_hidden(hand) or is_hidden(taker):
        if pile_size(hand) == 0:
            return None
        resource = chance.steal(hand)
        if resource is None:
            raise ValueError(f"seat {victim} holds cards, so a steal takes one")
        if is_hidden(hand):
            hand.add(-1)
        else:
            hand[resource] -= 1
        if is_hidden(taker):
            taker.add(1)
        else:
            taker[resource] += 1
        return resource if resource == UNSEEN else Resource(resource)
    resource = chance.steal(hand)
    if resource is None:
        return None
    hand[resource] -= 1
    taker[resource] += 1
    return Resource(resource)


def discard_count(state: GameState, player: int) -> int:
    """How many cards a player must discard when a seven is rolled: half, for
    a hand larger than the game type's discard limit."""
    held = hand_size(state, player)
    return held // 2 if held > state.rules.discard_limit else 0


def discard(
    state: GameState, player: int, cards: list[int], required: int | None = None
) -> None:
    """Return `cards`, counted per resource, from `player`'s hand to the bank.
    Raises `ValueError` unless `cards` totals `required` (`discard_count` when
    not given) and the hand holds them; of a hidden hand only the total is
    checked."""
    if required is None:
        required = discard_count(state, player)
    if sum(cards) != required:
        raise ValueError(f"player {player} must discard exactly {required}")
    hand = state.hands[player]
    if is_hidden(hand):
        # Discards are public; only the count can be checked against them.
        if len(hand) < required:
            raise ValueError(f"player {player} holds {len(hand)} cards, not {required}")
        for resource, count in enumerate(cards):
            if count:
                hand.move(resource, -count)
                state.bank[resource] += count
        return
    for resource, count in enumerate(cards):
        if count > hand[resource]:
            raise ValueError(f"player {player} lacks {count} of resource {resource}")
    for resource, count in enumerate(cards):
        hand[resource] -= count
        state.bank[resource] += count

