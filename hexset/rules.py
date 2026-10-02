# SPDX-License-Identifier: GPL-3.0-only
"""What varies between Catan game types, and which tables each is defined for.

`Rules` is the settings themselves:

`winning_points`: VPs that end the game, checked on the holder's turn.
`discard_limit`: holding *more* than this when a seven is rolled discards half.
`friendly_robber`: the robber may not take a hex occupied by a seat at or
below `robber.FRIENDLY_ROBBER_POINTS` *public* points, so a seat that is
visibly behind can be neither blocked nor robbed.
`balanced_dice`: rolls come from `chance.Balanced`, a dice deck, rather than
two independent uniform dice. It also moves `seven_odds`, the chance of a
seven on any one roll.
`road_building_min_roads`: road pieces a seat must still have to play Road
Building -- one by the printed rule, two at a table that refuses the card's
last-piece road.

`GameType` is a *contract*: one `Rules` together with the live seat counts it
is defined for, because the two are inseparable -- the 15-point game is a
two-seat game, and a four-seat game under its settings is not a format at
all. `hexset.game.start` checks the pair before it deals.

Two types ship, and they are the two that are played:

- `STANDARD_GAME`, 10 / 7 with no shield and independent dice, at 2, 3 or 4
  seats.
- `DUEL_VARIANT_GAME`, 15 / 9 with the shield on and the deck in use, at 2
  seats and only 2.

Anything else is a custom type, which is a consumer's to declare and not
this package's to ship. Build a `GameType` with the seat counts it is
defined for and `start` will honour it; what it does not do is let a shipped
type be dealt to a table it was never played on.

**Live seats, not dealt seats.** A table can retire seats -- the server closes
the empty ones, the arena seats fewer players than the table holds -- so the
count a type is checked against is what is actually playing. A four-seat deal
with two seats retired is a two-seat game and `DUEL_VARIANT_GAME` accepts it;
the same deal with nothing retired is a four-seat game and it does not.
"""
from __future__ import annotations

from dataclasses import dataclass

from .chance import BALANCED_SEVEN_ODDS, SEVEN_ODDS

__all__ = [
    "Rules",
    "STANDARD",
    "DUEL_VARIANT",
    "GameType",
    "STANDARD_GAME",
    "DUEL_VARIANT_GAME",
    "GAME_TYPES",
    "game_type",
]


@dataclass(frozen=True)
class Rules:
    """The rule settings one game is played to: points to win, the discard
    limit, the friendly robber, balanced dice and Road Building's piece
    floor."""

    winning_points: int = 10
    discard_limit: int = 7
    friendly_robber: bool = False
    balanced_dice: bool = False
    # Road pieces a player must still have to play Road Building. The printed
    # rule is one -- with a single piece left the card places a single road --
    # but a served table can refuse that last-piece road, and a bot that
    # plays the card there loses it for nothing; such a table sets two.
    road_building_min_roads: int = 1

    @property
    def seven_odds(self) -> float:
        """How often a roll under these rules is a seven: `chance.SEVEN_ODDS`
        for independent dice, `chance.BALANCED_SEVEN_ODDS` for the dice deck,
        whose rate is measured rather than derived (the constant says how)."""
        return BALANCED_SEVEN_ODDS if self.balanced_dice else SEVEN_ODDS

    def __post_init__(self) -> None:
        if self.winning_points < 1:
            raise ValueError(f"winning_points must be positive: {self.winning_points}")
        if self.discard_limit < 0:
            raise ValueError(f"discard_limit must be non-negative: {self.discard_limit}")
        if not 1 <= self.road_building_min_roads <= 2:   # the card's two roads
            raise ValueError(
                f"road_building_min_roads must be 1 or 2: {self.road_building_min_roads}")


STANDARD: Rules = Rules()
DUEL_VARIANT: Rules = Rules(
    winning_points=15, discard_limit=9, friendly_robber=True, balanced_dice=True
)


@dataclass(frozen=True)
class GameType:
    """A ruleset and the live seat counts it is played at.

    `seats` is the whole of the contract beyond `rules`: a type is defined
    for those table sizes and undefined everywhere else. `check` is the only
    place that judgement is made, so a caller cannot reach a ruleset without
    passing the table it is being dealt to.
    """

    name: str
    rules: Rules
    seats: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a game type needs a name")
        if not self.seats:
            raise ValueError(f"{self.name!r} declares no seat counts")
        if len(set(self.seats)) != len(self.seats):
            raise ValueError(f"{self.name!r} repeats a seat count: {self.seats}")
        if sorted(self.seats) != list(self.seats):
            raise ValueError(f"{self.name!r} lists seat counts out of order: {self.seats}")
        if self.seats[0] < 2:
            raise ValueError(f"{self.name!r} allows fewer than two seats: {self.seats}")

    def plays(self, live: int) -> bool:
        """Whether this type is defined for a table of `live` playing seats."""
        return live in self.seats

    def check(self, live: int, *, dealt: int | None = None) -> None:
        """Raise unless this type is defined for `live` playing seats.

        `dealt` is the table size when it differs from `live`, carried only so
        the message can say how the table got there -- a retired seat is easy
        to forget about and hard to see in a traceback.
        """
        if self.plays(live):
            return
        allowed = ", ".join(str(n) for n in self.seats)
        table = "" if dealt is None or dealt == live else (
            f" ({dealt} dealt, {dealt - live} retired)"
        )
        raise ValueError(
            f"the {self.name!r} game type is played at {allowed} seats, not "
            f"{live}{table}. Shipped types: "
            + "; ".join(
                f"{t.name} at {', '.join(str(n) for n in t.seats)}"
                for t in GAME_TYPES.values()
            )
            + ". Declare your own GameType for a format that is not one of these."
        )


STANDARD_GAME: GameType = GameType("standard", STANDARD, (2, 3, 4))
DUEL_VARIANT_GAME: GameType = GameType("duel-variant", DUEL_VARIANT, (2,))

#: The types this package ships, by name. Read by `hexset.game.start`; a
#: custom type is passed as an object, not registered here.
GAME_TYPES: dict[str, GameType] = {
    STANDARD_GAME.name: STANDARD_GAME,
    DUEL_VARIANT_GAME.name: DUEL_VARIANT_GAME,
}


def game_type(name: str) -> GameType:
    """The shipped type called `name`."""
    try:
        return GAME_TYPES[name]
    except KeyError:
        raise ValueError(
            f"unknown game type {name!r}; this package ships "
            + ", ".join(sorted(GAME_TYPES))
        ) from None
