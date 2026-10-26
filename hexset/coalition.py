# SPDX-License-Identifier: GPL-3.0-only
"""Public statements of alignment, with attributed resource and action intents.

These symbols record what a seat expressed. They neither certify another
seat's hand nor bind a trade or promise. ``None`` is an unresolved seat;
``TABLE`` is an explicit address to the table. Resource quantities may be
unknown. Text interpretation belongs to the client.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

__all__ = ["TABLE", "VERSION", "RESOURCES", "ACTIONS", "MODES", "ResourceIntent", "ActionIntent",
           "Coalition", "Event", "Transcript"]

TABLE = -1
VERSION = 1
RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")
ACTIONS = ("build_road", "build_settlement", "build_city", "buy_dev", "play_knight",
           "play_monopoly", "play_year_of_plenty", "play_road_building", "take_road",
           "defend_road", "take_army", "defend_army", "block", "rob", "plow", "no_trade")
MODES = ("request", "offer", "statement", "unknown")


def _seat(value: int | None) -> None:
    if value is not None and (type(value) is not int or value < TABLE):
        raise ValueError(f"invalid seat reference: {value!r}")


@dataclass(frozen=True)
class ResourceIntent:
    """A resource flow expressed by a seat, with unresolved arguments allowed."""
    resource: str
    quantity: int | None = None
    giver: int | None = None
    receiver: int | None = None
    mode: str = "unknown"

    def __post_init__(self):
        if self.resource not in RESOURCES or self.mode not in MODES:
            raise ValueError("unknown resource or intent mode")
        if self.quantity is not None and (type(self.quantity) is not int or self.quantity < 0):
            raise ValueError("a stated quantity is a nonnegative integer or unknown")
        _seat(self.giver)
        _seat(self.receiver)


@dataclass(frozen=True)
class ActionIntent:
    """An expressed game action and the seat asked or intending to perform it."""
    action: str
    actor: int | None = None
    target: int | None = None
    mode: str = "unknown"

    def __post_init__(self):
        if self.action not in ACTIONS or self.mode not in MODES:
            raise ValueError("unknown action or intent mode")
        _seat(self.actor)
        _seat(self.target)


@dataclass(frozen=True)
class Coalition:
    """One expressed alignment and its separately bound assistance arguments."""
    addressed_to: int | None = None
    supported: int | None = None
    opposed: int | None = None
    resources: tuple[ResourceIntent, ...] = ()
    actions: tuple[ActionIntent, ...] = ()

    def __post_init__(self):
        for value in (self.addressed_to, self.supported, self.opposed):
            _seat(value)
        if not isinstance(self.resources, tuple) or not all(isinstance(x, ResourceIntent) for x in self.resources):
            raise ValueError("resources must be a tuple of ResourceIntent")
        if not isinstance(self.actions, tuple) or not all(isinstance(x, ActionIntent) for x in self.actions):
            raise ValueError("actions must be a tuple of ActionIntent")


@dataclass(frozen=True)
class Event:
    """One utterance, possibly expressing several alignments.

    ``position`` is an observed event index, never an estimated timestamp.
    An unresolved position cannot be used to construct a historical prefix.
    """
    id: str
    speaker: int
    position: int | None
    symbols: tuple[Coalition, ...]

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("an event needs a stable id")
        _seat(self.speaker)
        if self.speaker is None or self.speaker == TABLE:
            raise ValueError("an event needs an identified speaker")
        if self.position is not None and (type(self.position) is not int or self.position < 0):
            raise ValueError("an event position is nonnegative or unknown")
        if not isinstance(self.symbols, tuple) or not all(isinstance(x, Coalition) for x in self.symbols):
            raise ValueError("symbols must be a tuple of Coalition")

    def to_dict(self) -> dict:
        return {"version": VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, row: Mapping) -> Event:
        row = dict(row)
        if row.pop("version", None) != VERSION:
            raise ValueError("unknown coalition protocol version")
        symbols = []
        for raw in row.pop("symbols"):
            raw = dict(raw)
            raw["resources"] = tuple(ResourceIntent(**x) for x in raw.get("resources", ()))
            raw["actions"] = tuple(ActionIntent(**x) for x in raw.get("actions", ()))
            symbols.append(Coalition(**raw))
        return cls(**row, symbols=tuple(symbols))

    def framed(self, seats: Sequence[int]) -> Event:
        """Map every seat reference into the supplied perspective order."""
        if len(set(seats)) != len(seats) or any(type(s) is not int or s < 0 for s in seats):
            raise ValueError("a frame names distinct seats")
        indices = {s: i for i, s in enumerate(seats)}

        def ref(s):
            if s is None or s == TABLE:
                return s
            if s not in indices:
                raise ValueError(f"seat {s} is missing from the frame")
            return indices[s]

        return replace(self, speaker=ref(self.speaker), symbols=tuple(
            replace(x, addressed_to=ref(x.addressed_to), supported=ref(x.supported), opposed=ref(x.opposed),
                    resources=tuple(replace(r, giver=ref(r.giver), receiver=ref(r.receiver)) for r in x.resources),
                    actions=tuple(replace(a, actor=ref(a.actor), target=ref(a.target)) for a in x.actions))
            for x in self.symbols))


class Transcript:
    """Deduplicated public utterances; opposition counts once per utterance-target."""

    def __init__(self):
        self._events: dict[str, Event] = {}
        self._against: dict[tuple[int, int], int] = {}
        self._last_position: int | None = None

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events.values())

    @property
    def against(self) -> dict[tuple[int, int], int]:
        return dict(self._against)

    def append(self, event: Event) -> bool:
        if event.id in self._events:
            if self._events[event.id] != event:
                raise ValueError("a repeated utterance id carries different content")
            return False
        if (event.position is not None and self._last_position is not None
                and event.position < self._last_position):
            raise ValueError("event positions must be nondecreasing")
        self._events[event.id] = event
        if event.position is not None:
            self._last_position = event.position
        for target in {s.opposed for s in event.symbols if s.opposed is not None and s.opposed != TABLE}:
            key = (event.speaker, target)
            self._against[key] = self._against.get(key, 0) + 1
        return True

    def before(self, position: int) -> tuple[Event, ...]:
        if any(e.position is None for e in self._events.values()):
            raise ValueError("cannot build a causal prefix with unresolved event positions")
        return tuple(e for e in self._events.values() if e.position < position)
