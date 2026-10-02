"""The game session behind a served table; its wire format is `wire`.

**Never build an action the engine did not offer.** `GameSession.submit`
checks the decoded wire action against a *fresh* `legal_actions`, not against
what was on offer at some earlier poll, so a UI bug, a stale page and a
tampered request all fail the same way. Clients echo back the literal wire
objects `state_view()` sent rather than constructing an `Action` from parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Sequence

from hexset.actions import (
    YEAR_OF_PLENTY_PAIRS,
    Action,
    ActionType,
    apply,
    legal_actions,
)
from hexset.board.board import Board
from hexset.board.terrain import NUM_RESOURCES, TERRAIN_RESOURCE
from hexset.cards import NUM_DEV_CARDS
from hexset.devcards import dev_count, holdings
from hexset.economy import hand_size, trade_ratios
from hexset.game import (
    Game, Phase, event_hand_sizes, is_over, lock_seat, may_act, pending_free_roads,
    players_owing_discards, publish_trade_event, to_move,
)
from hexset.ledger import PublicLedger
from hexset.roads import road_lengths
from hexset.state import GameState, copy_state, is_hidden
from hexset.trading import (
    RESPONSE_ACCEPT,
    RESPONSE_COUNTER,
    RESPONSE_PASS,
    Bundle,
    Offer,
    Response,
    Trade,
    apply_trades,
    execute_agreed,
    params_of,
)
from hexset import trading
from hexset.victory import public_victory_points, victory_points

from ._journal import Journal
from .wire import DEV_CARD_NAMES, RESOURCE_NAMES, action_to_wire, wire_to_action

class ResumeError(Exception):
    """A journalled game would not replay: its actions no longer describe a
    legal game under this engine. Recoverable — `api.reopen_session` deals a
    fresh game rather than failing the request."""


# --- Trading -------------------------------------------------------------------


@dataclass(frozen=True)
class PendingGate:
    """The gate of a manual seat. It never agrees to anything on its own:
    `offer` passes, `respond` records the offer to `game.pending` for the
    seat to answer later, `pick` declines, `gains_many` refuses."""

    game: "Game"
    seat: int

    def gains_many(self, view, received: Sequence[Bundle], counterparties: Sequence[int]) -> list[float]:
        del view, counterparties
        return [-1.0] * len(received)

    def offer(self, view, candidates):
        del view, candidates
        return None

    def respond(self, view, offer: Offer) -> Response:
        del view
        self.game.pending.append(Trade(offer.actor, self.seat, offer.received))
        return Response(self.seat, RESPONSE_PASS)

    def respond_any(self, view, offer: Offer) -> Response:
        """An open offer, recorded once like any other: the seat names its
        any cards in the counter it writes (`answer_round`)."""
        return self.respond(view, offer)

    def pick(self, view, responses):
        del view, responses
        return None


# --- Human-readable log -------------------------------------------------------


def _bundle_text(bundle: tuple[int, ...]) -> str:
    parts = [f"{n} {RESOURCE_NAMES[r]}" for r, n in enumerate(bundle) if n]
    return ", ".join(parts) if parts else "nothing"


def _hex_label(board: Board, hex_id: int) -> str:
    """A hex as a player reads it: its dice number and resource, e.g. "8
    Wood". The desert has neither."""
    resource = TERRAIN_RESOURCE[board.terrain[hex_id]]
    if resource is None:
        return "the desert"
    return f"{board.tokens[hex_id]} {RESOURCE_NAMES[resource]}"


def _resource_counts(counts: list[int]) -> str:
    """`[2, 1, 0, 0, 0]` -> `"2 Wood, 1 Brick"`. Unlike `_list_with_counts`
    this does not pluralise: "2 Sheep", never "2 Sheeps"."""
    return ", ".join(f"{n} {RESOURCE_NAMES[r]}" for r, n in enumerate(counts) if n)


def _hand_gains(before: list[int], after: list[int]) -> str | None:
    """What a hand gained between two snapshots, or `None`."""
    return _resource_counts([after[r] - before[r] for r in range(NUM_RESOURCES)]) or None


# (verb, noun) for every action `render_log` folds into a "placed/built" run.
_BUILD_KIND = {
    ActionType.SETUP_SETTLEMENT: ("placed", "settlement"),
    ActionType.SETUP_ROAD: ("placed", "road"),
    ActionType.BUILD_ROAD: ("built", "road"),
    ActionType.BUILD_SETTLEMENT: ("built", "settlement"),
    ActionType.BUILD_CITY: ("built", "city"),
}


def _list_with_counts(items: list[str]) -> str:
    """`['settlement', 'road']` -> `'a settlement and a road'`;
    `['road', 'road']` -> `'2 roads'` — one count per distinct item, in the
    order first seen, not one entry per occurrence."""
    order = list(dict.fromkeys(items))
    parts = [f"a {item}" if items.count(item) == 1 else f"{items.count(item)} {item}s" for item in order]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


@dataclass
class _Snapshot:
    hands: list[list[int]]
    held: list[list[int]]


# The actions a seat can take back, its own only. PLAY_ROAD_BUILDING and
# PLAY_KNIGHT are here so a card can be handed back in the instant after it is
# played but before what it opens has landed; a seven's own robber move is not,
# there being no card to hand back. A winning action is excluded by `apply_action`.
_UNDOABLE_BUILDS: frozenset[ActionType] = frozenset(
    {
        ActionType.SETUP_SETTLEMENT,
        ActionType.SETUP_ROAD,
        ActionType.BUILD_ROAD,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_CITY,
        ActionType.BANK_TRADE,
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.PLAY_KNIGHT,
    }
)


@dataclass
class _UndoPoint:
    """Everything one action could have touched, from just before it ran.
    `dev_card_played` lives on `Game`, not `GameState`, so `set_state` would
    otherwise leave it stuck true and lock the seat out of dev cards."""

    state: GameState
    # Snapshotted and restored with the state, and not derivable from it: an
    # undone BANK_TRADE that left it alone would certify a floor the hand no
    # longer supports.
    ledger: PublicLedger
    free_roads: int
    phase: Phase
    current_player: int
    setup_step: int
    last_settlement: int
    dev_card_played: bool
    events: int
    steps: int
    # Whose action this would take back; only its own actor may undo it.
    actor: int


def _snapshot(game: Game) -> _Snapshot:
    """Every seat's hand and dev-card holdings, unredacted."""
    # true state: the server's own omniscient observer snapshot.
    state = game.state(0, hidden=False)
    return _Snapshot(
        hands=[hand[:] for hand in state.hands],
        held=[holdings(state, p)[:] for p in range(state.num_players)],
    )


@dataclass
class _Event:
    """One applied action, held in full and redacted for nobody: the hiding
    happens per reader (`render_log`). `before` is kept alongside `after`
    because most log lines are a *difference*. `action` is `None` for a
    trade-round step (`note`) or a manual trade (`trades`)."""

    round_num: int
    actor: int
    action: Action | None
    before: _Snapshot
    after: _Snapshot
    last_roll: int | None
    # The exchanges cleared inside this action. The engine's own trade event
    # runs once a turn, on the way into MAIN, so in a game dealt here only the
    # action entering MAIN carries any -- except a manual trade's event, never
    # empty. A game journalled from a record (`journal.journal_of`) files an
    # exchange after whichever action it followed.
    trades: tuple[Trade, ...] = ()
    # One step of the trade round, collapsed by `render_log` into one line and
    # never redacted. Only set on an `action is None` event.
    note: RoundNote | None = None


class RoundNote(NamedTuple):
    """One step of a trade round. `kind` is `offer` (what the actor put to
    the table), `accept`, `counter` or `pass` (one seat's answer), `decline`
    (the actor turned the answers down) or `nobody` (every seat passed); `seat`
    is who did it (the actor for `offer`/`decline`/`nobody`, the answering
    seat otherwise); `actor` is the round's actor throughout; `bundle` is
    signed towards the actor for `offer`/`accept`/`counter`, else `None`;
    `any` is an `offer`'s any cards, signed the same way (`Offer.any`)."""

    kind: str
    seat: int
    actor: int
    bundle: Bundle | None
    any: int = 0


def _with_any(counts: tuple[int, ...], any_cards: int) -> str:
    """One side of an offer in words, its any cards included: "1 Wood and
    any 1 card", "any 2 cards"."""
    if any_cards <= 0:
        return _bundle_text(counts)
    extra = f"any {any_cards} card{'' if any_cards == 1 else 's'}"
    return extra if not any(counts) else f"{_bundle_text(counts)} and {extra}"


def _round_clause(note: RoundNote, labels: dict[int, str], *, standalone: bool) -> str | None:
    """The sentence one round step adds to the round's line — or, when that
    line is no longer the last written (`standalone`), one naming the offer it
    answers. `None` for a pass: the transcript says nothing about passes until
    everyone has passed (`nobody`)."""
    who = _who(note.seat, labels)
    actor = _who(note.actor, labels)
    if note.kind == "offer":
        gave = tuple(max(0, -n) for n in note.bundle or ())
        got = tuple(max(0, n) for n in note.bundle or ())
        return f"{who} offers {_with_any(gave, -note.any)} for {_with_any(got, note.any)}."
    if note.kind == "accept":
        return f"{who} accepts {actor}'s offer." if standalone else f"{who} accepts."
    if note.kind == "counter" and note.bundle is not None:
        # Signed towards the actor: positive is what the answerer hands over.
        gives = tuple(max(0, n) for n in note.bundle)
        wants = tuple(max(0, -n) for n in note.bundle)
        head = f"{who} counters {actor}'s offer with" if standalone else f"{who} counters with"
        return f"{head} {_bundle_text(gives)} for {_bundle_text(wants)}."
    if note.kind == "decline":
        return f"{who} declines every answer." if standalone else f"{who} declines."
    if note.kind == "nobody":
        return f"Everyone declines {actor}'s offer." if standalone else "Everyone declines."
    return None


class SeatLabels(dict):
    """Seat -> label, plus the seat's number among the seats still in the game:
    a closed seat keeps no number, so the table reads Player 1, 2, 3."""

    def __init__(self, labels: dict[int, str], locked: Iterable[int] = ()) -> None:
        super().__init__(labels)
        self.locked = frozenset(locked)

    def number(self, seat: int) -> int:
        """`seat`'s 1-based number among the seats still in the game."""
        return seat + 1 - sum(1 for s in self.locked if s < seat)


def _who(seat: int, labels: dict[int, str]) -> str:
    """"Player N (label)": N is the seat's number among the seats still in the
    game when `labels` is a `SeatLabels`, else the 1-indexed seat."""
    number = labels.number(seat) if isinstance(labels, SeatLabels) else seat + 1
    return f"Player {number} ({labels.get(seat, 'bot')})"


def _describe(
    event: _Event,
    board: Board,
    labels: dict[int, str],
    viewer: int | None,
    *,
    omniscient: bool = False,
) -> str:
    """One event as a sentence, told to `viewer`. The only thing `viewer`
    changes is what stays hidden — a card bought, a card stolen; everything
    else reads identically to everyone, `viewer=None` included. `omniscient`
    names all of those, and is only ever for a reader outside the game."""
    actor, action = event.actor, event.action
    before, after = event.before, event.after
    num_players = len(after.hands)
    kind = action.type
    who = _who(actor, labels)

    if kind is ActionType.ROLL:
        line = f"{who} rolled {event.last_roll}."
        gains = [
            f"{_who(p, labels)} collects {g}."
            for p in range(num_players)
            if (g := _hand_gains(before.hands[p], after.hands[p])) is not None
        ]
        return " ".join([line, *gains])

    if kind is ActionType.SETUP_SETTLEMENT:
        line = f"{who} placed a settlement."
        gains = _hand_gains(before.hands[actor], after.hands[actor])
        if gains:
            line += f" {who} collects {gains}."
        return line

    if kind is ActionType.SETUP_ROAD:
        return f"{who} placed a road."

    if kind is ActionType.BUILD_ROAD:
        return f"{who} built a road."

    if kind is ActionType.BUILD_SETTLEMENT:
        return f"{who} built a settlement."

    if kind is ActionType.BUILD_CITY:
        return f"{who} built a city."

    if kind is ActionType.BUY_DEV_CARD:
        gained = [c for c in range(NUM_DEV_CARDS) if before.held[actor][c] < after.held[actor][c]]
        if (omniscient or actor == viewer) and gained:
            return f"{who} bought a {DEV_CARD_NAMES[gained[0]]}."
        return f"{who} bought a development card."

    if kind is ActionType.PLAY_ROAD_BUILDING:
        return f"{who} played Road Building."

    if kind is ActionType.PLAY_KNIGHT:
        return f"{who} played a Knight."

    if kind is ActionType.MOVE_ROBBER:
        victim = action.b if action.b < num_players else None
        line = f"{who} moved the robber to {_hex_label(board, action.a)}"
        if victim is None:
            return line + "."
        stolen = _hand_gains(before.hands[victim], after.hands[victim])
        # Named only to the two seats who already know it.
        if (omniscient or viewer in (actor, victim)) and stolen:
            resource = next(
                RESOURCE_NAMES[r]
                for r in range(NUM_RESOURCES)
                if after.hands[victim][r] < before.hands[victim][r]
            )
            return line + f" and stole {resource} from {_who(victim, labels)}."
        return line + f" and stole a card from {_who(victim, labels)}."

    if kind is ActionType.PLAY_MONOPOLY:
        resource = RESOURCE_NAMES[action.a]
        swept = after.hands[actor][action.a] - before.hands[actor][action.a]
        return f"{who} played Monopoly on {resource} and collected {swept} card(s)."

    if kind is ActionType.PLAY_YEAR_OF_PLENTY:
        r0, r1 = YEAR_OF_PLENTY_PAIRS[action.a]
        return f"{who} played Year of Plenty for {RESOURCE_NAMES[r0]} and {RESOURCE_NAMES[r1]}."

    return f"{who} played {kind.name}."


def _trade_lines(event: _Event, labels: dict[int, str]) -> list[str]:
    """One sentence per exchange the engine cleared inside this action. Fully
    public — both seats and both bundles — since the ledger already certifies
    every card that moved."""
    out = []
    for trade in event.trades:
        got = tuple(max(0, n) for n in trade.received)
        gave = tuple(max(0, -n) for n in trade.received)
        out.append(
            f"{_who(trade.a, labels)} traded {_bundle_text(gave)} "
            f"to {_who(trade.b, labels)} for {_bundle_text(got)}."
        )
    return out


def _after_run(lines: list[str], run: dict, event: _Event, labels: dict[int, str]) -> dict | None:
    """The exchanges cleared inside a build or bank step, written after the
    run's line. They end the run, since a run grows by rewriting the last
    line, which would then be the trade's."""
    if not event.trades:
        return run
    for line in _trade_lines(event, labels):
        lines.append(f"{event.round_num}\t{line}")
    return None


def render_log(
    events: list[_Event],
    board: Board,
    labels: dict[int, str],
    viewer: int | None,
    *,
    omniscient: bool = False,
    discards_open: bool = False,
) -> list[str]:
    """Every event as the sidebar transcript `viewer` should see, as
    `<round>\t<text>` lines.

    A pure fold, run fresh per reader, which is what makes undo trivial:
    dropping the events drops their lines. Builds and consecutive bank trades
    of the same pair collapse into one run, rewritten in place as it grows.

    **Discards are held back until the whole round has resolved.** A seven's
    discards are simultaneous but resolve one submission at a time, so they
    accumulate and are written together, in seat order, on the first event that
    follows the round — or at the end of the fold if `discards_open` (the
    caller's live `any(game.discard_quota)`) says no seat still owes.

    Redaction is per reader: a seat sees the cards it lost named, everyone else
    a count, `omniscient` names them all.
    """
    lines: list[str] = []
    run: dict | None = None
    # The open discard round: seat -> what it has given up so far.
    owed: dict[int, list[int]] = {}
    owed_round = 0

    def flush_discards() -> None:
        # Seat order: a simultaneous round has no arrival order to report.
        nonlocal owed
        for actor in sorted(owed):
            counts = owed[actor]
            who = _who(actor, labels)
            if omniscient or actor == viewer:
                text = f"{who} discarded {_resource_counts(counts)}."
            else:
                # Which resources went is the discarding seat's line only.
                total = sum(counts)
                text = f"{who} discarded {total} card{'' if total == 1 else 's'}."
            lines.append(f"{owed_round}\t{text}")
        owed = {}

    def emit(round_num: int, text: str, continuing: bool) -> None:
        # A run is one line, rewritten in place as it grows.
        if continuing:
            lines.pop()
        lines.append(f"{round_num}\t{text}")

    for event in events:
        if owed and (event.action is None or event.action.type is not ActionType.DISCARD):
            # Any other event proves the discard round closed before it.
            flush_discards()
        if event.action is None:
            # No board action: a trade-round step or a manual trade. The round
            # is one line rewritten in place, passes are not written one by
            # one, and a late answer stands alone, naming the offer.
            if event.note is not None:
                note = event.note
                key = ("round", note.actor, event.round_num)
                continuing = run is not None and run["key"] == key and note.kind != "offer"
                clause = _round_clause(note, labels, standalone=not continuing and note.kind != "offer")
                if clause is None:
                    continue
                if note.kind == "offer":
                    run = {"key": key, "text": clause}
                    emit(event.round_num, clause, False)
                elif continuing:
                    run["text"] = f"{run['text']} {clause}"
                    emit(event.round_num, run["text"], True)
                else:
                    run = None
                    lines.append(f"{event.round_num}\t{clause}")
                continue
            trade_lines = _trade_lines(event, labels)
            key = ("round", event.actor, event.round_num)
            if run is not None and run["key"] == key and len(event.trades) == 1:
                # The bundle is already on the line.
                run["text"] = f"{run['text']} Traded with {_who(event.trades[0].b, labels)}."
                emit(event.round_num, run["text"], True)
                continue
            run = None
            for line in trade_lines:
                lines.append(f"{event.round_num}\t{line}")
            continue
        action, actor, round_num = event.action, event.actor, event.round_num
        kind = action.type
        who = _who(actor, labels)

        if kind in _BUILD_KIND:
            verb, item = _BUILD_KIND[kind]
            key = ("build", actor, round_num)
            continuing = run is not None and run["key"] == key
            if not continuing:
                run = {"key": key, "verb": verb, "items": [], "extra": []}
            run["items"].append(item)
            if kind is ActionType.SETUP_SETTLEMENT:
                gains = _hand_gains(event.before.hands[actor], event.after.hands[actor])
                if gains:
                    run["extra"].append(f"{who} collects {gains}.")
            line = f"{who} {run['verb']} {_list_with_counts(run['items'])}."
            emit(round_num, " ".join([line, *run["extra"]]), continuing)
            run = _after_run(lines, run, event, labels)
            continue

        if kind is ActionType.DISCARD:
            # The engine takes discards one card at a time, several seats at
            # once; they say nothing until `flush_discards` writes them.
            run = None  # a discard ends whatever build/bank run was open
            owed_round = round_num
            owed.setdefault(actor, [0] * NUM_RESOURCES)[action.a] += 1
            continue

        if kind is ActionType.BANK_TRADE:
            # The same pair traded in a row sums into one line.
            key = ("bank", actor, round_num, action.a, action.b)
            continuing = run is not None and run["key"] == key
            if not continuing:
                run = {"key": key, "given": 0, "got": 0}
            run["given"] += event.before.hands[actor][action.a] - event.after.hands[actor][action.a]
            run["got"] += 1
            line = (
                f"{who} traded {run['given']} {RESOURCE_NAMES[action.a]} "
                f"for {run['got']} {RESOURCE_NAMES[action.b]} with the bank."
            )
            emit(round_num, line, continuing)
            run = _after_run(lines, run, event, labels)
            continue

        run = None  # anything else ends whatever run was open

        if kind is ActionType.END_TURN:
            # The next line already implies the previous turn ended.
            continue

        lines.append(
            f"{round_num}\t{_describe(event, board, labels, viewer, omniscient=omniscient)}"
        )
        for line in _trade_lines(event, labels):
            lines.append(f"{round_num}\t{line}")

    # A round that finished on the last event applied is over all the same.
    if owed and not discards_open:
        flush_discards()
    return lines


def _is_split_knight_play(
    steps: list[tuple[int, Action | None, tuple[Trade, ...]]], index: int, actor: int
) -> bool:
    """Whether `steps[index]`, a `PLAY_KNIGHT` that did not end the game, is a
    bare card play followed by its own `MOVE_ROBBER` step, as against a
    journalled knight whose `a`/`b` carry that robber move. The *next* step is
    the exact test; operand values cannot distinguish the two shapes."""
    if index + 1 >= len(steps):
        return False
    next_actor, next_action, _ = steps[index + 1]
    return (
        next_actor == actor
        and next_action is not None
        and next_action.type is ActionType.MOVE_ROBBER
    )


# --- The session ---------------------------------------------------------------


@dataclass
class _OpenRound:
    """The current turn's offer still being negotiated; `None` when nothing
    is open. `responses` holds at most one answer per seat, a later one
    replacing it; `awaiting` is the manual seats yet to answer, and a bot
    actor's turn is held while it is non-empty."""

    offer: Offer
    responses: list[Response] = field(default_factory=list)
    awaiting: set[int] = field(default_factory=set)
    # Hand sizes as the round opened, for `publish_trade_event` on close.
    hand_sizes: tuple[int, ...] = ()



def _concrete(state, seat: int) -> bool:
    """Whether `seat`'s cards are in the state card by card, not as the
    counts an observed state holds for a seat it cannot see."""
    return not (is_hidden(state.hands[seat]) or is_hidden(state.dev_cards[seat])
                or is_hidden(state.new_dev_cards[seat]))

@dataclass
class GameSession:
    """One in-progress game: the engine state and which seats are claimed.
    Nothing here distinguishes a bot's seat from a person's. All mutation goes
    through `submit`, the one enforcement point, and nothing runs a further
    seat's turn on another's behalf."""

    game: Game
    # Every seat somebody is playing, whichever kind of client it is.
    claimed_seats: set[int]
    seed: int = 0
    journal: Journal | None = None
    # Seat -> the model-picker display name playing it. Bots only.
    bot_names: dict[int, str] = field(default_factory=dict)
    # Seat -> the entrant spec that built the bot on it, journalled so a
    # resumed game can put the same opponents back. Unused by play.
    bot_specs: dict[int, str] = field(default_factory=dict)
    # Seat -> whatever the person playing it registered as.
    player_names: dict[int, str] = field(default_factory=dict)
    # Seat -> `{"id", "kind"}` of the client that claimed it, journal only.
    clients: dict[int, dict] = field(default_factory=dict)
    # The join code of the table this game was dealt for, journalled so a
    # restart can find it again.
    code: str | None = None
    # Seat -> the dice total it rolled on its own most recent turn, which
    # `game.last_roll` (one global value) does not give.
    last_roll_by_seat: dict[int, int] = field(default_factory=dict)
    # Every action applied so far, unredacted; `render_log` folds the
    # transcript out of these per reader.
    events: list[_Event] = field(default_factory=list)
    # The step number the next action is journalled under: the journal's
    # numbering, not `len(events)`.
    steps: int = field(default=0, repr=False)
    # The `(turns, current_player)` a bot actor last broadcast in, so a second
    # entry into MAIN in one turn does not restart the round from scratch.
    _broadcast_turn: tuple[int, int] | None = field(default=None, repr=False)
    # This turn's own bundles already put to the table, and how many --
    # reset alongside `_broadcast_turn`. What a gate asking for more than one
    # offer a turn (`trade_offer_budget`, e.g. the fragmented trade policy) is
    # weighed against.
    _already_offered: set = field(default_factory=set, repr=False)
    _offers_made: int = field(default=0, repr=False)
    # Set the first time the game is over, so it is filed away once.
    ended: bool = field(default=False, repr=False)
    # The one action that could still be taken back, and by whom.
    _undo: _UndoPoint | None = field(default=None, repr=False)
    # The seat that has placed its setup road and not yet said it is done. A
    # setup road is the one handoff the engine makes on its own; holding the
    # turn here keeps the placement undoable, as in every other phase.
    awaiting_confirm: int | None = field(default=None)
    # Seat -> whatever answers that seat's private gate (`hexset.trading`).
    # `set_trader` is the one place that rewrites the engine's tuple.
    traders: dict[int, object] = field(default_factory=dict, repr=False)
    # This turn's own offer, still being negotiated; `None` before the
    # first, after one executes or is declined, and once the turn ends.
    open_round: "_OpenRound | None" = field(default=None, repr=False)

    def set_trader(self, seat: int, trader: object | None) -> None:
        """Seat (or unseat) what answers `seat`'s side of a trade's gate."""
        if trader is None:
            self.traders.pop(seat, None)
        else:
            self.traders[seat] = trader
        self.game.gates = tuple(
            self.traders.get(s) for s in range(self.game.num_players)
        )

    def confirm_mode(self, seat: int) -> None:
        """Gate `seat` with a `PendingGate`, at seat-up: a manual seat only
        ever trades through its own explicit answers on the round."""
        self.set_trader(seat, PendingGate(self.game, seat))

    def pending_for(self, seat: int) -> list[Trade]:
        """The offers standing against `seat`, unanswered:
        `PendingGate.respond` records one as `Trade(actor, seat, received)`,
        signed towards the actor."""
        return [t for t in self.game.pending if t.b == seat]

    def _any_of(self, pending: Trade) -> int:
        """The any cards of the open round a `pending` entry answers: a
        manual seat's pending entries are the open round's, and only its."""
        round_ = self.open_round
        if round_ is not None and round_.offer.actor == pending.a and round_.offer.received == pending.received:
            return round_.offer.any
        return 0

    def is_manual(self, seat: int) -> bool:
        """Whether `seat` answers trades through the API rather than a bot."""
        return isinstance(self.traders.get(seat), PendingGate)

    def trade_wait(self) -> list[int]:
        """The manual seats a bot actor's open round is still waiting on.
        Empty for a manual actor (a person picks whenever they like) and
        whenever no round is open."""
        round_ = self.open_round
        if round_ is None or self.is_manual(round_.offer.actor):
            return []
        return sorted(round_.awaiting)

    # -- the trade round (`hexset.trading`): the session drives it, so a round
    # survives between calls -- an answer lands long after the offer.

    def begin_round(self, *, entering: bool = True) -> None:
        """A bot actor's first offer of the turn, once a turn (keyed by
        `(turns, current_player)`, since a knight re-enters MAIN after its
        robber move), opened as the engine's own trade event opens
        (`hexset.game.run_trade_event`): on `entering` MAIN, or, for a gate
        with a `trade_now(game)` method, at the first main-phase decision
        point it says yes at -- entering MAIN or after any of its main-phase
        actions. Never while Road Building roads are still to place. A manual
        actor uses `open_round_for` instead. Further offers the same turn -- a
        gate asking for more than one through `trade_offer_budget` -- are
        picked up by `_try_broadcast` once each round it opens resolves."""
        game = self.game
        me = game.current_player
        if game.phase is not Phase.MAIN or pending_free_roads(game):
            return
        if self._broadcast_turn == (game.turns, me):
            return
        gate = self.traders.get(me)
        ask = None if me in game.locked else getattr(gate, "trade_now", None)
        if ask is None and not entering:
            return
        if ask is not None and not ask(game):
            return  # the window stays open until it says yes or the turn ends
        self._broadcast_turn = (game.turns, me)
        self._already_offered = set()
        self._offers_made = 0
        self._close_round()
        if me in game.locked:
            return
        if gate is None or isinstance(gate, PendingGate):
            return
        self._try_broadcast(gate, me)

    def _try_broadcast(self, gate: object, me: int) -> None:
        """Put `gate`'s next offer to the table, if it has one and this
        turn's offer budget -- `trade_offer_budget` as the engine reads it
        (`hexset.trading.params_of`), `-1` for a gate declaring no limit of
        its own -- is not yet spent. The opening is the engine's own (`hexset.trading.offer`): the
        gate's menu from its own view -- known, sampled, or the fragmented
        policy's next planned fragment -- and its pick."""
        del me  # the actor is the game's current player, which `offer` reads
        budget = params_of(gate).trade_offer_budget
        # `-1` is a gate declaring no limit of its own, not a spent budget: it
        # keeps broadcasting while `offer` still has a bundle the gate has
        # not already put to this table this turn.
        if 0 <= budget <= self._offers_made:
            return
        offered = trading.offer(self.game, self._gates(), already_offered=self._already_offered)
        if offered is None:
            return
        self._offers_made += 1
        self._broadcast(offered)

    def open_round_for(self, actor: int, received: Bundle, any_cards: int = 0) -> None:
        """A manual actor's own offer (`POST .../trade/round`), already
        validated by the caller, `any_cards` its any cards (`Offer.any`).
        Replaces any round it had open."""
        self._close_round()
        self._broadcast(Offer(actor, received, any_cards))

    def _broadcast(self, offer: Offer) -> None:
        """Put `offer` to every other seated gate. A bot answers at once and
        its answer -- a pass included -- is recorded and logged; a manual
        seat is `awaiting` and answers later through `answer_round`. The
        offer itself is the first line the log writes about the round."""
        game = self.game
        self.note(RoundNote("offer", offer.actor, offer.actor, offer.received, offer.any))
        responses: list[Response] = []
        awaiting: set[int] = set()
        for seat in range(game.num_players):
            if seat == offer.actor or seat in game.locked:
                continue
            gate = self.traders.get(seat)
            if gate is None:
                continue
            if isinstance(gate, PendingGate) and not any(game.state(0, hidden=False).hands[seat]):
                # true state: the engine is the referee for coverage. A seat
                # with no cards is passed at once rather than awaited.
                response = Response(seat, RESPONSE_PASS)
                responses.append(response)
                self.note(RoundNote(response.kind, seat, offer.actor, None))
                continue
            response = trading.respond(game, gate, seat, offer)
            if isinstance(gate, PendingGate):
                awaiting.add(seat)  # recorded to `game.pending`; answers through `answer_round`
            else:
                responses.append(response)
                self.note(RoundNote(response.kind, seat, offer.actor, response.bundle))
        self.open_round = _OpenRound(offer, responses, awaiting, event_hand_sizes(game))
        self._note_if_everyone_passed()
        self._resolve()

    def _gates(self) -> list[object | None]:
        """Every seat's gate in seat order, `None` where nobody is seated --
        the shape `hexset.trading`'s round stages take."""
        return [self.traders.get(seat) for seat in range(self.game.num_players)]

    def note(self, note: RoundNote, *, round_num: int | None = None) -> None:
        """One step of the trade round, into the event record and the journal.
        Nothing on the board moved, so the event carries the same snapshot
        before and after. Journalled so a restored session's log reads the
        same as the live one did. An offer, an acceptance or a counter is
        public, so it goes on the ledger too (`trading.show`), restored or
        live alike."""
        if round_num is None:
            round_num = self.round
        if note.kind in ("offer", "accept", "counter") and note.bundle is not None:
            mine = note.bundle if note.kind == "offer" else tuple(-n for n in note.bundle)
            trading.show(self.game, note.seat, mine)
        snapshot = _snapshot(self.game)
        self.events.append(
            _Event(
                round_num=round_num,
                actor=note.seat,
                action=None,
                before=snapshot,
                after=snapshot,
                last_roll=self.game.last_roll,
                note=note,
            )
        )
        if self.journal is not None:
            self.journal.note(step=self.steps, round_num=round_num, note=note)

    def _note_if_everyone_passed(self) -> None:
        """If the round is answered in full and every answer was a pass, say so
        once, the moment it is known."""
        round_ = self.open_round
        if round_ is None or round_.awaiting or not round_.responses:
            return
        if all(r.kind == RESPONSE_PASS for r in round_.responses):
            self.note(RoundNote("nobody", round_.offer.actor, round_.offer.actor, None))

    def _resolve(self) -> Trade | None:
        """A bot actor's pick once every manual seat has answered; the round
        closes either way. A no-op for a manual actor (which picks through
        `execute_round_choice`) or while `awaiting` is non-empty.

        Tells the actor's gate how its broadcast ended (`trade_round_finished`
        -- offer, answers, the executed `Trade` or `None`), the same as the
        arena's `trade_round`, and then gives it a chance to broadcast again
        this turn through `_try_broadcast`, bounded by its own
        `trade_offer_budget` and by its `candidates` menu -- for a plain gate this is a
        no-op, since its budget is spent after one."""
        round_ = self.open_round
        if round_ is None:
            return None
        actor = round_.offer.actor
        gate = self.traders.get(actor)
        if gate is None or isinstance(gate, PendingGate) or round_.awaiting:
            return None
        trade = None
        response = trading.pick(self.game, gate, actor, round_.responses)
        if response is not None:
            try:
                trade = self._execute_round_trade(actor, response.seat, response.bundle)
            except ValueError:
                trade = None
        finished_fn = getattr(gate, "trade_round_finished", None)
        if finished_fn is not None:
            finished_fn(self.game.state(actor), round_.offer, round_.responses, trade, turn=self.game.turns)
        self._close_round(traded=trade)
        if self.game.phase is Phase.MAIN and self.game.current_player == actor:
            self._try_broadcast(gate, actor)
        return trade

    def _close_round(self, *, traded: Trade | None = None) -> None:
        """Drop the open round and every unanswered `pending` entry it left
        with a manual seat, so an offer nobody can act on stops showing. A
        round closing with accepts or counters still on the table and nothing
        `traded` is recorded as the actor declining them, however it closed.
        Either way the round is published to every bot's `observe_trade`
        exactly as the engine's own loop publishes its event."""
        round_ = self.open_round
        if round_ is not None:
            offer = round_.offer
            if traded is None and any(r.kind != RESPONSE_PASS for r in round_.responses):
                self.note(RoundNote("decline", offer.actor, offer.actor, None))
            self.game.pending = [
                t for t in self.game.pending
                if not (t.a == offer.actor and t.received == offer.received)
            ]
            publish_trade_event(
                self.game, self._gates(), round_.hand_sizes, () if traded is None else (traded,)
            )
        self.open_round = None

    def answer_round(self, seat: int, actor: int, received: Bundle, kind: str, bundle: Bundle | None) -> None:
        """A manual `seat` answers `actor`'s open broadcast with `"accept"`,
        `"counter"` (`bundle`) or `"pass"`; `received` is the exact offer its
        `pending` showed, signed towards `actor`. Only a seat the round is
        waiting on answers it, once. `ValueError` for an offer no longer open,
        a seat it is not waiting on, or an accept or counter `seat` cannot
        cover -- checked here, against the answering seat's own hand, so a
        trade that could never execute is never put to the actor."""
        round_ = self.open_round
        if round_ is None or round_.offer.actor != actor or round_.offer.received != received:
            raise ValueError("that offer is no longer open")
        if seat not in round_.awaiting:
            raise ValueError("that offer is not waiting on your answer")
        if kind == RESPONSE_ACCEPT and trading.is_open(round_.offer):
            raise ValueError("an offer with any cards is answered by a counter naming them")
        if kind == RESPONSE_COUNTER:
            if bundle is None:
                raise ValueError("a counter needs a bundle")
            give = [max(0, n) for n in bundle]  # positive is towards the actor
            take = [max(0, -n) for n in bundle]
            if not any(give) or not any(take) or any(g and t for g, t in zip(give, take)):
                raise ValueError("a counter gives and gets on disjoint resources")
        if kind in (RESPONSE_ACCEPT, RESPONSE_COUNTER):
            # An accept moves exactly the offer; positive is towards the actor.
            gives = [max(0, n) for n in (received if kind == RESPONSE_ACCEPT else bundle)]
            # true state: the engine is the referee for coverage.
            hand = self.game.state(0, hidden=False).hands[seat]
            if any(hand[r] < n for r, n in enumerate(gives)):
                raise ValueError("you cannot cover your side of that trade")
        self.game.pending = [
            t for t in self.game.pending
            if not (t.a == actor and t.b == seat and t.received == received)
        ]
        round_.responses = [r for r in round_.responses if r.seat != seat]
        response = Response(
            seat, kind, None if kind == RESPONSE_PASS else (received if kind != RESPONSE_COUNTER else bundle)
        )
        round_.responses.append(response)
        self.note(RoundNote(kind, seat, actor, response.bundle))
        round_.awaiting.discard(seat)
        self._note_if_everyone_passed()
        self._resolve()

    def execute_round_choice(self, actor: int, seat: int, bundle: Bundle) -> Trade:
        """A manual `actor`'s own pick (`POST .../trade/round/choose`):
        `seat`'s recorded response whose `bundle` matches exactly, executed
        through `_execute_round_trade`. `ValueError` for no open round, no
        such response, or Road Building roads still to place, which come
        before any trade; the round closes on success."""
        round_ = self.open_round
        if round_ is None or round_.offer.actor != actor:
            raise ValueError("there is no open round to choose from")
        if pending_free_roads(self.game):
            raise ValueError("place the roads Road Building owes first")
        if not any(r.seat == seat and r.kind != RESPONSE_PASS and r.bundle == bundle for r in round_.responses):
            raise ValueError("no such response is open")
        trade = self._execute_round_trade(actor, seat, bundle)
        self._close_round(traded=trade)
        return trade

    def decline_round(self, actor: int) -> None:
        """`actor` declines every response on its open round. Nothing moves."""
        round_ = self.open_round
        if round_ is None or round_.offer.actor != actor:
            raise ValueError("there is no open round to decline")
        self._close_round()

    def _execute_round_trade(self, actor: int, counterparty: int, bundle: Bundle) -> Trade:
        """Execute an agreed exchange (`hexset.trading.execute_agreed`). Both
        sides have already agreed -- the responder by its answer, the actor by
        its offer or its pick, a manual side by its submission -- so neither
        gate is asked again; the referee checks only that both still cover
        their cards. Recorded as its own `_Event` and journal line, and clears
        any pending take-back, since cards moved."""
        round_num = self.round
        before = _snapshot(self.game)
        trade = execute_agreed(
            self.game, actor, counterparty, bundle,
            ask_actor=False, ask_counterparty=False,
            # A bot side's gain goes on the record, as the engine's own round
            # records it; a manual side has no price to record.
            price_actor=not self.is_manual(actor),
            price_counterparty=not self.is_manual(counterparty),
        )
        self.events.append(
            _Event(
                round_num=round_num,
                actor=actor,
                action=None,
                before=before,
                after=_snapshot(self.game),
                last_roll=self.game.last_roll,
                trades=(trade,),
            )
        )
        if self.journal is not None:
            self.journal.manual_trade(self.game, step=self.steps, round_num=round_num, trade=trade)
        self.steps += 1
        self._undo = None
        return trade

    def __post_init__(self) -> None:
        # Written at the deal, not on the first action: the header's point is
        # the shuffled deck, which the first BUY_DEV_CARD would have drawn on.
        if self.journal is not None:
            self.journal.start(
                self.game,
                seed=self.seed,
                # `setup_queue[0]` is `start`'s own `first`, read back.
                first=self.game.setup_queue[0],
                human_seats=sorted(self.claimed_seats),
                bot_names=self.bot_names,
                bot_specs=self.bot_specs,
                player_names=self.player_names,
                clients=self.clients,
                code=self.code,
            )

    @property
    def seat_labels(self) -> SeatLabels:
        """Seat -> what to call whoever holds it, people and bots in one map.
        A claimed seat with no bot label falls back to its registered name, so
        every seat has a label and `_who` never has to invent one."""
        labels = dict(self.bot_names)
        for seat in self.claimed_seats:
            if seat not in labels:
                # `api.py` names a seat at claim time; this is for a session
                # built directly with none.
                labels[seat] = self.player_names.get(seat) or "api"
        return SeatLabels(labels, self.game.locked)

    def claim(self, seat: int, name: str | None, client: dict | None = None) -> None:
        """A seat somebody joined after the deal, which the header could not
        have named. Journalled as `Journal.seated` with an empty `spec`, so a
        resumed table knows the seat was somebody's without needing its lost
        token back."""
        self.claimed_seats.add(seat)
        if name:
            self.player_names[seat] = name
        if client is not None:
            self.clients[seat] = client
        if self.journal is not None:
            self.journal.seated(seat=seat, name=name or "", spec="", client=client)

    @property
    def round(self) -> int:
        """One full lap of the table, 1-indexed, unlike `game.turns`, which
        counts per seat. 0 during setup. A lap is the seats *still in the
        game*, retired seats being skipped by turn rotation."""
        if self.game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            return 0
        # true state: `num_players` is a fixed, public board property.
        # `max(..., 1)` is for the table every seat has retired from.
        playing = self.game.state(0, hidden=False).num_players - len(self.game.locked)
        return self.game.turns // max(playing, 1) + 1

    def restore(
        self,
        steps: list[tuple[int, Action | None, tuple[Trade, ...]]],
        journal: Journal | None = None,
        notes: dict[int, list[tuple[int, RoundNote]]] | None = None,
        retired: dict[int, list[int]] | None = None,
    ) -> None:
        """Re-apply a journalled game's actions, bringing this session up to
        where it left off. Every step with a real `action` goes through
        `apply_action`, so the log, the per-seat rolls and the round numbering
        are rebuilt as a consequence of replaying; `action is None` is a
        manual trade, re-executed rather than re-gated. `notes` go back at the
        step each preceded, and each seat in `retired` (`journal.retirements`)
        is locked there, as the live game locked it. `journal` is attached only
        once the replay is done, or the game would be written in a second
        time."""
        if self.journal is not None:
            raise ValueError("restore would rewrite the journal it is reading")
        notes = notes or {}
        retired = retired or {}
        for index, (actor, action, trades) in enumerate(steps):
            for seat in retired.get(self.steps, ()):
                lock_seat(self.game, seat)
            for note_round, note in notes.get(self.steps, ()):
                self.note(note, round_num=note_round)
            if action is None:
                self.replay_trades(actor, trades)
                continue
            if action.type is ActionType.PLAY_KNIGHT:
                self._apply_knight(actor, action, trades, steps, index)
                continue
            if action not in legal_actions(self.game, actor):
                raise ResumeError(
                    f"step {self.steps}: {action} is not legal in {self.game.phase.name}"
                )
            self.apply_action(actor, action, replay=trades)
        for seat in retired.get(self.steps, ()):
            lock_seat(self.game, seat)
        for note_round, note in notes.get(self.steps, ()):
            self.note(note, round_num=note_round)
        self.journal = journal
        if journal is not None:
            journal.reopened(at_step=self.steps)

    def replay_trades(self, actor: int, trades: tuple[Trade, ...]) -> None:
        """Exchanges already agreed elsewhere, as a step of their own: a manual
        trade's journal line being restored, or a record's exchange being
        journalled (`journal.journal_of`). Re-executed, not re-gated."""
        round_num = self.round
        before = _snapshot(self.game)
        apply_trades(self.game, trades)
        self.events.append(
            _Event(
                round_num=round_num,
                actor=actor,
                action=None,
                before=before,
                after=_snapshot(self.game),
                last_roll=self.game.last_roll,
                trades=trades,
            )
        )
        if self.journal is not None:
            for trade in trades:
                self.journal.manual_trade(self.game, step=self.steps, round_num=round_num,
                                          trade=trade)
        self.steps += 1

    def _apply_knight(
        self,
        actor: int,
        action: Action,
        trades: tuple[Trade, ...],
        steps: list[tuple[int, Action | None, tuple[Trade, ...]]],
        index: int,
    ) -> None:
        """Replay one journalled `PLAY_KNIGHT` step: a bare card, or one whose
        `action.a`/`action.b` carry a robber move this engine resolves
        separately. Plays the bare card first and stops there if that ended the
        game, a winning knight never moving the robber; otherwise
        `_is_split_knight_play` decides whether the `MOVE_ROBBER` follows in the
        file (left to `restore`'s loop) or is synthesized here with the
        recorded step's `trades`. Both go through the ordinary `apply_action`, and
        `self.steps` is then folded back by one, the file having recorded this
        as a single step."""
        bare = Action(ActionType.PLAY_KNIGHT)
        if bare not in legal_actions(self.game, actor):
            raise ResumeError(
                f"step {self.steps}: {action} is not legal in {self.game.phase.name}"
            )
        self.apply_action(actor, bare, replay=())
        if is_over(self.game) or _is_split_knight_play(steps, index, actor):
            return
        move = Action(ActionType.MOVE_ROBBER, action.a, action.b)
        if move not in legal_actions(self.game, actor):
            raise ResumeError(
                f"step {self.steps}: {move} is not legal in {self.game.phase.name}"
            )
        self.apply_action(actor, move, replay=trades)
        self.steps -= 1

    def legal_wire_actions(self, viewer: int | None) -> list[dict]:
        """What `viewer` may play right now; empty unless it is their turn, and
        empty for a reader with no seat. "Their turn" is `may_act`, not
        `to_move`: a seven's discards are owed by several seats at once, so
        each owing seat is offered its own cards whatever the others do."""
        if is_over(self.game) or viewer is None:
            return []
        # A seat holding its setup turn open has one move: an ordinary
        # END_TURN, which `may_act` refuses, the snake having moved on.
        if self.awaiting_confirm == viewer:
            return [action_to_wire(Action(ActionType.END_TURN))]
        if not may_act(self.game, viewer):
            return []
        return [action_to_wire(a) for a in legal_actions(self.game, viewer)]

    def submit(self, seat: int, wire: dict) -> None:
        """Play `wire` as `seat`, which comes off a player token, not the
        request body. Every other check happens here, the one enforcement point
        every client funnels through. `END_TURN` is a submitted action like any
        other, and nothing runs a further seat's turn on this call's behalf."""
        if is_over(self.game):
            raise ValueError("the game is already over")
        if seat not in self.claimed_seats:
            raise ValueError(f"seat {seat} is not yours to play")
        # The held seat's own END_TURN, which releases the table. Checked
        # before `may_act` and the legality check, both of which refuse it.
        if self.awaiting_confirm == seat:
            if wire_to_action(wire).type is not ActionType.END_TURN:
                raise ValueError("end your setup turn first")
            self.awaiting_confirm = None
            return
        # Somebody else is still finishing their setup turn. Refused here, not
        # left to each client, so no one can outrace another seat's button.
        if self.awaiting_confirm is not None:
            raise ValueError(
                f"seat {self.awaiting_confirm} has not finished its setup turn"
            )
        if not may_act(self.game, seat):
            raise ValueError("it is not your turn to act")
        action = wire_to_action(wire)
        options = legal_actions(self.game, seat)
        if action not in options:
            raise ValueError(f"{action} is not a legal action right now")
        self.apply_action(seat, action)

    def apply_action(
        self, actor: int, action: Action, replay: tuple[Trade, ...] | None = None
    ) -> None:
        """Apply one action, recording, journalling and settling around it."""
        # Captured before apply(): `end_turn` increments `game.turns`, and so
        # `self.round`, which would misnumber the END_TURN line.
        round_num = self.round
        before = _snapshot(self.game)
        undoable = action.type in _UNDOABLE_BUILDS
        # Taken before apply(): it has to be the instant before this action.
        undo_point = (
            _UndoPoint(
                state=copy_state(self.game.state(0, hidden=False)),
                ledger=self.game.ledger.copy(),
                free_roads=self.game.free_roads,
                phase=self.game.phase,
                current_player=self.game.current_player,
                setup_step=self.game.setup_step,
                last_settlement=self.game.last_settlement,
                dev_card_played=self.game.dev_card_played,
                events=len(self.events),
                steps=self.steps,
                actor=actor,
            )
            if undoable
            else None
        )
        trades_before = len(self.game.trades)
        # `seat=actor` matters only for DISCARD, whose actor the position does
        # not fix: whoever submitted it loses the card.
        if replay is None:
            apply(self.game, action, seat=actor)
        else:
            # Replaying: the seats that published the vectors this game traded
            # on are gone, so the recorded exchanges are re-executed instead of
            # letting the engine's own event clear a different set.
            live, self.game.gates = self.game.gates, None
            try:
                apply(self.game, action, seat=actor)
            finally:
                self.game.gates = live
            apply_trades(self.game, replay)
        if action.type is ActionType.ROLL:
            self.last_roll_by_seat[actor] = self.game.last_roll
        if action.type is ActionType.END_TURN:
            self._close_round()  # a round is a per-turn thing
        self.events.append(
            _Event(
                round_num=round_num,
                actor=actor,
                action=action,
                before=before,
                after=_snapshot(self.game),
                last_roll=self.game.last_roll,
                trades=tuple(self.game.trades[trades_before:]),
            )
        )

        if self.journal is not None:
            self.journal.action(
                self.game,
                step=self.steps,
                round_num=round_num,
                actor=actor,
                action=action,
                before_hands=before.hands,
                before_held=before.held,
                trades=tuple(self.game.trades[trades_before:]),
            )
        self.steps += 1

        if is_over(self.game) and not self.ended:
            self.ended = True
            if self.journal is not None:
                self.journal.finish(self.game)

        # Every action decides this fresh: a qualifying placement that did not
        # win becomes the new and only undo point, anything else clears it. A
        # win is excluded because the result may already be on disk.
        self._undo = undo_point if (undo_point is not None and not is_over(self.game)) else None

        # Recomputed on every action: only a setup road that actually handed
        # the snake on leaves a turn owing a confirm. `to_move` still naming
        # `actor` is the turn of the snake, where a seat places twice.
        self.awaiting_confirm = (
            actor
            if (
                action.type is ActionType.SETUP_ROAD
                and not is_over(self.game)
                and to_move(self.game) != actor
                # A browser seat only: the hold protects the undo button. A
                # bot's `action_mask` has no END_TURN in this phase, so a held
                # bot seat would have no legal move at all, and an LLM seat
                # drives itself with nobody watching.
                and self.clients.get(actor, {}).get("kind") == "web"
            )
            else None
        )

        if replay is None and self.game.phase is Phase.MAIN:
            # A main-phase decision point may open this turn's round for a bot
            # actor, after this action's own event and journal line so a trade
            # it executes at once is its own step. Never on a replay, which
            # asks no gate.
            self.begin_round(entering=action.type in (ActionType.ROLL, ActionType.MOVE_ROBBER))

    def undo_last_build(self, seat: int) -> None:
        """Revert `seat`'s most recent placement, bank trade or Road
        Building/Knight play to exactly how the session stood before it, down
        to whose turn it is and the log line. `ValueError` unless the one
        pending undo point is this seat's — another seat's is never offered.
        Refused while the seat's own trade round is open, whose offer and
        answers were made against the position the undo would take away. The
        journal is append-only, so the undo goes in as its own entry."""
        if self._undo is None or self._undo.actor != seat:
            raise ValueError("nothing to undo")
        if not self.can_undo(seat):
            raise ValueError("close your open trade round first")
        point = self._undo
        self.game.set_state(point.state)
        self.game.ledger = point.ledger
        self.game.free_roads = point.free_roads
        self.game.phase = point.phase
        self.game.current_player = point.current_player
        self.game.setup_step = point.setup_step
        self.game.last_settlement = point.last_settlement
        self.game.dev_card_played = point.dev_card_played
        del self.events[point.events :]
        self.steps = point.steps
        if self.journal is not None:
            self.journal.undo(self.game, back_to=point.steps)
        self._undo = None
        # Undoing a setup road lands back before the handoff it was holding,
        # so there is no turn left to end.
        self.awaiting_confirm = None

    def can_undo(self, seat: int | None) -> bool:
        """Whether `undo_last_build(seat)` would succeed right now: the one
        pending undo point is `seat`'s, and no trade round of its own is
        open."""
        if seat is None or self._undo is None or self._undo.actor != seat:
            return False
        return self.open_round is None or self.open_round.offer.actor != seat

    def release(self, seat: int) -> None:
        """Let go of what a seat leaving the game (`api.Tables.leave_seat`)
        was holding: a setup turn it had not yet ended, which would hold the
        table for good, and its take-back, which would hand the turn back to
        a seat no longer playing."""
        if self.awaiting_confirm == seat:
            self.awaiting_confirm = None
        if self._undo is not None and self._undo.actor == seat:
            self._undo = None

    def rename(self, seat: int, name: str) -> None:
        """`seat`'s new display name, journalled so a reopened table keeps it
        (`journal.players`)."""
        self.player_names[seat] = name
        if self.journal is not None:
            self.journal.renamed(seat=seat, name=name)

    def _log_result(self, round_num: int) -> str:
        """The closing line, appended once the game is over, naming the winner
        the same way every other line names a seat."""
        winner = self.game.won_by
        if winner is None:
            return f"{round_num}\tGame over. Nobody won."
        who = _who(winner, self.seat_labels)
        # true state: victory points include hidden VP dev cards.
        points = victory_points(self.game.state(winner, hidden=False), winner)
        return f"{round_num}\t{who} wins with {points} points."

    def _public_mover(self) -> int:
        """Whose move it is. During `Phase.DISCARD` this is one owing seat out
        of possibly several, and a label rather than a permission: what a
        client may do comes off `legal_actions`, and `discard_quota` says which
        seats are outstanding."""
        game = self.game
        if is_over(game):
            return game.current_player
        # A seat holding its setup turn open still has the move, whatever the
        # engine says: the snake advanced when the road was placed, but the
        # table waits until that seat ends its turn. Answered here so every
        # reader of "whose move is it" agrees.
        if self.awaiting_confirm is not None:
            return self.awaiting_confirm
        return to_move(game)

    def state_view(
        self, viewer: int | None = None, *, omniscient: bool = False, finished: bool = False
    ) -> dict:
        """The whole game as `viewer` is allowed to see it.

        `viewer` is a seat, or `None` for someone watching without one, and it
        decides three things and nothing else: whose hand and true
        victory-point count come back in full, which seat's port ratios the
        trade panel gets, and how the transcript redacts. Everything else is
        public and identical to every reader.

        `omniscient` drops the first and third and is **only ever for a reader
        outside the game**: it raises if `viewer` is a seat, and one route
        passes it, the token-free `GET /api/table/<code>`. A finished game
        (`over`) gets the same drops, seat or no seat, and `finished` is that
        rule reaching a position that is not itself the end, which happens only
        in replay. Disclosure follows the game, not the position; `game_over`
        stays the position's own.
        """
        if omniscient and viewer is not None:
            raise ValueError("an omniscient view belongs to no seat")
        labels = self.seat_labels
        game = self.game
        # true state: the per-viewer filtering happens below, not here.
        state = game.state(0, hidden=False)
        over = is_over(game)
        # What may be disclosed, as against what the position is.
        revealed = over or finished
        players = []
        # Public: a route's length is visible on the board.
        lengths = road_lengths(state)
        for p in range(state.num_players):
            # A finished game discloses every seat -- as far as the state
            # holds it: on an observed one (`game.observe`) another seat's
            # cards are counts, and a count is all there is to disclose.
            reveal = (revealed or omniscient or p == viewer) and _concrete(state, p)
            seat_ledger = game.ledger.seats[p]
            entry = {
                "seat": p,
                "bot": self.bot_names.get(p),
                "name": labels.get(p),
                "last_roll": self.last_roll_by_seat.get(p),
                "victory_points": victory_points(state, p)
                if reveal
                else public_victory_points(state, p),
                "knights_played": state.knights_played[p],
                "road_length": lengths[p],
                "longest_road": state.longest_road_holder == p,
                "largest_army": state.largest_army_holder == p,
                "hand_size": hand_size(state, p),
                "dev_card_count": dev_count(state, p),
                # The public-knowledge ledger (`hexset.ledger`), public for
                # every seat, reveal or not: `known` is a certified
                # per-resource floor, `unknown` what the log cannot yet type.
                "known": dict(zip(RESOURCE_NAMES, seat_ledger.known)),
                "unknown": seat_ledger.unknown,
            }
            if reveal:
                entry["hand"] = dict(zip(RESOURCE_NAMES, state.hands[p]))
                entry["dev_cards"] = dict(zip(DEV_CARD_NAMES, holdings(state, p)))
            players.append(entry)

        return {
            "phase": game.phase.name,
            "current_player": game.current_player,
            "to_move": None if over else self._public_mover(),
            "seat": viewer,
            "claimed_seats": sorted(self.claimed_seats),
            # Seats closed outright and permanently retired from this game;
            # `api.Table.join` refuses one as it refuses an occupied seat.
            "locked": sorted(game.locked),
            # Whether the first move has been made. Until then a seat can be
            # closed, reopened or given a bot; after, the seats are fixed.
            "started": self.steps > 0,
            "winner": game.won_by,
            "game_over": over,
            # Whether `POST /api/undo` would succeed right now; a session
            # convenience, not a rule, so not in `legal_actions`.
            "can_undo": self.can_undo(viewer),
            # The seat whose setup turn is placed but not yet ended, visible
            # to every reader, not just a bool for itself.
            "awaiting_confirm": self.awaiting_confirm,
            # One lap of the table (the `round` property), matching the log
            # lines' own round-number tags.
            "round": self.round,
            # The rule the game is played to, so a client need not assume ten.
            "winning_points": state.rules.winning_points,
            "last_roll": game.last_roll,
            "robber": state.robber,
            "vertex_owner": state.vertex_owner,
            "vertex_building": state.vertex_building,
            "edge_owner": state.edge_owner,
            "bank": dict(zip(RESOURCE_NAMES, state.bank)),
            "dev_cards_remaining": len(state.deck),
            "discard_quota": list(game.discard_quota),
            "trade_ratios": dict(zip(RESOURCE_NAMES, trade_ratios(state, viewer)))
            if viewer is not None
            else {},
            "players": players,
            # What the engine cleared this turn: public, so not filtered.
            "trades": [
                {
                    "a": t.a,
                    "b": t.b,
                    "gave": [max(0, -n) for n in t.received],
                    "got": [max(0, n) for n in t.received],
                }
                for t in game.trades
            ],
            # The offers standing against `viewer` unanswered: `bundle` is
            # signed towards `actor` and echoed back verbatim by
            # `POST .../trade/round/answer`.
            "pending": [
                {"actor": t.a, "bundle": list(t.received),
                 **({"any": self._any_of(t)} if self._any_of(t) else {})}
                for t in self.pending_for(viewer)
            ]
            if viewer is not None
            else [],
            # `viewer`'s own open round, only for the seat that broadcast it:
            # the offer, every answer so far (a pass's `bundle` is null, so a
            # seat that declined is told apart from one still to answer) and
            # the manual seats still to answer.
            "trade_round": (
                {
                    "offer": {"actor": self.open_round.offer.actor,
                              "bundle": list(self.open_round.offer.received),
                              **({"any": self.open_round.offer.any} if self.open_round.offer.any else {})},
                    "responses": [
                        {"seat": r.seat, "kind": r.kind, "bundle": None if r.bundle is None else list(r.bundle)}
                        for r in self.open_round.responses
                    ],
                    "awaiting": sorted(self.open_round.awaiting),
                }
                if self.open_round is not None and self.open_round.offer.actor == viewer
                else None
            ),
            "legal_actions": self.legal_wire_actions(viewer),
            "log": self.log_for(viewer, omniscient=omniscient or revealed),
        }

    def log_for(self, viewer: int | None, *, omniscient: bool = False) -> list[str]:
        """The sidebar transcript as `viewer` should see it; `None` is a reader
        with no seat, who is owed the least, and `omniscient` the most. A
        discard round still in progress is told to nobody, spectator
        included."""
        # true state: the board is public.
        lines = render_log(
            self.events,
            self.game.state(0, hidden=False).board,
            self.seat_labels,
            viewer,
            omniscient=omniscient,
            discards_open=bool(players_owing_discards(self.game)),
        )
        if is_over(self.game):
            # The round the final action fell in, not `self.round`.
            lines.append(self._log_result(self.events[-1].round_num if self.events else 0))
        return lines
