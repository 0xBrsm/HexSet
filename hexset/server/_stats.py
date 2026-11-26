"""A game's dice and robber numbers, read from its journal (`_journal`).

Every action line carries every seat's hand after it, and a roll its number,
so the numbers follow the game as played, takebacks included (an `undo` line
drops the steps it takes back, as `_journal.replayable_rounds` does).

Per seat:

- `exp`: cards the seat's buildings pay on average on a roll other than 7,
  summed over the game's rolls other than 7, the robber's hex left out;
  `exp_all` the same with the robber's hex counted;
- `act`: cards the seat took on those rolls;
- `blocked`: cards a roll would have paid from the hex under the robber;
- `stolen`: cards the robber took from the seat, `gained` cards the seat took
  by it;
- `on`: robber moves by another seat onto a hex the seat has a building on, of
  `moves`, every robber move by another seat;
- `discarded`: cards the seat gave up to a seven.
"""

from __future__ import annotations

from hexset.actions import ActionType

#: Ways to roll each number with two dice.
WAYS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

_SETTLE = {ActionType.SETUP_SETTLEMENT.name, ActionType.BUILD_SETTLEMENT.name}


def played(events: list[dict]) -> list[dict]:
    """The action and trade lines of the game as it stands: an undo line drops
    every step from its `back_to` on."""
    steps: list[dict] = []
    for event in events:
        kind = event.get("kind")
        if kind in ("action", "trade"):
            steps.append(event)
        elif kind == "undo":
            del steps[event["back_to"]:]
    return steps


def game_stats(events: list[dict], board, upto: int | None = None) -> dict:
    """Every seat's numbers (module docstring) for the journal `events` of a
    game on `board` (the table's board: its tokens and which corners touch
    each hex), and `rounds`, the round of the newest step; `upto`, only its
    first `upto` steps (a past position's numbers)."""
    header = events[0] if events and events[0].get("kind") == "game" else {}
    n = int(header.get("num_players", 4))
    topology = board.topology
    keys = ("exp", "exp_all", "act", "blocked", "stolen", "gained", "on", "moves", "discarded")
    out = {key: [0.0 if key.startswith("exp") else 0 for _ in range(n)] for key in keys}
    building: dict[int, tuple[int, int]] = {}   # vertex -> (seat, 1 settlement / 2 city)
    robber = header.get("robber", next((h for h in range(board.num_hexes) if not board.tokens[h]), 0))
    hands = [0] * n
    rounds = 0
    for event in played(events)[:upto]:
        after = [sum(hand) for hand in event.get("hands", [])] or hands
        rounds = int(event.get("round", rounds))
        if event.get("kind") != "action":
            hands = after
            continue
        kind, actor, a = event["type"], event.get("actor"), event.get("a", 0)
        if kind == ActionType.ROLL.name and event.get("roll") not in (None, 7):
            roll = event["roll"]
            for h in range(board.num_hexes):
                token = board.tokens[h]
                if token not in WAYS:
                    continue
                for v in topology.hex_vertices[h]:
                    if v in building:
                        seat, size = building[v]
                        # Per roll other than 7: the chance of this number among them.
                        share = WAYS[token] / 30 * size
                        out["exp_all"][seat] += share
                        if h != robber:
                            out["exp"][seat] += share
                        elif token == roll:
                            out["blocked"][seat] += size
            for seat in range(n):
                out["act"][seat] += max(0, after[seat] - hands[seat])
        elif kind == ActionType.MOVE_ROBBER.name:
            here = {building[v][0] for v in topology.hex_vertices[a] if v in building}
            for seat in range(n):
                if seat != actor:
                    out["moves"][seat] += 1
                    out["on"][seat] += seat in here
                    out["stolen"][seat] += max(0, hands[seat] - after[seat])
            if actor is not None:
                out["gained"][actor] += max(0, after[actor] - hands[actor])
            robber = a
        elif kind in _SETTLE:
            building[a] = (actor, 1)
        elif kind == ActionType.BUILD_CITY.name:
            building[a] = (actor, 2)
        elif kind == ActionType.DISCARD.name and actor is not None:
            out["discarded"][actor] += max(0, hands[actor] - after[actor])
        hands = after
    out["rounds"] = rounds
    return out


def seat_stats(events: list[dict], board, seat: int, upto: int | None = None) -> dict | None:
    """`seat`'s numbers beside its opponents' as the page's review shows them,
    in cards: `you` and `them` (the one opponent, or the opponents' mean),
    each with `dice` (cards the rolls paid, the robber's blocks put back,
    less what they pay on average with no robber on the board), `robber`
    (cards taken by the seat's steals, less cards the robber blocked and
    cards stolen from the seat), `net` (the two together), `on` and `moves`
    (robber moves onto the seat of every move by another seat) and
    `discarded`; and `rounds`. `upto` stops after that many steps (the
    numbers at a past position). `None` for a game with no header."""
    if not events or events[0].get("kind") != "game":
        return None
    stats = game_stats(events, board, upto)
    header = events[0]
    locked = set(header.get("locked") or ())
    seats = [s for s in range(int(header.get("num_players", 4))) if s not in locked]
    others = [s for s in seats if s != seat]

    def one(s: int) -> dict:
        dice = stats["act"][s] + stats["blocked"][s] - stats["exp_all"][s]
        robber = stats["gained"][s] - stats["blocked"][s] - stats["stolen"][s]
        return {"dice": round(dice, 2), "robber": robber, "net": round(dice + robber, 2),
                "on": stats["on"][s], "moves": stats["moves"][s], "discarded": stats["discarded"][s]}

    def mean(rows: list[dict]) -> dict:
        out = {}
        for key in rows[0]:
            values = [r[key] for r in rows if r[key] is not None]
            out[key] = sum(values) / len(values) if values else None
        return out

    return {"you": one(seat), "them": mean([one(s) for s in others]) if others else None,
            "rounds": stats["rounds"]}
