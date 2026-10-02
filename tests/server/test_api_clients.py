"""Every client of the API plays a whole game at a table where bots trade.

A served table holds a bot's turn while its offer waits on a client's seat,
and refuses every move until that seat answers, so a client that does not
meet every duty `your_move` names stalls the game for everyone. Each client
here plays to the end of a game beside `test-trader` seats, which offer and
answer real trades.
"""

from __future__ import annotations

import random
import threading

from hexset.clients import botclient
from hexset.server import mcptools
from hexset.server._runner import LocalTransport

from conftest import new_tables

TRADER = "test-trader"
GAME_SECONDS = 240
# Player-turns before the game ends, as a win ends it: enough laps for every
# trade route to come up, short of the served default so the test is quick.
TURN_CAP = 40


class Recording:
    """A transport that keeps every POST its client made, and whether the
    server took it."""

    def __init__(self, inner):
        self.inner = inner
        self.posts: list[tuple[str, dict, bool]] = []

    def get(self, path, token):
        return self.inner.get(path, token)

    def post(self, path, token, body):
        reply = self.inner.post(path, token, body)
        self.posts.append((path, body, "error" not in reply))
        return reply


def _played(posts, suffix, **match):
    return [
        body for path, body, ok in posts
        if ok and path.endswith(suffix) and all(body.get(k) == v for k, v in match.items())
    ]


def test_two_example_clients_play_a_whole_game_beside_trading_bots():
    """Two `botclient` seats and two `test-trader` seats. The clients answer
    the bots' offers and each other's, make their own, and pick among the
    answers, so every trade route is played from both sides."""
    tables = new_tables()
    created = tables.handle("POST", "/api/games", {"bots": [TRADER, TRADER]}, None)
    tables.get(created["code"]).session.game.turn_cap = TURN_CAP
    joined = tables.handle("POST", "/api/join", {"code": created["code"]}, None)
    clients = [Recording(LocalTransport(tables)) for _ in range(2)]
    finals: list[dict] = []

    def run(transport, token, seed):
        finals.append(botclient.play(transport, token, rng=random.Random(seed), poll_interval=1.0))

    threads = [
        threading.Thread(target=run, args=(client, token, seed), daemon=True)
        for seed, (client, token) in enumerate(zip(clients, (created["token"], joined["token"])))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=GAME_SECONDS)
    assert not any(thread.is_alive() for thread in threads), "a client stalled before the game ended"
    assert len(finals) == 2 and all(view["game_over"] for view in finals)

    posts = [post for client in clients for post in client.posts]
    assert _played(posts, "/trade/round"), "no client offered"
    assert _played(posts, "/trade/round/answer", kind="accept"), "no client accepted"
    assert _played(posts, "/trade/round/answer", kind="pass"), "no client passed"
    assert [body for body in _played(posts, "/trade/round/choose") if "seat" in body], (
        "no client executed an answer to its own offer"
    )


# --- An MCP seat, through the tools alone -----------------------------------



def _rows(table: str) -> list[dict[str, str]]:
    """An MCP table, `(keys):row|row` behind an optional `KIND:`, as one dict
    of raw cells per row."""
    head, body = table[table.index("(") + 1:].split("):", 1)
    keys = head.split(",")
    return [dict(zip(keys, row.split(","))) for row in body.split("|")]


def _mcp_act(reply: dict, rng: random.Random) -> dict:
    """A build when one is legal, as the example client plays: `summary`
    names the settlement, city and robber moves with their `act` index, and
    `legal_actions` the rest."""
    summary = reply.get("summary") or {}
    if summary.get("spots"):
        return {"index": int(rng.choice(_rows(summary["spots"]))["index"])}
    legal = reply["legal_actions"]
    for kind in botclient.BUILDS:
        if kind in legal:
            return {"index": int(rng.choice(_rows(legal[kind]))["index"])}
    options = [int(row["index"]) for table in legal.values() for row in _rows(table)]
    if summary.get("robber"):
        options += [int(o.split(":")[0]) for row in _rows(summary["robber"]) for o in row["options"].split(";")]
    return {"index": rng.choice(options)}


def _mcp_discard(reply: dict) -> dict:
    me = next(p for p in reply["players"] if p["seat"] == reply["seat"])
    hand = dict(me.get("hand") or {})
    owed = reply["discard_quota"][reply["seat"]]
    cards: dict[str, int] = {}
    while owed:
        name = max(hand, key=hand.get)
        hand[name] -= 1
        cards[name] = cards.get(name, 0) + 1
        owed -= 1
    return {"cards": cards}


def test_an_mcp_seat_plays_a_whole_game_beside_trading_bots():
    """One MCP seat and three `test-trader` seats, the seat played only
    through the tools and the `your_move` each reply carries: it accepts
    every offer it can and passes the rest."""
    tables = new_tables()
    session = mcptools.Session()
    rng = random.Random(0)
    reply = mcptools.call_tool(tables, session, "new_game", {"identity": "conformance", "opponents": [TRADER] * 3})
    tables.get(session.code).session.game.turn_cap = TURN_CAP
    answered: list[str] = []
    for _ in range(20_000):
        move = reply["your_move"]
        if move == "game_over":
            break
        if move == "act":
            reply = mcptools.call_tool(tables, session, "act", _mcp_act(reply, rng))
        elif move == "discard":
            reply = mcptools.call_tool(tables, session, "discard", _mcp_discard(reply))
        elif move == "answer_trade":
            kind = "accept" if reply["pending"][0].get("can_accept") else "pass"
            answered.append(kind)
            reply = mcptools.call_tool(tables, session, "answer_trade", {"index": 0, "kind": kind})
        elif move == "choose_trade":
            reply = mcptools.call_tool(tables, session, "choose_trade", {"decline": True})
        else:
            reply = mcptools.call_tool(tables, session, "wait_for_turn", {})
    assert reply["your_move"] == "game_over"
    assert "accept" in answered, "the seat never had an offer it could take"
