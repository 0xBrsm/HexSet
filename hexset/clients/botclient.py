"""An example client of the HTTP API: one seat, played over `/api/*` alone.

It joins a game and, read by read, does what `GET /api/state`'s `your_move`
says the seat owes:

- `act` and `discard`: an action from `legal_actions`, a build when one is
  legal, else any. Once a turn in `MAIN` it first offers one card of the
  kind it holds most of for one of a kind it holds none of, and moves on
  only once every seat has answered.
- `answer_trade`: accepts an offer it can cover, counters an open offer
  with named cards it holds, and passes the rest.
- `choose_trade`: takes the first acceptance or counter it can cover, else
  declines them all.
- `wait`: parks on `?after=<version>` until the table moves.

It decides nothing well; it shows every route a seat plays through. It
imports nothing from the engine but the token header's name, so a client in
any language makes the same requests:

    python -m hexset.clients.botclient --url http://127.0.0.1:8770 --game abcdef
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from hexset.server.constants import TOKEN_HEADER

__all__ = [
    "client_of",
    "Transport",
    "HttpTransport",
    "BUILDS",
    "hand_of",
    "choose_action",
    "covers",
    "counter_for",
    "answer_for",
    "offer_for",
    "pick_for",
    "play",
]


def client_of(secret: str) -> dict:
    """The `client` wire field `POST /api/join` reads: kind `"api"`, `id` the
    sha256 of `secret`."""
    return {"id": hashlib.sha256(secret.encode("utf-8")).hexdigest(), "kind": "api"}


# --- Transport: the API's two verbs ------------------------------------------


class Transport(Protocol):
    """The API's two verbs: `get` and `post` a path with the seat's token,
    returning the JSON body, a refusal as `{"error": ...}`."""

    def get(self, path: str, token: str) -> dict: ...
    def post(self, path: str, token: str, body: dict) -> dict: ...


@dataclass
class HttpTransport:
    """A real client of a running server, over the public `/api/*` surface.
    A refusal comes back as its `{"error": ...}` body, whatever the status."""

    base_url: str
    timeout: float = 30.0

    def get(self, path: str, token: str) -> dict:
        return self._request("GET", path, token, None)

    def post(self, path: str, token: str, body: dict) -> dict:
        return self._request("POST", path, token, body)

    def _request(self, method: str, path: str, token: str, body: dict | None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if token:
            request.add_header(TOKEN_HEADER, token)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return json.loads(error.read().decode("utf-8"))


# --- Decisions: what to send, from the view alone ------------------------------
#
# Every bundle on the wire is five counts in the order the view's `hand`
# lists the resources, signed towards the offer's actor: positive counts go
# to the actor, negative ones leave it.

#: Preferred to any other legal action, best first.
BUILDS = ("BUILD_CITY", "BUILD_SETTLEMENT", "SETUP_SETTLEMENT", "BUY_DEV_CARD", "BUILD_ROAD", "SETUP_ROAD")


def hand_of(view: dict) -> list[int]:
    """This seat's own cards, in wire order."""
    me = next(p for p in view["players"] if p["seat"] == view["seat"])
    return list(me["hand"].values())


def choose_action(legal: list[dict], rng: random.Random) -> dict:
    """A build when one is legal, the best kind first; else any legal action.
    Sent back exactly as `legal_actions` listed it."""
    for kind in BUILDS:
        builds = [a for a in legal if a["type"] == kind]
        if builds:
            return rng.choice(builds)
    return rng.choice(legal)


def covers(hand: list[int], bundle: list[int], *, actor: bool) -> bool:
    """Whether `hand` holds its side of `bundle`: the actor gives the negative
    counts, the answerer the positive ones."""
    gives = [max(0, -n) if actor else max(0, n) for n in bundle]
    return all(held >= n for held, n in zip(hand, gives))


def counter_for(bundle: list[int], any_cards: int, hand: list[int]) -> list[int] | None:
    """An open offer's bundle with its `any_cards` named, on resources the
    offer leaves untouched, so both sides stay disjoint. A positive
    `any_cards` is cards the actor takes, which this seat names from its own
    hand; a negative one is cards the actor gives, which this seat asks for
    in the kind it holds fewest of. `None` where its hand cannot cover its
    side."""
    counter = list(bundle)
    free = [r for r, n in enumerate(bundle) if n == 0]
    if any_cards > 0:
        owed = any_cards
        for r in sorted(free, key=lambda r: -hand[r]):
            moved = min(hand[r], owed)
            counter[r] += moved
            owed -= moved
        if owed:
            return None
    else:
        if not free:
            return None
        counter[min(free, key=lambda r: hand[r])] += any_cards
    return counter if covers(hand, counter, actor=False) else None


def answer_for(offer: dict, hand: list[int]) -> dict:
    """The `.../trade/round/answer` body for one `pending` entry: accept what
    this seat covers, counter an open offer, pass the rest."""
    body = {"actor": offer["actor"], "received": offer["bundle"], "kind": "pass"}
    if offer.get("any"):
        counter = counter_for(offer["bundle"], offer["any"], hand)
        if counter is not None:
            body.update(kind="counter", bundle=counter)
    elif covers(hand, offer["bundle"], actor=False):
        body["kind"] = "accept"
    return body


def offer_for(hand: list[int], rng: random.Random) -> dict | None:
    """The `.../trade/round` body: one card of the kind this seat holds most
    of for one of a kind it holds none of. `None` with nothing to give or
    nothing missing."""
    missing = [r for r, n in enumerate(hand) if n == 0]
    if not missing or not any(hand):
        return None
    give = [0] * len(hand)
    want = [0] * len(hand)
    give[max(range(len(hand)), key=lambda r: hand[r])] = 1
    want[rng.choice(missing)] = 1
    return {"give": give, "want": want}


def pick_for(trade_round: dict, hand: list[int]) -> dict:
    """The `.../trade/round/choose` body for this seat's own answered round:
    the first acceptance or counter it covers, echoed exactly, else a
    decline."""
    for response in trade_round["responses"]:
        bundle = response["bundle"]
        if response["kind"] != "pass" and covers(hand, bundle, actor=True):
            return {"seat": response["seat"], "bundle": bundle}
    return {"decline": True}


# --- The loop -------------------------------------------------------------------


def play(
    transport: Transport,
    token: str,
    *,
    rng: random.Random | None = None,
    poll_interval: float = 10.0,
    stop: threading.Event | None = None,
) -> dict:
    """Play `token`'s seat until the game ends or `stop` is set, and return
    the last view read. A refused request is not retried as it was: the
    client waits for the table to move and reads it again."""
    rng = rng or random.Random()
    stop = stop or threading.Event()
    offered: set[int] = set()  # the rounds this seat has offered in
    view: dict = {}
    while not stop.is_set():
        view = transport.get("/api/state", token)
        if "error" in view:
            raise RuntimeError(view["error"])
        move = view["your_move"]
        if move == "game_over":
            break
        trades = f"/api/games/{view['code']}/trade/round"
        reply = None
        if move == "answer_trade":
            reply = transport.post(f"{trades}/answer", token, answer_for(view["pending"][0], hand_of(view)))
        elif move == "choose_trade":
            reply = transport.post(f"{trades}/choose", token, pick_for(view["trade_round"], hand_of(view)))
        elif move == "act" and view.get("trade_round"):
            pass  # its own offer is still out: wait for the answers
        elif move == "act" and view["phase"] == "MAIN" and view["round"] not in offered:
            offered.add(view["round"])
            body = offer_for(hand_of(view), rng)
            if body is None:
                continue  # nothing to offer: on to the turn's actions
            reply = transport.post(trades, token, body)
        elif move in ("act", "discard"):
            reply = transport.post("/api/action", token, {"action": choose_action(view["legal_actions"], rng)})
        if reply is None or "error" in reply:
            transport.get(f"/api/state?after={view['version']}&wait={poll_interval}", token)
    return view


def _main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Base URL of a running `python -m hexset.server.web`.")
    parser.add_argument("--game", required=True, help="The game's six-character code.")
    parser.add_argument("--name", default=None, help="Display name for this seat.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random choices.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Longest a parked read waits for the table to change (seconds).",
    )
    parser.add_argument(
        "--client-secret",
        default=None,
        help=(
            "Identifies this client for POST /api/reclaim (see api.py's "
            "module docstring): the seat's client.id is sha256 of this. "
            "Defaults to --name, or `botclient` without one."
        ),
    )
    args = parser.parse_args(argv)

    secret = args.client_secret or args.name or "botclient"
    transport = HttpTransport(args.url.rstrip("/"))
    joined = transport.post(
        "/api/join", "", {"code": args.game, "name": args.name, "client": client_of(secret)}
    )
    if "error" in joined:
        raise SystemExit(f"could not join {args.game}: {joined['error']}")
    print(f"joined {args.game} as seat {joined['seat']}", file=sys.stderr)
    try:
        final = play(transport, joined["token"], rng=random.Random(args.seed), poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        return
    print(f"game over, winner seat {final.get('winner')}", file=sys.stderr)


if __name__ == "__main__":
    _main()
