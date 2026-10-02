"""The example API client's decisions, read off a view alone. That it plays
a whole game through every route is `tests/server/test_api_clients.py`."""

from __future__ import annotations

import hashlib
import random

import pytest

from hexset.clients import botclient
from hexset.clients.botclient import (
    answer_for,
    choose_action,
    client_of,
    counter_for,
    offer_for,
    pick_for,
)


def test_client_of_hashes_the_secret_and_names_kind_api():
    assert client_of("s") == {"id": hashlib.sha256(b"s").hexdigest(), "kind": "api"}


@pytest.mark.parametrize(
    ("argv", "secret"),
    [([], "botclient"), (["--name", "Ada"], "Ada"), (["--name", "Ada", "--client-secret", "k"], "k")],
)
def test_main_joins_with_the_secret_it_was_given_or_its_name(monkeypatch, argv, secret):
    captured: dict = {}

    class FakeTransport:
        def __init__(self, base_url):
            captured["base_url"] = base_url

        def post(self, path, token, body):
            captured["path"] = path
            captured["body"] = body
            return {"error": "stop here: this test only checks the join payload"}

    monkeypatch.setattr(botclient, "HttpTransport", FakeTransport)
    with pytest.raises(SystemExit):
        botclient._main(["--url", "http://x/", "--game", "abcdef", *argv])
    assert captured["base_url"] == "http://x"
    assert captured["path"] == "/api/join"
    assert captured["body"]["client"] == client_of(secret)


def test_a_build_is_chosen_over_anything_else_and_the_best_build_first():
    legal = [{"type": "END_TURN"}, {"type": "BUILD_ROAD", "a": 3}, {"type": "BUILD_CITY", "a": 7}]
    assert choose_action(legal, random.Random(0)) == {"type": "BUILD_CITY", "a": 7}
    assert choose_action(legal[:2], random.Random(0)) == {"type": "BUILD_ROAD", "a": 3}
    assert choose_action(legal[:1], random.Random(0)) == {"type": "END_TURN"}


def test_an_offer_the_hand_covers_is_accepted_and_one_it_cannot_is_passed():
    wood_for_ore = {"actor": 1, "bundle": [-1, 0, 0, 0, 1]}  # the actor gives wood, gets ore
    assert answer_for(wood_for_ore, [0, 0, 0, 0, 1])["kind"] == "accept"
    assert answer_for(wood_for_ore, [3, 0, 0, 0, 0])["kind"] == "pass"
    assert answer_for(wood_for_ore, [0, 0, 0, 0, 1])["received"] == [-1, 0, 0, 0, 1]


def test_an_open_offer_is_countered_with_named_cards_on_untouched_resources():
    # "My sheep for any card": the actor takes one card this seat names.
    assert counter_for([0, 0, -1, 0, 0], 1, [0, 2, 1, 0, 0]) == [0, 1, -1, 0, 0]
    # "Any card for your ore": this seat names the card it wants.
    assert counter_for([0, 0, 0, 0, 1], -1, [2, 1, 3, 0, 1]) == [0, 0, 0, -1, 1]
    # A hand without the ore asked for, or without a card to name, passes.
    assert counter_for([0, 0, 0, 0, 1], -1, [2, 1, 3, 0, 0]) is None
    assert counter_for([0, 0, -1, 0, 0], 1, [0, 0, 1, 0, 0]) is None
    answered = answer_for({"actor": 0, "bundle": [0, 0, -1, 0, 0], "any": 1}, [0, 2, 1, 0, 0])
    assert answered == {"actor": 0, "received": [0, 0, -1, 0, 0], "kind": "counter", "bundle": [0, 1, -1, 0, 0]}


def test_the_offer_gives_the_most_held_kind_for_a_missing_one():
    body = offer_for([3, 1, 0, 2, 1], random.Random(0))
    assert body == {"give": [1, 0, 0, 0, 0], "want": [0, 0, 1, 0, 0]}
    assert offer_for([1, 1, 1, 1, 1], random.Random(0)) is None  # nothing missing
    assert offer_for([0, 0, 0, 0, 0], random.Random(0)) is None  # nothing to give


def test_the_pick_takes_the_first_answer_the_hand_covers_else_declines():
    trade_round = {
        "responses": [
            {"seat": 1, "kind": "pass", "bundle": None},
            {"seat": 2, "kind": "counter", "bundle": [-2, 0, 0, 0, 1]},
            {"seat": 3, "kind": "accept", "bundle": [-1, 0, 0, 0, 1]},
        ]
    }
    assert pick_for(trade_round, [1, 0, 0, 0, 0]) == {"seat": 3, "bundle": [-1, 0, 0, 0, 1]}
    assert pick_for(trade_round, [2, 0, 0, 0, 0]) == {"seat": 2, "bundle": [-2, 0, 0, 0, 1]}
    assert pick_for(trade_round, [0, 0, 0, 0, 0]) == {"decline": True}
