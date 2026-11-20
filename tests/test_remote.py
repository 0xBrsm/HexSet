# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.remote`: a seat whose server decides, and its fallback whenever the server cannot."""

from __future__ import annotations

import json
import random
import threading
import time

import pytest

from hexset import remote as R
from hexset.actions import legal_actions
from hexset.board.board import random_base_board
from hexset.game import observe, start


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(R, "HEARTBEAT_S", 0.05)
    monkeypatch.setattr(R, "LINK_DOWN_S", 0.4)
    monkeypatch.setattr(R, "RETRY_S", 0.05)
    monkeypatch.setattr(R, "REFUSED_RETRY_S", 0.2)
    monkeypatch.setattr(R, "CONNECT_S", 1.0)


class Bot:
    """Answers with its own name; `delay` seconds per choice; keeps its notes and a decision log."""

    def __init__(self, name, delay=0.0, started=None):
        self.name, self.delay, self.started = name, delay, started
        self.decision_log, self.notes, self.chosen = [], [], 0
        self.policy = None

    def choose(self, game):
        if self.started is not None:
            self.started.set()
        time.sleep(self.delay)
        self.chosen += 1
        self.decision_log.append({"by": self.name, "n": self.chosen})
        return (self.name, legal_actions(game)[0])

    def gains_many(self, game, bundles):
        return [self.name] * len(bundles)

    def seat_at(self, game):
        self.notes.append(("seat_at",))

    def seen(self, game, actor, action, viewer):
        self.notes.append(("seen", actor))


def game():
    played = start(random_base_board(random.Random(1)), 2, random.Random(1))
    return observe(played, 0)


def wait_up(bot, timeout=3.0):
    end = time.monotonic() + timeout
    while not bot.link_up():
        assert time.monotonic() < end, "the link never came up"
        time.sleep(0.01)


def serve(build, code="v1", port=0):
    return R.Server(build, port=port, code=code).start()


def test_the_server_decides_while_the_link_is_up_and_its_logs_come_back(tmp_path):
    built = []
    server = serve(lambda spec: built.append(spec) or Bot("server"))
    log = tmp_path / "calls.jsonl"
    bot = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "g1", {"model": "x"}, Bot("local"),
                      code="v1", log=log)
    wait_up(bot)
    who, action = bot.choose(game())
    assert who == "server" and built == [{"model": "x"}]
    assert bot.decision_log == [{"by": "server", "n": 1}]
    assert bot.gains_many(game(), [1, 2]) == ["server", "server"]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["source"] for r in rows] == ["remote", "remote"] and rows[0]["call"] == "choose"
    bot.close(); server.close()


def test_with_no_server_the_fallback_answers_at_once():
    bot = R.RemoteBot("127.0.0.1:1", "g1", {}, Bot("local"), code="v1")
    started = time.monotonic()
    assert bot.choose(game())[0] == "local"
    assert time.monotonic() - started < 0.5 and bot.counts == {"remote": 0, "local": 1}
    bot.close()


def test_a_link_that_drops_mid_decision_falls_back_within_the_heartbeat_window():
    started = threading.Event()
    server = serve(lambda spec: Bot("server", delay=10.0, started=started))
    bot = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "g1", {}, Bot("local"), code="v1")
    wait_up(bot)
    threading.Timer(0.3, server.close).start()
    t0 = time.monotonic()
    assert bot.choose(game())[0] == "local"
    assert time.monotonic() - t0 < 0.3 + R.LINK_DOWN_S + 0.5
    bot.close()


def test_a_slow_server_loses_the_move_to_the_clock(monkeypatch):
    monkeypatch.setattr(R, "CLOCK_MARGIN_S", 3.0)
    server = serve(lambda spec: Bot("server", delay=5.0))
    bot = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "g1", {}, Bot("local"), code="v1")
    wait_up(bot)
    deadline = time.monotonic() + 3.5
    bot.set_clock(lambda: deadline - time.monotonic())
    t0 = time.monotonic()
    assert bot.choose(game())[0] == "local"
    assert 0.3 < time.monotonic() - t0 < 1.5
    bot.close(); server.close()


def test_a_server_running_other_code_is_refused_and_the_fallback_plays():
    server = serve(lambda spec: Bot("server"), code="v2")
    bot = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "g1", {}, Bot("local"), code="v1")
    time.sleep(0.3)
    assert not bot.link_up() and bot.choose(game())[0] == "local"
    bot.close(); server.close()


def test_urgent_calls_go_before_others_waiting_in_the_queue():
    order, gate = [], threading.Event()

    class Recording(Bot):
        def choose(self, game):
            gate.wait(5)
            order.append(self.name)
            return self.name

    server = serve(lambda spec: Recording(spec["who"]))
    first = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "a", {"who": "first"}, Bot("l"), code="v1",
                        priority=R.OTHER)
    other = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "b", {"who": "other"}, Bot("l"), code="v1",
                        priority=R.OTHER)
    urgent = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "c", {"who": "urgent"}, Bot("l"),
                         code="v1", priority=R.URGENT)
    for bot in (first, other, urgent):
        wait_up(bot)
    g = game()
    threads = [threading.Thread(target=first.choose, args=(g,))]
    threads[0].start(); time.sleep(0.2)                  # running: holds the one job thread
    threads.append(threading.Thread(target=other.choose, args=(g,))); threads[-1].start(); time.sleep(0.1)
    threads.append(threading.Thread(target=urgent.choose, args=(g,))); threads[-1].start(); time.sleep(0.1)
    gate.set()
    for t in threads:
        t.join(5)
    assert order == ["first", "urgent", "other"]
    for bot in (first, other, urgent):
        bot.close()
    server.close()


def test_notes_reach_both_bots_in_order_and_a_fresh_server_bot_gets_them_all_again():
    local, made = Bot("local"), []
    server = serve(lambda spec: made.append(Bot("server")) or made[-1])
    port = server.address[1]
    bot = R.RemoteBot(f"127.0.0.1:{port}", "g1", {}, local, code="v1")
    wait_up(bot)
    g = game()
    bot.seat_at(g)
    bot.seen(g, 1, legal_actions(g)[0], 0)
    bot.choose(g)                          # a call after the notes arrives after them
    assert made[0].notes == [("seat_at",), ("seen", 1)] == local.notes
    server.close()
    time.sleep(R.LINK_DOWN_S + 0.2)
    bot.seen(g, 2, legal_actions(g)[0], 0)     # while down: the fallback sees it, the server later
    server = serve(lambda spec: made.append(Bot("server")) or made[-1], port=port)
    wait_up(bot)
    assert bot.choose(g)[0] == "server"
    assert made[1].notes == [("seat_at",), ("seen", 1), ("seen", 2)]
    bot.close(); server.close()


def test_a_game_crosses_the_wire_whole():
    seen = []

    class Echo(Bot):
        def choose(self, game):
            seen.append(game)
            return legal_actions(game)[0]

    server = serve(lambda spec: Echo("server"))
    bot = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "g1", {}, Bot("local"), code="v1")
    wait_up(bot)
    g = game()
    assert bot.choose(g) == legal_actions(g)[0]
    assert legal_actions(seen[0]) == legal_actions(g)
    bot.close(); server.close()


def test_non_callable_attributes_and_the_value_net_are_the_fallbacks():
    local = Bot("local")
    local.policy = "net"
    bot = R.RemoteBot("127.0.0.1:1", "g1", {}, local, code="v1", connect=False)
    assert bot.policy == "net" and object.__getattribute__(bot, "bot") is local
    assert not hasattr(bot, "no_such_method")


def test_a_seats_first_decision_waits_for_its_first_connection():
    server = serve(lambda spec: Bot("server"))
    bot = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "g1", {}, Bot("local"), code="v1")
    assert bot.choose(game())[0] == "server"          # straight after the seat is made
    bot.close(); server.close()


def test_a_game_opens_while_another_games_search_runs():
    started = threading.Event()
    server = serve(lambda spec: Bot(spec["who"], delay=spec.get("delay", 0.0), started=started))
    slow = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "a", {"who": "slow", "delay": 3.0}, Bot("l"), code="v1")
    wait_up(slow)
    threading.Thread(target=slow.choose, args=(game(),), daemon=True).start()
    assert started.wait(2)
    fresh = R.RemoteBot(f"127.0.0.1:{server.address[1]}", "b", {"who": "fresh"}, Bot("l"), code="v1")
    t0 = time.monotonic()
    wait_up(fresh, timeout=1.0)                        # opened while the slow search holds the job thread
    assert time.monotonic() - t0 < 1.0
    slow.close(); fresh.close(); server.close()
