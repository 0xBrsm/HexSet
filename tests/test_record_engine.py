# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from hexset.actions import Action, ActionType, apply
from hexset.board.board import random_base_board
from hexset.bots import RandomBot
from hexset.game import UNSTRUCTURED_TURN_CAP, is_over
from hexset.record import (
    Record,
    ReplayError,
    board_of,
    from_json,
    from_journal,
    read,
    record_game,
    replay,
    to_json,
    write,
)

SEED = 4  # the shortest finished game among the first dozen seeds
# `RandomBot` declines every offer, so under `"round"` its trade rounds are
# nearly all of a record's cost and change nothing: `"external"` asks no
# gate and plays the identical game.
NO_ROUNDS = "external"


def a_record(seed: int = SEED, **options) -> Record:
    board = random_base_board(random.Random(seed))
    bots = [RandomBot(random.Random(seed * 10 + s)) for s in range(4)]
    return record_game(bots, board, seed, turn_cap=UNSTRUCTURED_TURN_CAP,
                       trade_mode=NO_ROUNDS, **options)


@pytest.fixture(scope="module")
def record() -> Record:
    """One finished game, shared: every test here only reads it."""
    return a_record()


def test_the_board_survives_the_round_trip_through_a_record(record):
    board = random_base_board(random.Random(SEED))
    rebuilt = board_of(record)

    assert rebuilt.terrain == board.terrain
    assert rebuilt.tokens == board.tokens
    assert rebuilt.topology == board.topology
    assert rebuilt.hexes_by_roll == board.hexes_by_roll
    assert rebuilt.ports == board.ports


def test_replaying_reproduces_the_game_that_was_recorded(record):
    assert record.decided
    game = replay(record)
    assert is_over(game)
    assert game.won_by == record.winner
    assert game.turns == record.turns


def test_a_tampered_action_is_caught_rather_than_replayed(record):
    actions = list(record.actions)
    actions[0] = (int(ActionType.END_TURN), 0, 0)
    with pytest.raises(ReplayError, match="not legal"):
        replay(Record(**{**record.__dict__, "actions": tuple(actions)}))


def test_a_tampered_outcome_is_caught(record):
    with pytest.raises(ReplayError, match="record says"):
        replay(Record(**{**record.__dict__, "turns": record.turns + 1}))


def test_extra_chance_events_are_rejected(record):
    with pytest.raises(ReplayError, match="unconsumed chance"):
        replay(replace(record, chance=record.chance + (("roll", 6),)))


def test_records_round_trip_through_a_file(tmp_path, record):
    records = [record, a_record(seed=2, action_cap=40)]
    path = str(tmp_path / "games.jsonl")

    assert write(path, records) == 2
    assert list(read(path)) == records
    assert write(path, records[:1]) == 1
    assert len(list(read(path))) == 3


def test_a_version_1_line_is_refused_by_name(record):
    """A version-1 line has no `chance` and no `version`."""
    v1 = {k: v for k, v in json.loads(to_json(record)).items() if k not in ("version", "chance")}
    with pytest.raises(ValueError, match="version 1"):
        from_json(json.dumps(v1))


def test_a_record_replays_to_the_identical_terminal_state_with_no_seed(record):
    """A record carries its own chance stream: dice, deck and steals alike."""
    assert {"deck", "roll", "steal"} <= {kind for kind, _ in record.chance}
    seedless = Record(**{**record.__dict__, "seed": None})
    game = replay(seedless)
    assert is_over(game)
    assert game.won_by == record.winner
    assert game.turns == record.turns


def test_an_altered_roll_diverges_from_the_seed_and_raises(record):
    assert record.seed is not None
    events = list(record.chance)
    index = next(i for i, (kind, _) in enumerate(events) if kind == "roll")
    kind, value = events[index]
    tampered_value = 2 if value != 2 else 3
    events[index] = (kind, tampered_value)
    tampered = Record(**{**record.__dict__, "chance": tuple(events)})
    with pytest.raises(ReplayError, match="diverges"):
        replay(tampered)


#
# `Phase.DISCARD` is the one phase where the position does not say whose action
# it is: any owing seat may pay first. `Record.actors` carries that, and
# `replay` checks each action against the seat that took it rather than against
# `to_move`; without both, a real round replays flattened into seat order.


def _out_of_order_discard_record(seed: int, *, action_cap: int = 2000):
    """Stops once one discard is a card the *lowest* owing seat does not hold;
    `poisoned` says whether that happened. Seedless: the seat choices draw
    from their own rng.
    """
    from hexset.actions import legal_actions
    from hexset.board.terrain import NUM_RESOURCES
    from hexset.chance import Live, Recording
    from hexset.game import players_owing_discards, start, to_move
    from hexset.record import board_fields

    chance_rng = random.Random(seed)
    board = random_base_board(chance_rng)
    choices = random.Random(seed * 977 + 1)
    recording = Recording(Live(chance_rng))
    bots = [RandomBot(random.Random(seed * 10 + s)) for s in range(4)]
    game = start(board, 4, chance_rng, chance=recording)
    game.gates = tuple(bots)
    game.trade_mode = NO_ROUNDS

    actions: list[tuple[int, int, int]] = []
    actors: list[tuple[int, int]] = []
    trades: list[tuple[int, int, int, tuple[int, ...]]] = []
    poisoned = False

    while not is_over(game) and len(actions) < action_cap:
        owing = players_owing_discards(game)
        if len(owing) > 1:
            seat = owing[-1]
            hands = game.state(0, hidden=False).hands
            wanted = [
                r
                for r in range(NUM_RESOURCES)
                if hands[seat][r] > 0 and hands[owing[0]][r] == 0
            ]
            if wanted:
                poisoned = True
                action = Action(ActionType.DISCARD, choices.choice(wanted))
            else:
                action = choices.choice(legal_actions(game, seat))
            actors.append((len(actions), seat))
        else:
            seat = to_move(game)
            action = bots[seat].choose(game)

        before = len(game.trades)
        apply(game, action, seat)
        for trade in game.trades[before:]:
            trades.append((len(actions), trade.a, trade.b, tuple(trade.received)))
        actions.append((int(action.type), action.a, action.b))
        if poisoned and not players_owing_discards(game):
            break

    return (
        Record(
            num_players=4,
            seed=None,
            first=game.first,
            actions=tuple(actions),
            actors=tuple(actors),
            chance=tuple(recording.events),
            trades=tuple(trades),
            winner=game.won_by,
            turns=game.turns,
            **board_fields(board),
        ),
        poisoned,
    )


@pytest.fixture(scope="module")
def poisoned():
    for seed in range(1, 40):  # seed 1 is the first that poisons
        record, poisoned = _out_of_order_discard_record(seed)
        if poisoned:
            return record
    raise AssertionError("no seed produced a discard round with disjoint hands")


def test_a_discard_round_replays_in_the_order_it_actually_happened(poisoned):
    record = poisoned
    assert record.actors, "the fixture recorded no out-of-order discard"

    replayed = replay(record)
    assert (replayed.won_by, replayed.turns) == (record.winner, record.turns)


def test_the_same_round_without_its_actors_is_the_old_failure(poisoned):
    with pytest.raises(ReplayError):
        replay(replace(poisoned, actors=()))


def test_actors_survive_json_and_default_to_empty(poisoned, record):
    assert from_json(to_json(poisoned)) == poisoned
    assert from_json(to_json(poisoned)).actors == poisoned.actors
    older = json.loads(to_json(record))
    del older["actors"]
    restored = from_json(json.dumps(older))
    assert restored.actors == ()
    assert replay(restored).won_by == restored.winner


def test_moves_carries_each_step_with_its_actor_where_the_record_names_one(poisoned):
    from hexset.record import moves

    record = poisoned
    triples = list(moves(record))
    assert [(int(a.type), a.a, a.b) for _, a, _ in triples] == list(record.actions)
    assert sum(len(trades) for _, _, trades in triples) == len(record.trades)
    named = {step: seat for step, seat in record.actors}
    assert [actor for actor, _, _ in triples] == [
        named.get(step) for step in range(len(record.actions))
    ]


def test_a_journal_carries_its_discard_actors_into_the_record(tmp_path):
    from hexset.actions import ActionType as _AT, legal_actions
    from hexset.game import (
        is_over as _is_over,
        players_owing_discards,
        start as _start,
        to_move as _to_move,
    )
    from hexset.server._journal import Journal

    seed = 13  # a seed whose sevens do leave two seats owing at the same time
    board = random_base_board(random.Random(seed))
    rng = random.Random(seed)
    game = _start(board, 4, rng)
    bots = [RandomBot(random.Random(seed * 10 + s)) for s in range(4)]
    game.gates = tuple(bots)
    game.trade_mode = NO_ROUNDS
    choices = random.Random(7)

    journal = Journal(directory=str(tmp_path), game_id="discard-order")
    journal.start(
        game,
        seed=seed,
        first=0,
        human_seats=[],
        bot_names={s: "random" for s in range(4)},
        bot_specs={},
    )

    step = 0
    round_num = 0
    out_of_order = 0
    while not _is_over(game) and step < 600:
        owing = players_owing_discards(game)
        if len(owing) > 1:
            seat = owing[-1]
            action = choices.choice(legal_actions(game, seat))
            out_of_order += 1
        else:
            seat = _to_move(game)
            action = bots[seat].choose(game)
        before_hands = [hand[:] for hand in game.state(0, hidden=False).hands]
        before_held = [game.state(0, hidden=False).dev_cards[p][:] for p in range(4)]
        apply(game, action, seat)
        journal.action(
            game,
            step=step,
            round_num=round_num,
            actor=seat,
            action=action,
            before_hands=before_hands,
            before_held=before_held,
        )
        step += 1
        if action.type is _AT.END_TURN:
            round_num += 1
    journal.finish(game)

    assert out_of_order, "no seven in this game left two seats owing at once"

    record = from_journal(journal.path)
    assert len(record.actors) >= out_of_order
    for at_step, _seat in record.actors:
        assert record.actions[at_step][0] == int(_AT.DISCARD)

    replayed = replay(record)
    assert (replayed.won_by, replayed.turns) == (record.winner, record.turns)


def test_a_record_carries_the_ruleset_it_was_played_under():
    """A record that did not say which game it was would replay as a standard
    one: a 15-point game re-dealt at 10 ends at the wrong action."""
    from hexset.rules import DUEL_VARIANT, DUEL_VARIANT_GAME, STANDARD

    board = random_base_board(random.Random(7))
    bots = [RandomBot(random.Random(i)) for i in range(2)]
    record = record_game(bots, board, seed=7, game_type=DUEL_VARIANT_GAME, action_cap=60)

    assert record.rules == DUEL_VARIANT
    assert from_json(to_json(record)).rules == DUEL_VARIANT
    # The replay is dealt under the recorded rules, not the default ones.
    assert replay(record).state(0, hidden=False).rules == DUEL_VARIANT

    standard = record_game(bots, board, seed=7, action_cap=60)
    assert standard.rules == STANDARD
    assert replay(standard).state(0, hidden=False).rules == STANDARD


def test_a_record_written_before_rules_existed_reads_as_standard():
    """Every record on disk today is a standard game, so the absent field is
    not unknown -- it is known, and it is the default."""
    from hexset.rules import STANDARD

    board = random_base_board(random.Random(3))
    bots = [RandomBot(random.Random(i)) for i in range(2)]
    record = record_game(bots, board, seed=3, action_cap=60)

    raw = json.loads(to_json(record))
    del raw["rules"]
    assert from_json(json.dumps(raw)) == record
    assert from_json(json.dumps(raw)).rules == STANDARD


def test_a_replay_puts_the_offers_back_on_the_ledger():
    """Offers and answers move no card but go on the public ledger, so a
    record keeps them among the exchanges (`Record.shown`) and a replay --
    through JSON and back -- reaches the ledger that was played: certified
    cards, wants and wastes included."""
    import random as _random

    from hexset.arena import run_seated as _run, seat_bots as _seat
    from hexset.board.board import random_base_board
    from hexset.game import start
    from hexset.record import Tape, recording
    from hexset.rules import STANDARD
    from traders import ScarcityTrader

    rng = _random.Random(8)     # a game that ends with offers on the ledger
    board = random_base_board(rng)
    game = start(board, 4, rng, chance=recording(rng, STANDARD))
    bots = [ScarcityTrader(_random.Random(s)) for s in range(4)]
    _seat(game, bots, "round")
    tape = Tape()
    _run(game, bots, 3000, tape=tape)
    record = from_json(to_json(tape.sealed(game)))
    assert record.shown, "the game showed no offer; pick a seed that trades"
    replayed = replay(record)
    for live, back in zip(game.ledger.seats, replayed.ledger.seats):
        assert (back.known, back.unknown, back.want, back.waste) == \
            (live.known, live.unknown, live.want, live.waste)
    # Without them the replay is a different position.
    bare = replay(replace(record, shown=()))
    assert [(r.known, r.want, r.waste) for r in bare.ledger.seats] != \
        [(r.known, r.want, r.waste) for r in game.ledger.seats]
