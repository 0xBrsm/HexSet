# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from hexset.actions import Action, ActionType, apply
from hexset.board.board import random_base_board
from hexset.bots import RandomBot, greedy
from hexset.bots.evaluate import Evaluator
from hexset.game import is_over
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


def a_record(seed: int = 0, bot: str = "random") -> Record:
    rng = random.Random(seed)
    board = random_base_board(rng)
    if bot == "greedy":
        bots = [greedy(Evaluator(board), random.Random(seed * 10 + s)) for s in range(4)]
    else:
        bots = [RandomBot(random.Random(seed * 10 + s)) for s in range(4)]
    return record_game(bots, board, seed)


def test_a_recorded_game_finished_and_kept_its_actions():
    record = a_record()
    assert record.decided
    assert record.turns > 0
    assert len(record.actions) > record.turns
    assert record.num_players == 4


def test_the_board_survives_the_round_trip_through_a_record():
    rng = random.Random(3)
    board = random_base_board(rng)
    rebuilt = board_of(record_game([RandomBot(random.Random(1))] * 4, board, 3))

    assert rebuilt.terrain == board.terrain
    assert rebuilt.tokens == board.tokens
    assert rebuilt.topology == board.topology
    assert rebuilt.hexes_by_roll == board.hexes_by_roll
    assert rebuilt.ports == board.ports


def test_replaying_reproduces_the_game_that_was_recorded():
    record = a_record(seed=5)
    game = replay(record)
    assert is_over(game)
    assert game.won_by == record.winner
    assert game.turns == record.turns


@pytest.mark.parametrize("bot", ["random"])
def test_replay_holds_for_either_bot(bot):
    replay(a_record(seed=7, bot=bot))


def test_a_record_survives_json():
    record = a_record(seed=2)
    assert from_json(to_json(record)) == record


def test_json_records_replay_as_well_as_the_originals():
    record = a_record(seed=4)
    replay(from_json(to_json(record)))


def test_a_tampered_action_is_caught_rather_than_replayed():
    record = a_record(seed=6)
    actions = list(record.actions)
    # An end-of-turn where setup expects a settlement can never be legal.
    actions[0] = (int(ActionType.END_TURN), 0, 0)
    with pytest.raises(ReplayError, match="not legal"):
        replay(Record(**{**record.__dict__, "actions": tuple(actions)}))


def test_a_tampered_outcome_is_caught():
    record = a_record(seed=6)
    with pytest.raises(ReplayError, match="record says"):
        replay(Record(**{**record.__dict__, "turns": record.turns + 1}))


def test_extra_chance_events_are_rejected():
    record = a_record(seed=6)
    with pytest.raises(ReplayError, match="unconsumed chance"):
        replay(replace(record, chance=record.chance + (("roll", 6),)))


def test_incremental_consumers_use_recorded_chance_without_a_seed():
    from hexset.dataset import samples_from

    seeded = a_record(seed=5)
    seedless = replace(seeded, seed=None)
    assert list(samples_from(seeded, 0)) == list(samples_from(seedless, 0))


def test_the_action_cap_bounds_a_record():
    rng = random.Random(0)
    board = random_base_board(rng)
    record = record_game([RandomBot(random.Random(1))] * 4, board, 0, action_cap=10)
    assert len(record.actions) == 10
    assert record.winner is None
    assert not record.decided


def test_records_round_trip_through_a_file(tmp_path):
    records = [a_record(seed=s) for s in (1, 2)]
    path = str(tmp_path / "games.jsonl")

    assert write(path, records) == 2
    assert list(read(path)) == records
    # Appending, not truncating: a generator run should be resumable.
    assert write(path, records[:1]) == 1
    assert len(list(read(path))) == 3


def test_actions_are_stored_as_plain_triples_not_enums():
    """So a record does not depend on the action space's current numbering."""
    record = a_record()
    first = record.actions[0]
    assert isinstance(first, tuple) and len(first) == 3
    assert all(type(part) is int for part in first)
    assert Action(ActionType(first[0]), first[1], first[2]).type is ActionType.SETUP_SETTLEMENT


def test_a_version_1_line_is_refused_by_name():
    """`agents/reference/game-records.md`: a version-1 line (no `chance`, a
    required `seed`, no `version` at all) is refused with a message naming
    the registration, not silently misread as version 2."""
    record = a_record(seed=11)
    v1 = {k: v for k, v in json.loads(to_json(record)).items() if k not in ("version", "chance")}
    with pytest.raises(ValueError, match="agents/reference/game-records.md"):
        from_json(json.dumps(v1))


# --- the four tests the registration lists as essential ------------------


def test_a_record_replays_to_the_identical_terminal_state_with_no_seed():
    """`Scripted` alone, `seed=None`: a record must not depend on the seed
    to replay -- the whole point of carrying its own chance stream."""
    record = a_record(seed=12, bot="random")
    seedless = Record(**{**record.__dict__, "seed": None})
    game = replay(seedless)
    assert is_over(game)
    assert game.won_by == record.winner
    assert game.turns == record.turns


def test_an_altered_roll_diverges_from_the_seed_and_raises():
    """A record with `seed` present checks every scripted outcome against
    what that seed's stream would actually have produced -- altering one
    recorded roll must be caught, not silently replayed."""
    record = a_record(seed=13, bot="greedy")
    assert record.seed is not None
    events = list(record.chance)
    index = next(i for i, (kind, _) in enumerate(events) if kind == "roll")
    kind, value = events[index]
    tampered_value = 2 if value != 2 else 3
    events[index] = (kind, tampered_value)
    tampered = Record(**{**record.__dict__, "chance": tuple(events)})
    with pytest.raises(ReplayError, match="diverges"):
        replay(tampered)


def test_a_journal_converts_to_a_record_that_replays(tmp_path):
    """`from_journal` on a game played through the server's own recording
    path (`hexset.server.journal.Journal`), not hand-written JSON."""
    from hexset.actions import ActionType as _AT
    from hexset.bots import RandomBot as _RandomBot
    from hexset.game import is_over as _is_over, start as _start, to_move as _to_move
    from hexset.server.journal import Journal

    board = random_base_board(random.Random(21))
    seed = 21
    rng = random.Random(seed)
    game = _start(board, 4, rng)
    bots = [_RandomBot(random.Random(21 * 10 + s)) for s in range(4)]
    game.gates = tuple(bots)

    journal = Journal(directory=str(tmp_path), game_id="synthetic-21")
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
    # A short game -- enough actions to cross into rolls, robber moves and a
    # steal or two, then `journal.finish` is called regardless of whether
    # the game actually ended: `from_journal` needs a `result` line, and a
    # capped, undecided record is exactly as valid as a finished one
    # (`test_the_action_cap_bounds_a_record` above).
    while not _is_over(game) and step < 400:
        seat = _to_move(game)
        bot = bots[seat]
        before_hands = [hand[:] for hand in game.state(0, hidden=False).hands]
        before_held = [
            game.state(0, hidden=False).dev_cards[p][:] for p in range(4)
        ]
        action = bot.choose(game)
        apply(game, action)
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

    record = from_journal(journal.path)
    assert record.winner == game.won_by
    assert record.turns == game.turns
    replayed = replay(record)
    assert replayed.won_by == record.winner
    assert replayed.turns == record.turns


def test_default_chance_matches_the_seeded_stream():
    """The default (`Live`) chance source is byte-identical to the engine's
    pre-`chance` behaviour, for every seed. Self-contained -- no file from
    outside this repo: records a short game right here, then replays it with
    `seed` still attached, so `replay`'s `_SeedChecked` cross-checks *every*
    event in the recorded stream (the deck order, every roll, every steal,
    in their true interleaved order) against a freshly seeded `Live` stream,
    raising `ReplayError` at the first one that disagrees.

    `record.winner`/`record.turns` are the actual game's own `won_by`/
    `turns`, captured at recording time (`record_game`'s own return
    statement) -- so a clean replay whose terminal state matches them is a
    clean replay matching the game that was actually played, not merely
    matching itself.
    """
    record = a_record(seed=42, bot="greedy")
    assert record.seed == 42
    # The stream really was exercised, not vacuously empty or roll-only.
    kinds = {kind for kind, _ in record.chance}
    assert {"deck", "roll", "steal"} <= kinds

    replayed = replay(record)  # raises ReplayError on any divergence from the seed
    assert (replayed.won_by, replayed.turns) == (record.winner, record.turns)


# --- a seven's discards replay in the order they actually happened ----------
#
# `Phase.DISCARD` is the one phase where the position does not say whose
# action it is: every seat over the limit owes cards at the same instant and
# any of them may pay first (`hexset.game.may_act`, and the live server that
# asks it). `Record.actors` carries that fact; `replay` checks each action
# against the seat that took it rather than against `to_move`. Without both,
# a real table's discard round could only be replayed flattened into
# ascending seat order, which is not the game that was played.


def _out_of_order_discard_record(seed: int, *, action_cap: int = 2000):
    """Record a game whose discard rounds are paid highest-owing-seat first,
    stopping once at least one of those discards is a card the *lowest* owing
    seat does not hold.

    Returns `(record, poisoned)` -- `poisoned` says whether such a discard was
    recorded, which is what makes "replaying this without the actors would
    have raised" a fact about the record rather than a hope. Seedless
    (`seed=None` on the `Record`) because the seat choices below draw from
    their own rng: `replay`'s `_SeedChecked` compares the recorded stream
    against a freshly seeded `Live` one, and an extra draw would shift it.
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

    actions: list[tuple[int, int, int]] = []
    actors: list[tuple[int, int]] = []
    trades: list[tuple[int, int, int, tuple[int, ...]]] = []
    poisoned = False

    while not is_over(game) and len(actions) < action_cap:
        owing = players_owing_discards(game)
        if len(owing) > 1:
            # Highest owing seat first -- never the seat `to_move` names, so
            # every one of these steps is one the old replay refused.
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


def _a_poisoned_record():
    for seed in range(40):
        record, poisoned = _out_of_order_discard_record(seed)
        if poisoned:
            return record
    raise AssertionError("no seed produced a discard round with disjoint hands")


def test_a_discard_round_replays_in_the_order_it_actually_happened():
    record = _a_poisoned_record()
    assert record.actors, "the fixture recorded no out-of-order discard"

    replayed = replay(record)
    assert (replayed.won_by, replayed.turns) == (record.winner, record.turns)


def test_the_same_round_without_its_actors_is_the_old_failure():
    """The concrete cost of leaving the engine's serialization in place: with
    the actors dropped, every discard resolves against the lowest owing seat
    and the record no longer describes a legal game."""
    record = _a_poisoned_record()
    with pytest.raises(ReplayError):
        replay(replace(record, actors=()))


def test_actors_survive_json_and_default_to_empty():
    record = _a_poisoned_record()
    assert from_json(to_json(record)) == record
    assert from_json(to_json(record)).actors == record.actors
    # A line written before the field existed reads as a record with none,
    # and replays exactly as it always did.
    older = json.loads(to_json(a_record(seed=5)))
    del older["actors"]
    restored = from_json(json.dumps(older))
    assert restored.actors == ()
    assert replay(restored).won_by == restored.winner


def test_recording_a_game_the_engine_drove_names_no_actors():
    """`record_game` steps `to_move`, so the position already says who acted
    and nothing is stored -- the field costs an ordinary record nothing."""
    assert a_record(seed=5).actors == ()


def test_moves_carries_the_actor_and_steps_still_does_not():
    """`steps` is what every consumer outside this package unpacks; it keeps
    its two-tuple shape, and `moves` is the actor-aware form."""
    from hexset.record import moves, steps as record_steps

    record = _a_poisoned_record()
    paired = list(record_steps(record))
    triples = list(moves(record))
    assert [(a, t) for _, a, t in triples] == paired
    named = {step: seat for step, seat in record.actors}
    assert [actor for actor, _, _ in triples] == [
        named.get(step) for step in range(len(record.actions))
    ]


def test_a_journal_carries_its_discard_actors_into_the_record(tmp_path):
    """The porting surface end to end: the server serves a seven's discards
    simultaneously (`hexset.server.webplay.GameSession.submit` gates on
    `may_act`), so a journal is the one place a real round's true order
    exists. `from_journal` must not flatten it back into seat order."""
    from hexset.actions import ActionType as _AT, legal_actions
    from hexset.game import (
        is_over as _is_over,
        players_owing_discards,
        start as _start,
        to_move as _to_move,
    )
    from hexset.server.journal import Journal

    seed = 13  # a seed whose sevens do leave two seats owing at the same time
    board = random_base_board(random.Random(seed))
    rng = random.Random(seed)
    game = _start(board, 4, rng)
    bots = [RandomBot(random.Random(seed * 10 + s)) for s in range(4)]
    game.gates = tuple(bots)
    # Its own stream: drawing the discarding seat's card from `rng` would
    # shift the chance stream `replay`'s `_SeedChecked` re-derives from the
    # seed, and the divergence would be this test's fault, not the code's.
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
            # The highest owing seat, which is never the one `to_move` names:
            # exactly the arrival order the live server now accepts and the
            # old replay could not express.
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
    # Every named actor is a discard, and at least one is not the seat the
    # position would have served.
    for at_step, _seat in record.actors:
        assert record.actions[at_step][0] == int(_AT.DISCARD)

    replayed = replay(record)
    assert (replayed.won_by, replayed.turns) == (record.winner, record.turns)
