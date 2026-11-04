from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="hexset.clients.onnxbot needs onnxruntime installed")

from hexset.actions import apply, options_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.clients.committee import Committee, members_of  # noqa: E402
from hexset.clients.onnxmeta import with_metadata  # noqa: E402
from hexset.game import start, to_move  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
STUB8, STUB9 = FIXTURES / "stub-contract8.onnx", FIXTURES / "stub-contract9.onnx"


@pytest.fixture(autouse=True)
def _fresh_cache():
    yield
    from hexset.clients.onnxbot import _load_cached

    _load_cached.cache_clear()


def _committee(tmp_path, lead=STUB9, member=STUB8, **extra):
    """A lead file naming one member beside it."""
    (tmp_path / "member.onnx").write_bytes(member.read_bytes())
    path = tmp_path / "lead.onnx"
    path.write_bytes(with_metadata(lead.read_bytes(), {"committee": "member.onnx", **extra}))
    return path


def test_members_are_the_files_the_committee_key_names_beside_the_file():
    assert members_of({}) == ()
    assert members_of({"committee": "a.onnx+b.onnx"}) == ("a.onnx", "b.onnx")
    with pytest.raises(ValueError):
        members_of({"committee": "../a.onnx"})
    with pytest.raises(ValueError):
        members_of({"committee": "a.onnx++b.onnx"})


def test_a_file_naming_a_committee_loads_as_one_committee_led_by_itself(tmp_path):
    from hexset.clients.onnxbot import load

    board = random_base_board(random.Random(0))
    loaded = load(str(_committee(tmp_path)), board.topology)
    assert isinstance(loaded.policy, Committee) and len(loaded.policy.members) == 2
    assert loaded.players == load(str(STUB9), board.topology).players


def test_the_committee_plays_the_move_with_the_highest_mean_log_prior(tmp_path):
    """The vote, against the members read on their own: a product of their
    priors, so the move every member supports most."""
    from hexset.clients.onnxbot import load

    board = random_base_board(random.Random(0))
    committee = load(str(_committee(tmp_path)), board.topology).policy
    alone = [load(str(STUB9), board.topology).policy, load(str(STUB8), board.topology).policy]
    game = start(board, 4, random.Random(3))
    rng = random.Random(3)
    checked = 0
    while checked < 30 and game.won_by is None:
        seat, options = to_move(game), tuple(options_for(game))
        if len(options) > 1:
            priors = np.array([m.score_rows([(game, seat, options)])[0][0] for m in alone])
            with np.errstate(divide="ignore"):
                expected = options[int(np.argmax(np.log(priors).mean(0)))]
            assert committee.act_rows([(game, seat, options)]) == [expected]
            (prior, value), = committee.score_rows([(game, seat, options)])
            assert prior.sum() == pytest.approx(1.0) and len(prior) == len(options)
            reads = [m.value_rows([(game, seat)])[0] for m in alone]
            assert value == pytest.approx(tuple(np.mean(reads, axis=0)))
            checked += 1
        apply(game, rng.choice(options))
    assert checked == 30


def test_a_committee_file_spawns_a_seat_that_plays_legal_moves(tmp_path):
    from hexset.clients.onnxbot import spawn

    board = random_base_board(random.Random(0))
    bot = spawn(str(_committee(tmp_path)), board, rng=random.Random(1))
    game, rng = start(board, 4, random.Random(5)), random.Random(5)
    for _ in range(60):
        if game.won_by is not None:
            break
        options = tuple(options_for(game))
        action = bot.choose(game) if to_move(game) == 0 else rng.choice(options)
        assert action in options
        apply(game, action)


def test_a_missing_member_and_a_nested_committee_are_refused_at_load(tmp_path):
    from hexset.clients.onnxbot import load

    board = random_base_board(random.Random(0))
    missing = tmp_path / "missing.onnx"
    missing.write_bytes(with_metadata(STUB9.read_bytes(), {"committee": "absent.onnx"}))
    with pytest.raises(ValueError, match="not beside it"):
        load(str(missing), board.topology)
    nested = tmp_path / "nested"
    nested.mkdir()
    inner = _committee(nested)
    outer = nested / "outer.onnx"
    outer.write_bytes(with_metadata(STUB8.read_bytes(), {"committee": inner.name}))
    with pytest.raises(ValueError, match="of its own"):
        load(str(outer), board.topology)
