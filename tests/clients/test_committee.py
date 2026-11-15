from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="hexset.clients.onnxbot needs onnxruntime installed")

from hexset.actions import apply, options_for  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.clients.committee import Committee, members_of, weights_of  # noqa: E402
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


def test_weights_are_one_positive_number_per_member_the_file_first():
    assert weights_of({}, 3) is None
    assert weights_of({"committee_weights": "2+1+1"}, 3) == (2.0, 1.0, 1.0)
    for bad in ("2+1", "2+0+1", "2+x+1", "2+-1+1", "2+nan+1"):
        with pytest.raises(ValueError):
            weights_of({"committee_weights": bad}, 3)


class _Fixed:
    """A member whose prior is the same table at every row."""

    def __init__(self, prior, value=(0.5, 0.5)):
        self.prior, self.value, self.space = np.asarray(prior, dtype=np.float64), value, None

    def score_rows(self, rows):
        return [(self.prior, self.value) for _ in rows]

    def value_rows(self, rows):
        return [self.value for _ in rows]


def test_a_committee_takes_its_space_from_its_first_member_and_a_prior_from_all():
    first, second = _Fixed([0.5, 0.5]), _Fixed([0.5, 0.5])
    first.space = "space"
    assert Committee([first, second]).space == "space"
    assert Committee([first, second]).has_prior is True
    second.has_prior = False
    assert Committee([first, second]).has_prior is False
    whole = Committee([_Whole(1), _Whole(2)])
    assert whole.space is None and whole.plans is not None


def test_a_weighted_committee_plays_the_highest_weighted_mean_log_prior():
    """`2+1+1` is the nested vote: the lead against the even product of the
    other two. Equal weights keep the plain mean exactly."""
    lead, b, c = [0.6, 0.3, 0.1], [0.1, 0.5, 0.4], [0.1, 0.5, 0.4]
    row = (None, 0, ("x", "y", "z"))
    plain = Committee([_Fixed(lead), _Fixed(b), _Fixed(c)])

    assert plain.act_rows([row]) == ["y"]
    tilted = [0.9, 0.05, 0.05]
    w = np.array([2.0, 1.0, 1.0])
    logs = np.log(np.array([tilted, b, c]))
    expected = ("x", "y", "z")[int(np.argmax((w[:, None] * logs).sum(0) / w.sum()))]
    weighted = Committee([_Fixed(tilted), _Fixed(b), _Fixed(c)], (2, 1, 1))
    assert weighted.act_rows([row]) == [expected] == ["x"]
    assert Committee([_Fixed(tilted), _Fixed(b), _Fixed(c)]).act_rows([row]) == ["y"]
    (prior, _), = weighted.score_rows([row])
    want = np.exp((w[:, None] * logs).sum(0) / w.sum())
    np.testing.assert_allclose(prior, want / want.sum(), rtol=0, atol=1e-12)
    same = Committee([_Fixed(lead), _Fixed(b), _Fixed(c)], (3, 3, 3))
    with np.errstate(divide="ignore"):
        assert (same._vote(np.array([lead, b, c])) == np.log(np.array([lead, b, c])).mean(0)).all()
    with pytest.raises(ValueError):
        Committee([_Fixed(lead), _Fixed(b)], (1, 1, 1))


def test_a_file_weighting_its_committee_loads_with_those_weights(tmp_path):
    from hexset.clients.onnxbot import load

    board = random_base_board(random.Random(0))
    loaded = load(str(_committee(tmp_path, committee_weights="3+1")), board.topology)
    assert tuple(loaded.policy.weights) == (3.0, 1.0)
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(ValueError, match="2 files"):
        load(str(_committee(bad, committee_weights="2+1+1")), board.topology)
    lone = tmp_path / "lone.onnx"
    lone.write_bytes(with_metadata(STUB9.read_bytes(), {"committee_weights": "2+1"}))
    with pytest.raises(ValueError, match="names no"):
        load(str(lone), board.topology)


class _Whole:
    """A member that lists whole options (`option_rows`) by a fixed score per
    option, `salt` changing the score, and plays its most probable one a move
    at a time (`plans`)."""

    def __init__(self, salt: int = 0, *, options: bool = True) -> None:
        from hexset import decisions as D

        self.D, self.salt, self.has_options, self.plans, self.space = D, salt, options, D.Plans(), None

    def _listed(self, game, seat):
        listed = tuple(self.D.options(game, seat))
        scores = np.array([(hash((self.salt, self.D.key(o))) % 1_000_003) / 1_000_003 * 6.0 for o in listed])
        probs = np.exp(scores - scores.max())
        return listed, probs / probs.sum()

    def option_rows(self, rows):
        assert self.has_options
        return [self._listed(game, seat) for game, seat in rows]

    def _best(self, game, seat):
        listed, probs = self._listed(game, seat)
        return listed[int(np.argmax(probs))]

    def act_rows(self, rows):
        return [self.plans.act(game, seat, options, self._best, lambda g, s, o: o[0])
                for game, seat, options in rows]

    def score_rows(self, rows):
        out = []
        for game, seat, options in rows:
            listed, probs = self._listed(game, seat)
            mass = {}
            for option, p in zip(listed, probs):
                first = self.D.first_move(option)
                mass[first] = mass.get(first, 0.0) + p
            prior = np.array([mass.get(o, 0.0) for o in options]) + 1e-9
            out.append((prior / prior.sum(), (0.25, 0.25, 0.25, 0.25)))
        return out

    def value_rows(self, rows):
        return [(0.25, 0.25, 0.25, 0.25) for _ in rows]


def _play(policies, seed=7, moves=400):
    """Play one game, every seat asked of each policy in `policies` on its own
    copy, the first policy's move played on every copy; each policy's moves."""
    import copy

    game = start(random_base_board(random.Random(seed)), 4, random.Random(seed))
    copies = [copy.deepcopy(game) for _ in policies]
    seen = [[] for _ in policies]
    for _ in range(moves):
        if copies[0].won_by is not None:
            break
        picks = []
        for k, (policy, g) in enumerate(zip(policies, copies)):
            options = tuple(options_for(g))
            (move,) = policy.act_rows([(g, to_move(g), options)])
            assert move in options
            seen[k].append(move)
            picks.append(move)
        for g in copies:
            apply(g, picks[0])
    return seen


def test_a_committee_of_whole_members_chooses_whole_decisions_as_one_member_alone():
    """Two copies of one member vote exactly as that member plays alone, whole
    decisions and plans included."""
    alone, committee = _Whole(1), Committee([_Whole(1), _Whole(1)])
    assert committee.has_options and committee.plans is not None
    a, c = _play([alone, committee])
    assert a == c and len(a) > 50


def test_a_whole_vote_is_the_weighted_product_of_the_members_whole_options():
    from hexset.game import start as _start

    members = [_Whole(1), _Whole(2), _Whole(3)]
    committee = Committee(members, (2, 1, 1))
    game = _start(random_base_board(random.Random(0)), 4, random.Random(0))
    seat = to_move(game)
    ((listed, p),) = committee.option_rows([(game, seat)])
    rows = [m.option_rows([(game, seat)])[0] for m in members]
    assert all(tuple(r[0]) == listed for r in rows)
    w = np.array([2.0, 1.0, 1.0])
    want = np.exp((w[:, None] * np.log(np.array([r[1] for r in rows]))).sum(0) / w.sum())
    np.testing.assert_allclose(p, want / want.sum(), rtol=0, atol=1e-12)
    (move,) = committee.act_rows([(game, seat, tuple(options_for(game)))])
    from hexset import decisions as D

    best = listed[int(np.argmax(p))]
    assert move == D.first_move(best) and committee.plans.peek(game, seat) == tuple(best.parts[1:])


def test_a_committee_with_a_member_listing_no_whole_options_votes_a_move_at_a_time():
    committee = Committee([_Whole(1), _Whole(2, options=False)])
    assert not committee.has_options and committee.plans is None
    with pytest.raises(ValueError):
        committee.option_rows([(None, 0)])


def test_members_listing_different_options_are_refused():
    class Short(_Whole):
        def option_rows(self, rows):
            return [(listed[:-1], probs[:-1] / probs[:-1].sum()) for listed, probs in super().option_rows(rows)]

    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    with pytest.raises(ValueError, match="different whole options"):
        Committee([_Whole(1), Short(2)]).option_rows([(game, to_move(game))])


def test_a_search_opens_a_committee_decision_whole():
    from hexset.rollout import PairedRollouts

    committee = Committee([_Whole(1), _Whole(2)])
    search = PairedRollouts(committee, worlds=2, top=2)
    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    assert search._opens_whole(game, tuple(options_for(game)))
