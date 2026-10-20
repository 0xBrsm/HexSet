"""Which graph shape a checkpoint's `contract` metadata routes to, and whether
a record-contract graph loads and plays. The rule: a contract number names
exactly one graph shape, this repo reads it and never assigns it, and
anything else is refused **by name**. Contract 8 is current; contract 7, whose
record is contract 8's minus seven fields, and contract 6, contract 7's minus
`roads_to_take` and `to_move`, are still served. Contract 9 is the
contract-9 observation and a decision's arrays in, the decision out, run by
`DecisionPolicy` (`stub-contract9.onnx`, `fixtures/build_stub9.py`).

The contract-8 stub is 27-input, the contract-7 one 26 and the two contract-6
stubs 24 and 22: a loader that feeds a graph every field it happens to have, rather
than the ones it *declares*, breaks on the shorter ones.

Every fixture here is a stub. `dev-contract2.onnx` and `tiny.onnx` were
genuine exports with real weights until they were rebuilt by
`fixtures/build_refused_stubs.py`: nothing loads a refused contract to play,
so the weights were 1.1 MB nobody ran, and their metadata named a real
training checkpoint that no test read and the publish gate could not see.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime", reason="hexset.clients.onnxbot needs onnxruntime installed")

import onnxruntime as ort  # noqa: E402

from hexset.actions import apply  # noqa: E402
from hexset.board.board import random_base_board  # noqa: E402
from hexset.game import Phase, start  # noqa: E402

from hexset.clients.onnxbot import DecisionPolicy, V2Policy, load  # noqa: E402
from hexset.actions import options_for  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
STUB = FIXTURES / "stub-contract6.onnx"
STUB7 = FIXTURES / "stub-contract7.onnx"
STUB8 = FIXTURES / "stub-contract8.onnx"
STUB9 = FIXTURES / "stub-contract9.onnx"
STUB_PARTIAL = FIXTURES / "stub-contract6-partial.onnx"
DEV_CONTRACT2 = FIXTURES / "dev-contract2.onnx"
CONTRACT1 = FIXTURES / "tiny.onnx"


def _board():
    return random_base_board(random.Random(0))


def _metadata(path: Path) -> dict:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return dict(session.get_modelmeta().custom_metadata_map)


def _declared_inputs(path: Path) -> list[str]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return [i.name for i in session.get_inputs()]


# --- What the fixtures actually are -------------------------------------------


def test_the_contract_2_fixture_is_export_shaped():
    """Export-shaped, not an export: `build_refused_stubs.py` writes the
    provenance keys because a real contract-2 file carried them, with values
    that are visibly not real. There is deliberately no `source_checkpoint`.
    """
    meta = _metadata(DEV_CONTRACT2)
    assert meta["contract"] == "2"
    assert len(_declared_inputs(DEV_CONTRACT2)) == 23
    assert meta["exporter_commit"] and meta["checkpoint_sha256"]
    assert "source_checkpoint" not in meta


# --- Dispatch ------------------------------------------------------------------


@pytest.mark.parametrize("path", [STUB8, STUB7, STUB])
def test_a_contract_7_or_6_graph_routes_to_the_record_policy(path):
    assert isinstance(load(str(path), _board().topology).policy, V2Policy)


def test_the_served_contracts_are_the_record_ones_and_contract_9():
    from hexset.contract9 import CONTRACT
    from hexset.onnx_record import CONTRACT_VERSION
    from hexset.server.constants import RECORD_CONTRACTS

    assert CONTRACT_VERSION == "8.1" and CONTRACT == "9"
    assert RECORD_CONTRACTS == {"6", "7", "8", "8.1", "9"}
    assert _metadata(STUB9)["contract"] == "9"
    assert _metadata(STUB8)["contract"] == "8"
    assert _metadata(STUB7)["contract"] == "7"
    assert _metadata(STUB)["contract"] == "6"


def test_older_graphs_declare_a_subset_of_the_current_record():
    """What keeps contracts 6, 7 and 8 servable: every input they declare is
    still in the record, and what they lack is exactly what came after."""
    from hexset.onnx_record import RECORD_FIELDS

    six, seven, eight = (set(_declared_inputs(p)) for p in (STUB, STUB7, STUB8))
    assert set(RECORD_FIELDS) - (seven | eight) == {
        "winning_points", "discard_limit", "friendly_robber", "balanced_dice", "roll_odds",
    }
    assert seven - six == {"roads_to_take", "to_move"}
    assert eight - seven == {
        "reach", "own_matured", "dev_card_played", "expected_hand",
        "dev_ages", "knights_needed", "unseen_dev",
    }
    # A contract-8 graph declares only what it reads.
    assert seven - eight == {
        "knights_played", "longest_road_holder", "largest_army_holder",
        "hand_totals", "ledger_known", "ledger_unknown",
    }


@pytest.mark.parametrize("contract", ["5", "10"])
def test_a_contract_either_side_of_the_served_ones_is_refused_by_name(contract, tmp_path):
    onnx = pytest.importorskip("onnx")
    model = onnx.load(str(STUB7))
    for entry in model.metadata_props:
        if entry.key == "contract":
            entry.value = contract
    path = tmp_path / f"contract{contract}.onnx"
    onnx.save(model, str(path))
    with pytest.raises(ValueError) as caught:
        load(str(path), _board().topology)
    assert f"contract={contract!r}" in str(caught.value)


def test_a_file_declaring_no_contract_is_refused_as_declaring_none():
    """Not as `contract='1'`: the file declares nothing, and a message
    naming a contract it never wrote sends the reader looking for the key."""
    assert "contract" not in _metadata(CONTRACT1)
    with pytest.raises(ValueError) as caught:
        load(str(CONTRACT1), _board().topology)
    message = str(caught.value)
    assert "declares no contract" in message and "contract='" not in message
    assert "6" in message and "7" in message and "8" in message


def test_an_offer_protocol_contract_is_refused_by_name():
    with pytest.raises(ValueError) as caught:
        load(str(DEV_CONTRACT2), _board().topology)
    assert "contract='2'" in str(caught.value)


# --- Loading is not enough: it has to play ------------------------------------


@pytest.mark.parametrize(
    "path,expected_inputs", [(STUB8, 27), (STUB7, 26), (STUB, 24), (STUB_PARTIAL, 22)]
)
def test_a_record_contract_checkpoint_plays_legal_actions_from_every_phase(
    path, expected_inputs
):
    from hexset.clients.onnxbot import network_bot

    board = _board()
    assert len(_declared_inputs(path)) == expected_inputs

    bot = network_bot(str(path), board)
    game = start(board, 4, random.Random(3))
    seen = set()
    for _ in range(300):
        if game.won_by is not None:
            break
        action = bot.choose(game)
        assert action in options_for(game)
        seen.add(game.phase)
        apply(game, action)
    assert {Phase.SETUP_SETTLEMENT, Phase.ROLL, Phase.MAIN} <= seen


def test_a_checkpoint_plays_on_through_a_turn_the_engine_traded_in():
    from hexset.clients.onnxbot import network_bot
    from hexset.game import roll_dice, to_move

    class _Wants:
        """Prices a candidate positively iff it hands this seat more of `resource`."""

        trade_floor = 0.0

        def __init__(self, resource: int):
            self.resource = resource

        def gains_many(self, view, received, counterparties):
            return [1.0 if r[self.resource] > 0 else -1.0 for r in received]

    board = _board()
    bot = network_bot(str(STUB), board)
    game = start(board, 4, random.Random(11))
    while game.phase is not Phase.ROLL:
        apply(game, options_for(game)[0])

    mover = to_move(game)
    other = (mover + 1) % 4
    state = game.state(mover, hidden=False)
    for hand in state.hands:
        hand[:] = [0, 0, 0, 0, 0]
    state.hands[mover][0] = 1
    state.hands[other][4] = 1
    traders: list = [None] * 4
    traders[mover] = _Wants(4)
    traders[other] = _Wants(0)
    game.gates = tuple(traders)

    roll_dice(game, 8)
    assert game.trades, "the engine cleared nothing to play on through"
    assert bot.choose(game) in options_for(game)


# --- Contract 9 -------------------------------------------------------------------


def _stub9(tmp_path, name="stub.onnx", **kwargs):
    import sys

    sys.path.insert(0, str(FIXTURES))
    try:
        from build_stub9 import build
    finally:
        sys.path.remove(str(FIXTURES))
    return build(tmp_path / name, **kwargs)


def test_a_contract_9_graph_routes_to_the_decision_runner_and_plays_every_phase():
    from hexset.clients.onnxbot import network_bot

    board = _board()
    loaded = load(str(STUB9), board.topology)
    assert isinstance(loaded.policy, DecisionPolicy) and loaded.policy.has_prior
    assert "double_rows" not in loaded.policy.inputs
    bot = network_bot(str(STUB9), board)
    game = start(board, 4, random.Random(3))
    seen = set()
    for _ in range(600):
        if game.won_by is not None:
            break
        action = bot.choose(game)
        assert action in options_for(game)
        seen.add(game.phase)
        apply(game, action)
    assert {Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD, Phase.ROLL, Phase.MAIN} <= seen


def test_the_win_head_is_turned_from_the_rows_own_frame_to_board_seat_order():
    """The stub answers softmax(2, 1, 0, -1) in every row's own frame, the
    seat itself first: so each seat reads itself highest, and the seat after
    it in turn order next. Its `points` output is the graph's own: never read."""
    import numpy as np

    board = _board()
    policy = load(str(STUB9), board.topology).policy
    assert "points" in [o.name for o in policy.session.get_outputs()] and not hasattr(policy, "points_rows")
    game = start(board, 4, random.Random(3))
    want = np.exp([2.0, 1.0, 0.0, -1.0])
    want /= want.sum()
    for seat in range(4):
        (value,) = policy.value_rows([(game, seat)])
        assert [value[(seat + i) % 4] for i in range(4)] == pytest.approx(list(want), rel=1e-5)


def test_a_graph_that_declares_double_rows_is_asked_the_double_turn_as_one_decision(tmp_path):
    from hexset import decisions as D
    from hexset.game import to_move

    board = _board()
    policy = load(str(_stub9(tmp_path, joint=True)), board.topology).policy
    assert "double_rows" in policy.inputs
    game = start(board, 4, random.Random(3))
    while not D.is_double_turn(game):
        apply(game, policy.act_rows([(game, to_move(game), tuple(options_for(game)))])[0])
    option = policy._decide(game, to_move(game))
    assert isinstance(option, D.Compound) and option.kind is D.CompoundKind.SETUP_DOUBLE
    for part in option.parts:                          # played out of the plan, part by part
        assert policy.act_rows([(game, to_move(game), tuple(options_for(game)))])[0] == part
        apply(game, part)


def test_search_priors_are_the_graphs_prior_at_each_option_renormalised():
    import numpy as np

    from hexset import decisions as D
    from tests.test_deals import a_robber_move

    board = _board()
    policy = load(str(STUB9), board.topology).policy
    game = a_robber_move()
    seat = game.current_player
    options = tuple(options_for(game))
    ((prior, value),) = policy.score_rows([(game, seat, options)])
    decision = D.tensors(game, seat, policy.space)
    feed = policy._feed([(game, seat)], [decision])
    graph = policy.session.run(["prior"], feed)[0][0]
    weights = np.array([graph[D.flat_index(policy.space, o, seat, 4)] for o in options], dtype=np.float64)
    np.testing.assert_allclose(prior, weights / weights.sum(), rtol=1e-6)
    assert value == policy.value_rows([(game, seat)])[0]


@pytest.mark.parametrize("change,message", [
    ({"width": 1028}, "observation"),
    ({"extra": "mystery"}, "mystery"),
    ({"prior": False, "meta": {"search": "mcts"}}, "declares no `prior`"),
])
def test_a_decision_graph_that_does_not_fit_is_refused_at_load_by_name(change, message, tmp_path):
    path = _stub9(tmp_path, **change)
    with pytest.raises(ValueError, match=message):
        load(str(path), _board().topology)


def test_a_graph_without_a_prior_plays_but_is_refused_a_search(tmp_path):
    from hexset.clients.netbot import searcher_for
    from hexset.clients.onnxbot import network_bot

    path = _stub9(tmp_path, prior=False)
    board = _board()
    loaded = load(str(path), board.topology)
    assert not loaded.policy.has_prior
    game = start(board, 4, random.Random(3))
    assert network_bot(str(path), board).choose(game) in options_for(game)
    with pytest.raises(ValueError, match="declares no `prior`"):
        searcher_for(loaded)
    searched = load(str(_stub9(tmp_path, "searched.onnx", meta={"search": "mcts", "simulations": "8"})),
                    board.topology)
    assert searched.search.searches and searched.policy.has_prior


def test_a_stamped_contract_9_file_still_loads_and_names_itself(tmp_path):
    from hexset.clients.onnxmeta import read_metadata, stamp

    target = tmp_path / "named.onnx"
    stamp(STUB9, target, name="c9stub", modes=["standard"], version="1.0", commit="0123abcd",
          profile={"description": "A contract-9 stub."})
    meta = read_metadata(target.read_bytes())
    assert meta["contract"] == "9" and meta["name"] == "c9stub" and meta["version"] == "1.0"
    assert isinstance(load(str(target), _board().topology).policy, DecisionPolicy)
