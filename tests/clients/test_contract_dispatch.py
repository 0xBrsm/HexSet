"""Which graph shape a checkpoint's `contract` metadata routes to, and whether
a record-contract graph loads and plays. The rule: a contract number names
exactly one graph shape, this repo reads it and never assigns it, and
anything else is refused **by name**. Contract 6 is the only one served.

The two contract-6 stubs are 24- and 22-input: the pair exists because a
loader that feeds a graph every field it happens to have, rather than the ones
it *declares*, breaks on the shorter one.

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

from hexset.clients.onnxbot import V2Policy, load  # noqa: E402
from hexset.actions import options_for  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
STUB = FIXTURES / "stub-contract6.onnx"
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


@pytest.mark.parametrize("path", [STUB])
def test_a_contract_6_graph_routes_to_the_record_policy(path):
    assert isinstance(load(str(path), _board().topology).policy, V2Policy)


def test_a_contract_1_export_is_refused_by_name():
    assert "contract" not in _metadata(CONTRACT1)
    with pytest.raises(ValueError) as caught:
        load(str(CONTRACT1), _board().topology)
    assert "contract='1'" in str(caught.value)
    assert "6" in str(caught.value)


def test_an_offer_protocol_contract_is_refused_by_name():
    with pytest.raises(ValueError) as caught:
        load(str(DEV_CONTRACT2), _board().topology)
    assert "contract='2'" in str(caught.value)


# --- Loading is not enough: it has to play ------------------------------------


@pytest.mark.parametrize("path,expected_inputs", [(STUB, 24), (STUB_PARTIAL, 22)])
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
