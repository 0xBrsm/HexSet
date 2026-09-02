"""Which graph shape a checkpoint's `contract` metadata gets routed to, and
whether a genuine dev-HexNet export actually loads and plays.

This is the regression suite for the headline finding of the PI's review of
PR #2: the branch redefined contract `"2"` to mean the 29-input record while
`onnxbot._load_cached` still dispatched `V2Policy if contract == "2"`, so a
real 23-input contract-2 export died on its first move with `Invalid input
name: offer_proposer`, a 29-input contract-4 export was routed to the
contract-1 feature-tensor policy and died with `Required inputs ([terrain,
token, ...]) are missing`, and the only file that worked was the repo's own
re-stamped fixture — which is exactly why the suite was green. Every test
here would have failed on that branch.

Fixtures, and what each is:

* `dev-contract2.onnx` — a **genuine** dev-HexNet export (`tmp/export/
  linear2k.onnx`, exporter commit `36a8fa03`, 2026-08-31): contract `"2"`,
  23 declared inputs, `search=mcts`, real learned weights. The file PR #2
  broke.
* `stub-contract3.onnx`, `stub-contract4.onnx` — 27- and 29-input stubs
  (`fixtures/build_stub.py`). Real in shape, synthetic in weights: no genuine
  contract-3 or contract-4 export exists on any box this repo runs on, because
  producing one needs `hexset.export_onnx`, which needs torch. Stated plainly
  rather than papered over; `tests/test_onnx_record.py` pins the field names
  and shapes of all 29 fields against `hexset.onnx_record.RECORD_FIELDS`
  torch-free, which is the strongest check available without a real export.
* `tiny.onnx` — a real contract-1 export (no `contract` key at all). The
  owner dropped contract 1 on 2026-09-02
  (`docs/engine-divergence-2026-09-02.md`, B5): `encoding_v1.py` and the
  policy that read it are gone, so this fixture now exists only to pin that
  a contract-1 (or contract-less) file is refused by name at load, both here
  and in `RecordBrain`.
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

from hexset.server.constants import RECORD_CONTRACTS  # noqa: E402
from hexset.clients.onnxbot import V2Policy, load  # noqa: E402
from hexset.server.rules import options_for  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
DEV_CONTRACT2 = FIXTURES / "dev-contract2.onnx"
STUB_CONTRACT3 = FIXTURES / "stub-contract3.onnx"
STUB_CONTRACT4 = FIXTURES / "stub-contract4.onnx"
CONTRACT1 = FIXTURES / "tiny.onnx"


@pytest.fixture(autouse=True)
def _clear_loader_cache():
    from hexset.clients.onnxbot import _load_cached

    _load_cached.cache_clear()
    yield
    _load_cached.cache_clear()


def _board():
    return random_base_board(random.Random(0))


def _declared_inputs(path: Path) -> list[str]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return [i.name for i in session.get_inputs()]


def _metadata(path: Path) -> dict[str, str]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return dict(session.get_modelmeta().custom_metadata_map)


# --- What the fixtures actually are -------------------------------------------


def test_the_dev_fixture_is_a_real_contract_2_export_with_23_inputs():
    """Pinned so nobody quietly swaps a re-stamped stub back in. 23 inputs and
    a `checkpoint_sha256`/`exporter_commit` pair is what a genuine export from
    `hexset.export_onnx` looks like; the stub has neither."""
    meta = _metadata(DEV_CONTRACT2)
    assert meta["contract"] == "2"
    assert len(_declared_inputs(DEV_CONTRACT2)) == 23
    assert meta["exporter_commit"] and meta["checkpoint_sha256"]


def test_the_stub_fixtures_declare_27_and_29_inputs():
    assert _metadata(STUB_CONTRACT3)["contract"] == "3"
    assert len(_declared_inputs(STUB_CONTRACT3)) == 27
    assert _metadata(STUB_CONTRACT4)["contract"] == "4"
    assert len(_declared_inputs(STUB_CONTRACT4)) == 29


def test_no_fixture_stamps_the_29_input_graph_as_contract_2():
    """The specific mistake PR #2 made: one number naming two graphs. The
    contract number belongs to `hexset.export_onnx._CONTRACT_VERSION`; this
    repo reads it and never assigns it."""
    for path in (DEV_CONTRACT2, STUB_CONTRACT3, STUB_CONTRACT4):
        meta = _metadata(path)
        declared = len(_declared_inputs(path))
        assert (meta["contract"], declared) in {("2", 23), ("3", 27), ("4", 29)}


# --- Dispatch ------------------------------------------------------------------


@pytest.mark.parametrize("path", [DEV_CONTRACT2, STUB_CONTRACT3, STUB_CONTRACT4])
def test_every_record_contract_routes_to_the_record_policy(path):
    assert isinstance(load(str(path), _board().topology).policy, V2Policy)


def test_a_contract_1_export_is_refused_by_name():
    """`tiny.onnx` carries no `contract` key at all — the pre-metadata
    exports, which default to contract 1. The owner dropped contract 1
    2026-09-02: there is no feature-tensor policy left to route it to, so it
    is refused exactly like any other unsupported contract, naming what it
    found and what this server still serves."""
    assert "contract" not in _metadata(CONTRACT1)
    with pytest.raises(ValueError) as caught:
        load(str(CONTRACT1), _board().topology)
    assert "contract='1'" in str(caught.value)
    assert "2, 3, 4" in str(caught.value)


def test_an_unknown_contract_is_refused_by_name(tmp_path):
    """Not silently routed to a guessed graph shape, which is what PR #2 did
    with contract 4 — the failure then surfaced as a missing-input error
    naming tensors nobody had asked about, one second at a time, on a
    runner thread's stderr while the table hung."""
    import onnx

    model = onnx.load(str(STUB_CONTRACT4))
    for entry in model.metadata_props:
        if entry.key == "contract":
            entry.value = "99"
    future = tmp_path / "from-the-future.onnx"
    onnx.save(model, str(future))

    with pytest.raises(ValueError) as caught:
        load(str(future), _board().topology)
    assert "contract='99'" in str(caught.value)
    assert "2, 3, 4" in str(caught.value)


def test_the_contract_table_covers_what_the_policy_serves():
    assert RECORD_CONTRACTS == {"2", "3", "4"}


# --- Loading is not enough: it has to play ------------------------------------


@pytest.mark.parametrize(
    "path,expected_inputs",
    [(DEV_CONTRACT2, 23), (STUB_CONTRACT3, 27), (STUB_CONTRACT4, 29)],
)
def test_a_record_contract_checkpoint_plays_legal_actions_from_every_phase(
    path, expected_inputs
):
    """The test PR #2 could not have passed. `V2Policy._run` feeds the graph
    the fields *it declares*, so the same 29-field record drives a 23-, a 27-
    and a 29-input graph; feeding all 29 to the 23-input one raises
    `InvalidArgument: Invalid input name: offer_proposer` at the first move.

    `network_bot` rather than `spawn`, deliberately: `dev-contract2.onnx` says
    `search=mcts` in its metadata, and this test is about the feed, not about
    how many simulations the file asks for.
    """
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


def test_the_real_dev_export_survives_a_position_with_a_live_offer():
    """`offer_proposer`/`offer_give`/`offer_want`/`offer_answered` are the four
    fields contract 3 added and contract 2 does not declare — the exact names
    onnxruntime rejected. A contract-2 graph has to keep playing through a
    `TRADE_RESPOND` position, where the record carries them all."""
    from hexset.actions import ActionType
    from hexset.clients.onnxbot import network_bot

    board = _board()
    bot = network_bot(str(DEV_CONTRACT2), board)
    game = start(board, 4, random.Random(11))

    for _ in range(600):
        if game.won_by is not None:
            break
        if game.phase is Phase.TRADE_RESPOND:
            break
        options = options_for(game)
        offers = [a for a in options if a.type is ActionType.PROPOSE_TRADE]
        apply(game, offers[0] if offers else bot.choose(game))

    assert game.phase is Phase.TRADE_RESPOND, "never reached a live offer"
    assert game.offer is not None
    assert bot.choose(game) in options_for(game)


# --- The external client reads the same table ---------------------------------


@pytest.mark.parametrize("path", [DEV_CONTRACT2, STUB_CONTRACT3, STUB_CONTRACT4])
def test_recordbrain_accepts_every_record_contract(path):
    """`RecordBrain.load` refused anything but `"2"` and told the owner of a
    contract-4 file that it "is a contract=1 checkpoint". It now accepts the
    same set `onnxbot` routes to `V2Policy`, and a search-flagged file is
    still refused for being a search, not for its contract."""
    from hexset.clients.botclient import RecordBrain

    meta = _metadata(path)
    if meta.get("search") == "mcts":
        with pytest.raises(ValueError) as caught:
            RecordBrain.load(str(path))
        assert "search" in str(caught.value)
        return
    assert RecordBrain.load(str(path)).session is not None


def test_recordbrain_names_the_contract_it_actually_found():
    from hexset.clients.botclient import RecordBrain

    with pytest.raises(ValueError) as caught:
        RecordBrain.load(str(CONTRACT1))
    assert "contract='1'" in str(caught.value)


# --- The record against a real export's declared shapes ------------------------


def test_the_record_matches_a_real_dev_exports_declared_input_shapes():
    """A torch-free pin on the contract, and the strongest one available here.

    `dev-contract2.onnx` is a genuine `hexset.export_onnx` artefact, so its
    declared input shapes *are* dev's `_shapes` table for the 23 fields it
    carries — no transcription, no stub. This covers 23 fields; the full 29
    are pinned against `hexset.onnx_record.RECORD_FIELDS` by
    `tests/test_onnx_record.py`, torch-free, since `hexset.onnx_record` no
    longer needs torch to import.
    """
    import numpy as np

    from hexset.actions import build_space

    from hexset.onnx_record import record_from_game

    board = _board()
    topology = board.topology
    space = build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, 4)
    game = start(board, 4, random.Random(3))
    record = record_from_game(game, 0, space, tuple(options_for(game)))

    session = ort.InferenceSession(str(DEV_CONTRACT2), providers=["CPUExecutionProvider"])
    checked = 0
    for declared in session.get_inputs():
        assert declared.name in record, declared.name
        value = np.asarray(record[declared.name])
        # The graph's leading axis is the batch; the record is one row.
        assert list(value.shape) == list(declared.shape[1:]), declared.name
        expected = "tensor(bool)" if value.dtype == np.bool_ else "tensor(int64)"
        assert declared.type == expected, declared.name
        checked += 1
    assert checked == 23
