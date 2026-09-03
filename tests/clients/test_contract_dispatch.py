"""Which graph shape a checkpoint's `contract` metadata gets routed to, and
whether a record-contract graph actually loads and plays.

This is the regression suite for the headline finding of the PI's review of
PR #2: the branch redefined contract `"2"` to mean a wider record while
`onnxbot._load_cached` still dispatched on the number, so a real export died
on its first move with `Invalid input name: offer_proposer` and the only file
that worked was the repo's own re-stamped fixture — which is exactly why the
suite was green. The rule that came out of it, and what this file pins: a
contract number names exactly one graph shape, this repo reads it and never
assigns it, and anything else is refused **by name** rather than routed to a
guess.

**Contract 5** is the only one served. 2, 3 and 4 are the offer protocol's
contracts — 3 added four live-offer fields, 4 the two ledger fields, and all
three declare a `pair_mask` input and a `pair_index` output for the
one-for-one give/want heads. Trading is now one engine event with no actions
at all (`hexset.trading`), so those graphs describe a game this engine does
not play: there is no honest way to feed them and they are refused.

Fixtures, and what each is:

* `stub-contract5.onnx`, `stub-contract5-partial.onnx` — 25- and 23-input
  stubs (`fixtures/build_stub.py`). Real in shape, synthetic in weights: no
  genuine contract-5 export exists on any box this repo runs on, because
  producing one needs `hexset.export_onnx`, which needs torch. Stated plainly
  rather than papered over; `tests/test_onnx_record.py` pins the field names
  and shapes of every field against `hexset.onnx_record.RECORD_FIELDS`
  torch-free, which is the strongest check available without a real export.
  The pair exists because a loader that feeds a graph every field it happens
  to have, rather than the ones the graph *declares*, breaks on the shorter
  one (`onnxbot.V2Policy._run`).
* `dev-contract2.onnx` — a **genuine** dev-HexNet export (`tmp/export/
  linear2k.onnx`, exporter commit `36a8fa03`, 2026-08-31): contract `"2"`,
  23 declared inputs, real learned weights. Kept, now that contract 2 is no
  longer served, as the one fixture that proves a real file is refused by
  name rather than by crashing somewhere downstream.
* `tiny.onnx` — a real contract-1 export (no `contract` key at all). The
  owner dropped contract 1 on 2026-09-02
  (`docs/engine-divergence-2026-09-02.md`, B5); this pins that a
  contract-less file is refused by name too, here and in `RecordBrain`.
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
STUB = FIXTURES / "stub-contract5.onnx"
STUB_PARTIAL = FIXTURES / "stub-contract5-partial.onnx"
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


def test_the_stub_fixtures_declare_the_full_and_the_partial_record():
    assert _metadata(STUB)["contract"] == "5"
    assert len(_declared_inputs(STUB)) == 25
    assert _metadata(STUB_PARTIAL)["contract"] == "5"
    assert len(_declared_inputs(STUB_PARTIAL)) == 23
    assert set(_declared_inputs(STUB_PARTIAL)) < set(_declared_inputs(STUB))


def test_the_dev_fixture_is_still_a_real_contract_2_export():
    """Pinned so nobody quietly swaps a re-stamped stub back in. A
    `checkpoint_sha256`/`exporter_commit` pair is what a genuine export from
    `hexset.export_onnx` looks like; the stubs have neither."""
    meta = _metadata(DEV_CONTRACT2)
    assert meta["contract"] == "2"
    assert len(_declared_inputs(DEV_CONTRACT2)) == 23
    assert meta["exporter_commit"] and meta["checkpoint_sha256"]


def test_no_fixture_stamps_a_graph_with_a_contract_it_is_not():
    """The specific mistake PR #2 made: one number naming two graphs. The
    contract number belongs to `hexset.export_onnx._CONTRACT_VERSION`; this
    repo reads it and never assigns it."""
    for path, declared in ((DEV_CONTRACT2, 23), (STUB, 25), (STUB_PARTIAL, 23)):
        assert len(_declared_inputs(path)) == declared
    assert _metadata(DEV_CONTRACT2)["contract"] != _metadata(STUB)["contract"]


# --- Dispatch ------------------------------------------------------------------


@pytest.mark.parametrize("path", [STUB, STUB_PARTIAL])
def test_a_contract_5_graph_routes_to_the_record_policy(path):
    assert isinstance(load(str(path), _board().topology).policy, V2Policy)


def test_a_contract_1_export_is_refused_by_name():
    """`tiny.onnx` carries no `contract` key at all — the pre-metadata
    exports, which default to contract 1. There is no feature-tensor policy
    left to route it to, so it is refused exactly like any other unsupported
    contract, naming what it found and what this server still serves."""
    assert "contract" not in _metadata(CONTRACT1)
    with pytest.raises(ValueError) as caught:
        load(str(CONTRACT1), _board().topology)
    assert "contract='1'" in str(caught.value)
    assert "5" in str(caught.value)


def test_an_offer_protocol_contract_is_refused_by_name():
    """A real contract-2 export, refused loudly rather than fed a record it
    does not declare. The graph asks for `offer_proposer` and `pair_mask`,
    which a contract-5 record does not carry, and there is no honest way to
    supply them: the offer protocol they describe no longer exists."""
    with pytest.raises(ValueError) as caught:
        load(str(DEV_CONTRACT2), _board().topology)
    assert "contract='2'" in str(caught.value)


def test_an_unknown_contract_is_refused_by_name(tmp_path):
    """Not silently routed to a guessed graph shape, which is what PR #2 did
    with contract 4 — the failure then surfaced as a missing-input error
    naming tensors nobody had asked about, one second at a time, on a
    runner thread's stderr while the table hung."""
    import onnx

    model = onnx.load(str(STUB))
    for entry in model.metadata_props:
        if entry.key == "contract":
            entry.value = "99"
    future = tmp_path / "from-the-future.onnx"
    onnx.save(model, str(future))

    with pytest.raises(ValueError) as caught:
        load(str(future), _board().topology)
    assert "contract='99'" in str(caught.value)
    assert "5" in str(caught.value)


def test_the_contract_table_covers_what_the_policy_serves():
    assert RECORD_CONTRACTS == {"5"}


# --- Loading is not enough: it has to play ------------------------------------


@pytest.mark.parametrize("path,expected_inputs", [(STUB, 25), (STUB_PARTIAL, 23)])
def test_a_record_contract_checkpoint_plays_legal_actions_from_every_phase(
    path, expected_inputs
):
    """The test PR #2 could not have passed. `V2Policy._run` feeds the graph
    the fields *it declares*, so the same record drives both a full and a
    partial graph; feeding every field to the shorter one raises
    `InvalidArgument: Invalid input name: ledger_known` at the first move."""
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
    """A network seat publishes no valuation of its own yet (that is HexNet's
    side of the mechanic), so it never trades — but the *table* does, and the
    record it is fed carries every seat's vector and hands that a trade has
    moved. It has to keep playing through that."""
    from hexset.clients.onnxbot import network_bot
    from hexset.game import roll_dice, to_move
    from hexset.server.webplay import PostedValuation

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
    wants = (-1.0, 0.0, 0.0, 0.0, 1.0)
    traders = [None] * 4
    traders[mover] = PostedValuation(wants)
    traders[other] = PostedValuation(tuple(-v for v in wants))
    game.traders = tuple(traders)

    roll_dice(game, 8)
    assert game.trades, "the engine cleared nothing to play on through"
    assert bot.choose(game) in options_for(game)


# --- The external client reads the same table ---------------------------------


@pytest.mark.parametrize("path", [STUB, STUB_PARTIAL])
def test_recordbrain_accepts_every_record_contract(path):
    """`RecordBrain.load` refused anything but `"2"` and told the owner of a
    contract-4 file that it "is a contract=1 checkpoint". It now accepts the
    same set `onnxbot` routes to `V2Policy`."""
    from hexset.clients.botclient import RecordBrain

    assert RecordBrain.load(str(path)).session is not None


def test_recordbrain_names_the_contract_it_actually_found():
    from hexset.clients.botclient import RecordBrain

    for path, found in ((CONTRACT1, "1"), (DEV_CONTRACT2, "2")):
        with pytest.raises(ValueError) as caught:
            RecordBrain.load(str(path))
        assert f"contract='{found}'" in str(caught.value)


# --- The record against the stub's declared shapes -----------------------------


def test_the_record_matches_the_stubs_declared_input_shapes():
    """A torch-free pin on the contract. The stub's shapes are transcribed
    from `hexset.onnx_record.record_shapes` by `fixtures/build_stub.py`, so
    this is weaker than the same check against a genuine export was —
    `tests/test_onnx_record.py` pins every field against `RECORD_FIELDS`
    directly, which is the check that does not go through a transcription.
    """
    import numpy as np

    from hexset.actions import build_space
    from hexset.onnx_record import record_from_game

    board = _board()
    topology = board.topology
    space = build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, 4)
    game = start(board, 4, random.Random(3))
    record = record_from_game(game, 0, space, tuple(options_for(game)))

    session = ort.InferenceSession(str(STUB), providers=["CPUExecutionProvider"])
    checked = 0
    for declared in session.get_inputs():
        assert declared.name in record, declared.name
        value = np.asarray(record[declared.name])
        # The graph's leading axis is the batch; the record is one row.
        assert list(value.shape) == list(declared.shape[1:]), declared.name
        if value.dtype == np.bool_:
            expected = "tensor(bool)"
        elif value.dtype == np.float32:
            expected = "tensor(float)"
        else:
            expected = "tensor(int64)"
        assert declared.type == expected, declared.name
        checked += 1
    assert checked == 25
