# SPDX-License-Identifier: GPL-3.0-only
"""Export a trained checkpoint to ONNX, for hexset-ui's onnxruntime backend.

`hexset.netbot` loads a checkpoint straight into PyTorch: fine for the training
box, but heavier than a browser game on borrowed hardware needs. This script
runs the exact reconstruction path `netbot.load()` already trusts (same
shapes, same `state_dict` keys), traces it once with `torch.onnx.export`, and
embeds the handful of facts hexset-ui's `hexset_ui.onnxbot.load()` needs at
runtime as ONNX metadata props.

**Contract v2.** hexset-ui is an *interface*: it emits state, it accepts an
action, and nothing in that repo should know how a network reads a position.
Contract 1 broke that — `hexset_ui.encoding` was a 340-line reimplementation
of `hexset.encoding`, and `onnxbot.py` carried the masking, log-softmax,
give/want factorisation and seat un-rotation in numpy, both of which had to
stay bit-identical with this repo forever, in a language it does not run.
Contract 2 moves all of that into the graph: `record -> encoder -> HexNet
-> heads`, where `encoder` is `hexset.onnx_record.RecordEncoder` (the traced
mirror of `hexset.encoding`, see `onnx_record.py`'s own docstring) and `heads`
is masking, log-softmax, the give/want outer sum, argmax and value
un-rotation — lifted here out of `hexset_ui.onnxbot.OnnxPolicy`, whose own
math they still exactly mirror (`hexset.policy.masked_log_softmax`/
`pair_logits` are the same formulas onnxbot.py ported to numpy, and are
reused here rather than written a third time).

The line contract 2 draws is **the rules, not the network**: hexset-ui still
computes the position and what a seat may legally know (encoded as the
*information-set record*, `hexset.onnx_record.RECORD_FIELDS`), still enumer-
ates legal moves and builds `action_mask`/`pair_mask`, and still runs
`mcts.py`/`search2.py` over the rules. Everything downstream of "what is true
and visible" — how to read it — is now the graph's job, so `NetworkBot` can
read `action_index`/`pair_index` straight off one forward pass and a search
can read `prior`/`pair_prior`/`value` off the same one.

* **Inputs**, a dynamic leading batch axis, named exactly
  `hexset.onnx_record.RECORD_FIELDS` (27 names: board, position, information
  set, legality — see that module's docstring and `onnx-contract-v2.md`'s
  table). All int64 except `action_mask`/`pair_mask`, which are bool. `B` is
  1 for a single decision and a whole wave of leaves for the UI's MCTS
  `LeafEvaluator`, so the axis must really be dynamic.
* **Outputs**, in this order: `action_index` and `pair_index` (`(B,)` int64,
  argmax over the masked distributions), `prior` `(B, space.size)` and
  `pair_prior` `(B, NUM_PAIRS)` (float32, normalised over legal entries, zero
  elsewhere), and `value` `(B, players)` (float32, **board-seat order**,
  already un-rotated). One forward serves both callers: `NetworkBot` reads
  the two indices, a search reads the three distributions.
* **Metadata props** (`hexset_ui.modelmeta` and `onnxbot._load_cached`):
  `contract` is `"2"`; `players` and the `num_hexes`/`num_vertices`/
  `num_edges` fingerprint are required; `max_offers` (`""` for none) and
  `iteration` are read with defaults; and only when asked for on the command
  line, `search=mcts` with `simulations`/`wave`. Inference device is
  deliberately not metadata: the UI takes it from its own `--device`.
* **Provenance props**, which nothing reads at runtime and everything reads
  afterwards: `source_checkpoint` (repo-relative), `checkpoint_sha256`,
  `exported_at`, and `exporter_commit` when it can be determined. A deployed
  filename does not identify a file — `linear805.onnx` is lam095 at 805 and
  `mlp550.onnx` is twin-greedy at 550 — and `source_checkpoint` is normally
  `runs/<run>/latest.pt`, a moving pointer. The digest pins the weights and
  the commit pins the code, so an artifact can be reproduced from itself.
  See `_provenance`.
* **Scope cut, deliberate.** The stochastic/Gumbel sampling path is gone —
  every checkpoint served here plays argmax, so there is nothing to sample.
  A temperature input would be the way back in, if that is ever wanted.
* **Runtime**: onnxruntime >= 1.18 on the CPU provider, inside the UI's
  Python server — not onnxruntime-web; the browser only ever talks HTTP.
  Opset 18 is well inside that runtime's range.

A numerical parity check runs automatically after every export and raises
(non-zero exit) rather than warns: an export bug that silently degrades a
policy's quality is much more expensive to catch later, on a board, than
here, on real seeded positions.

Needs `torch` (not a declared dependency of this package — see
`hexset.netbot`'s own docstring for why) plus `onnx`/`onnxruntime`
(`pip install -e '.[export]'`). Run from `src/`::

    python -m hexset.export_onnx --checkpoint ../runs/lam095/latest.pt --out lam095.onnx
    python -m hexset.export_onnx --checkpoint ../runs/lam095/latest.pt \\
        --out mcts256.onnx --search mcts --simulations 256

Then drop the file into hexset-ui's `models/`; its stem is the name in the
in-game picker.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .actions import ActionSpace, build_space
from .board.board import random_base_board
from .board.terrain import NUM_RESOURCES
from .board.topology import Topology
from .cards import NUM_DEV_CARDS
from .encoding import StaticGraph, static_graph
from .game import Game, is_over, start, to_move
from .model import HexNet, config_from_args
from .onnx_record import RECORD_FIELDS, RecordEncoder, record_batch
from .play import step_randomly
from .policy import NUM_PAIRS, masked_log_softmax, pair_logits
from .policy import _OFF_DIAGONAL

_INPUT_NAMES = RECORD_FIELDS
_OUTPUT_NAMES = ("action_index", "pair_index", "prior", "pair_prior", "value")
_BATCH = "batch"

# `action_mask`/`pair_mask` are the only bool inputs; every other input is
# int64. Every output is float32 except the two argmaxed indices.
_BOOL_INPUTS = frozenset({"action_mask", "pair_mask"})
_INT_OUTPUTS = frozenset({"action_index", "pair_index"})

# The only value `hexset_ui.modelmeta.search_config` acts on; anything else in
# the `search` key is read as "no search", so writing anything else would be
# a silent no-op rather than a setting.
_SEARCHES = ("none", "mcts")

# `contract` tells `hexset_ui.onnxbot.load` which graph shape it is looking
# at (absent/`"1"` means the old feature-tensor-in, raw-logits-out shape;
# `"2"` the 23-input record). `"3"` is contract 2 plus the four live-offer
# record fields (trading design part 1) — same outputs, four more inputs.
# Bump this again only if the record or the output tuple changes shape.
_CONTRACT_VERSION = "3"


def _base_topology() -> Topology:
    """hexset-ui only ever plays the base map, and the base map's topology is
    seed-invariant — only terrain, tokens and ports are randomised, never the
    hex/vertex/edge graph (`random_base_board` always calls
    `build_topology(BASE_LAYOUT)`) — so any seed gives the same topology a
    checkpoint will actually see."""
    return random_base_board(random.Random(0)).topology


def _shapes(graph: StaticGraph, players: int, space: ActionSpace) -> dict[str, tuple]:
    """Every tensor's per-row shape, i.e. the contract minus the batch axis.
    One table for the sample inputs, the read-back check and the tests."""
    return {
        "terrain": (graph.num_hexes,),
        "token": (graph.num_hexes,),
        "port_code": (graph.num_vertices,),
        "robber": (),
        "vertex_owner": (graph.num_vertices,),
        "vertex_building": (graph.num_vertices,),
        "edge_owner": (graph.num_edges,),
        "bank": (NUM_RESOURCES,),
        "knights_played": (players,),
        "award_points": (players,),
        "longest_road_holder": (),
        "largest_army_holder": (),
        "phase": (),
        "free_roads": (),
        "deck_size": (),
        "turns": (),
        "perspective": (),
        "own_hand": (NUM_RESOURCES,),
        "hand_totals": (players,),
        "own_dev": (NUM_DEV_CARDS,),
        "dev_totals": (players,),
        "offer_give": (NUM_RESOURCES,),
        "offer_want": (NUM_RESOURCES,),
        "offer_proposer": (),
        "offer_answered": (players,),
        "action_mask": (space.size,),
        "pair_mask": (NUM_PAIRS,),
        "action_index": (),
        "pair_index": (),
        "prior": (space.size,),
        "pair_prior": (NUM_PAIRS,),
        "value": (players,),
    }


def _sample_games(players: int, count: int, seed: int = 0) -> list[tuple[Game, int]]:
    """`count` real `(game, perspective)` pairs from independent seeded
    playouts on the base topology.

    Not random gaussians, and not even random integers: the record is
    integers *with meaning* — `vertex_owner` in `[-1, players)`, a mask with
    at least one legal action per row — so anything else would trace and run
    the graph without ever exercising a mask that looks like one, which is
    exactly the case `action_index`/`pair_index`'s exact-match gate exists to
    stress. `perspective` is always `to_move(game)`, same as
    `hexset_ui.onnxbot.Request.seat`.
    """
    rng = random.Random(seed)
    pairs: list[tuple[Game, int]] = []
    while len(pairs) < count:
        game = start(random_base_board(rng), players, rng)
        for _ in range(rng.randrange(0, 150)):
            if is_over(game):
                break
            step_randomly(game, rng)
        if is_over(game):
            continue
        pairs.append((game, to_move(game)))
    return pairs


def _sample_inputs(space: ActionSpace, players: int, batch: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Real information-set records, for tracing and for the parity check
    alike — `_sample_games` plus `hexset.onnx_record.record_batch`."""
    return record_batch(_sample_games(players, batch, seed), space)


class _ExportWrapper(nn.Module):
    """`record -> encoder -> HexNet -> heads`.

    `torch.onnx.export` wants a tuple of tensors back, not `HexNet`'s own
    `Prediction` dataclass or a dict, so this both assembles the pipeline and
    flattens its output.
    """

    def __init__(self, encoder: RecordEncoder, net: HexNet, players: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.net = net
        self.players = players
        # Same array `hexset_ui.onnxbot`'s `_OFF_DIAGONAL` is — the diagonal
        # of an offer (give == want) is never legal, so it is masked out once
        # here rather than checked per offer. `hexset.policy` already defines
        # it for training; reused rather than rebuilt a third time.
        self.register_buffer("off_diagonal", torch.from_numpy(_OFF_DIAGONAL))

    def forward(
        self,
        terrain: Tensor,
        token: Tensor,
        port_code: Tensor,
        robber: Tensor,
        vertex_owner: Tensor,
        vertex_building: Tensor,
        edge_owner: Tensor,
        bank: Tensor,
        knights_played: Tensor,
        award_points: Tensor,
        longest_road_holder: Tensor,
        largest_army_holder: Tensor,
        phase: Tensor,
        free_roads: Tensor,
        deck_size: Tensor,
        turns: Tensor,
        perspective: Tensor,
        own_hand: Tensor,
        hand_totals: Tensor,
        own_dev: Tensor,
        dev_totals: Tensor,
        offer_give: Tensor,
        offer_want: Tensor,
        offer_proposer: Tensor,
        offer_answered: Tensor,
        action_mask: Tensor,
        pair_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        hexes, vertices, edges, globals_ = self.encoder(
            terrain,
            token,
            port_code,
            robber,
            vertex_owner,
            vertex_building,
            edge_owner,
            bank,
            knights_played,
            award_points,
            longest_road_holder,
            largest_army_holder,
            phase,
            free_roads,
            deck_size,
            turns,
            perspective,
            own_hand,
            hand_totals,
            own_dev,
            dev_totals,
            offer_give,
            offer_want,
            offer_proposer,
            offer_answered,
        )
        pred = self.net(hexes, vertices, edges, globals_)

        # `masked_log_softmax`/`pair_logits` are exactly
        # `hexset_ui.onnxbot._masked_log_softmax`/`_pair_logits`'s formulas,
        # already written once in `hexset.policy` for training-time sampling.
        slot_log_probs = masked_log_softmax(pred.logits, action_mask)
        offer_log_probs = masked_log_softmax(
            pair_logits(pred.give, pred.want), pair_mask & self.off_diagonal
        )
        action_index = slot_log_probs.argmax(dim=-1)
        pair_index = offer_log_probs.argmax(dim=-1)
        # Illegal entries sit at `NEG` before the softmax, so `exp` of the
        # shifted, normalised log-probs is already exactly zero there and
        # already sums to one over the legal entries -- nothing left to mask
        # or renormalise, unlike `LeafEvaluator._prior`'s option-subset gather.
        prior = slot_log_probs.exp()
        pair_prior = offer_log_probs.exp()

        # `hexset_ui.onnxbot._board_order`, as a gather: board seat `j`'s
        # value is the seat-relative value at `(j - perspective) % players`.
        board_seat = torch.arange(self.players, device=perspective.device)
        rotate = torch.remainder(
            board_seat.unsqueeze(0) - perspective.unsqueeze(1), self.players
        )
        value = pred.value.gather(1, rotate)

        return action_index, pair_index, prior, pair_prior, value


def _load_checkpoint(
    checkpoint: str, topology: Topology
) -> tuple[HexNet, ActionSpace, dict]:
    """The same reconstruction path `netbot.load` already trusts — so export
    uses the exact model-building logic the torch bot relies on, not a second
    implementation that could drift from it."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = state.get("args", {})
    players = int(args.get("players", 4))
    config = config_from_args(args)
    graph = static_graph(topology)
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, players
    )
    net = HexNet(space, graph, players, config)
    net.load_state_dict(state["net"])
    net.eval()
    # The default (non-fused) forward path aggregates neighbours with
    # `index_add_`, whose index tensor has duplicates by construction (many
    # vertices share a hex, many edges share a vertex) — the legacy
    # TorchScript ONNX exporter mishandles that ("does not support duplicated
    # values in 'index' field... will cause the ONNX model to be incorrect",
    # confirmed by a real mismatched export). `fused=True` swaps in the dense
    # adjacency-matmul path instead: "the same numbers up to float
    # reassociation" per its own docstring, with a dedicated equivalence test
    # already pinning that in torch — just architecturally the wrong default
    # to optimise for a training GPU (12-17% slower in CPU eager mode), which
    # is irrelevant for exporting a graph that runs once per move.
    net.fused = True
    return net, space, {
        "players": players,
        "max_offers": args.get("max_offers"),
        "iteration": int(state.get("iteration", 0)),
    }


def _repo_root() -> Path:
    """The checkout this package is being run from — `src/hexset/x.py` upward."""
    return Path(__file__).resolve().parents[2]


def _exporter_commit() -> str | None:
    """The commit that produced a file, or None rather than a guess.

    `git` is asked first but usually fails here: an export runs in a container
    that bind-mounts the worktree, and a worktree's `.git` is a file holding an
    absolute gitdir that does not exist inside it (the same trap the dev
    status journal records). `HEXSET_EXPORT_COMMIT` is the way in, set by
    whoever launches the container from a shell that *can* read the repo.

    Returns None when neither works. A missing key is honest; `"unknown"`
    stamped into an artifact is a lie that survives longer than the session.
    """
    env = os.environ.get("HEXSET_EXPORT_COMMIT", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _provenance(checkpoint: str) -> dict[str, str]:
    """What the artifact needs to describe itself once it has left this box.

    A filename cannot be trusted to do it — `linear805.onnx` on the deployment
    is lam095 at iteration 805, and `mlp550.onnx` is twin-greedy at 550 — so
    everything that identifies a file has to be inside it.

    `source_checkpoint` alone is not enough either: it is usually
    `runs/<run>/latest.pt`, a *moving* pointer whose bytes change the next time
    that run is resumed. `checkpoint_sha256` is what actually pins the weights,
    and `exporter_commit` pins the code that read them, so an artifact can be
    reproduced from itself without consulting anything external.
    """
    source = Path(checkpoint)
    try:
        relative = source.resolve().relative_to(_repo_root())
    except ValueError:
        # Outside the checkout entirely; the absolute path is all there is.
        relative = source
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    props = {
        "source_checkpoint": str(relative),
        "checkpoint_sha256": digest,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    commit = _exporter_commit()
    if commit:
        props["exporter_commit"] = commit
    return props


def _embed_metadata(
    path: Path,
    *,
    topology: Topology,
    players: int,
    max_offers: int | None,
    iteration: int,
    source_checkpoint: str,
    search: str,
    simulations: int | None,
    wave: int | None,
) -> None:
    """The facts `hexset_ui.onnxbot.load()` needs back at runtime: which
    contract the graph speaks, enough to build an `ActionSpace`, enough to
    refuse a topology mismatch loudly (the way `net.load_state_dict` already
    fails loudly on a shape mismatch today), and how the file wants to be
    played. Width/rounds/head-shape aren't included — those only matter for
    reconstructing the torch module, and tracing already baked them in.

    `search`/`simulations`/`wave` are written only when a search was asked
    for. `hexset_ui.modelmeta` would ignore a stray budget anyway ("a stale
    `simulations` left behind by an export cannot quietly turn a policy
    checkpoint into a search"), so leaving the keys out keeps the file's
    metadata saying exactly what it does."""
    import onnx

    props = {
        "contract": _CONTRACT_VERSION,
        "players": str(players),
        "num_hexes": str(topology.num_hexes),
        "num_vertices": str(topology.num_vertices),
        "num_edges": str(topology.num_edges),
        "max_offers": "" if max_offers is None else str(max_offers),
        "iteration": str(iteration),
        **_provenance(source_checkpoint),
    }
    if search == "mcts":
        props["search"] = "mcts"
        if simulations is not None:
            props["simulations"] = str(simulations)
        if wave is not None:
            props["wave"] = str(wave)

    model = onnx.load(str(path))
    onnx.helper.set_model_props(model, props)
    onnx.save(model, str(path))


def _check_tensor(tensor, shapes: dict[str, tuple], expected_dtype) -> None:
    import onnx

    kind = tensor.type.tensor_type
    if kind.elem_type != expected_dtype:
        name = onnx.TensorProto.DataType.Name(expected_dtype)
        raise ValueError(f"{tensor.name} is not {name}")
    dims = [d.dim_param or d.dim_value for d in kind.shape.dim]
    expected = [_BATCH, *shapes[tensor.name]]
    if dims != expected:
        raise ValueError(f"{tensor.name} is shaped {dims}, not {expected}")


def _verify_contract(onnx_path: Path, shapes: dict[str, tuple]) -> None:
    """Read the graph back and hold it to the names, dtypes and shapes
    `hexset_ui.onnxbot`'s v2 path hard-codes — the parity check below would
    pass a graph whose outputs were merely *renamed*, and the UI would then
    fail at the first move with an onnxruntime error rather than here."""
    import onnx

    model = onnx.load(str(onnx_path))
    found = {
        "input": [t.name for t in model.graph.input],
        "output": [t.name for t in model.graph.output],
    }
    expected = {"input": list(_INPUT_NAMES), "output": list(_OUTPUT_NAMES)}
    if found != expected:
        raise ValueError(f"graph signature {found} is not hexset-ui's {expected}")
    for tensor in model.graph.input:
        dtype = onnx.TensorProto.BOOL if tensor.name in _BOOL_INPUTS else onnx.TensorProto.INT64
        _check_tensor(tensor, shapes, dtype)
    for tensor in model.graph.output:
        dtype = onnx.TensorProto.INT64 if tensor.name in _INT_OUTPUTS else onnx.TensorProto.FLOAT
        _check_tensor(tensor, shapes, dtype)


def _verify_parity(
    net: HexNet,
    encoder: RecordEncoder,
    space: ActionSpace,
    players: int,
    onnx_path: Path,
    samples: int = 8,
    seed: int = 1,
) -> None:
    """Fails loudly (raises) rather than warns — there's no other test
    coverage of the ONNX path, so this is the one thing standing between a
    silent export bug and a bot that just plays worse for no logged reason.

    The reference path is `hexset.encoding.encode` (numpy) feeding the same
    `net`, `masked_log_softmax` and `pair_logits` the wrapper's `forward`
    uses — the exact math `hexset_ui.onnxbot`'s numpy port mirrors, computed
    here in eager torch rather than a fourth reimplementation. What this
    check actually exercises is tracing and onnxruntime execution: does the
    exported graph, run through the runtime hexset-ui actually uses, agree
    with the eager computation on real positions.

    `action_index`/`pair_index` must match exactly — argmax on a near-tie is
    the one failure this whole check exists to catch, and a tolerance would
    hide exactly that. `prior`/`pair_prior`/`value` keep the wider tolerance
    calibrated against a real checkpoint for the v1 contract (a
    width=64/rounds=2 net tripped a 1e-4 bound on ordinary fp32 accumulation
    drift, not a bug): `rtol=1e-3, atol=1e-4`.
    """
    import onnxruntime as ort

    pairs = _sample_games(players, samples, seed)
    record = record_batch(pairs, space)

    from .encoding import encode

    observations = [encode(game, seat) for game, seat in pairs]
    hexes = torch.from_numpy(np.stack([o.hexes for o in observations]))
    vertices = torch.from_numpy(np.stack([o.vertices for o in observations]))
    edges = torch.from_numpy(np.stack([o.edges for o in observations]))
    globals_ = torch.from_numpy(np.stack([o.globals for o in observations]))
    mask = torch.from_numpy(record["action_mask"])
    pair = torch.from_numpy(record["pair_mask"])
    perspective = torch.from_numpy(record["perspective"])
    off_diagonal = torch.from_numpy(_OFF_DIAGONAL)

    with torch.no_grad():
        pred = net(hexes, vertices, edges, globals_)
        slots = masked_log_softmax(pred.logits, mask)
        offers = masked_log_softmax(pair_logits(pred.give, pred.want), pair & off_diagonal)
        ref_action_index = slots.argmax(dim=-1).numpy()
        ref_pair_index = offers.argmax(dim=-1).numpy()
        ref_prior = slots.exp().numpy()
        ref_pair_prior = offers.exp().numpy()
        board_seat = torch.arange(players)
        rotate = torch.remainder(board_seat.unsqueeze(0) - perspective.unsqueeze(1), players)
        ref_value = pred.value.gather(1, rotate).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = dict(zip(_OUTPUT_NAMES, session.run(list(_OUTPUT_NAMES), record)))

    np.testing.assert_array_equal(onnx_out["action_index"], ref_action_index, err_msg="action_index")
    np.testing.assert_array_equal(onnx_out["pair_index"], ref_pair_index, err_msg="pair_index")
    for name, ref in (("prior", ref_prior), ("pair_prior", ref_pair_prior), ("value", ref_value)):
        np.testing.assert_allclose(onnx_out[name], ref, rtol=1e-3, atol=1e-4, err_msg=name)


def export(
    checkpoint: str,
    out: Path,
    *,
    topology: Topology | None = None,
    opset: int = 18,
    search: str = "none",
    simulations: int | None = None,
    wave: int | None = None,
) -> Path:
    """Load `checkpoint`, trace it to `out`, verify contract and parity, return `out`.

    `search="mcts"` (with an optional `simulations`/`wave` budget) makes the
    file ask hexset-ui to search over its own priors; the default plays one
    forward pass per move. A budget without a search is refused rather than
    written, since the UI would read the file as a plain policy regardless.
    """
    if search not in _SEARCHES:
        raise ValueError(f"search must be one of {_SEARCHES}, not {search!r}")
    if search == "none" and (simulations is not None or wave is not None):
        raise ValueError("simulations/wave only mean something with search='mcts'")

    topology = topology or _base_topology()
    net, space, meta = _load_checkpoint(checkpoint, topology)
    players = meta["players"]
    graph = static_graph(topology)
    shapes = _shapes(graph, players, space)

    encoder = RecordEncoder(graph, players)
    encoder.eval()
    wrapper = _ExportWrapper(encoder, net, players)
    wrapper.eval()

    dummy = _sample_inputs(space, players, batch=1, seed=0)
    dummy_tensors = tuple(torch.from_numpy(dummy[name]) for name in _INPUT_NAMES)

    torch.onnx.export(
        wrapper,
        dummy_tensors,
        str(out),
        # The classic TorchScript-based exporter, not torch's newer
        # dynamo-based default (`dynamo=True` since ~2.7, needing the extra
        # `onnxscript` dependency) — nothing in the encoder or `HexNet`'s
        # forward has data-dependent control flow, so tracing is safe and
        # there's nothing the dynamo path would buy here.
        dynamo=False,
        input_names=list(_INPUT_NAMES),
        output_names=list(_OUTPUT_NAMES),
        dynamic_axes={n: {0: _BATCH} for n in (*_INPUT_NAMES, *_OUTPUT_NAMES)},
        opset_version=opset,
    )
    _embed_metadata(
        out,
        topology=topology,
        source_checkpoint=checkpoint,
        search=search,
        simulations=simulations,
        wave=wave,
        **meta,
    )
    _verify_contract(out, shapes)
    _verify_parity(net, encoder, space, players, out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a .pt checkpoint (as hexset.train writes)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .onnx path (default: the checkpoint's own path with a .onnx suffix).",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--search",
        choices=_SEARCHES,
        default="none",
        help="How hexset-ui should play the file: one forward pass (none) or a "
        "PUCT search over its own priors (mcts). Written as metadata.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="MCTS descents per decision (hexset-ui defaults to 128, caps at 4096).",
    )
    parser.add_argument(
        "--wave",
        type=int,
        default=None,
        help="MCTS leaves batched per expansion (hexset-ui defaults to 16, caps at 256).",
    )
    args = parser.parse_args(argv)
    if args.search == "none" and (args.simulations is not None or args.wave is not None):
        parser.error("--simulations/--wave need --search mcts")

    checkpoint = Path(args.checkpoint)
    out = args.out or checkpoint.with_suffix(".onnx")
    path = export(
        str(checkpoint),
        out,
        opset=args.opset,
        search=args.search,
        simulations=args.simulations,
        wave=args.wave,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
