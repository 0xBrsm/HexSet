# SPDX-License-Identifier: GPL-3.0-only
"""The policy and value network over the encoded graph.

Message passing runs over two relations, hex-vertex and vertex-edge, in both
directions, with a global token that reads and writes every type. `StaticGraph`
also carries `hex_hex` and `vertex_vertex`; both are left out until they are
shown to pay, since hex adjacency is reachable through shared vertices and
vertex adjacency through shared edges, and the point of the extra relation is
only to shorten that path by a round.

Adjacency is fixed at construction, so a batch is dense `(B, nodes, features)`
against one shared index set rather than a block-diagonal union of graphs. That
holds because every game on a given layout shares a topology — only the
features differ. A second layout needs a second model, which is what would
happen anyway.

The heads emit exactly the widths `hexset.readout` declares and are scattered
into the flat action space by its plan, so the policy is masked and applied
with the same indices the engine already uses.

`ModelConfig.value_head` and `.policy_head` exist because of one measurement.
The value head is globally calibrated and locally blind: explained variance
+0.474 on its own positions, and `benchmarks.rank` puts its sibling ranking at
r=+0.574 with a 42.5% top-1 rate against 28.5% chance. It reads a single
64-vector `g` that saw the board only through `h.mean(1), v.mean(1), e.mean(1)`,
so two positions one settlement apart differ in it by about a fifty-fourth of a
mean, and one `nn.Linear` then has to turn that into the difference a search is
asking about. Whether that shape is the cause is a hypothesis, not a finding,
and the shapes below exist to be ablated against each other rather than because
any of them is known to be better. `"linear"` on both is every run on record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .actions import ActionSpace
from .board.terrain import NUM_RESOURCES
from .encoding import (
    HEX_FEATURES,
    Observation,
    StaticGraph,
    edge_features,
    global_features,
    vertex_features,
)
from .readout import EDGES, GLOBALS, HEXES, VERTICES, plan


VALUE_HEADS = ("linear", "mlp", "pooled", "mlp_pooled", "attn", "quantile")
POLICY_HEADS = ("linear", "mlp")

# Shapes that read the node embeddings rather than only the global token, and
# shapes that put a hidden layer in front of their output. `mlp_pooled` is both.
_POOLED = frozenset({"pooled", "mlp_pooled"})
_DEEP = frozenset({"mlp", "mlp_pooled"})


@dataclass(frozen=True)
class ModelConfig:
    """`width` is capacity; the two head fields are shape, and only shape.

    Every checkpoint under `runs/` was written by the defaults, so the defaults
    are load-bearing: `"linear"` on both has to build the same modules under the
    same names as before these fields existed, and `test_model` pins the
    `state_dict` keys to make sure it stays that way.

    The value shapes, and what each one would show if it won an ablation:

    `"mlp"` — the head is too shallow to read `g`. Cheapest of the five and the
    only one that leaves the head's input unchanged, so it separates "the
    readout is too weak" from "the readout is looking at the wrong thing".

    `"pooled"` — mean pooling washed out the extremes. `g` is built from
    `h.mean(1), v.mean(1), e.mean(1)`; this concatenates max-pools of the same
    three onto it, which preserves the best vertex rather than the average one.
    A settlement placed on the best remaining spot moves a max and barely moves
    a mean.

    `"mlp_pooled"` — both, and the control for reading the two above together.

    `"attn"` — the value needs to look at *specific* nodes, and which ones
    depends on the position. A single query off `g` attends over the 54 vertex
    embeddings; a max-pool is the same idea with the choice frozen.

    `"quantile"` — the variance screen's candidate 3. `"linear"`'s shape
    widened to `players x quantiles` outputs, read as one distribution per
    seat. **The mean of the quantiles is `V`**: `forward` returns the
    `players`-vector mean and nothing downstream sees the spread, so GAE, the
    search and `benchmarks.*` are unchanged by construction. What differs is
    the loss — `hexset.ppo` trains it with the per-seat quantile Huber loss
    instead of squared error — and the mechanism under test is that loss
    shaping the shared trunk, since Gate A2 measured the mean itself flat
    (bias^2 down 1.0% against a 20% line). `quantiles` below is its width.

    The policy shapes are `"linear"` and `"mlp"`, applied per node type. The
    policy is not the head that fails — the raw policy beats `search2-offers3`
    at 56.9% while the value head in the same tree loses 13.3/86.7 — so this is
    here to hold capacity fixed across an ablation of the value head, not
    because there is a defect to chase.
    """

    width: int = 64
    rounds: int = 2
    value_head: str = "linear"
    policy_head: str = "linear"
    # How many quantiles a `"quantile"` value head emits per seat. Inert under
    # every other shape, so the default is behaviour-neutral for every
    # checkpoint on record. 32 is QR-DQN's own N and `benchmarks.head_swap`'s
    # Gate A2 default, kept so the heat's head is the head the gate measured.
    quantiles: int = 32

    def __post_init__(self) -> None:
        # A mistyped shape has to fail before a run starts, not silently fall
        # back to the default and be discovered when the ablation reads flat.
        if self.value_head not in VALUE_HEADS:
            raise ValueError(f"unknown value_head {self.value_head!r}: {VALUE_HEADS}")
        if self.policy_head not in POLICY_HEADS:
            raise ValueError(f"unknown policy_head {self.policy_head!r}: {POLICY_HEADS}")
        if self.quantiles < 1:
            raise ValueError(f"a quantile head needs at least one level, got {self.quantiles}")


def config_from_args(args: Mapping[str, object]) -> ModelConfig:
    """The shape a checkpoint's stored `args` says it was trained with.

    Width and rounds have always been read back this way; the head shapes have
    to be, and were not everywhere. A checkpoint that records a shape and is
    rebuilt without it fails in `load_state_dict` on the key names -- the loud
    failure, and the right one, but it fired twice in `benchmarks/` on the first
    `--policy-head mlp` lineage because two net builders read only width and
    rounds. One reader, so it cannot drift a third time.

    Missing keys default to what every checkpoint written before the field
    existed was trained with, so a namespace that predates a knob means "the
    default shape" rather than an error.
    """
    return ModelConfig(
        width=int(args.get("width", 64)),  # type: ignore[arg-type]
        rounds=int(args.get("rounds", 2)),  # type: ignore[arg-type]
        value_head=str(args.get("value_head", "linear")),
        policy_head=str(args.get("policy_head", "linear")),
        quantiles=int(args.get("quantiles", 32)),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class Prediction:
    """`logits` indexes the flat action space; mask it with `legal_mask`.

    `give` and `want` are the offer that rides alongside `PROPOSE_TRADE`, whose
    own slot in `logits` says only that proposing is available. Each is a
    categorical over resources, so v1 proposes one-for-one trades — the same
    shape `legal_actions` samples, and the majority of what actually gets
    traded. The `ask` order is not modelled: partner choice is worth nothing to
    the baseline, so there is no target to learn from yet.

    `value` is one number per seat, seat-relative like the encoder, so the
    search's max^n backup and `hexset.bots.STANCES` both read it unchanged.

    `quantiles` is `None` for every head shape but `"quantile"`, where it is
    the `(B, players, Q)` tensor whose last-axis mean *is* `value`. It exists
    so `hexset.ppo` can put the quantile loss on the same forward pass the
    policy already paid for; nothing that reads `value` has to know it is
    there, which is what keeps every other caller on the old shape.
    """

    logits: Tensor
    give: Tensor
    want: Tensor
    value: Tensor
    quantiles: Tensor | None = None


def _mlp(in_dim: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, width), nn.SiLU(), nn.Linear(width, width))


class Round(nn.Module):
    """One round of message passing: every type reads its neighbours and the global."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.hex = _mlp(3 * width, width)
        self.vertex = _mlp(4 * width, width)
        self.edge = _mlp(3 * width, width)
        self.globe = _mlp(4 * width, width)


def _head(in_dim: int, out_dim: int, width: int, *, deep: bool) -> nn.Module:
    """A head that is a bare `nn.Linear` when it is one.

    Not a wrapper module with a `.out` inside it, which is the obvious way to
    write this and would rename every head's parameters — `heads.vertices.weight`
    becomes `heads.vertices.out.weight` and no checkpoint in `runs/` loads again.
    The shallow case has to *be* the layer.
    """
    if not deep:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, width), nn.SiLU(), nn.Linear(width, out_dim))


def _output(head: nn.Module) -> nn.Linear:
    """The layer whose initialisation gain sets a head's output scale."""
    if isinstance(head, QuantileValueHead):
        head = head.module
    return head if isinstance(head, nn.Linear) else head[-1]


# The Huber width of the quantile value loss, in reward units, and a constant
# rather than a knob. `benchmarks.head_swap` calibrated it for Gate A2 and the
# register fixed it there; a flag would invite retuning it mid-screen, which
# would make Gate B's arms differ in two things instead of one. One lattice
# step of the return: terminal victory points are integers and
# `hexset.rewards.relative_points` divides by ten, so 1/30 of a reward unit is
# the finest resolution the label actually has.
QUANTILE_HUBER_KAPPA = 1.0 / 30.0


def quantile_levels(count: int) -> Tensor:
    """The midpoint levels `(i + 0.5) / Q`, as the register specifies.

    Midpoints rather than `i / (Q - 1)` because the pinball loss at level 0 or 1
    is one-sided: it is minimised by the sample minimum or maximum, which no
    finite sample estimates stably, and those two atoms would then drag the
    mean-of-quantiles that everything downstream reads as `V`. Midpoints also
    make the levels symmetric about 0.5, which is what makes the mean of the
    quantiles equal the mean of a symmetric distribution exactly rather than
    approximately.
    """
    if count < 1:
        raise ValueError(f"a quantile head needs at least one level, got {count}")
    return (torch.arange(count, dtype=torch.float32) + 0.5) / count


def quantile_huber_loss(
    predicted: Tensor, target: Tensor, levels: Tensor, kappa: float
) -> Tensor:
    """Pinball loss at `levels`, Huberised below `kappa`, meaned over everything.

    `predicted` is `(B, players, Q)`, `target` is `(B, players)`: one sampled
    return per seat per position, which is the same one-sample-per-transition
    setting QR-DQN trains in. The quantile levels are not constrained to come
    out sorted and are not sorted here -- everything downstream reads their
    mean, which no reordering changes.

    **`kappa` is small on purpose and this is the one number that had to be
    chosen rather than inherited.** Huberising the pinball loss replaces it,
    within `kappa` of the target, with a quadratic -- and a quadratic weighted
    by `|tau - 1{u<0}|` is minimised at the *expectile*, not the quantile. QR-
    DQN's kappa=1 is small against Atari returns in the hundreds; against this
    project's returns, which `hexset.rewards.relative_points` divides by ten into
    a range of about +-1 with a standard deviation near 0.2, kappa=1 would make
    the loss quadratic everywhere and quietly turn the head into an expectile
    head. The default is one lattice step of the return (1/30 of a reward unit,
    a third of a victory point, the resolution the label actually has), so the
    minimiser is displaced from the true quantile by at most one quantum of a
    label that has no finer resolution than that. `kappa=0` is the exact
    pinball loss.

    Meaned over Q rather than summed, so the loss sits at the same magnitude as
    the squared-error arm's and one learning rate serves both. Summing would
    multiply the quantile arm's gradient by Q and make the two arms differ in
    effective step size as well as in loss, which is precisely the confound the
    experiment is built to avoid.
    """
    if kappa < 0.0:
        raise ValueError(f"a Huber width cannot be negative, got {kappa}")
    difference = target.unsqueeze(-1) - predicted
    magnitude = difference.abs()
    if kappa > 0.0:
        element = torch.where(
            magnitude <= kappa,
            0.5 * difference * difference,
            kappa * (magnitude - 0.5 * kappa),
        ) / kappa
    else:
        element = magnitude
    # The indicator is piecewise constant in `predicted`, so it carries no
    # gradient; taken off the detached difference to say so rather than to rely
    # on a bool cast happening not to build one.
    below = difference.detach().lt(0.0).to(element.dtype)
    return ((levels - below).abs() * element).mean()


class QuantileValueHead(nn.Module):
    """`players x Q` outputs whose forward is the `players`-vector mean.

    The wrapper exists so that the *only* thing the rest of the system can see
    is `V`. `forward` returns the mean of the quantiles, so `_read_value`,
    `Prediction.value`, GAE, `lambda_returns`, `hexset.mcts` and every
    `benchmarks.*` reader keep the shape and the meaning they had; `spread`
    hands the full tensor to the one caller that needs it, the value loss in
    `hexset.ppo`. Gate A2 measured the mean flat, so the mechanism on trial is
    the loss reaching the shared trunk -- which an offline head swap cannot
    express and which this wiring is built to allow.

    Built by `hexset.model._head` at the same width and depth as the scalar head
    it replaces, widened only in its output, so the two differ in their loss
    and in nothing else. Its parameters live under `value.module.*`: a wrapper
    would rename `value.weight` on every recorded checkpoint, which is why the
    other five shapes must not have one -- this shape is new, has no
    checkpoints of its own, and pays no such cost.
    """

    def __init__(
        self,
        value_in: int,
        players: int,
        width: int,
        *,
        deep: bool,
        quantiles: int,
        kappa: float = QUANTILE_HUBER_KAPPA,
    ) -> None:
        super().__init__()
        self.players = players
        self.quantiles = quantiles
        self.kappa = kappa
        self.module = _head(value_in, players * quantiles, width, deep=deep)
        # A buffer, so `.to(device)` moves the levels with the parameters.
        # `persistent=False` for the same reason the adjacency buffers use it:
        # the levels are derived from `quantiles`, not learned. Q reaches a
        # rebuild through the checkpoint's `args`, which is where every other
        # shape parameter already lives, and keeping the levels out of the
        # state_dict is what lets `quantile_warm_start` hand over a scalar
        # head's weights without inventing one.
        self.register_buffer("levels", quantile_levels(quantiles), persistent=False)

    def spread(self, features: Tensor) -> Tensor:
        """`(B, players, Q)`. Output `p * Q + q` is seat `p`'s level `q`."""
        return self.module(features).view(-1, self.players, self.quantiles)

    def forward(self, features: Tensor) -> Tensor:
        """`V`, exactly the mean of the quantiles."""
        return self.spread(features).mean(-1)

    def loss(self, spread: Tensor, target: Tensor) -> Tensor:
        """The value term, given a spread this head already produced."""
        return quantile_huber_loss(spread, target, self.levels, self.kappa)


def quantile_warm_start(
    weights: "dict[str, Tensor]", players: int, quantiles: int
) -> "dict[str, Tensor]":
    """A scalar value head's weights, as a quantile head that predicts the same `V`.

    Every level starts at the scalar head's own output, so `V` at iteration 0
    is the checkpoint's own and the treatment arm of a heat starts from the
    same policy *and* the same critic as its control. Without this a quantile
    arm would spend its first iterations refitting a value head from orthogonal
    noise, and the contrast would read that as the treatment.

    "The same" here is to one float32 rounding, not bit-exact: averaging `Q`
    copies of a float rounds for `Q > 4`, measured at most 2.4e-7 absolute on
    real weights against a value scale of ~1. That is four orders of magnitude
    below the label's own 1/30 lattice, so it cannot be what a heat reads.

    The identical levels are not a symmetry trap: the pinball weight
    `|tau_q - 1{u<0}|` differs per level, so the first gradient already
    separates them.
    """
    prefix = "value.2." if "value.2.weight" in weights else "value."
    out: dict[str, Tensor] = {}
    for key, tensor in weights.items():
        if not key.startswith("value."):
            out[key] = tensor
            continue
        if key.startswith(prefix) and key[len(prefix) :] in ("weight", "bias"):
            if tensor.shape[0] != players:
                raise ValueError(
                    f"{key} emits {tensor.shape[0]} rows, not one per seat "
                    f"({players}); this is not a scalar value head"
                )
            # `repeat_interleave`, not `repeat`: `spread` views the output as
            # `(players, Q)`, so seat `p`'s levels are rows `p*Q .. p*Q+Q-1`.
            tensor = tensor.repeat_interleave(quantiles, dim=0)
        out["value.module." + key[len("value.") :]] = tensor
    return out


def _counts(index: Tensor, size: int) -> Tensor:
    out = torch.zeros(size)
    out.index_add_(0, index, torch.ones(index.numel()))
    return out.clamp(min=1.0).unsqueeze(-1)


def _mean_adjacency(dest: Tensor, src: Tensor, rows: int, cols: int) -> Tensor:
    """Row-normalised incidence: `out @ features` is the mean over neighbours.

    A zero-degree destination keeps an all-zero row, matching `_pass`, whose
    sum is 0 and whose count clamps to 1.
    """
    out = torch.zeros(rows, cols)
    out[dest, src] = 1.0
    return out / out.sum(-1, keepdim=True).clamp(min=1.0)


def _split(linear: nn.Linear, width: int) -> list[Tensor]:
    """The first trunk linear's weight, as per-input-block column views.

    `cat([a, b, c], -1) @ W.T` is `a @ Wa.T + b @ Wb.T + c @ Wc.T` for the
    column blocks of `W` — the fused forward computes the right side so the
    concat never materialises, and the global block's term is computed once
    per batch instead of once per node. Views, not copies: the parameters and
    their state_dict keys are untouched.
    """
    weight = linear.weight
    return [weight[:, i : i + width] for i in range(0, weight.shape[1], width)]


class HexNet(nn.Module):
    def __init__(
        self,
        space: ActionSpace,
        graph: StaticGraph,
        players: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.players = players
        self.readout = plan(space)
        self.num_hexes = graph.num_hexes
        self.num_vertices = graph.num_vertices
        self.num_edges = graph.num_edges
        width = self.config.width

        self.embed_hex = nn.Linear(HEX_FEATURES, width)
        self.embed_vertex = nn.Linear(vertex_features(players), width)
        self.embed_edge = nn.Linear(edge_features(players), width)
        self.embed_global = nn.Linear(global_features(players), width)

        self.rounds = nn.ModuleList(Round(width) for _ in range(self.config.rounds))

        deep_policy = self.config.policy_head in _DEEP
        self.heads = nn.ModuleDict(
            {
                head.source: _head(width, head.width, width, deep=deep_policy)
                for head in self.readout.heads
            }
        )
        self.trade_give = nn.Linear(width, NUM_RESOURCES)
        self.trade_want = nn.Linear(width, NUM_RESOURCES)

        shape = self.config.value_head
        if shape in _POOLED:
            # `g` plus a max-pool of each node type. Max rather than mean
            # because `g` already carries the three means, so a second copy of
            # them would add nothing the head could not already read.
            value_in = 4 * width
        elif shape == "attn":
            value_in = 2 * width
            # The query is a trunk layer by every property that matters to
            # `_initialise`: it feeds a softmax, not a prediction.
            self.value_query = nn.Linear(width, width)
        else:
            value_in = width
        if shape == "quantile":
            self.value: nn.Module = QuantileValueHead(
                value_in,
                players,
                width,
                deep=shape in _DEEP,
                quantiles=self.config.quantiles,
            )
        else:
            self.value = _head(value_in, players, width, deep=shape in _DEEP)
        # Plain attribute, not a buffer or a config field: it changes which
        # gradients flow, never a parameter or a shape, so a checkpoint written
        # with it on loads identically with it off.
        self.detach_value = False

        self._initialise()

        hex_vertex = torch.from_numpy(graph.hex_vertex)
        vertex_edge = torch.from_numpy(graph.vertex_edge)
        self.register_buffer("hv_hex", hex_vertex[0])
        self.register_buffer("hv_vertex", hex_vertex[1])
        self.register_buffer("ve_vertex", vertex_edge[0])
        self.register_buffer("ve_edge", vertex_edge[1])
        self.register_buffer("n_vertex_from_hex", _counts(hex_vertex[1], graph.num_vertices))
        self.register_buffer("n_hex_from_vertex", _counts(hex_vertex[0], graph.num_hexes))
        self.register_buffer("n_edge_from_vertex", _counts(vertex_edge[1], graph.num_edges))
        self.register_buffer("n_vertex_from_edge", _counts(vertex_edge[0], graph.num_vertices))

        # The same neighbour means as dense row-normalised adjacency matmuls,
        # for the fused forward below: `A_vh @ h` is `index_select` +
        # `index_add_` + divide collapsed into one GEMM. On a 19/54/72-node
        # board the matrices are tiny and the win is kernel count, not FLOPs.
        # `persistent=False` because every checkpoint on record predates them —
        # they are derived from the graph, not learned, and must not become
        # state_dict keys old checkpoints fail strict loading over.
        # Same numbers up to float reassociation; the equivalence test
        # compares the two paths. Default OFF: on CPU eager the dense GEMMs do
        # ~6x the gather path's arithmetic and measured 12-17% slower (43->50
        # ms/forward at batch 128, 1 thread), so the fused path is an opt-in
        # for the dispatch-bound GPU update, where fewer kernels is the bet —
        # `--fused` on `hexset.train`, decided by a measured A/B there.
        self.register_buffer(
            "A_vh", _mean_adjacency(hex_vertex[1], hex_vertex[0], graph.num_vertices, graph.num_hexes), persistent=False
        )
        self.register_buffer(
            "A_hv", _mean_adjacency(hex_vertex[0], hex_vertex[1], graph.num_hexes, graph.num_vertices), persistent=False
        )
        self.register_buffer(
            "A_ev", _mean_adjacency(vertex_edge[1], vertex_edge[0], graph.num_edges, graph.num_vertices), persistent=False
        )
        self.register_buffer(
            "A_ve", _mean_adjacency(vertex_edge[0], vertex_edge[1], graph.num_vertices, graph.num_edges), persistent=False
        )
        # Plain attribute like `detach_value`: wiring, not architecture.
        self.fused = False

        # The heads are concatenated in `readout.heads` order and permuted into
        # the flat space in one gather. The scatter is a bijection, so its
        # argsort is exactly that permutation.
        destinations = np.concatenate(
            [head.scatter.ravel() for head in self.readout.heads]
        )
        self.register_buffer("flat_gather", torch.from_numpy(np.argsort(destinations)))

    def _initialise(self) -> None:
        """Orthogonal weights: gain √2 through the trunk, 0.01 on every head.

        Nothing in this package initialised anything before this: all 27
        `nn.Linear` layers ran on PyTorch's default `kaiming_uniform_(a=√5)`,
        which is the most-cited missing item on the standard PPO
        implementation-details list. Two distinct consequences were measured on
        this architecture at width 64, and **they pull in opposite directions,
        which is why this is one method and not two changes.**

        *The trunk was suppressed.* A default-initialised residual branch here
        carries ~0.265 of its skip connection's magnitude, so each `Round`
        contributed about a quarter of what it should and the trunk RMS barely
        moved across the two rounds (`h` 0.3195 → 0.3329 → 0.3422). At
        initialisation the network was very nearly a linear readout of the raw
        embeddings — the message passing, which is the entire point of the
        architecture, started at quarter strength and had to be learned up.
        Orthogonal √2 puts the branch at ~1.025 of the skip, which is the point
        of the gain.

        *The heads have to go the other way.* The default heads happened to
        produce a near-uniform policy (99.98% of `log 54` on a real legal mask)
        **only because they sat on those cold activations**. Raising the trunk
        ~4× while leaving the heads hot would manufacture a peaked initial
        policy — the exact defect that was, by luck, absent. So every
        logit-producing head gets gain 0.01, the standard PPO value, and the
        value head gets 1.0 because it predicts at its target's scale. Biases
        are zeroed everywhere; the default gave them `U(±1/√fan_in)` too.

        A multi-layer head splits that argument rather than escaping it. Only
        the layer that emits the number is at the head's scale; a hidden layer
        inside a head is trunk by every property this method cares about — it
        feeds a SiLU, its output is not a logit and not a value — so it keeps
        √2 from the loop and `_output` picks out the one layer that does not.
        Giving a whole `Sequential` gain 0.01 would be the other reading and it
        is the wrong one: it would suppress the hidden layer for a reason that
        only applies to logits, and a two-layer head would start weaker than
        the one-layer head it is meant to be tested against.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        # Re-init the outputs only; the loop above already zeroed their biases.
        for head in self.heads.values():
            nn.init.orthogonal_(_output(head).weight, gain=0.01)
        nn.init.orthogonal_(self.trade_give.weight, gain=0.01)
        nn.init.orthogonal_(self.trade_want.weight, gain=0.01)
        nn.init.orthogonal_(_output(self.value).weight, gain=1.0)

    def _pass(
        self,
        src: Tensor,
        gather: Tensor,
        scatter: Tensor,
        size: int,
        counts: Tensor,
    ) -> Tensor:
        """Mean of `src` over each destination node."""
        message = src.index_select(1, gather)
        out = src.new_zeros(src.shape[0], size, src.shape[-1])
        out.index_add_(1, scatter, message)
        return out / counts

    def _value_features(self, g: Tensor, h: Tensor, v: Tensor, e: Tensor) -> Tensor:
        """Assemble whatever `config.value_head` reads.

        Dispatched on a string per forward rather than resolved once into a
        submodule, because which of the four tensors a head reads differs per
        shape, and hiding that behind a uniform call would mean wrapping the
        default head in a module and renaming its parameters. A Python branch on
        an interned string is nothing against a forward that message-passes over
        54 vertices.

        Split out from the read itself so the quantile head can be asked for its
        spread and its mean off one assembly rather than two — the same tensors,
        the same order, so every other shape's arithmetic is untouched.
        """
        shape = self.config.value_head
        if shape in _POOLED:
            return torch.cat(
                [g, h.max(1).values, v.max(1).values, e.max(1).values], -1
            )
        if shape == "attn":
            # One query, so one softmax over vertices — this is a pooling
            # operation, not a layer of self-attention, and it costs a single
            # `(B, 54)` matmul. Scaled by √width for the usual reason: the dot
            # product of two width-64 vectors grows with the width, and an
            # unscaled softmax over it starts near one-hot.
            query = self.value_query(g).unsqueeze(1)
            scores = (v * query).sum(-1) / v.shape[-1] ** 0.5
            pooled = (torch.softmax(scores, -1).unsqueeze(-1) * v).sum(1)
            return torch.cat([g, pooled], -1)
        return g

    def _read_value(
        self, g: Tensor, h: Tensor, v: Tensor, e: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """`(V, spread)`. `spread` is `None` for every shape but `"quantile"`."""
        features = self._value_features(g, h, v, e)
        if isinstance(self.value, QuantileValueHead):
            spread = self.value.spread(features)
            # The mean the rest of the system reads as `V`, taken here rather
            # than through `forward` so the spread is not recomputed.
            return spread.mean(-1), spread
        return self.value(features), None

    def forward(
        self, hexes: Tensor, vertices: Tensor, edges: Tensor, globals_: Tensor
    ) -> Prediction:
        h = self.embed_hex(hexes)
        v = self.embed_vertex(vertices)
        e = self.embed_edge(edges)
        g = self.embed_global(globals_)

        if self.fused:
            for round_ in self.rounds:
                h, v, e, g = self._fused_round(round_, h, v, e, g)
            return self._emit(h, v, e, g, globals_)

        for round_ in self.rounds:
            v_from_h = self._pass(
                h, self.hv_hex, self.hv_vertex, self.num_vertices, self.n_vertex_from_hex
            )
            h_from_v = self._pass(
                v, self.hv_vertex, self.hv_hex, self.num_hexes, self.n_hex_from_vertex
            )
            e_from_v = self._pass(
                v, self.ve_vertex, self.ve_edge, self.num_edges, self.n_edge_from_vertex
            )
            v_from_e = self._pass(
                e, self.ve_edge, self.ve_vertex, self.num_vertices, self.n_vertex_from_edge
            )
            broadcast = g.unsqueeze(1)

            h = h + round_.hex(torch.cat([h, h_from_v, broadcast.expand_as(h)], -1))
            v = v + round_.vertex(
                torch.cat([v, v_from_h, v_from_e, broadcast.expand_as(v)], -1)
            )
            e = e + round_.edge(torch.cat([e, e_from_v, broadcast.expand_as(e)], -1))
            g = g + round_.globe(torch.cat([g, h.mean(1), v.mean(1), e.mean(1)], -1))

        return self._emit(h, v, e, g, globals_)

    def _fused_round(
        self, round_: Round, h: Tensor, v: Tensor, e: Tensor, g: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One round as GEMMs — the same numbers as the `_pass` path up to
        float reassociation, with no gathers, no materialised concats, and
        the global block's linear applied once per batch rather than once per
        node. All four messages read the pre-round features and `g` reads the
        post-update means, exactly as the reference loop orders it."""
        v_from_h = torch.matmul(self.A_vh, h)
        h_from_v = torch.matmul(self.A_hv, v)
        e_from_v = torch.matmul(self.A_ev, v)
        v_from_e = torch.matmul(self.A_ve, e)

        width = self.config.width
        hex_first, vertex_first, edge_first = (
            round_.hex[0],
            round_.vertex[0],
            round_.edge[0],
        )
        hw = _split(hex_first, width)
        vw = _split(vertex_first, width)
        ew = _split(edge_first, width)

        pre_h = (
            h @ hw[0].T
            + h_from_v @ hw[1].T
            + (g @ hw[2].T + hex_first.bias).unsqueeze(1)
        )
        pre_v = (
            v @ vw[0].T
            + v_from_h @ vw[1].T
            + v_from_e @ vw[2].T
            + (g @ vw[3].T + vertex_first.bias).unsqueeze(1)
        )
        pre_e = (
            e @ ew[0].T
            + e_from_v @ ew[1].T
            + (g @ ew[2].T + edge_first.bias).unsqueeze(1)
        )
        # [1] is each MLP's own SiLU, [2] its second linear: the modules the
        # reference path runs, minus the first linear this method just applied
        # blockwise.
        h = h + round_.hex[2](round_.hex[1](pre_h))
        v = v + round_.vertex[2](round_.vertex[1](pre_v))
        e = e + round_.edge[2](round_.edge[1](pre_e))
        g = g + round_.globe(torch.cat([g, h.mean(1), v.mean(1), e.mean(1)], -1))
        return h, v, e, g

    def _emit(
        self, h: Tensor, v: Tensor, e: Tensor, g: Tensor, globals_: Tensor
    ) -> Prediction:
        emitted = {
            HEXES: self.heads[HEXES](h),
            VERTICES: self.heads[VERTICES](v),
            EDGES: self.heads[EDGES](e),
            GLOBALS: self.heads[GLOBALS](g),
        }
        batch = globals_.shape[0]
        stacked = torch.cat(
            [emitted[head.source].reshape(batch, -1) for head in self.readout.heads],
            dim=1,
        )

        # `detach_value` cuts the value head off the trunk without cutting it
        # off the optimiser: the head keeps fitting the moving features, but
        # stops voting on where they move. Distillation wants that because the
        # value term is an unweighted mean over every row while the policy term
        # sees only the contested ones, so the trunk's largest gradient comes
        # from a head the duel never reads. Every tensor the head reads is cut,
        # not just `g`, or a pooled head would keep the path the flag exists to
        # remove.
        cut = self.detach_value
        # Built in this order, and it is load-bearing. `g` feeds four heads, so
        # its gradient is a sum of four terms and autograd accumulates them in
        # the order the forward created the nodes. Hoisting the value read
        # above `trade_give`/`trade_want` reassociates that sum: the loss stays
        # bit-identical and the *gradient* moves in the last couple of bits,
        # which was enough to make a default-config update diverge from the
        # pre-quantile build after a single step. Measured, not theorised.
        logits = stacked.index_select(1, self.flat_gather)
        give = self.trade_give(g)
        want = self.trade_want(g)
        value, quantiles = self._read_value(
            g.detach() if cut else g,
            h.detach() if cut else h,
            v.detach() if cut else v,
            e.detach() if cut else e,
        )
        return Prediction(
            logits=logits, give=give, want=want, value=value, quantiles=quantiles
        )


def collate(observations: list[Observation]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Stack observations sharing one topology into batched inputs."""
    return tuple(
        torch.from_numpy(np.stack([getattr(obs, name) for obs in observations]))
        for name in ("hexes", "vertices", "edges", "globals")
    )


@dataclass(frozen=True)
class Packing:
    """Where each input block sits inside one flat per-position row.

    A copy to this device costs a fixed ~250 µs per tensor whatever its size,
    so moving four blocks separately costs three of those for nothing. Packing
    also leaves the search one buffer to reuse rather than four, which is what
    CUDA graphs want.
    """

    width: int
    blocks: tuple[tuple[str, int, int, tuple[int, ...]], ...]


def packing(graph: StaticGraph, players: int) -> Packing:
    shapes = (
        ("hexes", (graph.num_hexes, HEX_FEATURES)),
        ("vertices", (graph.num_vertices, vertex_features(players))),
        ("edges", (graph.num_edges, edge_features(players))),
        ("globals", (global_features(players),)),
    )
    blocks = []
    start = 0
    for name, shape in shapes:
        size = int(np.prod(shape))
        blocks.append((name, start, start + size, shape))
        start += size
    return Packing(width=start, blocks=tuple(blocks))


def pack(layout: Packing, observations: list[Observation]) -> Tensor:
    """Stack observations, reusing a batch encoder buffer when available."""
    batch = len(observations)
    if observations:
        source = observations[0]._packed
        if (
            source is not None
            and source.shape[1] == layout.width
            and all(observation._packed is source for observation in observations)
        ):
            rows = np.asarray(
                [observation._row for observation in observations], dtype=np.intp
            )
            if batch == source.shape[0] and np.array_equal(rows, np.arange(batch)):
                return torch.from_numpy(source)
            # Opponent mixing can hand one policy a subset of a tick. One
            # contiguous gather is still cheaper than stacking four blocks.
            return torch.from_numpy(source[rows])

    out = np.empty((batch, layout.width), dtype=np.float32)
    for name, start, stop, shape in layout.blocks:
        destination = out[:, start:stop].reshape(batch, *shape)
        if destination.base is None:
            raise AssertionError(f"{name} slice did not reshape to a view")
        np.stack([getattr(obs, name) for obs in observations], out=destination)
    return torch.from_numpy(out)


def unpack(layout: Packing, packed: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Views back onto the four blocks, so nothing is copied on the device."""
    batch = packed.shape[0]
    return tuple(
        packed[:, start:stop].view(batch, *shape)
        for _, start, stop, shape in layout.blocks
    )
