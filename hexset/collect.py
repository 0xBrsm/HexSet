# SPDX-License-Identifier: GPL-3.0-only
"""Parallel self-play collection: the torch-free Collector, sharded.

One collector process measures ~166 ms a tick at 512 lanes and leaves the
box's other cores idle, because everything a tick does — `legal_actions`,
`encode`, the engine step, a scripted opponent's one-ply search — is Python
compute holding the GIL. That rules out the usual vectorised-env fan-out, a
thread pool over native engine subprocesses whose `step()` blocks on pipe I/O
and releases the GIL; ours never releases it, so the shard has to be a
process.

Each worker owns a slice of the lanes and a CPU copy of the network, and
plays its games end to end: worker `w` of `K` deals game indices
`base + w, base + w + K, ...` (the Collector's `stride`), so the workers'
index sets are disjoint by construction, every game stays the same pure
function of `(seed, index)` it always was, and the caster — also pure in the
index — casts identically at any worker count. Inference stays per-worker on
CPU because the crossover is measured: below batch ~64 the CPU beats the GPU
outright, and a worker's slice is a couple dozen lanes. The GPU stays free
for the update, and the learner re-syncs weights once per iteration — 159k
parameters, under a millisecond a worker.

What crosses processes is one weights dict per iteration going out and
finished episodes coming back — never per-tick observations. The other shape
considered, SampleFactory-style central inference, was rejected for this
model: it reintroduces a global tick barrier plus megabytes a tick over the
pipes, to buy GPU batching that only pays above the batch sizes a shard sees.

On a resume, `games_started()` reports the *max* over the workers' counters.
Indices between the slowest and fastest worker's next deal are skipped rather
than replayed — unused seeds, not lost games — and the same rule makes a
resume safe across a change in worker count.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

import numpy as np

from .actions import build_space
from .board.board import Board, random_base_board
from .bots import greedy
from .encoding import Observation, static_graph
from .evaluate import Evaluator
from .model import HexNet, ModelConfig, Packing, packing
from .policy import NetworkPolicy
from .selfplay import BatchPolicy, BotPolicy, Collector, Episode, Transition


def frozen(path: str, device: str, board: Board, players: int) -> NetworkPolicy:
    """A checkpoint as a fixed greedy policy — a ladder rung or a lane opponent.

    Greedy argmax rather than sampling, so these numbers stay comparable with
    the arena's `network:` entrants, and `.eval()` with no optimiser anywhere
    means nothing here can train. Width, rounds and the head shapes come from
    the checkpoint's own recorded args, not the run's, so a differently-sized or
    differently-shaped parent still loads.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    stored = state.get("args", {})
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, players
    )
    graph = static_graph(topology)
    net = HexNet(
        space,
        graph,
        players,
        ModelConfig(
            width=int(stored.get("width", 64)),
            rounds=int(stored.get("rounds", 2)),
            value_head=str(stored.get("value_head", "linear")),
            policy_head=str(stored.get("policy_head", "linear")),
            quantiles=int(stored.get("quantiles", 32)),
        ),
    ).to(device)
    net.load_state_dict(state["net"])
    net.eval()
    return NetworkPolicy(
        net, space, packing(graph, players), device=device, greedy=True
    )


def greedy_opponent(seed: int, max_offers: int | None, lanes: int) -> BotPolicy:
    """`greedy-offers3` as a lane opponent: fitted weights, relative stance.

    One bot per board because the evaluator caches per-vertex pips; capacity
    covers double the lane count so a live board is never thrashed out.
    """
    return BotPolicy(
        lambda board: greedy(
            Evaluator(board), random.Random(seed), max_offers=max_offers
        ),
        capacity=max(256, 2 * lanes),
    )


def named_opponent(name: str, seed: int, lanes: int) -> BotPolicy:
    """Any arena entrant by name as a lane or ladder opponent.

    Routed through `hexset.arena.spawn` rather than constructing the bot here, so
    a rung is *literally* the entrant the arena scores. That is the whole point:
    `search2-offers3` on the ladder is the same bot whose 400-game results are on
    record, which is what makes a ladder reading comparable to them at all.

    `search2-offers3` is the interesting one. It is measured at parity with
    catanatron's `AB:2` — the chain the campaign's external numbers rest on — it
    lives in this repo, and unlike catanatron **it plays the trading game**, which
    catanatron's ruleset does not model at all. So it can see the dimension the
    external benchmark is structurally blind to, at no protocol cost.

    The import is function-scoped because `hexset.arena` reaches for torch on some
    entrant kinds and `hexset.selfplay` imports this module; the collector is
    tested and timed where torch cannot be installed.
    """
    from .arena import entrant_from_name, spawn

    try:
        entrant = entrant_from_name(name)
    except KeyError:
        # Matching `--mix`'s style: a mistyped rung should fail with a sentence
        # before the run starts, not a bare KeyError traceback out of a lambda.
        raise SystemExit(f"unknown entrant: {name}") from None
    return BotPolicy(
        lambda board: spawn(entrant, board, random.Random(seed)),
        capacity=max(256, 2 * lanes),
    )


# `--mix` names that are not arena entrants, and must never become them.
#
# Both predate `named_opponent`, 34 recorded runs collected against them, and a
# lane opponent whose strength moved silently would invalidate every one of
# those numbers without changing a line of their configuration. So they are
# resolved here, by name, exactly as they were before an entrant spec was
# accepted at all.
#
# `greedy` is the trap. It is **not** the arena's `greedy` preset: it takes the
# run's own `--max-offers`, which defaults to 3, so what 34 runs actually played
# is `greedy-offers3`. The arena preset means `max_offers=None` — the engine's
# whole eight-offer budget — and greedy saturates that cap, so the two are
# different bots at different strengths. Routing the name through
# `named_opponent` for tidiness would re-point every recorded mix run's opponent.
RESERVED_MIX = ("greedy", "parent")


def mix_opponents(
    mix: Sequence[tuple[str, float]],
    *,
    seed: int,
    max_offers: int | None,
    lanes: int,
    parent: Callable[[], BatchPolicy] | None = None,
) -> list[BatchPolicy]:
    """The `--mix` names as lane opponents, in caster id order.

    Id `k + 1` is `mix[k]` — the contract `mixed_caster` casts against and
    `Collector._answers` dispatches on — so this list's order is load-bearing.

    Reserved names resolve as they always have (see `RESERVED_MIX`). Every other
    name is an arena entrant spec resolved by `named_opponent`, which is what
    makes `search2-offers3` or `mcts:<ckpt>@64` a training opponent without a
    branch here per bot, and makes the opponent a run trained against *literally*
    the entrant the arena scores.

    `parent` arrives as a thunk rather than a policy because the trainer has
    already loaded that checkpoint for its ladder rung and would hand the same
    object back, while a worker has to build its own on its own board — and a
    `torch.load` of a checkpoint nothing casts is not free.
    """
    out: list[BatchPolicy] = []
    for k, name in enumerate(mix_names(mix)):
        if name == "greedy":
            out.append(greedy_opponent(seed + 77, max_offers, lanes))
        elif name == "parent":
            if parent is None:
                raise ValueError("the 'parent' mix opponent needs a parent checkpoint")
            out.append(parent())
        else:
            # A stream per entry, so two entrants in one mix do not break their
            # tie-breaks in lockstep. 77 stays where it was.
            out.append(named_opponent(name, seed + 700 + k, lanes))
    return out


def check_mix(mix: Sequence[tuple[str, float]], *, have_parent: bool) -> None:
    """Refuse an unusable `--mix` before the run starts.

    Every failure here would otherwise surface as a traceback out of a collector
    subprocess, after the manifest was frozen and the box was committed. The
    checkpoint existence check earns its place: `mcts:<path>@64` resolves to an
    entrant whatever `<path>` says, and the `torch.load` that finds out is inside
    a worker.

    Function-scoped import for the same reason `named_opponent` has one:
    `hexset.arena` reaches for torch on some entrant kinds, and this module is
    tested and timed where torch cannot be installed.
    """
    from .arena import entrant_from_name

    for name in mix_names(mix):
        if name == "parent":
            if not have_parent:
                raise SystemExit("--mix parent needs --parent <checkpoint>")
            continue
        if name == "greedy":
            continue
        try:
            entrant = entrant_from_name(name)
        except (KeyError, ValueError):
            raise SystemExit(f"unknown mix opponent: {name}") from None
        if isinstance(entrant.weights, str) and not Path(entrant.weights).exists():
            raise SystemExit(
                f"--mix {name} names a checkpoint that is not there: {entrant.weights}"
            )


def alternating(players: int, flip: bool = False):
    """The duel caster: the reference on every other seat, pairs swapping by
    game parity.

    The old docstring claimed this made "seat effects cancel across a cohort of
    even size". That is true of the *mean* seat effect and false of the
    parity-correlated part, which is what made a swapped duel fail to negate:
    the cast is a function of the game index, and the board is a function of
    `(seed, index)` too, so exchanging the two entrants moves each of them onto
    the complementary seat-pair of every board rather than relabelling one
    experiment. Measured on this repo: a checkpoint duelled against *itself*
    read -0.77 VP over 48 games and -0.08 over 800, and 55% of a single-order
    duel's variance was that residual, reported as if it were ordinary noise.

    `flip` names the complementary assignment, which is what `train.versus`
    needs to play one board both ways and difference the seat term out. See
    `versus(..., antithetic=True)`.
    """
    offset = 1 if flip else 0

    def caster(index: int) -> tuple[int, ...]:
        cast = [0] * players
        for seat in range(1 - (index + offset) % 2, players, 2):
            cast[seat] = 1
        return tuple(cast)

    return caster


def league_caster(learners: int, players: int, order: Sequence[int] | None = None):
    """Rotate learner ids over the seats by game index, so no learner owns a
    seat and the shares balance over any index window. Pure in the index, the
    same law every caster obeys.

    `order` permutes the learner ids before the rotation, which changes *who
    sits next to whom* while leaving every learner's share of every board seat
    exactly as balanced. Rotation alone cannot do that: the cyclic order round
    the table is always `0,1,2,3`, so learner 0's turn-order successor is
    learner 1 in every game ever played. That fixed adjacency is the leading
    suspect for the two-tight-pairs structure both noise heats produced --
    `{0,1}` against `{2,3}` is exactly a perfect matching of the 4-cycle, one
    of the two disjoint-adjacent-pair splits, contrary to the note in 191087b
    that called it unnatural. `order=(0,2,1,3)` reseats the cycle as 0,2,1,3;
    if the pairs become `{0,2}`/`{1,3}` the structure is adjacency, and if they
    stay `{0,1}`/`{2,3}` it is keyed to learner index somewhere else.
    """
    seq = tuple(order) if order is not None else tuple(range(learners))
    if sorted(seq) != list(range(learners)):
        raise ValueError(f"learner order must be a permutation of 0..{learners-1}: {seq}")

    def caster(index: int) -> tuple[int, ...]:
        return tuple(seq[(seat + index) % learners] for seat in range(players))

    return caster


def paired_caster(caster):
    """The board pairing's casting half: games `2k` and `2k+1` get `caster(k)`,
    so within a pair the same policy holds the same seat on the same board.
    That identity is what makes the mate game's reward a baseline for a seat
    rather than a comparison of two different players — `hexset.ppo`'s
    `pair_baseline` conditions on (board, seat, policy), and this wrapper is
    the policy leg. Still a pure function of the index, the law every caster
    obeys.

    The balance consequence: `league_caster` gives every learner an equal
    share of every seat over any `learners`-game window; doubled up, the same
    exact balance holds over any `2*learners`-game window instead. Nothing is
    lost over a whole iteration — the games-per-iteration counts in use are
    multiples of both — the cadence is just twice as long.
    """
    return lambda index: caster(index // 2)


# A mix entry whose name reads `table(a|b|c)` casts a *table*: the learner in
# exactly one seat and the other seats each drawn independently from the pool
# `a`, `b`, `c` — with replacement, so three copies of one bot is a possible
# table, which is the catanatron bridge's own design (one network against three
# `AB:2`). This is the geometry the harness-path check named as the only
# seating-free strength referent, and until 2026-08-29 no run could collect in
# it: a plain entry gives its opponent 2 of 4 seats on alternating parity, so
# the learner always had a twin at the table and its measured gains turned out
# to be specific to that seating.
TABLE = "table("


def table_pool(name: str) -> list[str] | None:
    """The pool a `table(...)` entry draws from, or None for a plain entry."""
    if not name.startswith(TABLE):
        return None
    if not name.endswith(")"):
        raise ValueError(f"unclosed table entry: {name}")
    pool = [member.strip() for member in name[len(TABLE) : -1].split("|")]
    if not pool or any(not member for member in pool):
        raise ValueError(f"a table entry needs at least one opponent: {name}")
    return pool


def parse_mix(spec: str) -> list[tuple[str, float]]:
    """`"greedy=0.15,search2-offers3=0.1"` — the share of games each opponent plays.

    A name is a reserved one (`RESERVED_MIX`), any arena entrant spec, or a
    `table(a|b|c)` pool (see `TABLE`); only the shares and the table syntax are
    checked here, because resolving a name needs `hexset.arena` and this parser
    runs where torch may not be installed. `check_mix` is the resolution check,
    and the trainers call it.

    Table entries live in the spec string rather than behind a flag on purpose:
    `run.manifest.load` refuses a config whose keys are not exactly the mode's
    parameter set, so one new argparse dest would make every frozen config on
    disk unloadable. Everything new is inside `--mix`'s value.
    """
    if not spec:
        return []
    out: list[tuple[str, float]] = []
    for part in spec.split(","):
        name, _, value = part.rpartition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"a mix entry needs a name: {part!r}")
        table_pool(name)
        out.append((name, float(value)))
    if any(f <= 0 for _, f in out) or sum(f for _, f in out) > 1.0 + 1e-9:
        raise ValueError(f"mix fractions must be positive and sum to at most 1: {spec}")
    return out


def mix_names(mix: Sequence[tuple[str, float]]) -> list[str]:
    """The distinct opponents a mix seats, in caster id order: id `k + 1` is
    `mix_names(mix)[k]`.

    Plain entries contribute their own name and a table entry contributes its
    pool, each name once at its first appearance — so a mix with no table
    entries and no repeated names lists exactly `[name for name, _ in mix]`,
    which is the id law every recorded run was cast under. `mix_opponents`
    builds this list and `mix_caster` casts against it; keeping them on one
    function is what stops the two from drifting.
    """
    out: list[str] = []
    for name, _ in mix:
        for member in table_pool(name) or [name]:
            if member not in out:
                out.append(member)
    return out


def mix_caster(mix: Sequence[tuple[str, float]], players: int, seed: int):
    """Game index -> per-seat policy ids for any `--mix`, pure in the index.

    Without a table entry this *is* `mixed_caster` — same rng key, same
    cumulative walk over the shares, same alternating seat pairs — and the
    identity is pinned by test, because 34 recorded runs were cast by that
    function and a resumed one must still deal the games it would have dealt.

    A table entry, when the walk lands on it, seats the learner once and fills
    every other seat with an independent draw from its pool. The learner's seat
    and the draws come off the same per-index stream as the entry draw, so a
    cast is still a function of `(seed, index)` alone. The seat is drawn rather
    than taken as `index % players`: the board is keyed to the index too, and
    `alternating`'s history is what a parity-correlated cast costs.
    """
    names = mix_names(mix)
    plan: list[tuple[float, int | tuple[int, ...]]] = []
    for name, fraction in mix:
        pool = table_pool(name)
        if pool is None:
            plan.append((fraction, names.index(name) + 1))
        else:
            plan.append((fraction, tuple(names.index(member) + 1 for member in pool)))

    def caster(index: int) -> tuple[int, ...]:
        rng = random.Random(f"{seed}:{index}:cast")
        draw = rng.random()
        cumulative = 0.0
        for fraction, who in plan:
            cumulative += fraction
            if draw < cumulative:
                if isinstance(who, int):
                    cast = [0] * players
                    for seat in range(1 - index % 2, players, 2):
                        cast[seat] = who
                    return tuple(cast)
                learner = rng.randrange(players)
                return tuple(
                    0 if seat == learner else rng.choice(who) for seat in range(players)
                )
        return (0,) * players

    return caster


def mixed_caster(fractions: Sequence[float], players: int, seed: int):
    """Game index -> per-seat policy ids, a pure function of the index.

    Opponent `k + 1` takes every other seat — pairs alternating by parity, so
    neither side owns a seat — for a `fractions[k]` share of games; the rest
    stay pure self-play. Pure in the index so a resumed run casts the games it
    would have cast, and the cast of a logged episode is recomputable.

    **One draw a game, so one opponent a game.** Three opponents at a tenth each
    is three *kinds of game*, not a table holding three kinds of opponent: 30% of
    games are cast, and each cast game seats a single id on 2 of the 4 seats. Two
    consequences, and neither is a defect for what `--mix` is for.

    The marginal distribution of opponents the learner faces is fully
    expressible, and that is what an opponent-distribution hypothesis is about.
    What is not expressible is a *heterogeneous table*, and by this repo's own
    evidence that is a separate and large effect — the noise decomposition put
    92% of cross-heat variance on table-mates rather than seeds. So it is a
    second experiment, not a missing feature of this one, and folding both into
    one change would make either result unreadable.

    A fraction also buys less exposure than it reads like. In a cast game the
    learner holds 2 seats, so 2 of a learner seat's 3 opponents are the cast bot
    and the third is itself: exposure is `fraction * 2/3`, and the canonical
    `greedy=0.15` is 10% of the learner's opponent-facing interactions.

    Adding heterogeneous tables needs no new flag and should not get one — a new
    argparse dest changes `run.manifest.parameters` and every frozen config then
    fails `load`. It belongs in the spec string, as a `+` between names sharing
    one share: `"greedy+search2-offers3=0.1"` keeps every existing spec
    byte-identical, keeps the shares exact because there is still one draw a
    game, and turns each entry's cast from an id into a pair of ids. The seat
    assignment within the pair is the part that needs care rather than taste —
    see `alternating`, where a parity-correlated cast read as -0.77 VP of
    nothing.
    """

    def caster(index: int) -> tuple[int, ...]:
        draw = random.Random(f"{seed}:{index}:cast").random()
        cumulative = 0.0
        for k, fraction in enumerate(fractions):
            cumulative += fraction
            if draw < cumulative:
                cast = [0] * players
                for seat in range(1 - index % 2, players, 2):
                    cast[seat] = k + 1
                return tuple(cast)
        return (0,) * players

    return caster


@dataclass(frozen=True)
class WorkerSpec:
    """Everything a worker needs to rebuild its shard. Picklable by design,
    the same rule the arena's entrants follow: descriptions cross processes,
    never built objects."""

    seed: int
    players: int
    lanes: int
    action_cap: int
    max_offers: int | None
    first_game: int
    stride: int
    width: int
    rounds: int
    torch_seed: int
    # Shape as well as size: a worker builds its own net and then has the
    # learner's parameters pushed into it, so a shape mismatch here would fail
    # at the first sync rather than at construction.
    value_head: str = "linear"
    policy_head: str = "linear"
    # Inert unless `value_head == "quantile"`; see `ModelConfig.quantiles`.
    quantiles: int = 32
    mix: tuple[tuple[str, float], ...] = ()
    parent: str = ""
    cohort: bool = True
    # The table league: ids 0..learners-1 are learner nets sharing every game,
    # each recording its own seats. 1 is every run before gen3. Mutually
    # exclusive with `mix`: both allocate the caster's id space, and a
    # per-learner mix share is incoherent when learners share a table.
    learners: int = 1
    # A permutation of 0..learners-1 applied by `league_caster` before its
    # rotation, so table adjacency can be varied independently of seat share.
    learner_order: tuple[int, ...] | None = None
    # Board-paired collection (variance screen, candidate 1): games 2k and
    # 2k+1 share 2k's board and — via `paired_caster` — its cast, so each
    # game's mate differs in dice and play only. One flag drives both wires
    # because either alone breaks what `ppo.pair_baseline` conditions on.
    pair_boards: bool = False
    # Searched collection. `simulations` of 0 keeps the plain policy, which is
    # every PPO run on record. Above 0 the worker wraps its policy in a
    # `SearchPolicy`, so each transition carries the search's `Target` and the
    # corpus is distillable.
    #
    # The searched path is the one that needed this: a single process spends
    # ~64 games an hour at 256 simulations on 3 of 32 cores, because batching
    # amortises the network call and nothing else -- the engine, `encode` over
    # ~151 leaves a decision, and the tree descent are all single-threaded
    # Python. 30 processes were measured at ~689 games an hour.
    simulations: int = 0
    wave: int = 16
    exploration: float = 1.25
    stance: str = "relative"
    root_noise: float = 0.0
    noise_fraction: float = 0.25
    play_temperature: float = 1.0


def _build(spec: WorkerSpec) -> tuple[list[NetworkPolicy], Collector]:
    # One thread per worker: the whole point is many workers, and torch's
    # default of one thread per core would have them fighting for the box.
    torch.set_num_threads(1)
    torch.manual_seed(spec.torch_seed)

    board = random_base_board(random.Random(spec.seed))
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, spec.players
    )
    graph = static_graph(topology)
    net = HexNet(
        space,
        graph,
        spec.players,
        ModelConfig(
            width=spec.width,
            rounds=spec.rounds,
            value_head=spec.value_head,
            policy_head=spec.policy_head,
            quantiles=spec.quantiles,
        ),
    )
    policy = NetworkPolicy(
        net,
        space,
        packing(graph, spec.players),
        device="cpu",
        generator=torch.Generator().manual_seed(spec.torch_seed),
    )

    if spec.learners > 1 and spec.mix:
        # Two separate blockers, and the id space is only the first.
        #
        # `opponents` is one flat list indexed by cast id, learners first, so a
        # mix's ids would start at `learners` while `mixed_caster` emits `k + 1`
        # — a mix meaning `greedy` would silently seat *learner 1*. That much is
        # an offset away from being fixed.
        #
        # The second is not. `league_caster` fills **every** seat with a learner
        # at any learner count, so a mixed game has to displace one, and the
        # league's premise is that every learner is seated in every game: that
        # is what makes the arms paired, what licenses the seat split under the
        # noise scale, and what lets `standings` score every learner off every
        # game. A combined caster would have to rotate *which* learner is
        # displaced along with the seats, so that every learner's seat share and
        # every learner's exposure to each mix opponent both balance over an
        # index window — and `standings` and `owned` would have to stop assuming
        # a learner appears in every episode.
        raise ValueError("league workers and mix opponents share the caster's "
                         "id space; run one or the other")
    fellow_learners: list[NetworkPolicy] = []
    for k in range(1, spec.learners):
        # Same architecture, its own sampling stream: a learner explores with
        # its own dice, or two identical configs would play identical games.
        fellow = HexNet(
            space,
            graph,
            spec.players,
            ModelConfig(
                width=spec.width,
                rounds=spec.rounds,
                value_head=spec.value_head,
                policy_head=spec.policy_head,
                quantiles=spec.quantiles,
            ),
        )
        fellow_learners.append(
            NetworkPolicy(
                fellow,
                space,
                packing(graph, spec.players),
                device="cpu",
                generator=torch.Generator().manual_seed(spec.torch_seed + 7000 + k),
            )
        )

    opponents: list[BatchPolicy] = list(fellow_learners)
    opponents.extend(
        mix_opponents(
            spec.mix,
            seed=spec.seed,
            max_offers=spec.max_offers,
            lanes=spec.lanes,
            parent=lambda: frozen(spec.parent, "cpu", board, spec.players),
        )
    )
    if spec.learners > 1:
        caster = league_caster(spec.learners, spec.players, spec.learner_order)
    elif spec.mix:
        caster = mix_caster(spec.mix, spec.players, spec.seed)
    else:
        caster = None
    if spec.pair_boards and caster is not None:
        caster = paired_caster(caster)

    acting: BatchPolicy = policy
    if spec.simulations > 0:
        # Imported here rather than at module scope: `hexset.expert` pulls in the
        # search, and a plain PPO worker has no use for it.
        from .expert import SearchPolicy
        from .mcts import Search
        from .netbot import LeafEvaluator

        acting = SearchPolicy(
            Search(
                LeafEvaluator(policy=policy, space=space),
                simulations=spec.simulations,
                wave=spec.wave,
                exploration=spec.exploration,
                stance=spec.stance,
                max_offers=spec.max_offers,
                root_noise=spec.root_noise,
                noise_fraction=spec.noise_fraction,
                rng=random.Random(spec.seed + 991),
            ),
            temperature=spec.play_temperature,
            rng=random.Random(spec.seed + 992),
        )

    collector = Collector(
        acting,
        lanes=spec.lanes,
        fill=not spec.cohort,
        players=spec.players,
        seed=spec.seed,
        action_cap=spec.action_cap,
        max_offers=spec.max_offers,
        first_game=spec.first_game,
        stride=spec.stride,
        opponents=opponents,
        caster=caster,
        learners=tuple(range(spec.learners)),
        pair_boards=spec.pair_boards,
    )
    return [policy, *fellow_learners], collector


class Flattened:
    """A cohort of episodes in wire form: a few large arrays instead of tens of
    thousands of pickled objects.

    A cohort crosses the worker pipes as ~0.9 GB of per-transition ndarrays —
    one observation row, one mask, several scalars each — and the parent
    deserialises all sixteen workers' worth serially, inside `collect_seconds`.
    `Observation.__reduce__` strips the shared tick buffer on purpose, so every
    row also arrives as its own allocation and `pack` loses its gather path.

    This container flattens each worker's trajectories into contiguous blocks
    where the time is parallel, and `episodes()` rebuilds the identical
    `Episode` objects on the far side. Observations come back as views into one
    shared `(positions, width)` buffer with `_packed`/`_row` set, so `pack`
    gathers the whole cohort in one strided copy instead of stacking 145k rows.
    Same numbers, different container: the rebuilt cohort assembles to a
    byte-identical `Batch`, and `test_collect` pins that.

    `action`, `aux` and `value` stay object lists: actions carry structured
    trade bundles, `aux` is whatever the policy attached, and a search policy
    may record `()` for a forced move's value — a ragged column cannot be an
    array without changing what it holds.
    """

    def __init__(self, episodes: Sequence[Episode], layout: Packing) -> None:
        self.layout = layout
        self.meta = [(e.index, e.seed, e.players, e.outcome, e.cast) for e in episodes]
        self.counts = [[len(seat) for seat in e.trajectories] for e in episodes]
        transitions = [t for e in episodes for seat in e.trajectories for t in seat]
        n = len(transitions)

        graphs: list = []
        graph_ids: dict[int, int] = {}
        graph_index = np.empty(n, dtype=np.int32)
        for i, t in enumerate(transitions):
            graph = t.observation.graph
            slot = graph_ids.get(id(graph))
            if slot is None:
                slot = len(graphs)
                graph_ids[id(graph)] = slot
                graphs.append(graph)
            graph_index[i] = slot
        self.graphs = graphs
        self.graph_index = graph_index

        self.buffer = np.empty((n, layout.width), dtype=np.float32)
        for name, start, stop, shape in layout.blocks:
            destination = self.buffer[:, start:stop].reshape(n, *shape)
            for i, t in enumerate(transitions):
                destination[i] = getattr(t.observation, name)
        self.mask = (
            np.stack([t.mask for t in transitions])
            if transitions
            else np.empty((0, 0), dtype=bool)
        )
        self.seat = np.array([t.seat for t in transitions], dtype=np.int64)
        self.step = np.array([t.step for t in transitions], dtype=np.int64)
        self.chosen = np.array([t.index for t in transitions], dtype=np.int64)
        self.log_prob = np.array([t.log_prob for t in transitions], dtype=np.float64)
        self.action = [t.action for t in transitions]
        self.value = [t.value for t in transitions]
        self.aux = [t.aux for t in transitions]

    def episodes(self) -> list[Episode]:
        """The identical episodes back, observations as views into one buffer."""
        out: list[Episode] = []
        cursor = 0
        for (index, seed, players, outcome, cast), counts in zip(
            self.meta, self.counts
        ):
            trajectories = []
            for count in counts:
                seat_transitions = []
                for k in range(cursor, cursor + count):
                    row = self.buffer[k]
                    components = {
                        name: row[start:stop].reshape(shape)
                        for name, start, stop, shape in self.layout.blocks
                    }
                    seat_transitions.append(
                        Transition(
                            seat=int(self.seat[k]),
                            step=int(self.step[k]),
                            observation=Observation(
                                graph=self.graphs[self.graph_index[k]],
                                _packed=self.buffer,
                                _row=k,
                                **components,
                            ),
                            mask=self.mask[k],
                            action=self.action[k],
                            index=int(self.chosen[k]),
                            log_prob=float(self.log_prob[k]),
                            value=self.value[k],
                            aux=self.aux[k],
                        )
                    )
                trajectories.append(tuple(seat_transitions))
                cursor += count
            out.append(
                Episode(
                    index=index,
                    seed=seed,
                    players=players,
                    trajectories=tuple(trajectories),
                    outcome=outcome,
                    cast=cast,
                )
            )
        return out


def _flat(episodes: list[Episode], policy) -> object:
    """Wrap for the pipe when the worker's policy carries a layout."""
    layout = getattr(policy, "layout", None)
    if layout is None:
        return episodes
    return Flattened(episodes, layout)


def _serve(spec: WorkerSpec, connection) -> None:
    """The worker loop: build once, then answer commands until told to stop."""
    try:
        policies, collector = _build(spec)
        policy = policies[0]
        while True:
            kind, payload = connection.recv()
            if kind == "weights":
                # `policy` is the `NetworkPolicy`, which is what a `SearchPolicy`
                # wraps and what its evaluator holds — so syncing here reaches
                # the searched path too, and `_build` returns it deliberately
                # rather than returning whatever is acting. A list is the
                # league: one dict per learner, in id order.
                states = payload if isinstance(payload, list) else [payload]
                if len(states) != len(policies):
                    raise ValueError(
                        f"{len(states)} weight dicts for {len(policies)} learners"
                    )
                for learner, state in zip(policies, states):
                    learner.net.load_state_dict(state)
                connection.send(("ok", None))
            elif kind == "collect":
                connection.send(("episodes", _flat(collector.collect(payload), policy)))
            elif kind == "cohort":
                connection.send(("episodes", _flat(collector.cohort(payload), policy)))
            elif kind == "counter":
                connection.send(("counter", collector.games_started()))
            elif kind == "stop":
                connection.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown command: {kind}")
    except EOFError:
        return
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except Exception:
            pass


class ParallelCollector:
    """`WorkerSpec`s made processes, behind the slice of `Collector` the
    trainer uses: `collect`, `games`, `games_started` — plus `sync`, which
    ships the learner's current weights out once per iteration.

    `collect` forwards to each worker's `Collector.cohort` unless the specs ask
    for streaming, so the on-policy guarantee holds per worker: a worker deals
    its own quota, plays all of it out, and ends with empty lanes. The cost is
    that the iteration waits on the slowest game of the slowest worker."""

    def __init__(self, specs: Sequence[WorkerSpec]) -> None:
        if not specs:
            raise ValueError("a parallel collector needs at least one worker")
        context = mp.get_context("spawn")
        self._command = "cohort" if all(spec.cohort for spec in specs) else "collect"
        self.games = 0
        self.last_collect_seconds = 0.0
        self._pending_quotas: list[int] | None = None
        self._collect_started = 0.0
        self._connections = []
        self._processes = []
        for spec in specs:
            ours, theirs = context.Pipe()
            process = context.Process(target=_serve, args=(spec, theirs), daemon=True)
            process.start()
            theirs.close()
            self._connections.append(ours)
            self._processes.append(process)

    def _hear(self, connection, wanted: str):
        kind, payload = connection.recv()
        if kind == "error":
            raise RuntimeError(f"a collector worker failed:\n{payload}")
        if kind != wanted:
            raise RuntimeError(f"expected {wanted}, a worker sent {kind}")
        return payload

    def sync(self, net: torch.nn.Module) -> None:
        self.sync_many([net])

    def sync_many(self, nets: Sequence[torch.nn.Module]) -> None:
        """Ship every learner's weights, in id order — the league's sync."""
        if self._pending_quotas is not None:
            raise RuntimeError("cannot sync while collection is in flight")
        states = [
            {k: v.detach().cpu() for k, v in net.state_dict().items()} for net in nets
        ]
        payload = states if len(states) > 1 else states[0]
        for connection in self._connections:
            connection.send(("weights", payload))
        for connection in self._connections:
            self._hear(connection, "ok")

    def _quotas(self, episodes: int) -> list[int]:
        share, extra = divmod(episodes, len(self._connections))
        return [
            share + (1 if worker < extra else 0)
            for worker in range(len(self._connections))
        ]

    def start_collect(self, episodes: int) -> None:
        """Dispatch a fixed cohort without waiting for workers to finish it."""
        if self._pending_quotas is not None:
            raise RuntimeError("collection is already in flight")
        quotas = self._quotas(episodes)
        for connection, quota in zip(self._connections, quotas):
            if quota:
                connection.send((self._command, quota))
        self._pending_quotas = quotas
        self._collect_started = time.perf_counter()

    def finish_collect(self) -> list[Episode]:
        """Wait for the cohort dispatched by `start_collect`."""
        if self._pending_quotas is None:
            raise RuntimeError("there is nothing in flight to finish")
        quotas = self._pending_quotas
        self._pending_quotas = None
        out: list[Episode] = []
        for connection, quota in zip(self._connections, quotas):
            if quota:
                payload = self._hear(connection, "episodes")
                out.extend(
                    payload.episodes() if isinstance(payload, Flattened) else payload
                )
        self.last_collect_seconds = time.perf_counter() - self._collect_started
        self.games += len(out)
        return out

    def collect(self, episodes: int) -> list[Episode]:
        """Each worker plays out an equal share of the quota.

        Fixed shares rather than first-`n`-across-workers, so the cohort is
        decided before anyone plays — taking whichever games finish first
        selects for short games, the same bias the in-process collector's
        `deal` bound exists to prevent.
        """
        self.start_collect(episodes)
        return self.finish_collect()

    def games_started(self) -> int:
        """The next safe base index: above it, no worker has dealt anything."""
        if self._pending_quotas is not None:
            raise RuntimeError("cannot read counters while collection is in flight")
        for connection in self._connections:
            connection.send(("counter", None))
        return max(
            self._hear(connection, "counter") for connection in self._connections
        )

    def close(self) -> None:
        if self._pending_quotas is not None:
            try:
                self.finish_collect()
            except Exception:
                pass
        for connection in self._connections:
            try:
                connection.send(("stop", None))
                self._hear(connection, "ok")
            except Exception:
                pass
            connection.close()
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
