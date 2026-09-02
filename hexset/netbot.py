# SPDX-License-Identifier: GPL-3.0-only
"""A trained checkpoint as a `hexset.bots.Bot`, so the arena can score it.

The network already speaks `hexset.selfplay.BatchPolicy` — batched, one call per
tick — and the tournament code speaks `hexset.bots.Bot`, one position at a time.
This is the adapter, built in that direction because `hexset.arena` already has
the seat rotation, the Wilson intervals, the process pool and the offer budget,
and none of that would be cheap to grow on the collector.

Four things here are not decoration:

*The checkpoint is loaded once per process, not once per game.* `arena.spawn`
is called per game per worker, and a `torch.load` per game would dominate the
thing being measured. The cache is keyed on the topology as well as the path,
because a model's adjacency buffers are baked in at construction: a second
layout is a second network, not the same one on a different board.

*Intraop threading is turned off.* A duel is thirty worker processes each
running a batch of one, so torch's default of one thread per core would have
thirty processes fighting over thirty-two cores and the measurement would be of
the thrash.

*The policy is greedy.* Sampling is the behaviour distribution PPO needed, not
the policy worth scoring; `NetworkPolicy(greedy=True)` takes the argmax.

*The offer budget defaults to the one the checkpoint trained under*, which is
recorded in its `args`. Three offers a turn is what this run saw, and scoring it
at the engine's eight would measure it on a horizon it never played.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .actions import Action, ActionSpace, build_space, within_offer_budget
from .arena import (
    register_checkpoint_loader,
    register_entrant_kind,
    register_evaluator_provider,
    register_leaf_evaluator_factory,
)
from .board.board import Board
from .board.topology import Topology
from .bots import options_for
from .encoding import encode, static_graph
from .game import Game, to_move
from .mcts import Search
from .model import HexNet, config_from_args, packing
from .policy import NetworkPolicy, pair_index, pair_mask
from .selfplay import Request, action_mask


def _check_players(game: Game, players: int) -> None:
    if game.state.num_players != players:
        raise ValueError(
            f"network was trained for {players} players, "
            f"not {game.state.num_players}"
        )


def _board_order(value: np.ndarray, seat: int) -> tuple[float, ...]:
    """Undo the encoder's seat rotation and restore board-seat order."""
    players = len(value)
    return tuple(
        float(value[(board_seat - seat) % players])
        for board_seat in range(players)
    )


@dataclass(frozen=True)
class Loaded:
    """A checkpoint made playable, plus what the run it came from was doing."""

    policy: NetworkPolicy
    space: ActionSpace
    players: int
    max_offers: int | None
    iteration: int


@lru_cache(maxsize=4)
def load(
    path: str,
    topology: Topology,
    device: str = "cpu",
    compile_mode: str = "none",
) -> Loaded:
    """The network at `path`, ready to act on boards of this topology.

    Cached per process: a worker plays hundreds of games and every one of them
    would otherwise pay for the same `torch.load`.
    """
    # One thread, because there are thirty of these processes. Set here rather
    # than at import so it is attached to the decision that needs it.
    torch.set_num_threads(1)

    state = torch.load(path, map_location=device, weights_only=False)
    args = state.get("args", {})
    players = int(args.get("players", 4))
    # The head shapes read the same way width and rounds do, and default to the
    # shape every checkpoint written before they existed was trained with. A
    # checkpoint that records a shape has to be rebuilt with it or
    # `load_state_dict` fails on the keys, which is the loud failure and the one
    # worth having.
    config = config_from_args(args)

    graph = static_graph(topology)
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, players
    )
    net = HexNet(space, graph, players, config)
    net.load_state_dict(state["net"])
    net = net.to(device).eval()
    if compile_mode != "none":
        net = torch.compile(net, mode=compile_mode)

    return Loaded(
        policy=NetworkPolicy(
            net, space, packing(graph, players), device=device, greedy=True
        ),
        space=space,
        players=players,
        max_offers=args.get("max_offers"),
        iteration=int(state.get("iteration", 0)),
    )


@dataclass
class NetworkBot:
    """`HexNet` answering one position at a time.

    A batch of one costs the network's whole dispatch toll, which is why this is
    for scoring and not for collecting. On CPU that toll is small — about 0.5 ms
    against the GPU's 1.5 ms compiled — and a duel is thirty processes wide, so
    the arena is the right place for it and the GPU is not.
    """

    policy: NetworkPolicy
    space: ActionSpace
    players: int
    max_offers: int | None = None

    def choose(self, game: Game) -> Action:
        _check_players(game, self.players)
        seat = to_move(game)
        # The same filter, in the same order, as `Collector._ask`: the policy is
        # never shown a slot the budget forbids, in training or in scoring.
        options = within_offer_budget(game, options_for(game), self.max_offers)
        request = Request(
            lane=0,
            seat=seat,
            observation=encode(game, seat),
            mask=action_mask(self.space, options),
            options=tuple(options),
        )
        return self.policy.act([request])[0].action

@dataclass
class NetworkEvaluator:
    """The value head as `hexset.bots.SearchBot`'s leaf evaluation.

    The head emits one number per seat, which is the shape max^n backs up, and
    that was the point of building it that way. So the cheapest real use of a
    trained network inside the existing search is to leave the search alone and
    swap what it scores leaves with.

    **Read the caveat before reading the result.** A PPO value head is trained
    on-policy: it saw the positions its own sampling policy reached, and nothing
    else. A search puts it positions chosen to be extreme — the best reply to
    the best reply — which is exactly where an on-policy head is least reliable.
    If learned leaves lose to handcrafted ones, that is the first explanation to
    reach for and it is a result, not a bug.

    `evaluate_game` rather than `evaluate`, because the observation needs the
    whole `Game`: the phase, the turn count and the free-road counter are all
    encoded and none of them live on `GameState`. The handcrafted evaluations
    need only the state and keep the cheaper call.
    """

    policy: NetworkPolicy
    players: int
    max_offers: int | None = None

    def evaluate_game(self, game: Game, seat: int) -> list[float]:
        _check_players(game, self.players)
        value = self.policy.values([encode(game, seat)])[0]
        return list(_board_order(value, seat))


@dataclass
class LeafEvaluator:
    """A whole wave of `hexset.mcts` leaves in one forward.

    The prior and the value come off the same trunk, so a search that wanted
    them separately would pay the dispatch toll twice for one position.

    **The trade slot is one slot and the search has many trade options.** The
    flat categorical says only "propose something"; the offer heads say what.
    So the slot's probability is split across the legal offers by the pair
    distribution, which is the same joint `hexset.policy.act` samples from and
    the same one PPO takes its ratios against. Leaving the mass undivided would
    give a single arbitrary offer the whole of the policy's appetite for
    trading, and this policy proposes about 66 times a game.
    """

    policy: NetworkPolicy
    space: ActionSpace
    pad_to: int | None = None

    def __post_init__(self) -> None:
        if self.pad_to is not None and self.pad_to < 1:
            raise ValueError("pad_to must be positive")

    def evaluate(self, leaves):
        if not leaves:
            return []
        count = len(leaves)
        padded = list(leaves)
        if self.pad_to is not None and count < self.pad_to:
            padded.extend([leaves[-1]] * (self.pad_to - count))
        observations = [encode(leaf.game, leaf.seat) for leaf in padded]
        masks = np.stack([action_mask(self.space, leaf.options) for leaf in padded])
        pairs = np.stack([pair_mask(leaf.options) for leaf in padded])
        slots, offers, values = self.policy.score(observations, masks, pairs)
        return [
            (
                self._prior(leaf.options, slots[row], offers[row]),
                self._value(values[row], leaf.seat),
            )
            for row, leaf in enumerate(leaves)
        ]

    def _prior(self, options, slots: np.ndarray, offers: np.ndarray) -> np.ndarray:
        trade = self.policy.trade_slot
        log_probs = np.empty(len(options))
        for i, option in enumerate(options):
            index = self.space.index(option)
            log_probs[i] = slots[index]
            if index == trade:
                log_probs[i] += offers[pair_index(option.give, option.want)]
        prior = np.exp(log_probs)
        total = prior.sum()
        # Normalised rather than trusted: the mask makes this 1 up to float
        # error, and a search cannot use probability mass parked anywhere else.
        if total <= 0:
            return np.full(len(options), 1.0 / len(options))
        return prior / total

    def _value(self, value: np.ndarray, seat: int) -> tuple[float, ...]:
        return _board_order(value, seat)


def searcher(
    path: str,
    board: Board,
    *,
    simulations: int = 128,
    wave: int = 16,
    max_offers: int | None = None,
    device: str = "cpu",
    compile_mode: str = "none",
    inference_batch: int | None = None,
    rng=None,
) -> Search:
    """The checkpoint at `path` as a batched PUCT search, playing on `board`.

    `Search` already satisfies `hexset.bots.Bot`, so this needs no wrapper: the
    arena can seat the return value directly. `max_offers` reads `None` the way
    `network_bot` does — the budget the checkpoint trained under, not the
    engine's whole eight.
    """
    loaded = load(path, board.topology, device, compile_mode)
    budget = loaded.max_offers if max_offers is None else max_offers
    return Search(
        LeafEvaluator(
            policy=loaded.policy,
            space=loaded.space,
            pad_to=inference_batch,
        ),
        simulations=simulations,
        wave=wave,
        max_offers=budget,
        rng=rng,
    )


def network_evaluator(
    path: str, board: Board, *, device: str = "cpu"
) -> NetworkEvaluator:
    """The checkpoint at `path` as a leaf evaluation for the search."""
    loaded = load(path, board.topology, device)
    return NetworkEvaluator(
        policy=loaded.policy, players=loaded.players, max_offers=loaded.max_offers
    )


def network_bot(
    path: str, board: Board, *, max_offers: int | None = None, device: str = "cpu"
) -> NetworkBot:
    """The checkpoint at `path`, playing on `board`.

    `max_offers` of `None` means the budget the checkpoint recorded training
    under rather than the engine's whole eight — the default that measures the
    policy on the horizon it learned on. Pass a number to override it, which is
    what a duel about the budget itself would do.
    """
    loaded = load(path, board.topology, device)
    return NetworkBot(
        policy=loaded.policy,
        space=loaded.space,
        players=loaded.players,
        max_offers=loaded.max_offers if max_offers is None else max_offers,
    )


def _spawn_network(entrant, board: Board, rng) -> NetworkBot:
    """`hexset.arena`'s "network" entrant kind, wired through the registry in
    `hexset.arena` so a torch-free process never imports this module."""
    if not isinstance(entrant.weights, str):
        raise ValueError("a network entrant's weights is a checkpoint path")
    return network_bot(entrant.weights, board, max_offers=entrant.max_offers)


def _spawn_mcts(entrant, board: Board, rng) -> Search:
    """`hexset.arena`'s "mcts" entrant kind, same registry."""
    if not isinstance(entrant.weights, str):
        raise ValueError("an mcts entrant's weights is a checkpoint path")
    return searcher(
        entrant.weights,
        board,
        simulations=entrant.simulations,
        wave=entrant.wave,
        max_offers=entrant.max_offers,
        rng=rng,
    )


def _network_evaluator_provider(weights: object, board: Board) -> NetworkEvaluator:
    """`hexset.arena`'s "network" evaluator, same registry."""
    if not isinstance(weights, str):
        raise ValueError("a network evaluator's weights is a checkpoint path")
    return network_evaluator(weights, board)


def _leaf_evaluator_factory(policy, space, pad_to=None) -> LeafEvaluator:
    return LeafEvaluator(policy=policy, space=space, pad_to=pad_to)


# Registered at import so any process that imports `hexset.netbot` -- directly,
# or via `hexnet.train`/`hexnet.league`/`hexnet.collect`/`hexnet.duel` -- can
# spawn a network-backed entrant through `hexset.arena.spawn` without
# `hexset.arena` itself ever importing torch or this module.
register_entrant_kind("network", _spawn_network)
register_entrant_kind("mcts", _spawn_mcts)
register_evaluator_provider("network", _network_evaluator_provider)
register_checkpoint_loader(load)
register_leaf_evaluator_factory(_leaf_evaluator_factory)
