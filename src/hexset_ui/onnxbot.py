"""A dropped-in ONNX checkpoint as a `hexset_ui.bots.Bot`.

This module is the whole boundary between the game and a model. Everything
that knows a network exists — the observation encoding, the flat action
space, masking, sampling, the give/want pair distribution, and how a
checkpoint wants to be played — lives behind here. The rest of the package
hands over a `Game` and gets an `Action` back, and `spawn` below is the only
entry point it needs.

`NetworkBot`, `NetworkEvaluator` and `LeafEvaluator` are thin adapters over
`OnnxPolicy`; they only ever call methods on `self.policy` and never inspect
a network directly. `OnnxPolicy` is an onnxruntime `InferenceSession` with
the masking and sampling math in numpy — mechanical, since none of it has
learned parameters.

The ONNX graph is currently "observation in, raw logits/give/want/value out"
— see the training repo's `export_onnx`, which produces it. Masking stays in
Python because a position's legal moves are a fact about the rules, not
about the network, and only the engine can compute them.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import onnxruntime as ort

from .actions import Action, ActionSpace, ActionType, build_space, within_offer_budget
from .board.terrain import NUM_RESOURCES
from .board.topology import Topology
from .bots import options_for
from .encoding import Observation, encode
from .game import Game, to_move
from .mcts import Search
from .modelmeta import SearchConfig, search_config

# The off-diagonal (give, want) pairs, flattened as `give * NUM_RESOURCES +
# want`. The diagonal is never legal — `legal_actions` skips `wanted ==
# given` — so it is masked out once here rather than checked per offer.
# Same constant the training repo's policy module defines; redefined rather
# than shared because that module imports torch, which this one must not.
NUM_PAIRS = NUM_RESOURCES * NUM_RESOURCES
_OFF_DIAGONAL = ~np.eye(NUM_RESOURCES, dtype=bool).reshape(NUM_PAIRS)

# Large and finite rather than -inf, so a row whose mask is entirely False
# produces a diagnosable uniform distribution instead of NaN — same
# reasoning as the training repo's own NEG.
NEG = -1e9


@dataclass(frozen=True)
class Request:
    """One position put to the network.

    `options` rides along with `mask` because the two are not interchangeable:
    an offer is ten numbers, so `PROPOSE_TRADE` occupies one slot in the flat
    space and its bundles cannot be recovered from an index. Choosing that slot
    still leaves the offer itself to name.

    `seat` is `to_move`, which is not always `current_player` — discarding on a
    seven and answering an offer belong to somebody else — and it is the
    perspective `observation` was encoded from.
    """

    seat: int
    observation: Observation
    mask: np.ndarray
    options: tuple[Action, ...]


def action_mask(space: ActionSpace, options: Sequence[Action]) -> np.ndarray:
    """Mark already-enumerated actions without enumerating them again."""
    mask = np.zeros(space.size, dtype=bool)
    for action in options:
        mask[space.index(action)] = True
    return mask


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
        float(value[(board_seat - seat) % players]) for board_seat in range(players)
    )


def _pair_index(give, want) -> int:
    """The flat pair slot for a one-for-one offer's two one-hot bundles."""
    return give.index(1) * NUM_RESOURCES + want.index(1)


def _pair_mask(options) -> np.ndarray:
    """Which one-for-one offers were legal, as a flat `(NUM_PAIRS,)` bool."""
    mask = np.zeros(NUM_PAIRS, dtype=bool)
    for option in options:
        if option.type is ActionType.PROPOSE_TRADE:
            mask[_pair_index(option.give, option.want)] = True
    return mask


def _one_hot(resource: int) -> tuple[int, ...]:
    return tuple(1 if r == resource else 0 for r in range(NUM_RESOURCES))


def _masked_log_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`log_softmax` over the legal entries of each row."""
    filled = np.where(mask, logits, NEG)
    shifted = filled - filled.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _pair_logits(give: np.ndarray, want: np.ndarray) -> np.ndarray:
    """`(B, NUM_PAIRS)` outer sum, so the two heads factor the joint offer."""
    batch = give.shape[0]
    return (give[:, :, None] + want[:, None, :]).reshape(batch, NUM_PAIRS)


def _providers_for(device: str) -> list[str]:
    """onnxruntime's own provider names, not torch's device strings — `cpu`
    is the only one this repo actually runs on (no GPU on the boxes it
    targets), but the `--device` flag stays functional for anyone with a
    build that has a GPU execution provider available, falling back to CPU
    if it doesn't."""
    if not device or device == "cpu":
        return ["CPUExecutionProvider"]
    return [f"{device.upper()}ExecutionProvider", "CPUExecutionProvider"]


class OnnxPolicy:
    """An onnxruntime `InferenceSession` behind the four-member interface the
    training repo's policy class presents (`.act`, `.values`, `.score`,
    `.trade_slot`).

    Always greedy in practice: every checkpoint served here plays argmax, so
    there is no behaviour distribution anything actually reads. The stochastic
    path exists anyway, as the honest port of what the interface supports.
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        space: ActionSpace,
        *,
        greedy: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.session = session
        self.space = space
        self.greedy = greedy
        self.rng = rng or np.random.default_rng()
        self.trade_slot = space.offsets[ActionType.PROPOSE_TRADE]

    def _forward(self, observations) -> dict[str, np.ndarray]:
        inputs = {
            "hexes": np.stack([o.hexes for o in observations]).astype(np.float32),
            "vertices": np.stack([o.vertices for o in observations]).astype(np.float32),
            "edges": np.stack([o.edges for o in observations]).astype(np.float32),
            "globals": np.stack([o.globals for o in observations]).astype(np.float32),
        }
        logits, give, want, value = self.session.run(
            ["logits", "give", "want", "value"], inputs
        )
        return {"logits": logits, "give": give, "want": want, "value": value}

    def _sample(self, log_probs: np.ndarray) -> np.ndarray:
        if self.greedy:
            return log_probs.argmax(axis=-1)
        # Gumbel-max: a fully vectorised categorical draw over log-probs,
        # with no second normalising pass needed since the rows are already
        # log-softmax'd.
        gumbel = -np.log(-np.log(self.rng.random(log_probs.shape)))
        return (log_probs + gumbel).argmax(axis=-1)

    def _log_probs(
        self, prediction: dict[str, np.ndarray], mask: np.ndarray, pair: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        slots = _masked_log_softmax(prediction["logits"], mask)
        offers = _masked_log_softmax(
            _pair_logits(prediction["give"], prediction["want"]),
            pair & _OFF_DIAGONAL,
        )
        return slots, offers

    def act(self, requests: Sequence[Request]) -> list[Action]:
        """One action per request, in order."""
        if not requests:
            return []

        masks = np.stack([request.mask for request in requests])
        pairs = np.stack([_pair_mask(request.options) for request in requests])
        prediction = self._forward([request.observation for request in requests])
        slot_log_probs, offer_log_probs = self._log_probs(prediction, masks, pairs)
        chosen = self._sample(slot_log_probs)
        offer = self._sample(offer_log_probs)

        out = []
        for row in range(len(requests)):
            index = int(chosen[row])
            if index == self.trade_slot:
                slot = int(offer[row])
                out.append(
                    Action(
                        ActionType.PROPOSE_TRADE,
                        give=_one_hot(slot // NUM_RESOURCES),
                        want=_one_hot(slot % NUM_RESOURCES),
                    )
                )
            else:
                out.append(self.space.decode(index))
        return out

    def values(self, observations) -> np.ndarray:
        """The value head alone, `(B, players)`, in each row's own frame."""
        return self._forward(list(observations))["value"]

    def score(
        self, observations, masks: np.ndarray, pairs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Masked slot log-probs, masked offer log-probs and values for a
        batch — what a batched search needs and `act` does not give it."""
        prediction = self._forward(list(observations))
        slots, offers = self._log_probs(prediction, masks, pairs)
        return slots, offers, prediction["value"]


@dataclass(frozen=True)
class Loaded:
    """A checkpoint made playable, plus what the run it came from was doing."""

    policy: OnnxPolicy
    space: ActionSpace
    players: int
    max_offers: int | None
    iteration: int
    search: SearchConfig = SearchConfig()


@lru_cache(maxsize=4)
def _load_cached(path: str, topology: Topology, device: str, mtime_ns: int) -> Loaded:
    session = ort.InferenceSession(str(path), providers=_providers_for(device))
    meta = session.get_modelmeta().custom_metadata_map

    players = int(meta["players"])
    # The topology fingerprint `hexset_ui.export_onnx` embeds: a graph traced
    # for one board shape fails silently if fed another (wrong-shaped
    # inputs, or worse, right-shaped-but-meaningless ones) rather than
    # loudly the way `net.load_state_dict` fails on a shape mismatch today
    # — so this is the loud failure standing in for that one.
    fingerprint = (
        int(meta["num_hexes"]),
        int(meta["num_vertices"]),
        int(meta["num_edges"]),
    )
    actual = (topology.num_hexes, topology.num_vertices, topology.num_edges)
    if fingerprint != actual:
        raise ValueError(
            f"{path} was exported for a board shaped {fingerprint} "
            f"(hexes, vertices, edges), not this table's {actual}"
        )

    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, players
    )
    max_offers = meta.get("max_offers") or None
    return Loaded(
        policy=OnnxPolicy(session, space),
        space=space,
        players=players,
        max_offers=int(max_offers) if max_offers is not None else None,
        iteration=int(meta.get("iteration", 0)),
        search=search_config(meta),
    )


def load(path: str, topology: Topology, device: str = "cpu") -> Loaded:
    """The checkpoint at `path`, ready to act on boards of this topology.

    Cache key folds in the file's mtime, unlike the training repo's loader: its
    `.pt` checkpoints under `runs/` are effectively immutable per-run
    artifacts, but hexset-ui's whole pitch is replacing a file in `models/`
    by name — without the mtime, a same-named replacement would silently
    keep serving the old in-memory session.
    """
    return _load_cached(path, topology, device, os.stat(path).st_mtime_ns)


@dataclass
class NetworkBot:
    """`OnnxPolicy` answering one position at a time."""

    policy: OnnxPolicy
    space: ActionSpace
    players: int
    max_offers: int | None = None

    def choose(self, game: Game) -> Action:
        _check_players(game, self.players)
        seat = to_move(game)
        options = within_offer_budget(game, options_for(game), self.max_offers)
        request = Request(
            seat=seat,
            observation=encode(game, seat),
            mask=action_mask(self.space, options),
            options=tuple(options),
        )
        return self.policy.act([request])[0]


@dataclass
class NetworkEvaluator:
    """The value head as `hexset_ui.bots.SearchBot`'s leaf evaluation."""

    policy: OnnxPolicy
    players: int
    max_offers: int | None = None

    def evaluate_game(self, game: Game, seat: int) -> list[float]:
        _check_players(game, self.players)
        value = self.policy.values([encode(game, seat)])[0]
        return list(_board_order(value, seat))


@dataclass
class LeafEvaluator:
    """A whole wave of `hexset_ui.mcts` leaves in one forward."""

    policy: OnnxPolicy
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
        pairs = np.stack([_pair_mask(leaf.options) for leaf in padded])
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
                log_probs[i] += offers[_pair_index(option.give, option.want)]
        prior = np.exp(log_probs)
        total = prior.sum()
        if total <= 0:
            return np.full(len(options), 1.0 / len(options))
        return prior / total

    def _value(self, value: np.ndarray, seat: int) -> tuple[float, ...]:
        return _board_order(value, seat)


def searcher(
    path: str,
    board,
    *,
    simulations: int = 128,
    wave: int = 16,
    max_offers: int | None = None,
    device: str = "cpu",
    inference_batch: int | None = None,
    rng=None,
) -> Search:
    """The checkpoint at `path` as a batched PUCT search, playing on `board`."""
    loaded = load(path, board.topology, device)
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


def network_evaluator(path: str, board, *, device: str = "cpu") -> NetworkEvaluator:
    """The checkpoint at `path` as a leaf evaluation for the search."""
    loaded = load(path, board.topology, device)
    return NetworkEvaluator(
        policy=loaded.policy, players=loaded.players, max_offers=loaded.max_offers
    )


def network_bot(
    path: str, board, *, max_offers: int | None = None, device: str = "cpu"
) -> NetworkBot:
    """The checkpoint at `path`, playing on `board`."""
    loaded = load(path, board.topology, device)
    return NetworkBot(
        policy=loaded.policy,
        space=loaded.space,
        players=loaded.players,
        max_offers=loaded.max_offers if max_offers is None else max_offers,
    )


def spawn(
    path: str,
    board,
    *,
    rng: random.Random | None = None,
    device: str = "cpu",
    max_offers: int | None = None,
):
    """The checkpoint at `path` as something with `.choose(game) -> Action`.

    The only entry point the rest of the package needs. Whether the file plays
    a single forward pass or a search over its own priors is the file's own
    business, declared in its metadata and read here — a caller passes a path
    and gets a bot, and never learns which it got.

    `device` is deliberately not read from metadata: it is a fact about the
    machine serving the game, not about the checkpoint, and a model file has no
    business demanding an accelerator its host may not have.
    """
    loaded = load(path, board.topology, device)
    if not loaded.search.searches:
        return network_bot(path, board, max_offers=max_offers, device=device)
    return searcher(
        path,
        board,
        simulations=loaded.search.simulations,
        wave=loaded.search.wave,
        max_offers=max_offers,
        device=device,
        rng=rng,
    )
