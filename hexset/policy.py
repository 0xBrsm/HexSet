# SPDX-License-Identifier: GPL-3.0-only
"""The torch `BatchPolicy`: one forward per tick, masked, sampled.

This is the seam between `hexset.selfplay` and `hexset.model`, and its shape is
decided by one measurement. A forward costs a fixed dispatch toll — 12.5 ms
eager, ~1.5 ms compiled — plus ~25 µs per position, so the cost of asking the
network anything at all dwarfs the cost of asking it about five hundred things.
`Collector` therefore gathers one `Request` per lane and calls `act` once; this
module answers the whole batch with a single `forward`, a single host-to-device
copy and a single read-back. Everything else here exists to keep it to one.

## `PROPOSE_TRADE` is one slot and ten numbers, and that is the whole problem

Every other action is recoverable from its flat index: `ActionSpace.decode`
turns the index back into exactly the `Action` that `legal_actions` emitted. An
offer cannot be, because an offer is ten numbers and `PROPOSE_TRADE` occupies a
single slot meaning "an offer is available". So when the flat categorical picks
that slot the policy has chosen to propose *something* and has not yet said
what.

**It names the offer from its own `give` and `want` heads, masked to the offers
that were actually legal.** Three alternatives were rejected and it is worth
recording why, because each fails somewhere that would not show up as a crash:

- *Pick uniformly from the trade options.* The offer heads then receive no
  gradient ever, and `hexset.model` grew them for this and nothing else. The
  policy could learn when to trade but never what to trade.
- *Sample the heads unmasked and let the engine reject an illegal offer.* There
  is no rejection path — `apply` carries out any well-formed offer — so this
  does not fail loudly, it silently plays offers `legal_actions` had ruled out
  because nobody at the table could cover them.
- *Sample the heads masked but record only the slot's log-prob.* This is the
  subtle one. PPO's importance ratio has to be taken against the distribution
  that actually generated the data, and here that distribution is the joint one
  over (slot, offer). Recording half of it makes every ratio on a trading
  transition wrong by the offer's log-prob, and the error is systematic rather
  than noisy — it is largest exactly where the policy is most confident.

So `log_prob` is the joint: `log P(slot) + log P(give, want | slot)`, the second
term present only when the slot is the trade slot. The pair distribution is the
outer sum of the two heads' logits, which makes `give` and `want` conditionally
independent given the decision to trade. That is a real modelling restriction
and it is deliberate for v1: `legal_actions` only ever enumerates one-for-one
offers, so the reachable set is the 20 off-diagonal pairs and a factored
distribution covers all of it.

The mask over those pairs is not recoverable from a stored transition — it
depends on what the opponents could cover, which the observation deliberately
does not contain — so it rides along in `Choice.aux` for the update to reuse.

## What is not optimised here, and why

Measured on the box today: engine step 18.8 µs, `encode` 25.2 µs,
`legal_actions` 1.8 µs — about 46 µs of Python per position against 23 µs of
GPU forward at batch 512. The rollout is plumbing-bound by roughly 2:1, so
shaving the torch side of a tick cannot buy more than a third of it even if it
went to zero. More throughput comes from running collectors in parallel
processes, not from anything in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .actions import Action, ActionSpace, ActionType
from .board.terrain import NUM_RESOURCES
from .model import HexNet, Packing, pack, packing, unpack
from .selfplay import Choice, Request

# The off-diagonal (give, want) pairs, flattened as `give * NUM_RESOURCES +
# want`. The diagonal is never legal — `legal_actions` skips `wanted == given`
# — so it is masked out once here rather than checked per offer.
NUM_PAIRS = NUM_RESOURCES * NUM_RESOURCES

_OFF_DIAGONAL = ~np.eye(NUM_RESOURCES, dtype=bool).reshape(NUM_PAIRS)

# Large and finite rather than -inf. A row whose mask is entirely False would
# make `log_softmax` produce NaN under -inf and merely produce a uniform
# distribution here, which is recoverable and diagnosable. `Collector` raises
# `Stuck` before an empty flat mask can reach us, but the pair mask is empty on
# every non-trading position, which is most of them.
NEG = -1e9


def pair_index(give: Sequence[int], want: Sequence[int]) -> int:
    """The flat pair slot for a one-for-one offer's two one-hot bundles."""
    return give.index(1) * NUM_RESOURCES + want.index(1)


def pair_mask(options: Sequence[Action]) -> np.ndarray:
    """Which one-for-one offers were legal, as a flat `(NUM_PAIRS,)` bool.

    Empty for a position where proposing is not available, which is the common
    case and is why the caller must not assume any bit is set.
    """
    mask = np.zeros(NUM_PAIRS, dtype=bool)
    for option in options:
        if option.type is ActionType.PROPOSE_TRADE:
            mask[pair_index(option.give, option.want)] = True
    return mask


def masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """`log_softmax` over the legal entries of each row."""
    return torch.log_softmax(logits.masked_fill(~mask, NEG), dim=-1)


def pair_logits(give: Tensor, want: Tensor) -> Tensor:
    """`(B, NUM_PAIRS)` outer sum, so the two heads factor the joint offer."""
    batch = give.shape[0]
    return (give.unsqueeze(2) + want.unsqueeze(1)).reshape(batch, NUM_PAIRS)


def _entropy(log_probs: Tensor) -> Tensor:
    return -(log_probs.exp() * log_probs).sum(-1)


@dataclass(frozen=True)
class Evaluation:
    """What the PPO update needs recomputed under the current parameters."""

    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    # `None` for every value head but `"quantile"`, where it is the
    # `(B, players, Q)` tensor whose mean is `value`. Carried through so the
    # value loss can be taken on the forward pass the ratio already paid for;
    # `hexset.ppo` is the only reader.
    quantiles: Tensor | None = None


class NetworkPolicy:
    """`HexNet` behind `hexset.selfplay.BatchPolicy`.

    Sampling is stochastic by default because PPO is on-policy and needs the
    behaviour distribution it will later take ratios against. `greedy=True`
    takes the argmax instead, which is for evaluation duels only — a greedy
    policy's `log_prob` is not the one that generated anything.
    """

    def __init__(
        self,
        net: HexNet,
        space: ActionSpace,
        layout: Packing,
        *,
        device: torch.device | str = "cpu",
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        self.net = net
        self.space = space
        self.layout = layout
        self.device = torch.device(device)
        self.greedy = greedy
        self.generator = generator
        self.trade_slot = space.offsets[ActionType.PROPOSE_TRADE]
        self._off_diagonal = torch.from_numpy(_OFF_DIAGONAL).to(self.device)

    def _sample(self, log_probs: Tensor) -> Tensor:
        if self.greedy:
            return log_probs.argmax(dim=-1)
        # `multinomial` over probabilities rather than Gumbel over logits: the
        # rows are already normalised by the masked log-softmax, so this needs
        # no second pass and stays one kernel.
        return torch.multinomial(
            log_probs.exp(), 1, generator=self.generator
        ).squeeze(-1)

    def _log_probs(
        self, prediction, mask: Tensor, pair: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The two distributions shared by acting, search, and PPO updates."""
        slots = masked_log_softmax(prediction.logits, mask)
        offers = masked_log_softmax(
            pair_logits(prediction.give, prediction.want),
            pair & self._off_diagonal,
        )
        return slots, offers

    def _chosen_log_prob(
        self,
        slots: Tensor,
        offers: Tensor,
        chosen: Tensor,
        offer: Tensor,
    ) -> Tensor:
        log_prob = slots.gather(1, chosen.unsqueeze(1)).squeeze(1)
        offer_log_prob = offers.gather(1, offer.unsqueeze(1)).squeeze(1)
        return log_prob + torch.where(
            chosen == self.trade_slot,
            offer_log_prob,
            torch.zeros_like(log_prob),
        )

    def act(self, requests: Sequence[Request]) -> list[Choice]:
        if not requests:
            return []

        masks = np.stack([request.mask for request in requests])
        pairs = np.stack([pair_mask(request.options) for request in requests])

        buffer = pack(self.layout, [request.observation for request in requests])
        mask = torch.from_numpy(masks).to(self.device, non_blocking=True)
        pair = torch.from_numpy(pairs).to(self.device, non_blocking=True)

        with torch.no_grad():
            prediction = self.net(*unpack(self.layout, buffer.to(self.device)))

            slot_log_probs, offer_log_probs = self._log_probs(prediction, mask, pair)
            chosen = self._sample(slot_log_probs)

            # Computed for every row rather than only the trading ones: a
            # gather over the whole batch is one kernel, where selecting the
            # trading subset first would be a device-to-host sync per tick to
            # find out which rows those are.
            offer = self._sample(offer_log_probs)
            log_prob = self._chosen_log_prob(
                slot_log_probs, offer_log_probs, chosen, offer
            )

            # One read-back rather than four. A transfer costs a fixed ~250 µs
            # per tensor whatever its size, so the concatenation is free and
            # the three extra crossings are not.
            read = torch.cat(
                [
                    chosen.unsqueeze(1).to(prediction.value.dtype),
                    offer.unsqueeze(1).to(prediction.value.dtype),
                    log_prob.unsqueeze(1),
                    prediction.value,
                ],
                dim=1,
            ).cpu().numpy()

        out = []
        for row, request in enumerate(requests):
            index = int(read[row, 0])
            action = self.space.decode(index)
            # Recorded on every row, not only the ones that proposed. The pair
            # mask is a property of the *position* — which trades are legal —
            # not of the action drawn, and `evaluate` weights the offer entropy
            # by P(trade) on every row it sees. Attaching it only to proposing
            # rows left `hexset.ppo._pair_mask` substituting an all-False mask
            # everywhere else, which `masked_log_softmax` turns into a uniform
            # over all 25 slots: a constant `log 25 = 3.219` of offer entropy
            # carrying no gradient to `trade_give`/`trade_want` and — because it
            # enters weighted by P(trade) — a standing entropy bonus for
            # proposing worth `entropy_coefficient * 3.219` per unit of
            # probability on every trading-legal position. The *ratio* was never
            # affected, since the offer log-prob is gated on
            # `chosen == trade_slot` in both `act` and `evaluate`, so this only
            # ever surfaced as inflated entropy and a drift toward proposing —
            # which is the mechanism behind ppo3's accept-canary overshoot when
            # the entropy coefficient was doubled.
            aux = pairs[row]
            if index == self.trade_slot:
                slot = int(read[row, 1])
                action = Action(
                    ActionType.PROPOSE_TRADE,
                    give=_one_hot(slot // NUM_RESOURCES),
                    want=_one_hot(slot % NUM_RESOURCES),
                )
                aux = pairs[row]
            out.append(
                Choice(
                    action=action,
                    log_prob=float(read[row, 2]),
                    value=tuple(read[row, 3:].tolist()),
                    aux=aux,
                )
            )
        return out

    def values(self, observations: Sequence) -> np.ndarray:
        """The value head alone, `(B, players)`, in each row's own frame.

        For a search, which scores leaves and has no use for the policy logits.
        The forward computes them anyway — the heads share a trunk and splitting
        them would cost more than the gather does — so the saving here is the
        read-back and the sampling, not the network.

        Takes a list because that is the seam a leaf-batching search will want,
        even though `hexset.bots.SearchBot` currently hands over one at a time.
        """
        buffer = pack(self.layout, list(observations))
        with torch.no_grad():
            prediction = self.net(*unpack(self.layout, buffer.to(self.device)))
        return prediction.value.cpu().numpy()

    def score(
        self,
        observations: Sequence,
        masks: np.ndarray,
        pairs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Masked slot log-probs, masked offer log-probs and values for a batch.

        What a batched search needs and `act` does not give it: the distribution
        itself rather than a draw from it. A search wants the prior over every
        option, and `values` alone is not enough — a PUCT bonus without a prior
        is an untrained search. One forward and one read-back, for the same
        reason `act` is one of each.
        """
        buffer = pack(self.layout, list(observations))
        mask = torch.from_numpy(masks).to(self.device, non_blocking=True)
        pair = torch.from_numpy(pairs).to(self.device, non_blocking=True)
        with torch.no_grad():
            prediction = self.net(*unpack(self.layout, buffer.to(self.device)))
            slots, offers = self._log_probs(prediction, mask, pair)
            read = torch.cat([slots, offers, prediction.value], dim=1).cpu().numpy()
        width = slots.shape[1]
        return (
            read[:, :width],
            read[:, width : width + NUM_PAIRS],
            read[:, width + NUM_PAIRS :],
        )

    def distributions(
        self, buffer: Tensor, mask: Tensor, pair: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Both masked log-prob rows and the values, with grad.

        The third reader of `_log_probs`, and it wants what the other two throw
        away. `score` is these same quantities without grad and as numpy, for
        the search; `evaluate` reduces them to the one action PPO took. A
        cross-entropy against a visit distribution is defined over the whole
        row, so it can use neither.
        """
        prediction = self.net(*unpack(self.layout, buffer))
        slots, offers = self._log_probs(prediction, mask, pair)
        return slots, offers, prediction.value

    def evaluate(
        self,
        buffer: Tensor,
        mask: Tensor,
        pair: Tensor,
        chosen: Tensor,
        offer: Tensor,
    ) -> Evaluation:
        """Recompute a stored batch's log-probs, entropy and values, with grad.

        The mirror of `act`, and the two have to agree exactly or PPO's ratio is
        wrong at step zero rather than after it has learned something.
        `test_evaluate_reproduces_the_log_prob_act_recorded` pins that.
        """
        prediction = self.net(*unpack(self.layout, buffer))

        slot_log_probs, offer_log_probs = self._log_probs(prediction, mask, pair)
        log_prob = self._chosen_log_prob(
            slot_log_probs, offer_log_probs, chosen, offer
        )

        # The joint entropy of a two-stage decision, not the sum of two
        # marginals: the offer is only drawn if the slot came out as the trade
        # slot, so its entropy enters weighted by that probability. Adding it
        # unweighted would push the policy to keep its offer distribution wide
        # on the great majority of positions where it will never propose at all.
        trade_prob = slot_log_probs[:, self.trade_slot].exp()
        entropy = _entropy(slot_log_probs) + trade_prob * _entropy(offer_log_probs)

        return Evaluation(
            log_prob=log_prob,
            entropy=entropy,
            value=prediction.value,
            quantiles=prediction.quantiles,
        )


def _one_hot(resource: int) -> tuple[int, ...]:
    return tuple(1 if r == resource else 0 for r in range(NUM_RESOURCES))


def build(
    net: HexNet,
    space: ActionSpace,
    graph,
    players: int,
    **kwargs,
) -> NetworkPolicy:
    """A policy over `net`, with the packing layout derived from the graph."""
    return NetworkPolicy(net, space, packing(graph, players), **kwargs)
