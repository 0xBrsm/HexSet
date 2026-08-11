from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan.actions import ActionType, legal_actions, space_for  # noqa: E402
from catan.board.board import random_base_board  # noqa: E402
from catan.encoding import static_graph  # noqa: E402
from catan.game import start  # noqa: E402
from catan.model import CatanNet, ModelConfig, packing  # noqa: E402
from catan.play import step_randomly  # noqa: E402
from catan.policy import (  # noqa: E402
    NUM_PAIRS,
    NetworkPolicy,
    masked_log_softmax,
    pair_index,
    pair_mask,
)
from catan.selfplay import Collector  # noqa: E402


def a_policy(players: int = 4, seed: int = 0, **kwargs):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    space = space_for(game)
    graph = static_graph(board.topology)
    torch.manual_seed(seed)
    net = CatanNet(space, graph, players, ModelConfig(width=16, rounds=1))
    return NetworkPolicy(net, space, packing(graph, players), **kwargs)


class Counting:
    """Wraps the net so a test can say how many forwards a tick actually ran."""

    def __init__(self, net):
        self.net = net
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.net(*args)

    def parameters(self):
        return self.net.parameters()


def test_a_tick_runs_exactly_one_forward_however_many_lanes_there_are():
    policy = a_policy()
    counter = Counting(policy.net)
    policy.net = counter
    lanes = 12
    collector = Collector(policy, lanes=lanes, seed=1)

    collector.tick()
    collector.tick()

    # Anti-vacuity: one forward per tick is trivially true at one lane, and the
    # whole point of the collector's shape is that it stays true at many.
    assert lanes > 1
    assert collector.steps == 2 * lanes
    assert counter.calls == 2, f"{counter.calls} forwards for 2 ticks"


def test_every_action_it_picks_is_one_the_engine_offered():
    policy = a_policy(seed=2)
    collector = Collector(policy, lanes=8, seed=2)

    kinds = set()
    for _ in range(60):
        requests = [collector._ask(lane, slot) for slot, lane in enumerate(collector._lanes)]
        for request, choice in zip(requests, policy.act(requests)):
            assert choice.action in request.options, choice.action
            kinds.add(choice.action.type)
        collector.tick()

    # Anti-vacuity: a run that only ever saw ROLL and END_TURN would pass this
    # while exercising none of the index arithmetic it is meant to check.
    assert len(kinds) >= 5, f"only saw {sorted(k.name for k in kinds)}"


def test_it_never_picks_an_action_the_mask_ruled_out():
    policy = a_policy(seed=3)
    collector = Collector(policy, lanes=8, seed=3)

    excluded = 0
    checked = 0
    for _ in range(40):
        requests = [collector._ask(lane, slot) for slot, lane in enumerate(collector._lanes)]
        for request, choice in zip(requests, policy.act(requests)):
            index = policy.space.index(choice.action)
            assert request.mask[index]
            excluded += int((~request.mask).sum())
            checked += 1
        collector.tick()

    # Anti-vacuity: if the mask excluded nothing there would be nothing to obey.
    assert checked > 0
    assert excluded > 0


def test_every_legal_action_but_an_offer_is_recoverable_from_its_index():
    # The policy reconstructs its choice with `space.decode` rather than by
    # searching the options, which is only sound if decode is exact for every
    # kind except PROPOSE_TRADE. That is the assumption; this is the pin.
    rng = random.Random(4)
    game = start(random_base_board(rng), 4, rng)
    space = space_for(game)

    kinds = set()
    for _ in range(1500):
        for action in legal_actions(game):
            if action.type is ActionType.PROPOSE_TRADE:
                continue
            kinds.add(action.type)
            assert space.decode(space.index(action)) == action, action
        step_randomly(game, rng)

    assert len(kinds) >= 8, f"only saw {sorted(k.name for k in kinds)}"


def test_a_proposed_offer_is_one_the_engine_would_have_enumerated():
    policy = a_policy(seed=5)
    collector = Collector(policy, lanes=16, seed=5)

    proposals = 0
    for _ in range(120):
        requests = [collector._ask(lane, slot) for slot, lane in enumerate(collector._lanes)]
        for request, choice in zip(requests, policy.act(requests)):
            if choice.action.type is not ActionType.PROPOSE_TRADE:
                continue
            proposals += 1
            offers = [
                o for o in request.options if o.type is ActionType.PROPOSE_TRADE
            ]
            assert choice.action in offers, choice.action
            assert isinstance(choice.aux, np.ndarray)
            assert choice.aux[pair_index(choice.action.give, choice.action.want)]
        collector.tick()

    # Anti-vacuity: the loop above is silent if the policy never proposed once.
    assert proposals > 0, "the policy never proposed a trade"


def test_the_offer_mask_matches_the_offers_that_were_legal():
    rng = random.Random(6)
    game = start(random_base_board(rng), 4, rng)

    seen = 0
    for _ in range(1200):
        options = legal_actions(game)
        offers = [o for o in options if o.type is ActionType.PROPOSE_TRADE]
        mask = pair_mask(options)
        assert int(mask.sum()) == len(offers)
        for offer in offers:
            assert mask[pair_index(offer.give, offer.want)]
        seen += len(offers)
        step_randomly(game, rng)

    assert seen > 0, "no offer was ever legal"


def test_a_trade_records_the_offer_in_its_log_prob_and_a_plain_action_does_not():
    policy = a_policy(seed=7)
    collector = Collector(policy, lanes=16, seed=7)

    trades, plains = [], []
    for _ in range(120):
        requests = [collector._ask(lane, slot) for slot, lane in enumerate(collector._lanes)]
        choices = policy.act(requests)
        with torch.no_grad():
            buffer, mask, pair, chosen, offer = _as_batch(policy, requests, choices)
            slot_only = masked_log_softmax(
                policy.net(*_unpacked(policy, buffer)).logits, mask
            ).gather(1, chosen.unsqueeze(1)).squeeze(1)
        for row, choice in enumerate(choices):
            if choice.action.type is ActionType.PROPOSE_TRADE:
                trades.append((choice.log_prob, float(slot_only[row])))
            else:
                plains.append((choice.log_prob, float(slot_only[row])))
        collector.tick()

    assert trades, "the policy never proposed a trade"
    assert plains

    # A plain action's joint log-prob is exactly its slot's.
    for joint, slot in plains:
        assert joint == pytest.approx(slot, abs=1e-4)
    # A trade's is strictly smaller, because naming the offer costs probability.
    for joint, slot in trades:
        assert joint < slot - 1e-6


def _unpacked(policy, buffer):
    from catan.model import unpack

    return unpack(policy.layout, buffer)


def _as_batch(policy, requests, choices):
    from catan.model import pack

    buffer = pack(policy.layout, [r.observation for r in requests])
    mask = torch.from_numpy(np.stack([r.mask for r in requests]))
    pair = torch.from_numpy(
        np.stack(
            [
                c.aux if isinstance(c.aux, np.ndarray) else np.zeros(NUM_PAIRS, bool)
                for c in choices
            ]
        )
    )
    chosen = torch.tensor(
        [policy.space.index(c.action) for c in choices], dtype=torch.int64
    )
    offer = torch.tensor(
        [
            pair_index(c.action.give, c.action.want)
            if c.action.type is ActionType.PROPOSE_TRADE
            else 0
            for c in choices
        ],
        dtype=torch.int64,
    )
    return buffer, mask, pair, chosen, offer


def test_evaluate_reproduces_exactly_what_act_recorded():
    # PPO's ratio is `exp(evaluate - act)`, so if these two disagree the very
    # first update is already taking ratios against the wrong distribution.
    policy = a_policy(seed=8)
    collector = Collector(policy, lanes=16, seed=8)

    trades = 0
    for _ in range(40):
        requests = [collector._ask(lane, slot) for slot, lane in enumerate(collector._lanes)]
        choices = policy.act(requests)
        buffer, mask, pair, chosen, offer = _as_batch(policy, requests, choices)
        with torch.no_grad():
            evaluation = policy.evaluate(buffer, mask, pair, chosen, offer)

        recorded = torch.tensor([c.log_prob for c in choices])
        assert torch.allclose(evaluation.log_prob, recorded, atol=1e-4)
        values = torch.tensor([c.value for c in choices])
        assert torch.allclose(evaluation.value, values, atol=1e-4)
        trades += sum(
            c.action.type is ActionType.PROPOSE_TRADE for c in choices
        )
        collector.tick()

    # Anti-vacuity: the offer half of the joint is the part most likely to
    # disagree, so agreeing on a batch with no trades in it proves little.
    assert trades > 0, "no trade was sampled, so the offer term went unchecked"


def test_the_offer_distribution_never_puts_mass_on_a_swap_for_itself():
    # `legal_actions` skips `wanted == given`, so the diagonal is masked out
    # once at import rather than checked per offer. If that broke, the policy
    # could propose trading wood for wood.
    policy = a_policy(seed=9)
    logits = torch.zeros(3, 5)
    pair = torch.ones(3, NUM_PAIRS, dtype=torch.bool)
    from catan.policy import pair_logits

    log_probs = masked_log_softmax(
        pair_logits(logits, logits), pair & policy._off_diagonal
    )
    probabilities = log_probs.exp().reshape(3, 5, 5)
    assert torch.allclose(probabilities.diagonal(dim1=1, dim2=2), torch.zeros(3, 5), atol=1e-6)
    assert probabilities.sum() == pytest.approx(3.0, abs=1e-4)


def test_a_greedy_policy_makes_the_same_choice_twice():
    policy = a_policy(seed=10, greedy=True)
    collector = Collector(policy, lanes=8, seed=10)
    requests = [collector._ask(lane, slot) for slot, lane in enumerate(collector._lanes)]

    first = policy.act(requests)
    second = policy.act(requests)

    assert [c.action for c in first] == [c.action for c in second]
    assert len({c.action for c in first}) > 1, "every lane picked the same action"


def test_an_empty_batch_is_answered_without_touching_the_network():
    policy = a_policy(seed=11)
    counter = Counting(policy.net)
    policy.net = counter
    assert policy.act([]) == []
    assert counter.calls == 0
