# SPDX-License-Identifier: GPL-3.0-only
"""Atomic seats on native lanes: the same games, decisions and labels as on Python lanes.

A seat that chooses one atomic move at a time (`PolicySeat`) reads each row's position: on a native lane a view of
the lane (`hexset.native.playout.NativeCopy`). The policies here read contract-8 rows (`hexset.encoding`), contract-9
rows (`hexset.contract9`) and the decision's tensors, and score each option from a hash of everything they read, so a
single differing byte changes the game.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from hexset import contract9, decisions as D, encoding
from hexset.actions import space_for
from hexset.arena import deal_game
from hexset.gym.driver import Driver
from hexset.gym.lanes import LaneEnv
from hexset.gym.native import available
from hexset.gym.seats import Labels, Play, PolicySeat
from hexset.record import to_json

from test_driver import SEED, Uniform

pytestmark = pytest.mark.skipif(not available(), reason="optional native extension")


class Reader:
    """A policy reading `contract` at each row, and the row's decision tensors."""

    def __init__(self, contract: str, players: int) -> None:
        self.native_contract = contract
        self.space = space_for(deal_game(SEED, 0, players))
        self.players = players

    def score_rows(self, rows):
        games, seats = [game for game, _, _ in rows], [seat for _, seat, _ in rows]
        if self.native_contract in encoding.OBSERVATION_CONTRACTS:
            feats = [np.concatenate([o.hexes.ravel(), o.vertices.ravel(), o.edges.ravel(), o.globals])
                     for o in encoding.encode_batch(games, seats, self.native_contract)]
        else:
            feats = [contract9.observation_row(game, seat, self.native_contract) for game, seat in zip(games, seats)]
        out = []
        for (game, seat, options), feat in zip(rows, feats):
            decision = D.decision_at(game, seat, self.space, double_turn=False)
            digest = hashlib.blake2b(feat.tobytes() + decision.flat.tobytes() + decision.setup.tobytes()
                                     + repr(options).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            prior = rng.random(len(options)) + 0.05
            out.append((prior / prior.sum(), tuple(rng.random(self.players))))
        return out


def play(players, kind, *, mixed=False, games=4, order="random"):
    contracts = ("8", "9", "8.1") if players == 4 else ("8", "9d", "9")
    seat = PolicySeat([Reader(c, players) for c in contracts],
                      {0: Play(members=(0, 1, 2), temperature=0.7), 2: Play(members=(1,), temperature=1.0)},
                      record=True)
    cast = (lambda i: (0, 1, 2, 1)) if mixed and players == 4 else (lambda i: (0, 2, 0, 2)[:players])
    env = LaneEnv(players, SEED, 2, deal=games, action_cap=500, discard_order=order, caster=cast,
                  decisions={1} if mixed else ())
    seats = {0: seat, 2: seat}
    if mixed:
        seats[1] = Uniform(players, "9")
    labels, kept, decided = Labels(), {}, []

    def on_decision(request, answer):
        labels(request, answer)
        decided.append((request.index, request.step, request.seat, request.decision_id, request.part,
                        driver.targets().points[request.row]))

    driver = Driver(env, seats, on_decision=on_decision, lanes=kind,
                    keep=lambda e: kept.update({e.index: (to_json(e.record), e.outcome,
                                                          [(x.seat, x.chosen, x.priors.tobytes(), x.values.tobytes())
                                                           for x in labels.take(e.index)])}))
    driver.play()
    assert driver.lanes == kind
    return kept, sorted(decided), seat.rows


@pytest.mark.parametrize("players", [2, 4])
@pytest.mark.parametrize("order", ["seat", "random"])
def test_atomic_seats_play_the_same_games_and_labels_on_native_lanes(players, order):
    python, decided_a, rows_a = play(players, "python", order=order)
    native, decided_b, rows_b = play(players, "native", order=order)
    assert sorted(python) == sorted(native) and len(python) == 4
    assert rows_a == rows_b > 100
    for index, game in python.items():
        assert game == native[index]
    assert decided_a == decided_b


def test_atomic_and_whole_decision_seats_share_native_lanes():
    python, decided_a, _ = play(4, "python", mixed=True)
    native, decided_b, _ = play(4, "native", mixed=True)
    assert python == native and decided_a == decided_b


def test_auto_takes_native_lanes_only_where_every_atomic_policy_reads_them():
    seat = PolicySeat([Reader("8", 4)], {0: Play(members=(0,))})
    env = LaneEnv(4, SEED, 2, deal=2, action_cap=50)
    assert Driver(env, {0: seat}, keep=lambda e: None, lanes="auto").lanes == "native"
    blind = Reader("8", 4)
    blind.native_contract = None
    env = LaneEnv(4, SEED, 2, deal=2, action_cap=50)
    assert Driver(env, {0: PolicySeat([blind], {0: Play(members=(0,))})}, keep=lambda e: None,
                  lanes="auto").lanes == "python"
    with pytest.raises(ValueError, match="reads 7"):
        Driver(LaneEnv(4, SEED, 2, deal=2), {0: PolicySeat([Reader("7", 4)], {0: Play(members=(0,))})},
               keep=lambda e: None, lanes="native")
