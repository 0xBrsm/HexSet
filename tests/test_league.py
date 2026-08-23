from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from catan.league import OVERRIDES, parse_learner, standings  # noqa: E402
from catan.ppo import PPOConfig  # noqa: E402
from catan.selfplay import Episode, Outcome  # noqa: E402


def test_learner_overrides_apply_and_unknown_keys_refuse():
    base = PPOConfig()
    varied = parse_learner("entropy=0.05,lr=6e-4,epochs=2", base)
    assert varied.entropy_coefficient == 0.05
    assert varied.learning_rate == 6e-4
    assert varied.epochs == 2
    assert varied.clip == base.clip, "untouched fields keep the base"
    assert parse_learner("", base) == base
    with pytest.raises(SystemExit):
        parse_learner("games=256", base)


def test_standings_count_wins_and_vp_by_cast():
    def game(winner, cast, points):
        return Episode(
            index=0,
            seed=0,
            players=4,
            trajectories=((), (), (), ()),
            outcome=Outcome(
                winner=winner, points=points, turns=10, actions=40, truncated=False
            ),
            cast=cast,
        )

    episodes = [
        game(0, (0, 1, 0, 1), (10, 2, 3, 4)),
        game(0, (1, 0, 1, 0), (10, 2, 3, 4)),
        game(None, (0, 1, 0, 1), (5, 5, 5, 5)),
    ]
    # Game one's winner seat 0 is learner 0's; game two's seat 0 is learner
    # 1's under the rotated cast — one win each, by cast and not by seat.
    wins, vp = standings(episodes, 2)
    assert wins == [1, 1]
    assert vp[0] == pytest.approx(vp[1]), "symmetric fixtures score level"



def test_every_override_key_names_a_real_config_field():
    fields = set(PPOConfig().__dataclass_fields__)
    assert all(name in fields for name, _ in OVERRIDES.values())
