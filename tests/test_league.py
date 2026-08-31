# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from hexset.league import OVERRIDES, nudged, parse_learner, standings  # noqa: E402
from hexset.ppo import PPOConfig  # noqa: E402
from hexset.selfplay import Episode, Outcome  # noqa: E402


def test_learner_overrides_apply_and_unknown_keys_refuse():
    base = PPOConfig()
    varied, target, gain = parse_learner("entropy=0.05,lr=6e-4,epochs=2,eps=1e-8", base)
    assert target is None
    assert gain == 0.10, "default gain, even though this seat has no controller"
    assert varied.entropy_coefficient == 0.05
    assert varied.learning_rate == 6e-4
    assert varied.epochs == 2
    assert varied.adam_eps == 1e-8
    assert varied.clip == base.clip, "untouched fields keep the base"
    assert parse_learner("", base) == (base, None, 0.10)
    with pytest.raises(SystemExit):
        parse_learner("games=256", base)


def test_target_entropy_arms_the_controller_without_touching_the_config():
    base = PPOConfig()
    config, target, gain = parse_learner("target_entropy=0.65", base)
    assert config == base and target == 0.65 and gain == 0.10


def test_gain_tunes_the_controller_and_does_nothing_without_a_target():
    base = PPOConfig()
    config, target, gain = parse_learner("target_entropy=0.62,gain=0.025", base)
    assert config == base and target == 0.62 and gain == 0.025


def test_the_entropy_controller_nudges_toward_the_target_and_clamps():
    assert nudged(0.02, entropy=0.5, target=0.65) == pytest.approx(0.022)
    assert nudged(0.02, entropy=0.8, target=0.65) == pytest.approx(0.02 / 1.1)
    assert nudged(0.10, entropy=0.1, target=0.65) == 0.10, "clamped above"
    assert nudged(0.005, entropy=0.9, target=0.65) == 0.005, "clamped below"


def test_a_damped_gain_moves_the_coefficient_less_per_iteration():
    hot = nudged(0.02, entropy=0.5, target=0.65, gain=0.10)
    damped = nudged(0.02, entropy=0.5, target=0.65, gain=0.025)
    assert 0.02 < damped < hot, "Heat 5's whole point: less ring per iteration"


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


def test_pair_boards_defaults_off_so_recorded_heats_replay():
    from hexset.league import build_parser

    args = build_parser().parse_args(["--learner", "", "--checkpoint-dir", "x"])
    assert args.pair_boards is False


def test_the_value_head_override_defaults_to_the_base_checkpoint_s_own_shape():
    """Empty means "keep the base's shape", so every heat on record replays.

    A heat's model shape has always come from the base checkpoint's stored
    args. `--value-head` is the first flag that can override it, and its
    default has to be the absence of an override rather than any particular
    shape — `"linear"` as a default would silently rebuild an `mlp` base.
    """
    from hexset.league import build_parser

    args = build_parser().parse_args(["--learner", "", "--checkpoint-dir", "x"])

    assert args.value_head == ""
    assert args.quantiles == 32


def test_the_quantile_head_is_selectable_on_a_heat():
    from hexset.league import build_parser

    args = build_parser().parse_args(
        ["--learner", "", "--checkpoint-dir", "x", "--value-head", "quantile"]
    )

    assert args.value_head == "quantile"


def test_both_head_knobs_are_frozen_into_a_league_manifest():
    """`hexset.run.init` freezes whatever the parser defines, so a heat that
    ran the quantile head cannot be reconstructed as one that did not."""
    from hexset.run import parameters

    assert {"value_head", "quantiles"} <= parameters("league")
