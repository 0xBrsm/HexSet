# SPDX-License-Identifier: GPL-3.0-only
"""A seven's discards through the gym: any owing seat, in any order.

`docs/gym-design.md` §2. AEC still names exactly one agent per `step()` --
that contract is untouched -- but during `Phase.DISCARD` the agent it names is
chosen among the seats still owing rather than fixed at the lowest-indexed
one, which is a serialization no table has and the live server no longer
imposes (`hexset.game.may_act`).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from hexset.actions import Action, ActionType  # noqa: E402
from hexset.board.terrain import NUM_RESOURCES, Resource  # noqa: E402
from hexset.game import Phase, players_owing_discards, to_move  # noqa: E402
from hexset.gym.aec import HexSetAEC, agent_name  # noqa: E402
from hexset.gym.env import HexSetEnv  # noqa: E402


def _owing_position(game, *, roller: int = 1) -> None:
    """Park `game` in `Phase.DISCARD` with seats 0 and 3 each owing two cards
    out of disjoint hands, and the seat that rolled owing none.

    The same position `tests/test_actions.py::a_game_owing` builds, so what
    the engine, the server and this environment do with it can be compared
    directly. Disjoint hands are the point: a discard applied to the wrong
    seat is then not merely wrong but illegal, so a mix-up cannot pass
    silently.
    """
    game.phase = Phase.DISCARD
    game.current_player = roller
    for seat in range(4):
        game._state.hands[seat] = [0] * NUM_RESOURCES
    game._state.hands[0] = [4, 0, 0, 0, 0]  # WOOD
    game._state.hands[3] = [0, 0, 0, 0, 4]  # ORE
    game.discard_quota = [2, 0, 0, 2]


def _env_owing(*, discard_order: str = "random", seed: int = 0) -> HexSetAEC:
    env = HexSetAEC(discard_order=discard_order)
    env.reset(seed=seed)
    _owing_position(env._game)
    env.agent_selection = env._next_agent(env._game)
    return env


def _discard(env: HexSetAEC, seat: int) -> None:
    """Have `seat` discard through the environment, the way a caller that
    owns the order does: name the seat, then step it."""
    env.select_agent(agent_name(seat))
    mask = env.observe(agent_name(seat))["action_mask"]
    index = int(mask.argmax())
    assert mask[index], f"{agent_name(seat)} was offered nothing to discard"
    env.step(index)


def _position(env: HexSetAEC) -> tuple:
    game = env._game
    return (
        [hand[:] for hand in game._state.hands],
        game._state.bank[:],
        game.discard_quota[:],
        game.phase,
        game.current_player,
    )


def test_an_out_of_order_discard_round_reaches_the_same_position():
    """The whole claim, driven through `HexSetAEC` rather than the engine:
    seat 3 may clear its quota before seat 0 starts, and the round ends where
    the ascending one does."""
    ascending = _env_owing()
    for seat in (0, 0, 3, 3):
        _discard(ascending, seat)

    descending = _env_owing()
    for seat in (3, 3, 0, 0):
        _discard(descending, seat)

    interleaved = _env_owing()
    for seat in (3, 0, 3, 0):
        _discard(interleaved, seat)

    assert _position(ascending) == _position(descending) == _position(interleaved)
    assert ascending._game.phase is Phase.ROBBER
    assert ascending._game.discard_quota == [0, 0, 0, 0]


def test_the_highest_owing_seat_may_act_first():
    """The behaviour the environment used to make impossible: `step()` landed
    on `players_owing_discards(game)[0]` whatever `agent_selection` said."""
    env = _env_owing()
    _discard(env, 3)

    assert env._game.discard_quota == [2, 0, 0, 1]
    assert env._game._state.hands[3][Resource.ORE] == 3
    assert env._game._state.hands[0][Resource.WOOD] == 4  # untouched
    assert env._game.phase is Phase.DISCARD


def test_the_mask_is_the_selected_seats_own_hand():
    """`observe` answers for the agent about to act, not for whichever seat
    `to_move` would serialize to."""
    env = _env_owing()
    space = env._space

    env.select_agent(agent_name(3))
    mask = env.observe(agent_name(3))["action_mask"]
    assert [i for i, on in enumerate(mask) if on] == [
        space.index(Action(ActionType.DISCARD, Resource.ORE))
    ]

    env.select_agent(agent_name(0))
    mask = env.observe(agent_name(0))["action_mask"]
    assert [i for i, on in enumerate(mask) if on] == [
        space.index(Action(ActionType.DISCARD, Resource.WOOD))
    ]


def test_a_seat_that_is_not_selected_still_sees_an_empty_mask():
    """PettingZoo's convention survives: one active agent per `step()`."""
    env = _env_owing()
    env.select_agent(agent_name(3))
    assert not env.observe(agent_name(0))["action_mask"].any()


def test_select_agent_refuses_a_seat_the_engine_would_refuse():
    env = _env_owing()
    # Seat 1 rolled the seven and owes nothing; seat 2 owes nothing either.
    for seat in (1, 2):
        with pytest.raises(ValueError):
            env.select_agent(agent_name(seat))
    with pytest.raises(ValueError):
        env.select_agent("seat_9")


def test_select_agent_outside_a_discard_round_is_still_one_seat():
    env = HexSetAEC()
    env.reset(seed=0)
    mover = to_move(env._game)
    env.select_agent(agent_name(mover))
    for seat in range(4):
        if seat != mover:
            with pytest.raises(ValueError):
                env.select_agent(agent_name(seat))


def test_seat_order_is_the_engines_own_serialization():
    env = _env_owing(discard_order="seat")
    for _ in range(4):
        assert env.agent_selection == agent_name(to_move(env._game))
        mask = env.observe(env.agent_selection)["action_mask"]
        env.step(int(mask.argmax()))
    assert env._game.phase is Phase.ROBBER


def test_random_order_does_not_always_name_the_lowest_owing_seat():
    env = _env_owing(discard_order="random")
    named = {env._next_agent(env._game) for _ in range(50)}
    assert named == {agent_name(0), agent_name(3)}
    assert players_owing_discards(env._game) == [0, 3]


def test_random_order_is_reproducible_from_the_reset_seed():
    """A fixed seed still deals a fixed game *and* a fixed discard order: the
    order is drawn from its own stream, never from the game's rng."""
    def order(seed: int) -> list[str]:
        env = _env_owing(discard_order="random", seed=seed)
        return [env._next_agent(env._game) for _ in range(20)]

    assert order(4) == order(4)
    assert order(4) != order(5)


def test_the_board_a_seed_deals_is_unchanged_by_the_order_stream():
    """The discard rng must not be drawn from the game's, or the same seed
    would deal a different board than it did before this existed."""
    left = HexSetAEC()
    left.reset(seed=11)
    right = HexSetAEC(discard_order="seat")
    right.reset(seed=11)
    assert left._game.state(0, hidden=False).board.terrain == (
        right._game.state(0, hidden=False).board.terrain
    )
    assert left._game.state(0, hidden=False).board.tokens == (
        right._game.state(0, hidden=False).board.tokens
    )


def test_an_unknown_discard_order_is_refused_by_name():
    with pytest.raises(ValueError, match="discard order"):
        HexSetAEC(discard_order="ascending")


# --- the Gymnasium wrapper --------------------------------------------------


def test_the_learner_is_not_queued_behind_a_lower_numbered_bot_seat():
    """`HexSetEnv` hands the learner control the moment it owes cards, rather
    than after every lower-numbered bot seat has cleared its own quota."""
    env = HexSetEnv(opponents=("random", "random", "random"), learner_seat=3)
    env.reset(seed=2)
    _owing_position(env._aec._game)
    env._aec.agent_selection = agent_name(0)

    env._auto_play_opponents()

    assert env._aec.agent_selection == agent_name(3)
    # Seat 0 owes exactly what it owed: nothing was played on its behalf.
    assert env._aec._game.discard_quota == [2, 0, 0, 2]

    observation, info = env._observe_learner()
    del observation
    space = env._aec._space
    assert [i for i, on in enumerate(info["action_mask"]) if on] == [
        space.index(Action(ActionType.DISCARD, Resource.ORE))
    ]

    observation, reward, terminated, truncated, info = env.step(
        space.index(Action(ActionType.DISCARD, Resource.ORE))
    )
    assert not terminated and not truncated and reward == 0.0
    # The learner's card, out of the learner's hand.
    assert env._aec._game._state.hands[3][Resource.ORE] == 3
    assert env._aec._game._state.hands[0][Resource.WOOD] == 4
