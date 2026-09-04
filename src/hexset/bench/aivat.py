# SPDX-License-Identifier: GPL-3.0-only
"""AIVAT's chance-correction term, measured on games this project already ran.

The estimator, in one line. For a game whose history contains chance events
`k = 1..K`, each drawn from a distribution the *evaluator* knows:

    X_aivat = margin(z) - sum_k [ V(h_k . c_k) - E_{c ~ p_k} V(h_k . c) ]

Every bracket has conditional expectation zero given the history up to just
before its own draw, so `E[X_aivat] = E[margin(z)]` for **any** value function
`V`. That is the whole correctness argument, and it is a martingale-difference
argument on the chance filtration alone: it never mentions the number of
players, zero-sum payoffs, or whether `V` is any good. Accuracy of `V` buys
variance reduction; it cannot buy or lose unbiasedness. Burch, Schmid,
Moravčík, Morrill & Bowling, AAAI 2018 (arXiv:1612.06915).

**This is the chance term only.** No action-correction terms and no imaginary
observations: those need the acting players' strategies and an information
partition in which the opponents' reach cancels, and Pluribus is explicit that
the second half is what stopped AIVAT being applied to its human seats.

Three things about this engine make the chance term implementable at all, and
each of them is load-bearing:

*Every chance event sits behind an explicit action* (`afterstate-audit.md`), so
there is a state at which the distribution can be enumerated and a copy played
forward. `ROLL` carries 2d6; `BUY_DEV_CARD` carries the deck draw;
`MOVE_ROBBER` and `PLAY_KNIGHT` carry the steal.

*Every one of those distributions is known to an evaluator holding the full
state* -- 11 weighted dice outcomes, the remaining deck as a multiset, the
victim's hand as a multiset -- even though two of the three are hidden from the
players. AIVAT needs the *estimator's* knowledge of the chance law, not the
players'. What the players may know constrains `V`, not `p_k`.

*Nothing here perturbs the game's own random stream.* Outcomes are enumerated
on `imagine` copies fed a separate generator, so an instrumented replay of a
recorded duel is bit-identical to it -- which is how this gets measured on the
800-game cells in `runs/eval/` without playing new games.
`--check` asserts that identity against the recorded verdict.

    python -m hexset.bench.aivat network:/w/runs/lam095/latest.pt greedy \\
        --games 800 --duel-seed 20000 --workers 26 \\
        --value network:/w/runs/lam095/latest.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from hexset.actions import Action, ActionType, apply, victim_of
from hexset.arena import MAX_ACTIONS, entrant_from_name, load_checkpoint, seat_of, spawn
from hexset.board.board import Board, random_base_board
from hexset.chance import Forced, Live
from hexset.encoding import encode
from hexset.game import ROLL_ODDS, Game, imagine, is_over, roll_dice, start, to_move
from hexset.mcts import draws_hidden
from hexset.trading import publish_valuation
from hexset.victory import WINNING_POINTS, victory_points

# Which chance families the correction covers. Separable because they are not
# equally worth their compute: `roll` fires every turn and moves every seat's
# income, `deck` fires a few dozen times a game and can hand over a literal
# victory point, `steal` moves one card. A run reports how many events of each
# it actually corrected, so a term that bought nothing is visible as such.
TERMS = ("roll", "deck", "steal")

# Below this a correction is float noise from `V(o) - sum p.V`, not a signal.
_NEGLIGIBLE = 1e-12

# The four transitions that embed a draw; see `hexset.mcts.draws_hidden`.
CHANCE_ACTIONS = frozenset(
    {
        ActionType.ROLL,
        ActionType.BUY_DEV_CARD,
        ActionType.MOVE_ROBBER,
        ActionType.PLAY_KNIGHT,
    }
)


def margin_scale(players: int) -> float:
    """Terminal VP margin per unit of the value head's relative-points margin.

    `rewards.relative_points` is `(own - mean of others) / 10`, so for a seat
    `v = (P.p - T) / ((P-1).W)`. Inverting and differencing two disjoint,
    equal-sized seat groups leaves `(P-1).W/P` -- 7.5 at four players and ten
    winning points. The estimator is unbiased at any scale; getting it right is
    what makes the control variate *fit* the statistic it is subtracted from.
    """
    return WINNING_POINTS * (players - 1) / players


def _thief(game: Game) -> int:
    """Who a robber or knight action steals for. Neither ends the turn."""
    return game.current_player


def _outcome_key(game: Game, action: Action) -> object:
    """A signature of the chance outcome, read off the state after the action.

    Read back rather than snapshotted because `apply` returns nothing, and
    because the same reader then serves both the enumerated copies and the real
    game -- so the observed outcome is matched to its own enumerated twin by
    construction instead of by a parallel bookkeeping that could drift.

    true state throughout this module: AIVAT's counterfactual variance
    reduction enumerates every hidden outcome exactly, which needs the true
    hand/deck contents, not an information-set estimate of them.
    """
    thief = _thief(game)
    if action.type is ActionType.ROLL:
        return game.last_roll
    if action.type is ActionType.BUY_DEV_CARD:
        return tuple(game.state(thief, hidden=False).new_dev_cards[thief])
    return tuple(game.state(thief, hidden=False).hands[thief])


def _term_of(action: Action) -> str:
    if action.type is ActionType.ROLL:
        return "roll"
    if action.type is ActionType.BUY_DEV_CARD:
        return "deck"
    return "steal"


def chance_outcomes(
    game: Game, action: Action, rng: random.Random
) -> list[tuple[object, float, Game]]:
    """Every outcome of a chance action as `(key, probability, successor)`.

    Empty for an action that resolves nothing hidden, and empty for a
    single-outcome draw, whose correction is exactly zero and not worth a
    forward pass.

    `game` and its random stream are untouched: every successor is an
    `imagine` copy drawing from `rng`.

    Every successor is also re-seated with `game.gates`, which `imagine`
    deliberately does not copy (`hexset.game.Game.gates`): this is an
    instrumented *replay* of a real game, not a bot's hypothetical, so its
    successors must clear the same trades the real one would.

    Neither a roll's nor a steal's successor forces that event to run: it
    stays pending on `child` exactly as it stays pending on the real
    `game` at this same point in `instrumented`'s own loop (`roll_dice`/
    `move_robber_to` only *arm* the turn's first event now,
    `Game.event_pending` -- the PI amendment "publish points and the event
    trigger" -- rather than running it eagerly), so a child and the real
    game it stands in for are read at the identical point relative to that
    event -- both before it, consistently -- whether that read is
    `_outcome_key` below or a value function scoring `child` for the
    correction. Forcing it on either one and not the other was tried and
    was wrong twice over: forcing only the children left `_outcome_key`
    unable to find the real game's (still pre-event) outcome among the
    (post-event) enumerated ones, and forcing the real game too, right
    after `apply`, consumed its `event_pending` before that seat's own
    `Game.publish_due` check -- reached one loop iteration later -- ever
    saw it, so the seat silently never got to publish that turn at all.
    `Game.state`'s own docstring is what actually keeps a value function's
    `hidden=False` read (`child.state(0, hidden=False)`, scoring, not a
    decision) from firing anything: only `hidden=True` is a trigger.
    """
    if action.type is ActionType.ROLL:
        out = []
        for roll, probability in ROLL_ODDS:
            child = imagine(game, rng, randomize_deck=False)
            child.gates = game.gates
            roll_dice(child, roll)
            out.append((roll, probability, child))
        return out

    if not draws_hidden(game, action):
        return []

    if action.type is ActionType.BUY_DEV_CARD:
        deck = game.state(0, hidden=False).deck
        remaining = Counter(deck)
        if len(remaining) < 2:
            return []
        out = []
        for card, count in sorted(remaining.items()):
            child = imagine(game, rng, randomize_deck=False)
            child.gates = game.gates
            # `devcards.buy` pops the end of the deck, so moving the forced card
            # there and calling the real `apply` keeps the rules -- the payment
            # and the victory-point card's win check -- exactly as played.
            child_deck = child.state(0, hidden=False).deck
            child_deck.remove(card)
            child_deck.append(card)
            apply(child, action)
            out.append((_outcome_key(child, action), count / len(deck), child))
        return out

    # A steal. `robber.steal` takes one card uniformly from the victim's *cards*,
    # so the outcome distribution is the victim's hand normalised.
    victim = victim_of(game, action.b)
    assert victim is not None  # `draws_hidden` already ruled the no-victim case out
    hand = game.state(victim, hidden=False).hands[victim]
    total = sum(hand)
    if total == 0 or sum(1 for n in hand if n) < 2:
        return []

    # Each outcome is played out in full rather than patched afterwards.
    # Patching the played child's two hand entries was exact while a steal
    # moved one card and nothing else; it is not any more, because the steal
    # resolves the robber and so opens the main phase (`move_robber_to`
    # only *arms* that turn's first trade event, per this function's own
    # docstring -- it stays pending on `child`, unrun, matching the real
    # game at this same point). `chance.Forced` stands in for the child's
    # chance source exactly as `hexset.bots.heximax` does it: returning the
    # wanted resource directly makes the draw deterministic, and nothing
    # else on the path consults `child.chance`.
    out = []
    for resource, count in enumerate(hand):
        if not count:
            continue
        child = imagine(game, rng, randomize_deck=False)
        child.gates = game.gates
        child.chance = Forced(resource)
        apply(child, action)
        child.chance = Live(rng)
        out.append((_outcome_key(child, action), count / total, child))
    return out


def _moved(before: list[int], after: list[int]) -> int:
    for index, (old, new) in enumerate(zip(before, after)):
        if new > old:
            return index
    raise AssertionError("nothing moved: this action drew no card")


class Valuer:
    """`V` as the terminal-VP margin between two seat groups, from one seat's view.

    The perspective matters for how much variance comes off, and not at all for
    whether the estimate is unbiased. `hexset.encoding` is information-set
    correct, so a value read from seat `s` cannot resolve a chance outcome that
    seat `s` may not see: an opponent's dev-card draw raises that opponent's card
    *count* whichever card it was, so all of its successors encode identically
    and its correction is exactly zero. The dice are the opposite case -- income
    lands in own hand exactly and in every opponent's counts -- which is why the
    `roll` term is the one worth buying first.
    """

    def __init__(
        self,
        path: str,
        board: Board,
        *,
        perspective: int,
        ours: list[int],
        theirs: list[int],
        players: int,
    ) -> None:
        self.loaded = load_checkpoint(path, board.topology, "cpu")
        self.perspective = perspective
        self.scale = margin_scale(players)
        self.ours = [(s - perspective) % players for s in ours]
        self.theirs = [(s - perspective) % players for s in theirs]

    def margins(self, games: list[Game]) -> np.ndarray:
        value = self.loaded.policy.values(
            [encode(game, self.perspective) for game in games]
        )
        return self.scale * (
            value[:, self.ours].mean(axis=1) - value[:, self.theirs].mean(axis=1)
        )


class StubValuer:
    """A value function chosen to be wrong, for the unbiasedness test.

    Unbiasedness must not depend on `V`, so the test that matters is run with a
    `V` that is deterministic, large, and unrelated to the position. A stub that
    returned zero would pass the test while checking nothing.
    """

    def __init__(
        self, path: str = "", board: object = None, *, scale: float = 20.0, **_: object
    ) -> None:
        self.scale = scale

    def margins(self, games: list[Game]) -> np.ndarray:
        return np.array(
            [
                self.scale
                * (
                    hash(
                        (
                            # true state: a deterministic stub, unrelated to
                            # any bot's information set by design.
                            tuple(map(tuple, game.state(0, hidden=False).hands)),
                            -1 if game.last_roll is None else game.last_roll,
                            len(game.state(0, hidden=False).deck),
                            int(game.phase),
                        )
                    )
                    % 1000
                    / 500.0
                    - 1.0
                )
                for game in games
            ],
            dtype=np.float64,
        )


def instrumented(
    entrants: tuple,
    index: int,
    seed: int,
    *,
    action_cap: int = MAX_ACTIONS,
    antithetic: bool = True,
    value: str = "",
    terms: tuple[str, ...] = TERMS,
    detail: bool = False,
) -> dict:
    """Play game `index` of a tournament, accumulating the chance correction.

    A deliberate line-for-line twin of `arena._play_one` -- same board stream,
    same rotation, same per-entrant streams, same `{seed}:{board_index}:game`
    generator -- so a recorded duel can be re-derived rather than re-sampled.
    The instrumentation only ever reads copies, so the sequence of draws the
    real game takes is the sequence it took when the verdict was written.
    """
    seats = len(entrants)
    if antithetic:
        pair, half = divmod(index, 2)
        board_index = pair
        rotation = pair + half * (seats // 2)
    else:
        board_index = rotation = index
    board = random_base_board(random.Random(f"{seed}:{board_index}:board"))
    seats_taken = [seat_of(e, rotation, seats) for e in range(seats)]

    lineup = [None] * seats
    for e, entrant in enumerate(entrants):
        lineup[seats_taken[e]] = spawn(
            entrant, board, random.Random(f"{seed}:{board_index}:{e}")
        )

    ours = [seats_taken[e] for e in range(seats // 2)]
    theirs = [seats_taken[e] for e in range(seats // 2, seats)]
    factory = StubValuer if value == "stub" else Valuer
    valuer = factory(
        value,
        board,
        perspective=ours[0],
        ours=ours,
        theirs=theirs,
        players=seats,
    )

    game = start(board, seats, random.Random(f"{seed}:{board_index}:game"))
    # Seated as `arena.play` seats them, so the engine's one trade event a
    # turn asks the same bots the same questions here as it does there --
    # without this the replay would be a no-trade game against a trading one.
    game.gates = tuple(lineup)
    # Its own stream, so the enumeration cannot be blamed for a divergence and
    # a rerun at a different `--terms` still plays the same game.
    aux = random.Random(f"aivat:{seed}:{board_index}")

    correction = 0.0
    events = {term: 0 for term in TERMS}
    priced = {term: 0 for term in TERMS}
    per_event: list[dict] = []
    actions = 0
    while not is_over(game) and actions < action_cap:
        seat = to_move(game)
        bot = lineup[seat]
        # A line-for-line twin of `arena.play`: the acting seat publishes
        # once a turn, when the engine says it is due (`Game.publish_due`),
        # not after every action, so this replay's vectors match a real
        # duel's bit-for-bit rather than staying at whatever they were when
        # the game started.
        if game.publish_due(seat):
            publish_valuation(game, seat, bot)
        action = bot.choose(game)
        chance = action.type in CHANCE_ACTIONS
        term = _term_of(action) if chance else ""
        outcomes = chance_outcomes(game, action, aux) if term in terms else []
        apply(game, action)
        actions += 1
        if len(outcomes) > 1:
            keys, probabilities, children = zip(*outcomes)
            values = valuer.margins(list(children))
            observed = keys.index(_outcome_key(game, action))
            delta = float(values[observed] - np.dot(probabilities, values))
            correction += delta
            events[term] += 1
            # "Priced" means the value function could actually tell the
            # outcomes apart. An information-set-correct `V` cannot: an
            # opponent's dev-card draw looks the same whichever card it was, so
            # its correction is zero and the event bought nothing.
            priced[term] += abs(delta) > _NEGLIGIBLE
            if detail:
                per_event.append(
                    {
                        "term": term,
                        "delta": delta,
                        "probabilities": list(probabilities),
                        "values": [float(v) for v in values],
                    }
                )

    # true state: the verdict's victory points include hidden VP dev cards.
    points = tuple(
        victory_points(game.state(seats_taken[e], hidden=False), seats_taken[e])
        for e in range(seats)
    )
    half = seats // 2
    margin = sum(points[:half]) / half - sum(points[half:]) / half
    winner = None if game.won_by is None else seats_taken.index(game.won_by)
    return {
        "index": index,
        "board": board_index,
        "points": points,
        "margin": margin,
        "correction": correction,
        "winner": winner,
        "won": winner is not None and winner < half,
        "actions": actions,
        "events": events,
        "priced": priced,
        "per_event": per_event,
    }


def _job(argument):
    entrants, index, seed, action_cap, antithetic, value, terms, detail = argument
    return instrumented(
        entrants,
        index,
        seed,
        action_cap=action_cap,
        antithetic=antithetic,
        value=value,
        terms=terms,
        detail=detail,
    )


def replay(
    entrants: tuple,
    games: int,
    *,
    seed: int,
    value: str,
    terms: tuple[str, ...] = TERMS,
    workers: int = 1,
    antithetic: bool = True,
    action_cap: int = MAX_ACTIONS,
    detail: bool = False,
) -> list[dict]:
    jobs = [
        (entrants, i, seed, action_cap, antithetic, value, terms, detail)
        for i in range(games)
    ]
    if workers > 1:
        with Pool(workers) as pool:
            return pool.map(_job, jobs, chunksize=1)
    return [_job(job) for job in jobs]


def _spread(sample: list[float]) -> float:
    return statistics.stdev(sample) if len(sample) > 1 else 0.0


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or not _spread(left) or not _spread(right):
        return 0.0
    return statistics.correlation(left, right)


def _fisher(rho: float, n: int) -> tuple[float, float]:
    """A 95% interval on a correlation, through the z transform."""
    if n < 4 or abs(rho) >= 1.0:
        return (rho, rho)
    z = 0.5 * math.log((1 + rho) / (1 - rho))
    half = 1.959964 / (n - 3) ** 0.5
    return tuple(math.tanh(z + sign * half) for sign in (-1, 1))


def _bootstrap_ratio(
    plain: list[float], corrected: list[float], *, draws: int = 4000, seed: int = 7
) -> tuple[float, float]:
    """Percentile interval on `SD(corrected) / SD(plain)`, resampled in pairs.

    Paired, because the two estimators are computed on the *same* games: a ratio
    resampled independently would carry variance that the comparison does not
    have.
    """
    rng = random.Random(seed)
    n = len(plain)
    ratios = []
    for _ in range(draws):
        picks = [rng.randrange(n) for _ in range(n)]
        a = [plain[i] for i in picks]
        b = [corrected[i] for i in picks]
        base = _spread(a)
        if base > 0:
            ratios.append(_spread(b) / base)
    ratios.sort()
    if not ratios:
        return 0.0, 0.0
    return ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))]


def summarise(rows: list[dict], *, target: int = 6900) -> dict:
    """Both estimators, per game and per antithetic board pair, plus the bill."""
    by_board: dict[int, list[dict]] = {}
    for row in rows:
        by_board.setdefault(row["board"], []).append(row)
    pairs = [v for v in by_board.values() if len(v) == 2]

    game_plain = [row["margin"] for row in rows]
    game_aivat = [row["margin"] - row["correction"] for row in rows]
    board_plain = [statistics.fmean(r["margin"] for r in p) for p in pairs]
    board_aivat = [
        statistics.fmean(r["margin"] - r["correction"] for r in p) for p in pairs
    ]

    corrections = [statistics.fmean(r["correction"] for r in p) for p in pairs]
    bias = statistics.fmean(corrections) if corrections else 0.0
    bias_se = _spread(corrections) / len(corrections) ** 0.5 if len(corrections) > 1 else 0.0

    sd_plain, sd_aivat = _spread(board_plain), _spread(board_aivat)
    ratio = sd_aivat / sd_plain if sd_plain else 0.0
    low, high = _bootstrap_ratio(board_plain, board_aivat)

    # The diagnostic that separates "the head has nothing to say about a chance
    # outcome" from "it has something to say and AIVAT's unit coefficient is the
    # wrong amount of it". AIVAT subtracts the correction at beta = 1, which is
    # only variance-optimal for a calibrated `V`; the best any coefficient could
    # do is `sqrt(1 - rho^2)`, and that is the ceiling of what this control
    # variate holds. It is also the direct analogue of Gate A's rho_v readout,
    # measured here per chance event instead of per game.
    rho = _correlation(board_plain, corrections)
    variance = _spread(corrections) ** 2
    beta = (
        rho * _spread(board_plain) * _spread(corrections) / variance if variance else 0.0
    )
    ceiling = (1.0 - rho**2) ** 0.5

    # The same three quantities before the antithetic average, because the
    # pairing shares the whole chance stream between a board's two halves and
    # therefore removes part of what the chance term is trying to remove. If
    # `rho` is much below `rho_game`, the existing machinery got there first.
    per_game = [row["correction"] for row in rows]
    rho_game = _correlation(game_plain, per_game)
    # How much of a board's variance the pairing itself takes out. Two games of
    # one board are not independent draws: `Var(mean) = Var(x)(1 + r)/2`, so `r`
    # is what the pairing bought over and above having played two games.
    within = (
        2.0 * _spread(board_plain) ** 2 / _spread(game_plain) ** 2 - 1.0
        if _spread(game_plain)
        else 0.0
    )

    events = {term: sum(row["events"][term] for row in rows) for term in TERMS}
    priced = {term: sum(row["priced"][term] for row in rows) for term in TERMS}
    return {
        "games": len(rows),
        "boards": len(pairs),
        "wins": sum(1 for row in rows if row["won"]),
        "win_rate": sum(1 for row in rows if row["won"]) / len(rows) if rows else 0.0,
        "unfinished": sum(1 for row in rows if row["winner"] is None),
        "events": events,
        "events_per_game": {t: events[t] / len(rows) for t in TERMS} if rows else {},
        "priced": priced,
        # The unbiasedness readout on real games: the two estimators differ by
        # the mean correction and nothing else, so this is the whole test.
        "mean_correction": bias,
        "mean_correction_se": bias_se,
        "mean_correction_t": bias / bias_se if bias_se else 0.0,
        "paired_vp": statistics.fmean(board_plain) if board_plain else 0.0,
        "paired_vp_aivat": statistics.fmean(board_aivat) if board_aivat else 0.0,
        "game_sd_plain": _spread(game_plain),
        "game_sd_aivat": _spread(game_aivat),
        "board_sd_plain": sd_plain,
        "board_sd_aivat": sd_aivat,
        "sd_ratio": ratio,
        "sd_ratio_low": low,
        "sd_ratio_high": high,
        "sd_reduction": 1.0 - ratio,
        "sd_reduction_low": 1.0 - high,
        "sd_reduction_high": 1.0 - low,
        "correction_sd": _spread(corrections),
        "rho": rho,
        "rho_low": _fisher(rho, len(corrections))[0],
        "rho_high": _fisher(rho, len(corrections))[1],
        "beta_optimal": beta,
        "sd_ratio_optimal": ceiling,
        "sd_reduction_optimal": 1.0 - ceiling,
        "target_games_optimal": target * ceiling**2,
        "rho_game": rho_game,
        "sd_reduction_optimal_game": 1.0 - (1.0 - rho_game**2) ** 0.5,
        "within_pair_correlation": within,
        "antithetic_sd_gain": 1.0 - (1.0 + within) ** 0.5,
        # What the whole exercise is for. Variance scales the sample linearly,
        # so the bill moves with the *square* of the SD ratio.
        "target_games": target,
        "target_games_aivat": target * ratio**2,
        "target_games_aivat_low": target * low**2,
        "target_games_aivat_high": target * high**2,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", help="entrant spec for the two seats being scored")
    p.add_argument("b", help="entrant spec for the two reference seats")
    p.add_argument("--games", type=int, default=800)
    p.add_argument("--duel-seed", type=int, default=20_000)
    p.add_argument("--workers", type=int, default=26)
    p.add_argument(
        "--value",
        default=None,
        help="checkpoint spec for V, e.g. /w/runs/lam095/latest.pt, or `stub` "
        "for the deliberately-wrong function the unbiasedness test uses. "
        "Defaults to side A's own checkpoint when A is a network entrant",
    )
    p.add_argument(
        "--terms",
        default=",".join(TERMS),
        help=f"which chance families to correct, from {TERMS}",
    )
    p.add_argument("--target", type=int, default=6900, help="games per cell to price")
    p.add_argument(
        "--check",
        default=None,
        help="a runs/eval verdict to assert the replay reproduces: same games, "
        "or the SD comparison is not on the recorded cell",
    )
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    entrants = tuple(
        entrant_from_name(name).renamed(f"{name}#{i}")
        for i, name in enumerate([args.a, args.a, args.b, args.b])
    )
    value = args.value
    if value is None:
        if not args.a.startswith("network:"):
            p.error("--value is required unless side A is a network entrant")
        value = args.a[len("network:") :]
    terms = tuple(t for t in args.terms.split(",") if t)
    for term in terms:
        if term not in TERMS:
            p.error(f"unknown term {term!r}: {TERMS}")

    started = time.monotonic()
    rows = replay(
        entrants,
        args.games,
        seed=args.duel_seed,
        value=value,
        terms=terms,
        workers=args.workers,
    )
    result = summarise(rows, target=args.target)
    result.update(
        a=args.a,
        b=args.b,
        value=value,
        terms=list(terms),
        duel_seed=args.duel_seed,
        workers=args.workers,
        seconds=time.monotonic() - started,
    )

    if args.check:
        recorded = [json.loads(line) for line in Path(args.check).open()]
        match = [
            row
            for row in recorded
            if row["games"] == args.games
            and row["duel_seed"] == args.duel_seed
            # On the specs, not on the labels: a label is whatever the caller
            # typed and several cells in one file share `--games` and `--seed`.
            and row["a_path"] == args.a
            and row["b_path"] == args.b
        ]
        if not match:
            p.error(f"no verdict in {args.check} for {args.a} vs {args.b}")
        verdict = match[-1]
        result["checked_against"] = args.check
        result["recorded_paired_vp"] = verdict["paired_vp"]
        result["recorded_win_rate"] = verdict["win_rate"]
        result["reproduces"] = (
            abs(verdict["paired_vp"] - result["paired_vp"]) < 1e-9
            and abs(verdict["win_rate"] - result["win_rate"]) < 1e-9
        )

    print(json.dumps(result, indent=1))
    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        print(f"\nappended to {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
