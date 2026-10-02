# SPDX-License-Identifier: GPL-3.0-only
"""Pure planning helpers for bounded signed-trade fragmentation.

This module does not run a game or move a card. A caller supplies the
candidate pool, each row `(partner, bundle, raw, estimate)` with the
proposer's own raw score and its estimate of the partner's gain; one full
target is selected against the cutoff, then it may be represented by at most
`max_fragments` sequential legal fragments. `hexset.trading._policy` is the
state machine built on top, and it records each round only after the round
has a response and the engine has confirmed execution -- in particular, a
second identical signed fragment is admitted only after the first fragment of
this target actually executed, and an identical *refused* offer remains
excluded.

The limits arrive as arguments and have no defaults here: they are one gate's
own config, and this module does not know whose. `MAX_FRAGMENTS` is the one
number it keeps, and it is structural rather than a policy --
`fragment_partitions` enumerates two-way splits only.

Ties between equal scores break on canonical bundle order (the counts
negated, compared resource by resource), then the lower partner seat, for
determinism only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ._params import MAX_FRAGMENTS

Bundle = tuple[int, ...]


def _bundle(value: Sequence[int]) -> Bundle:
    out = tuple(value)
    if not out or any(isinstance(n, bool) or not isinstance(n, int) for n in out):
        raise ValueError("signed bundle must be a non-empty integer tuple")
    if not any(n > 0 for n in out) or not any(n < 0 for n in out):
        raise ValueError("a legal exchange needs cards moving in both directions")
    return out


def _tie(bundle: Sequence[int]) -> tuple:
    """The tie-break part of a ranking key, for `max`: canonical bundle
    order, the counts negated and compared resource by resource."""
    return (tuple(-int(n) for n in bundle),)


def side_counts(bundle: Sequence[int]) -> tuple[int, int]:
    b = _bundle(bundle)
    return sum(max(n, 0) for n in b), sum(max(-n, 0) for n in b)


def legal_fragment(bundle: Sequence[int], *, max_cards: int) -> bool:
    """Whether signed `bundle` is one legal fragment: cards move both ways, each
    side is at most `max_cards`, and one side is a single card."""
    try:
        receive, give = side_counts(bundle)
    except ValueError:
        return False
    return 0 < receive <= max_cards and 0 < give <= max_cards and min(receive, give) == 1


def _allocations(target: Bundle) -> Iterable[Bundle]:
    """Yield every vector that can be the first signed fragment."""
    choices: list[list[int]] = []
    for n in target:
        choices.append(list(range(n + 1)) if n > 0 else list(range(n, 1)))

    current = [0] * len(target)

    def walk(index: int):
        if index == len(target):
            yield tuple(current)
            return
        for value in choices[index]:
            current[index] = value
            yield from walk(index + 1)
        current[index] = 0

    yield from walk(0)


def fragment_partitions(
    target: Sequence[int], *, max_cards: int, max_fragments: int = MAX_FRAGMENTS,
) -> tuple[tuple[Bundle, ...], ...]:
    """Return deterministic one/two-fragment exact partitions of ``target``.

    Every returned fragment is a signed legal exchange and the component sum
    equals the target.  Two-fragment orders are returned in both directions so
    the caller can choose the best positive first fragment. A target that
    cannot be split into two non-empty exchanges falls back to one fragment
    only when it already fits the per-fragment limit, and so does every target
    at ``max_fragments=1``.
    """
    target_b = _bundle(target)
    receive, give = side_counts(target_b)
    if max_fragments < 2:
        return ((target_b,),) if legal_fragment(target_b, max_cards=max_cards) else ()
    if receive > 2 * max_cards or give > 2 * max_cards:
        return ()
    out: set[tuple[Bundle, Bundle]] = set()
    for first in _allocations(target_b):
        second = tuple(a - b for a, b in zip(target_b, first))
        if legal_fragment(first, max_cards=max_cards) and legal_fragment(second, max_cards=max_cards):
            out.add((first, second))
            if first != second:
                out.add((second, first))
    # Prefer two sequential exchanges whenever the full target can be split.
    # A one-fragment target is the fallback for 1-card sides or other
    # structurally indivisible bundles.
    if out:
        return tuple(sorted(out))
    if receive <= max_cards and give <= max_cards:
        return ((target_b,),)
    return ()


@dataclass(frozen=True)
class InitialSelection:
    partner: int
    bundle: Bundle
    raw_score: float
    estimate: float
    partition: tuple[Bundle, ...]


def _frozen_u64(*parts: object) -> float:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def choose_initial(
    scored, *, cutoff: float, draw_key: object,
    max_cards: int, max_fragments: int = MAX_FRAGMENTS,
    fit: Callable[[int, Bundle], int] | None = None,
) -> InitialSelection | None:
    """Gate on the highest raw parent, then rank the union fragment pool.

    `fit(partner, bundle)`, when given, is how far a row fits what its
    partner has shown it wants and will give up, and the rows that fit best
    go first: the selection is made among the parents fitting at least as
    well as the best, then at least as well as the next best, and so on down
    to every parent, which is the selection without `fit`; within a level,
    fragments rank by fit before value."""
    if fit is None:
        return _choose_initial(scored, cutoff=cutoff, draw_key=draw_key, max_cards=max_cards,
                               max_fragments=max_fragments)
    scored = list(scored)
    levels = sorted({fit(int(p), tuple(b)) for p, b, _, _ in scored}, reverse=True)
    for level in levels:
        selected = _choose_initial(
            scored, cutoff=cutoff, draw_key=draw_key, max_cards=max_cards,
            max_fragments=max_fragments, fit=fit, level=level,
        )
        if selected is not None:
            return selected
    return None


def _choose_initial(
    scored, *, cutoff: float, draw_key: object, max_cards: int, max_fragments: int,
    fit: Callable[[int, Bundle], int] | None = None, level: int = 0,
) -> InitialSelection | None:
    """`choose_initial` at one fit level: only parents fitting at least
    `level` open a plan."""
    rank = (lambda partner, bundle: fit(int(partner), tuple(bundle))) if fit else (lambda p, b: 0)
    try:
        seed, index, turn, seat = draw_key
    except (TypeError, ValueError):
        seed, index, turn, seat = draw_key, 0, 0, 0
    scored = list(scored)
    parents = []
    for ordinal, (partner, bundle, raw, estimate) in enumerate(scored):
        if estimate is None or float(estimate) <= 0:
            continue
        passed = float(raw) > cutoff or (
            float(raw) == cutoff and
            _frozen_u64("thin-followup-retention", seed, index, turn, seat, 0) < .5
        )
        if passed and rank(partner, bundle) >= level:
            parents.append((int(partner), tuple(bundle), float(raw), float(estimate), ordinal))
    if not parents:
        return None
    top = max(parents, key=lambda x: (x[2], *_tie(x[1]), -x[0]))
    # Every scored row that can open a plan, indexed by its bundle and kept in
    # `scored` order, so each partition's first fragment is one lookup rather
    # than a pass over the whole pool.
    openers: dict[tuple, list] = {}
    for partner, bundle, raw, estimate in scored:
        if (float(raw) > 0 and float(estimate) > 0
                and legal_fragment(bundle, max_cards=max_cards)):
            openers.setdefault(tuple(bundle), []).append((partner, bundle, raw, estimate))
    options = []
    for parent in parents:
        for part in fragment_partitions(
            parent[1], max_cards=max_cards, max_fragments=max_fragments
        ):
            first = part[0]
            for partner, bundle, raw, estimate in openers.get(tuple(first), ()):
                fragment_key = (rank(partner, bundle), float(raw), *_tie(bundle), -int(partner))
                parent_key = (parent[2], *_tie(parent[1]), -parent[0], tuple(part))
                options.append((fragment_key, parent_key, tuple(part), int(partner),
                                tuple(bundle), float(raw), float(estimate), parent))
    # The top raw parent is a hard gate: another parent cannot rescue it when
    # it has no positive legal first fragment.
    if not any(o[-1] == top for o in options):
        return None
    best_fragment_key = max(o[0] for o in options)
    same = [o for o in options if o[0] == best_fragment_key]
    _, _, part, partner, bundle, raw, estimate, _ = max(same, key=lambda o: o[1])
    return InitialSelection(partner, bundle, raw, estimate, part)


def choose_remainder(scored, remainder: Sequence[int], *, max_cards: int,
                     fit: Callable[[int, Bundle], int] | None = None):
    """The live re-pricing of a planned remainder: the highest-scoring scored
    candidate equal to it that still stands on its own value and is still
    positive for the recipient, or `None` to drop the plan. Under `fit` the
    partner fitting best goes first."""
    rank = (lambda partner, bundle: fit(int(partner), tuple(bundle))) if fit else (lambda p, b: 0)
    options = [
        x for x in scored
        if tuple(x[1]) == tuple(remainder) and float(x[2]) > 0 and float(x[3]) > 0
        and legal_fragment(x[1], max_cards=max_cards)
    ]
    if not options:
        return None
    return max(options, key=lambda x: (rank(x[0], x[1]), float(x[2]), *_tie(x[1]), -int(x[0])))
