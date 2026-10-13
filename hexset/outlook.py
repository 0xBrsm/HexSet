# SPDX-License-Identifier: GPL-3.0-only
"""Every seat's points over the coming round, counted: what each seat *can* score.

The round is the mover's turn under way, then each other seat's next turn in
order. For each seat, `Outlook` counts by arithmetic and dice odds alone, with
no model of how anybody plays:

- the hand it holds at its turn: its hand now plus what every roll before and
  on that turn produces for it (two dice; the robber where it stands). A seven
  produces nothing: a seat holding more than the discard limit discards half,
  from its largest piles; on its own roll it steals one card from the seat
  holding most, drawn by that hand's make-up;
- with that hand, at its bank and port rates and out of the bank's stock, the
  most points it can pay for: cities on its settlements, settlements on free
  spots at its road ends or `SPOT_ROADS` new roads further, the roads that take
  Longest Road (`roads.roads_to_take`, within `LONGEST_ROADS`), and up to
  `CARD_BUYS` development cards, each worth the chance it is a victory point
  card as the reader sees the unseen cards;
- at most one development card played (none bought this turn; on the mover's
  turn none if it has played one): a knight steals a card and takes Largest
  Army where it would, Road Building lays two roads, Year of Plenty takes two
  cards, Monopoly takes every other seat's cards of one kind;
- each event's chance on its own -- a settlement, a city, a card bought, Longest
  Road taken, Largest Army taken -- and each award holder's chance that another
  seat can take its award.

A seat taking an award takes 2 from its holder. Seats are read independently:
one seat's turn does not change another's. Player trades are left out, and so is
every choice another seat makes about this one (who it robs, where the robber
goes).

`Outlook.can` is an upper bound on what seats do: it assumes every seat makes
the best purchase its cards allow. `Outlook.will` scales each event by `Rates`,
the share of the time a seat that can do it does.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from .cards import DECK_COMPOSITION, DevCard
from .devcards import dev_count
from .economy import COSTS, Purchase, trade_ratios
from .roads import ROADS_TO_TAKE_CAP, roads_to_take
from .state import (
    MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, NO_OWNER, GameState, city_count, production, road_count,
    settlement_count, settlement_placeable,
)
from .victory import LARGEST_ARMY_VP, LONGEST_ROAD_VP, MIN_LARGEST_ARMY, victory_points
from .view import View

__all__ = [
    "CARD_BUYS",
    "EVENTS",
    "LONGEST_ROADS",
    "Outlook",
    "Rates",
    "SPOT_ROADS",
    "outlook",
    "outlook_of",
    "standing",
]

SPOT_ROADS = 2       # the most new roads a settlement spot is looked for at
LONGEST_ROADS = 3    # the most new roads Longest Road is taken with in a turn
CARD_BUYS = 2        # the most development cards bought in a turn
FREE_ROADS = 2       # Road Building
FREE_CARDS = 2       # Year of Plenty
EVENTS = ("settlement", "city", "card", "longest_road", "largest_army")
_ROLLS = tuple((r, (6 - abs(r - 7)) / 36) for r in range(2, 13))
_CITY, _SETTLEMENT, _CARD, _ROAD = (COSTS[Purchase.CITY], COSTS[Purchase.SETTLEMENT],
                                    COSTS[Purchase.DEV_CARD], COSTS[Purchase.ROAD])
_NR = len(_ROAD)


@dataclass(frozen=True)
class Rates:
    """The share of the time a seat that can do each event does it, and the
    share of the time an award is lost where another seat can take it."""

    settlement: float
    city: float
    card: float
    longest_road: float
    largest_army: float
    lose_longest_road: float
    lose_largest_army: float


@dataclass(frozen=True)
class Outlook:
    """The round's outlook, board order. `can[s]`: seat `s`'s expected points
    change, each seat making its best purchase; `events[s]`: its chance of
    being able to do each of `EVENTS`; `lose[s]`: its chance that another seat
    can take its (Longest Road, Largest Army); `card_vp`: a bought card's
    chance of being a victory point card."""

    can: tuple[float, ...]
    events: tuple[tuple[float, ...], ...]
    lose: tuple[tuple[float, float], ...]
    card_vp: float

    def will(self, rates: Rates) -> tuple[float, ...]:
        """Each seat's expected points change with every event scaled by `rates`."""
        gain = (rates.settlement, rates.city, rates.card * self.card_vp,
                rates.longest_road * LONGEST_ROAD_VP, rates.largest_army * LARGEST_ARMY_VP)
        return tuple(
            sum(g * p for g, p in zip(gain, ev))
            - LONGEST_ROAD_VP * rates.lose_longest_road * lo[0]
            - LARGEST_ARMY_VP * rates.lose_largest_army * lo[1]
            for ev, lo in zip(self.events, self.lose)
        )


# -- what the board allows each seat, the same in every world ----------------


def _spot_roads(st: GameState, seat: int) -> tuple[int, ...]:
    """New roads to each free spot `seat` could settle, cheapest first: 0 at its
    road ends, else up to `SPOT_ROADS` over unowned edges, never through another
    seat's building."""
    topo = st.board.topology
    own = {v for e, o in enumerate(st.edge_owner) if o == seat for v in topo.edges[e]}
    own |= {v for v, o in enumerate(st.vertex_owner) if o == seat}
    dist = {v: 0 for v in own}
    frontier = [v for v in own if st.vertex_owner[v] in (NO_OWNER, seat)]
    for d in range(1, SPOT_ROADS + 1):
        nxt = []
        for v in frontier:
            for e in topo.vertex_edges[v]:
                if st.edge_owner[e] != NO_OWNER:
                    continue
                a, b = topo.edges[e]
                u = b if a == v else a
                if u in dist:
                    continue
                dist[u] = d
                if st.vertex_owner[u] == NO_OWNER:
                    nxt.append(u)
        frontier = nxt
    return tuple(sorted(
        d for v, d in dist.items()
        if (settlement_placeable(st, seat, v) if d == 0 else settlement_placeable(st, seat, v, connected=False))
    ))


@dataclass(frozen=True)
class _Seat:
    """What the board allows one seat: its yield per roll, rates, the bundles it
    could buy (`_bundles`) with and without free roads, and each event's cost."""

    prods: tuple
    rates: np.ndarray
    bundles: tuple
    bundles_rb: tuple
    event_costs: tuple   # (event indices, their costs, their costs with Road Building's roads)


@lru_cache(maxsize=1 << 12)
def _bundles(cities_left: int, settled: int, settles_left: int, spots: tuple, lr: int, roads_left: int,
             card_vp: float, deck: int, free: int) -> tuple:
    """Every purchase set, best value first, as arrays: (values, costs, takes Longest Road)."""
    out = []
    for t in ((0, 1) if lr else (0,)):
        for s in range(min(settles_left, len(spots)) + 1):
            roads = (lr if t else 0) + sum(spots[:s])
            if roads > roads_left:
                break
            paid = max(0, roads - free)
            for c in range(min(cities_left, settled + s) + 1):
                for d in range(min(CARD_BUYS, deck) + 1):
                    v = c + s + LONGEST_ROAD_VP * t + card_vp * d
                    if v > 0:
                        out.append((v, tuple(c * a + s * b + paid * r + d * x
                                             for a, b, r, x in zip(_CITY, _SETTLEMENT, _ROAD, _CARD)), t))
    out.sort(key=lambda b: -b[0])
    return (np.array([b[0] for b in out], dtype=float), np.array([b[1] for b in out], dtype=np.int64).reshape(-1, _NR),
            np.array([b[2] for b in out], dtype=float))


def _seat(st: GameState, seat: int, prods_all: dict, card_vp: float) -> _Seat:
    need = roads_to_take(st, seat)
    lr = need if 0 < need <= LONGEST_ROADS and need < ROADS_TO_TAKE_CAP else 0
    spots = _spot_roads(st, seat)
    settled, cities = settlement_count(st, seat), city_count(st, seat)
    key = (MAX_CITIES - cities, settled, MAX_SETTLEMENTS - settled, spots, lr, MAX_ROADS - road_count(st, seat),
           card_vp, len(st.deck))
    costs = []
    if spots and settled < MAX_SETTLEMENTS:
        costs.append((0, tuple(a + spots[0] * r for a, r in zip(_SETTLEMENT, _ROAD)), spots[0]))
    if settled and cities < MAX_CITIES:
        costs.append((1, _CITY, 0))
    if len(st.deck):
        costs.append((2, _CARD, 0))
    if lr:
        costs.append((3, tuple(lr * r for r in _ROAD), lr))
    prods = tuple((r, q, tuple(prods_all[r][seat]) if r != 7 and any(prods_all[r][seat]) else None)
                  for r, q in _ROLLS)
    index = np.array([e for e, _, _ in costs], dtype=np.int64)
    plain = np.array([c for _, c, _ in costs], dtype=np.int64).reshape(-1, _NR)
    paved = np.array([tuple(x - min(roads, FREE_ROADS) * r for x, r in zip(c, _ROAD)) for _, c, roads in costs],
                     dtype=np.int64).reshape(-1, _NR)
    return _Seat(prods, np.array(trade_ratios(st, seat), dtype=np.int64), _bundles(*key, 0),
                 _bundles(*key, FREE_ROADS), (index, plain, paved))


# -- one seat's turn in one world ----------------------------------------------


def _pays(hands: np.ndarray, costs: np.ndarray, rates: np.ndarray, bank: np.ndarray, wild: int = 0) -> np.ndarray:
    """`(hands, costs)`: whether each hand pays each cost at `rates` out of `bank`, `wild` free cards of its
    choice besides. A shortfall within `wild` pays; otherwise the bank must stock what is short, and the
    surplus, traded at `rates`, must cover what `wild` does not."""
    diff = hands[:, None, :] - costs[None, :, :]
    lack = np.maximum(-diff, 0)
    short = lack.sum(-1)
    surplus = (np.maximum(diff, 0) // rates).sum(-1)
    return (short <= wild) | ((lack <= bank).all(-1) & (short - wild <= surplus))


def _best(hands: np.ndarray, rates, bank, bundles, wild: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Each hand's best purchase set: (its value, whether it takes Longest Road); 0 where none pays."""
    values, costs, takes = bundles
    if not len(values):
        z = np.zeros(len(hands))
        return z, z
    ok = _pays(hands, costs, rates, bank, wild)
    first = ok.argmax(1)
    some = ok.any(1)
    return np.where(some, values[first], 0.0), np.where(some, takes[first], 0.0)


def _discard(hand: tuple, limit: int) -> tuple:
    n = sum(hand)
    if n <= limit:
        return hand
    h = list(hand)
    for _ in range(n // 2):
        r = max(range(_NR), key=lambda i: (h[i], -i))
        h[r] -= 1
    return tuple(h)


def _plus(hand: tuple, r: int, k: int = 1) -> tuple:
    return tuple(x + k if i == r else x for i, x in enumerate(hand))


@lru_cache(maxsize=1 << 14)
def _hands(hand: tuple, prods: tuple, rolls: int, own_last: bool, victim: tuple, limit: int) -> tuple:
    """The distribution of `hand` after `rolls` rolls, as ((hand, chance), ...)."""
    dist = {hand: 1.0}
    for j in range(rolls):
        own = own_last and j == rolls - 1
        nxt: dict = {}
        for h, p in dist.items():
            for r, q, y in prods:
                if r == 7:
                    kept = _discard(h, limit)
                    if own and victim:
                        for c, pc in victim:
                            k = _plus(kept, c)
                            nxt[k] = nxt.get(k, 0.0) + p * q * pc
                    else:
                        nxt[kept] = nxt.get(kept, 0.0) + p * q
                    continue
                k = tuple(a + b for a, b in zip(h, y)) if y else h
                nxt[k] = nxt.get(k, 0.0) + p * q
        dist = nxt
    return tuple(dist.items())


def _victim(st: GameState, seat: int) -> tuple:
    """The make-up of the hand a steal by `seat` draws from: the other seat holding most."""
    others = [s for s in range(st.num_players) if s != seat]
    vic = max(others, key=lambda s: (sum(st.hands[s]), -s))
    total = sum(st.hands[vic])
    return tuple((c, st.hands[vic][c] / total) for c in range(_NR) if st.hands[vic][c]) if total else ()


def _turn(st: GameState, seat: int, hand: tuple, info: _Seat, rolls: int, rolls_now: bool, matured: bool,
          card_played: bool, limit: int) -> tuple[float, float, float, list[float]]:
    """One seat's turn in world `st` holding `hand`: (expected best points, chance the best takes Longest
    Road, chance it takes Largest Army, each event's chance). Every hand it may hold is read at once."""
    n = st.num_players
    cards = [st.dev_cards[seat][k] + (st.new_dev_cards[seat][k] if matured else 0) for k in range(5)]
    if card_played and not matured:
        cards = [0] * 5
    victim = _victim(st, seat)
    pool = [sum(st.hands[s][r] for s in range(n) if s != seat) for r in range(_NR)]
    holder = st.largest_army_holder
    kp = st.knights_played[seat] + 1
    la = bool(cards[DevCard.KNIGHT]) and holder != seat and kp >= MIN_LARGEST_ARMY and (
        holder == NO_OWNER or kp > st.knights_played[holder])
    rates, bank = info.rates, np.array(st.bank, dtype=np.int64)
    index, plain, paved = info.event_costs
    dist = _hands(hand, info.prods, rolls, rolls_now, victim, limit)
    H = np.array([h for h, _ in dist], dtype=np.int64).reshape(-1, _NR)
    P = np.array([p for _, p in dist])
    m = len(H)

    def afford(hands, wild=0, costs=plain) -> np.ndarray:
        out = np.zeros((m, len(EVENTS)))
        if len(index):
            out[:, index] = _pays(hands, costs, rates, bank, wild)
        return out

    # Each option: (values, takes LR, takes LA, events), one row per hand.
    v, t = _best(H, rates, bank, info.bundles)
    options = [(v, t, 0.0, afford(H))]
    if cards[DevCard.ROAD_BUILDING]:
        v, t = _best(H, rates, bank, info.bundles_rb)
        options.append((v, t, 0.0, afford(H, costs=paved)))
    if cards[DevCard.YEAR_OF_PLENTY]:
        v, t = _best(H, rates, bank, info.bundles, FREE_CARDS)
        options.append((v, t, 0.0, afford(H, FREE_CARDS)))
    if cards[DevCard.MONOPOLY]:
        for r in range(_NR):
            if pool[r]:
                Hm = H.copy()
                Hm[:, r] += pool[r]
                v, t = _best(Hm, rates, bank, info.bundles)
                options.append((v, t, 0.0, afford(Hm)))
    if cards[DevCard.KNIGHT]:
        vk, tk, ak = np.zeros(m), np.zeros(m), np.zeros((m, len(EVENTS)))
        for c, pc in (victim or ((None, 1.0),)):
            Hk = H if c is None else H + np.eye(_NR, dtype=np.int64)[c]
            v, t = _best(Hk, rates, bank, info.bundles)
            vk += pc * v
            tk += pc * t
            ak += pc * afford(Hk)
        if la:
            ak[:, EVENTS.index("largest_army")] = 1.0
        options.append((vk + (LARGEST_ARMY_VP if la else 0), tk, 1.0 if la else 0.0, ak))
    values = np.stack([o[0] for o in options])
    pick = values.argmax(0)
    rows = np.arange(m)
    best = values[pick, rows]
    takes = np.stack([o[1] for o in options])[pick, rows]
    army = np.array([o[2] for o in options])[pick]
    events = np.stack([o[3] for o in options]).max(0)
    return float(P @ best), float(P @ takes), float(P @ army), (P @ events).tolist()


# -- the round ------------------------------------------------------------------


def _card_vp(st: GameState, reader: int) -> float:
    unseen = len(st.deck) + sum(dev_count(st, s) for s in range(st.num_players) if s != reader)
    vp = (DECK_COMPOSITION[DevCard.VICTORY_POINT] - st.dev_cards[reader][DevCard.VICTORY_POINT]
          - st.new_dev_cards[reader][DevCard.VICTORY_POINT])
    return vp / unseen if unseen else 0.0


class _Round:
    """The board's part of a round from one position, read once and shared by every world: the turns in
    order and each seat's `_Seat`."""

    def __init__(self, st: GameState, mover: int, rolled: bool, card_played: bool, card_vp: float,
                 through: int | None = None) -> None:
        n = st.num_players
        prods_all = {r: (production(st, r) if r != 7 else None) for r, _ in _ROLLS}
        self.seats = [_seat(st, s, prods_all, card_vp) for s in range(n)]
        first = 0 if rolled else 1
        # (seat, rolls before its purchases, whether it rolls on this turn, whether its new cards matured)
        if through is None:
            self.turns = [(mover, first, first == 1, False)] + [
                ((mover + i) % n, first + i, True, True) for i in range(1, n)]
        else:
            # Through `through`'s next turn, without the rest of its turn under way.
            self.turns = [] if mover == through else [(mover, first, first == 1, False)]
            s, rolls = mover, first
            while True:
                s, rolls = (s + 1) % n, rolls + 1
                self.turns.append((s, rolls, True, True))
                if s == through:
                    break
        self.card_played, self.card_vp, self.limit, self.n = card_played, card_vp, st.rules.discard_limit, n

    def turn(self, world: GameState, seat: int, hand: tuple) -> tuple:
        for s, rolls, now, matured in self.turns:
            if s == seat:
                return _turn(world, s, hand, self.seats[s], rolls, now, matured, self.card_played, self.limit)
        return 0.0, 0.0, 0.0, [0.0] * len(EVENTS)   # no turn in the horizon

    def world(self, world: GameState, hands: dict | None = None) -> tuple:
        """Every seat's turn in `world`, `hands` overriding some seats' hands."""
        return tuple(self.turn(world, s, tuple((hands or {}).get(s, world.hands[s]))) for s in range(self.n))

    def combine(self, world: GameState, turns: Sequence) -> tuple:
        """(can, events, lose) for one world from its seats' turns."""
        can = [t[0] for t in turns]
        lose = [[0.0, 0.0] for _ in range(self.n)]
        for a, (holder, take) in enumerate(((world.longest_road_holder, 1), (world.largest_army_holder, 2))):
            if holder == NO_OWNER:
                continue
            keep = able = 1.0
            for s in range(self.n):
                if s != holder:
                    keep *= 1.0 - turns[s][take]
                    able *= 1.0 - turns[s][3][3 + a]
            can[holder] -= (LONGEST_ROAD_VP if a == 0 else LARGEST_ARMY_VP) * (1.0 - keep)
            lose[holder][a] = 1.0 - able
        return can, [t[3] for t in turns], lose


def _average(parts: Sequence[tuple], n: int, card_vp: float) -> Outlook:
    k = len(parts)
    return Outlook(
        can=tuple(sum(p[0][s] for p in parts) / k for s in range(n)),
        events=tuple(tuple(sum(p[1][s][e] for p in parts) / k for e in range(len(EVENTS))) for s in range(n)),
        lose=tuple(tuple(sum(p[2][s][a] for p in parts) / k for a in range(2)) for s in range(n)),
        card_vp=card_vp,
    )


def outlook_of(state: GameState, mover: int, *, rolled: bool = True, card_played: bool = False,
               reader: int | None = None) -> Outlook:
    """The round's outlook on one concrete world, every hand as `state` holds it. `mover` is the seat on
    turn, `rolled` whether it has rolled, `card_played` whether it has played a development card this turn;
    a bought card's victory point chance is read as `reader` (default `mover`) sees the unseen cards."""
    card_vp = _card_vp(state, mover if reader is None else reader)
    rnd = _Round(state, mover, rolled, card_played, card_vp)
    return _average([rnd.combine(state, rnd.world(state))], state.num_players, card_vp)


def outlook(view: View, *, rolled: bool = True, worlds: int = 8, rng: random.Random | None = None) -> Outlook:
    """The round's outlook as `view`'s seat can know it: the average over `worlds` worlds drawn from its
    information set (`View.sample`), its own hand exact. `view.mover` must be set (`View.from_game`)."""
    if view.mover is None:
        raise ValueError("outlook needs the seat on turn: build the view with View.from_game")
    rng = rng or random.Random(hash((view.perspective, view.signature())))
    card_vp = _card_vp(view.state, view.perspective)
    rnd = _Round(view.state, view.mover, rolled, view.card_played, card_vp)
    parts = []
    for _ in range(worlds):
        w = view.sample(rng)
        parts.append(rnd.combine(w, rnd.world(w)))
    return _average(parts, view.state.num_players, card_vp)


def standing(state: GameState, seat: int, rates: Rates, *, mover: int, rolled: bool = True,
             card_played: bool = False) -> tuple[float, ...]:
    """Every seat's points now plus its `Outlook.will` from now through the end of `seat`'s next turn,
    on one concrete world, board order. The horizon leaves out the rest of `seat`'s turn under way, so two
    positions `seat` could move to -- having built, or having ended its turn -- are read over the same turns:
    the other seats' next turns and its own. A seat with no turn in it counts only the awards it may lose."""
    card_vp = _card_vp(state, seat)
    rnd = _Round(state, mover, rolled, card_played, card_vp, through=seat)
    can, events, lose = rnd.combine(state, rnd.world(state))
    will = _average([(can, events, lose)], rnd.n, card_vp).will(rates)
    return tuple(victory_points(state, s) + will[s] for s in range(rnd.n))
