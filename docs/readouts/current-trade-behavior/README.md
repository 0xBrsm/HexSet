# Current Heximax trade volume and bundle sizes

Four current Heximax seats averaged **38.97 executed player exchanges per
game**, with a game-level approximate 95% confidence interval for the mean
of **38.09–39.84**. **39.66% of exchanges involved three cards on at least
one side.** This describes current behavior; it does not show that fewer or
smaller trades improve strength.

The 400-game native self-play baseline uses adaptive move weights, exchange
robber_risk +0.075 and spare_card 0.15, floor zero and no trade-count cap.
All four seats use the same current policy. This is a trading-heavy opponent
mix and is not an estimate of human or mixed-opponent games.

## Trade volume

| Metric | Result |
| --- | ---: |
| Games / executed exchanges | 400 / 15,586 |
| Exchanges per game, whole table | 38.97 |
| Exchanges per game, median / 90th / 95th percentile | 38 / 51 / 55 |
| Largest game exchange count | 73 |
| Trade participations per player per game | 19.48 |
| Player-turns reaching a trade event | 25,098 |
| Exchanges per such player-turn | 0.621 |
| Player-turns with any exchange | 47.38% |
| Player-turns with multiple exchanges | 11.93% |
| Exchanges per turn that traded | 1.31 |
| Most exchanges in a single event | 6 |

An exchange has two participants; the per-player average includes trading
during another player's turn. Player-turn denominators count turns reaching
the post-roll/robber trade event, including events with zero exchanges; setup
and a terminal turn that ends before that event are excluded. There were
24,576 events eligible under the adaptive activity rule (actor and some
other seat each held at least two cards before clearing).

## Executed bundle sizes

Sizes are cards **per side**, unordered: 2x3 includes both two-for-three and
three-for-two. These are player exchanges; bank/port trades are not counted.

| Bundle | Exchanges | Share |
| --- | ---: | ---: |
| 1x1 | 1,937 | 12.43% |
| 1x2 | 5,675 | 36.41% |
| 1x3 | 76 | 0.49% |
| 2x2 | 1,792 | 11.50% |
| 2x3 | 5,130 | 32.91% |
| 3x3 | 976 | 6.26% |

Overall, 6,182 exchanges (39.66%) had a three-card side. Most of these were
2x3 rather than 3x3. Each exchange moved an average of **3.84 cards total
across both sides**, or 1.92 per side. The table moved **149.69 cards per
game** (95% CI for the mean 146.27–153.11). Each player gave and received
37.42 cards on average. These are gross transfers: a card exchanged repeatedly
is counted repeatedly, not treated as a new resource.

Trade participation started with a hand above seven cards 8,535 times out
of 31,172 participations. Exchanges crossed a hand above seven 2,300 times
and down to seven or fewer 238 times. These are descriptive hand transitions,
not an estimate of excess discard losses or the causal value of a trade.

## What remains to test

Trade volume and bundle size are separate controls. Their interaction is
plausible: smaller bundles could require more exchanges. A subsequent bounded
comparison should hold current weights fixed and vary the two controls in a
small factorial design, measuring strength and these behavior metrics together.
No cap experiment or parameter change is part of this baseline measurement.

The native arena uses **automatic clearing**, not broadcast offers. It clears
mutually accepted exchanges until nothing clears, a position repeats, or an
explicit count cap is reached. This readout measures completed exchanges,
not initial offers, rejections or counters. The
[historical initial-offer cap study](../offer-cap-history/README.md) used the
served protocol and therefore answers a different cap question. Its modest
cap-two numerical lead was inconclusive.

## Provenance and verification

[Plan](PLAN.md) and [runner](run.py) frozen before launch in `52bbe19`.
Baseline policy commit: `270b9270ea7684c9db60dd346b1f3444e1b9fd0c`.
Source SHA256: `d9cd0c7c833130e401a8d0fbf468a7b6113824e6cb3a88310b47df90410411e2`.
Fresh seed 732000000, indices 0–399; cyclic seat rotations, no antithetic
repeats. Stock search, standard 10-VP rules. Wintermute Python 3.12.14, 30
workers, one BLAS/OpenMP thread each, networking disabled; no Catanatron imports.

All 400 games finished in **60.40 seconds**, within the 400-game / 300-second
hard cap. No retries or additional games. Independent replay verified all
100,535 legal actions and recorded chance events, boards, seating, outcomes
and final scores, then reconstructed trade-event boundaries and hand changes.
Bundle and participation totals were checked against the raw trade tapes.
Confidence intervals for means use 1.96 times the standard error across
independent games; no individual-trade independence is assumed.

[Manifest](manifest.json), [completion](completion.json), [audit/metrics](audit.json),
[per-game metrics](per-game-metrics.json), [runtime](runtime-receipt.json),
[raw records](records.tar.gz), [checksums](SHA256SUMS).

```sh
git worktree add /tmp/trade-behavior-baseline 52bbe19
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/trade-behavior-baseline/src timeout --signal=KILL 300s \
  python docs/readouts/current-trade-behavior/run.py --out /tmp/trade-behavior --workers 30
PYTHONPATH=/tmp/trade-behavior-baseline/src \
  python docs/readouts/current-trade-behavior/audit.py /tmp/trade-behavior
```
