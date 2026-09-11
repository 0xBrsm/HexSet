# Current trade behavior measurement

User identified trade volume and bundle size as the remaining Heximax trade
questions and requested measuring current behavior. This is descriptive,
not a strength test or a tuning sweep. No production parameters change.

Run exactly 400 stock Heximax self-play games, all four seats current adaptive
Heximax with fixed exchange robber_risk +0.075 and spare_card 0.15. Native
HexSet automatic clearing, floor zero, max_trades=None, three cards per side,
standard 10-VP rules and stock search. Baseline source is 270b927, SHA256
d9cd0c7c833130e401a8d0fbf468a7b6113824e6cb3a88310b47df90410411e2.

Fresh seed 732000000, indices 0–399, cyclic seat rotations, no antithetic
repeats. 30 workers, one BLAS/OpenMP thread per worker, networking disabled.
Hard cap: 400 attempted games and 300 seconds evaluation wall time. No
replacement games, retries, tuning or outcome-dependent extension. Stop and
report partial if the cap is reached. Preserve all action/chance records.

Independently replay every game. Report executed exchanges per game and
player-turn reaching the trade event, turns with any/multiple exchanges,
per-game distribution, cards moved per exchange, symmetric bundle-size
histogram (1x1, 1x2, 1x3, 2x2, 2x3, 3x3), participant card-hand threshold
crossings, and the fraction of exchanges involving three cards on either
side. Report game-level confidence intervals for mean volume, with games as
independent sampling units. Counters and denominators must be explicit.

This is a trading-heavy all-Heximax baseline. It does not estimate human or
mixed-opponent behavior or establish excessive trade volume causally. Arena
automatic clearing does not broadcast initial offers: this measures executed
bundles, not offered/rejected bundles or the served-protocol initial-offer
cap from the historical Clio study.
