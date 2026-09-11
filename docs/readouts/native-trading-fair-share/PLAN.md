# Trading-enabled profile comparison

User request: 400 games with one new no-trade-profile seat versus three old
no-trade-profile seats; another 400 versus three trading-profile seats.
Player-to-player trading and bank/port exchange are enabled for all seats.

Native HexSet source is PR141 at 396e227, source fingerprint
0254eb3dbe0d14bea1b2d9eb42ec18a553d38c2f9100dd3477edb62347778c3e.
Use the pinned campaign Python image, 30 workers and 30 CPUs. No Catanatron
opponents or hosting participate. Native public-history ledger, standard
rules, independent boards, antithetic=False, seed 717000000, indices 0–399
per matchup. The same board schedule is used across matchups; each has 100
candidate games per seat. Complete all 800, with no result-based stopping.

All policies explicitly use mode=honest, max_trades=None, depth=2, width=6,
600 leaves, k=1, win stance, temperature 2.476644394795811, placement enabled,
and the standard trade floor .0197. Explicit weight and expansion overrides
separate the profile choice from permission to trade. New no-trade means the
validated road=0 profile plus expansion=.25. Old no-trade means the original
road=.1237 weights and expansion=0. Trading means TRADING_WEIGHTS and
expansion=0. No new tuning or defaults adoption follows automatically.

The first game of each requested matchup verifies real domestic exchanges
before the pool continues; both count toward the requested 400. Factories
assert every effective setting, real decisions assert native host and enabled
trading, and every game saves trade-gate call counts, actual exchanges,
profile identities, source/runner hashes, action digest and result. Exact
checkpoint identity is required for reuse. An unfinished game is an error.

Report win rates and two-sided 95% Wilson intervals against 25% fair share,
plus actual trade participation. This is a direct comparison of full profiles,
including the expansion feature. A nonsignificant result alone does not
establish equivalence or justify removing a weight set. These two matchups
measure the requested new profile against both opponent populations; they
are not a complete test of every profile under trading on and off.

Preflight correction before the batch: original first game completed with
zero trades despite active gate calls from every seat (25/23/25/24). The
runner incorrectly required a completed exchange in that one game. Version
2 removes only that outcome requirement and adds offer-gain diagnostics;
it retains all original policy settings and the fixed sample/seed schedule.
Run all 800 records under v2 identity in a separate directory, including
replaying index 0. Preserve the original preflight record separately; it is
not an additional statistical sample. No trade floor or policy is changed.
