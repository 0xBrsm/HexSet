# Adaptive versus fixed exchange weights: fixed 2v2 test

Run 400 independent four-player games in native HexSet. Two seats use the
current adaptive coefficient vector for exchange valuation; two use stock
fixed TRADING_WEIGHTS for exchanges. Both sides use current unpinned adaptive
move search. Each bot still plays to win individually; sides group win counts.

- Production source 38ac537, fingerprint c5b426a446f78e5686bca51c66f053bef44f58e924911bbb38a53eb1f743d9ef.
- Stock search depth 2, width 6, 600 leaves, k=1, default stance/temperature
  and opening prior. Standard 10-VP rules. No Catanatron imports or host.
- Trading enabled for all seats, floor zero, exchange expansion credit zero
  on both sides. Only the exchange coefficient vector differs.
- Candidate exchange coefficients use its latest public eight-eligible-turn
  activity at each valuation batch; event observations arrive after clearing.
  Move weights update on choose exactly as in production on both sides.
- Seed 728000000, indices 0–399, antithetic=False. Entrants A,A,F,F rotate
  through all four seats, 100 games per entrant in each seat. This measures
  the adjacent 2v2 lineup; the alternating seat pattern is outside this test.
- 30 workers, one BLAS/OpenMP thread each, same pinned runtime image as the
  current AB2 benchmark, networking disabled. Frozen sample, no replacements,
  exclusions, tuning, outcome-driven extension or automatic policy adoption.
- Preflight seed 728900000, indices 0–3: compare complete four-fixed-wrapper
  traces against stock Heximax; four additional mixed games compare every
  trade gain against an independently constructed evaluator. Exclude all 12.
- Preserve full records, trade activity observations and gate/move alpha counts.
  Replay every strength game, validate observations and recount independently.
- Primary result: adaptive-exchange side wins / all 400, descriptive 95%
  Wilson interval against equal share 50%. Unfinished games stay in denominator.
  A positive/negative interval relative to 50 supports this specific matchup;
  overlap is inconclusive. Report exchange counts and observed activity to
  characterize the actual environment; do not generalize to all opponents.
