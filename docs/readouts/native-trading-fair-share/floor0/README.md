# Corrected fair-share comparison: trade floor zero

The user clarified that the intended trading floor was zero for all seats.
These are the requested results, superseding the parent readout's .0197-floor
run for the weight-profile question. All real games ran in native HexSet.

One seat uses new no-trade weights plus expansion=.25. Three seats use the
specified opponent profile with expansion=0. Every seat has max_trades=None,
mode=honest and trade_floor=0.0; each actual bot's settings are asserted.
Bank/port exchanges are available throughout.

| Three opponents | Candidate wins | Win rate | 95% Wilson interval | Domestic exchanges | Games with exchanges | Candidate exchanges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old no-trade weights | 110 / 400 | 27.50% | 23.35–32.07% | 601 | 277 | 316 |
| Trading weights | 64 / 400 | 16.00% | 12.73–19.91% | 580 | 292 | 281 |

The first interval includes 25% fair share; it does not establish an advantage
or equivalence against the old no-trade profile with active trading. The
second lies wholly below 25%: the new no-trade profile is disadvantaged against
three trading-profile opponents under this protocol. These results support
retaining distinct profiles for now; they do not justify replacing trading
weights with the new no-trade profile. They compare full configurations,
including the expansion bonus, so they do not isolate which weight or feature
causes the difference and do not establish that no universal profile exists.

Both matchups use independent generated boards, seed 717000000, indices
0–399, antithetic=False, 100 candidate games in each seat. The board schedule
matches across matchups and across the previous floor run. Source is PR141
at 396e227; exact source/runner hashes, Python runtime and all profiles are
in manifest.json and every game record. Depth 2, width 6, 600 leaves, k=1,
win temperature 2.476644394795811, standard opening prior, standard game
rules and native public-history ledger. No Catanatron host or opponents.

Thirty workers completed all 800 games in 103.07 seconds (7.76 games/second),
including retained initial verification games and pool startup. Trade activity
was observed before the pool continued. No result-based stopping, discarded
games, tuning, or default-profile changes occurred.

Independent verification checked all 800 complete unique indices, exact
identities, zero floor, balanced candidate seating, all-seat trade gate calls,
exchange totals, candidate participation, positive gains on both sides of
every executed trade, and Wilson intervals (within the rounding of arena's
normal quantile). The archive holds all 800 per-game records and the original
manifest and summary. Its SHA256 is in archive.sha256.

Reproduce in the pinned campaign image with one OpenBLAS/OMP/MKL thread per
worker and PYTHONHASHSEED=0:

```
PYTHONPATH=/study/src python /study/docs/readouts/native-trading-fair-share/run_floor0.py --out /out/floor0
```

See ../PLAN-floor0.md and ../runtime-floor0.txt. Previous floor-.0197 records
remain preserved separately and are excluded from this 800-game sample.
