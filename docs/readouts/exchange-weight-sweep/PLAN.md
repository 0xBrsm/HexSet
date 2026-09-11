# Bounded exchange-only sweep

One screen and at most one fresh confirmation. User-approved hard caps:
1,600 complete or attempted games and 900 seconds of evaluation wall time.
No candidate combinations, expanded ranges, seed replacements, reruns,
second finalists or automatic extensions. If any stage errors or times out,
report incomplete and retain the incumbent. Raw partial games remain preserved.

Freeze the current adaptive move policy, search budget, trading floor zero,
win temperature, exchange expansion zero, and all other exchange coefficients.
Native HexSet standard 10-VP games; two candidate seats versus two incumbent
seats, adjacent A,A,C,C rotated cyclically. Every bot plays individually.
30 workers, one BLAS/OpenMP thread, networking disabled. Production source
38ac537, SHA256 c5b426a446f78e5686bca51c66f053bef44f58e924911bbb38a53eb1f743d9ef.

Exactly six candidates, in tie-break order: buy_progress, spare_card,
robber_risk, each multiplied by 0.75 and 1.25 in that order. Each changes
only that exchange coefficient; the move profiles are unchanged. Candidate
and incumbent exchange coefficients remain fixed throughout each game.

1. Preflight: 8 stock/wrapped control games (seed 729900000, indices 0–3),
   plus 2 games per candidate (seed 729900100 + candidate index, indices 0–1)
   independently checking every exchange gain. 20 games total.
2. Screen: 128 independent boards per candidate, shared seed 729000000,
   indices 0–127, antithetic=False. 768 games total. Rank by candidate-side
   wins, breaking ties by the registered order. If no candidate wins more
   than 64 games, stop and retain the incumbent.
3. One finalist only: 800 independent fresh boards, seed 729100000, indices
   0–799. Selection is recorded before confirmation begins. Promote only if
   its two-sided 95% Wilson lower bound is strictly above 50%, followed by
   a successful independent replay/provenance audit. Otherwise retain the
   incumbent. Screening games never enter confirmation inference.

Each screen entrant occupies each seat 32 times; confirmation 200 times.
Unfinished games remain in the denominator. The study uses at most 1,588 games;
12 further games are reserved solely for adoption behavior checks if needed,
within the same 1,600-game / 900-second cumulative evaluation budget. No such
checks may change the statistical verdict or provide a second selection.
An external process-group timeout enforces the run's 900-second limit.

Preserve all records, per-bot weights, activity observations, source hashes,
selection, and runtime receipt. Independently replay all strength records,
reconstruct public activity, verify candidate coefficients and balanced seats,
and recompute screening selection and the confirmation interval. Conclusions
apply to this native adjacent-2v2 trading matchup. Stop after reporting the
registered verdict; further optimization needs a separate decision.
