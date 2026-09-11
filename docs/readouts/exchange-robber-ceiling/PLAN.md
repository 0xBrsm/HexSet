# Bounded exchange robber-weight sign and ceiling test

User requested exploring zero and positive robber weights after −0.225 beat
−0.30. This is a new bounded experiment, not an extension of that sample.
Hard cap: 1,600 attempted games and 900 seconds cumulative evaluation wall
time, including reserved adoption checks. No retries, replacements, extra
candidates or automatic range extension. On error/time cap, retain incumbent
and report incomplete. At the upper edge, report an unlocated ceiling.

Baseline: PR #149 source b328d2d, SHA256
6b60259813b77ecfca12316ab46de7838b6a93b335f7b79c80955ee385fb2d49.
Current exchange robber weight is exactly −0.30 * 0.75. Both move endpoints
stay −0.30. Only the fixed exchange robber coefficient changes. All other
exchange coefficients, adaptive move search, zero floor, zero exchange
expansion, default temperature and opening prior remain fixed.

Native HexSet standard 10-VP games, adjacent A,A,B,B cyclic rotations,
antithetic=False. Bots play individually; side win baseline is 50%. 30
workers, one BLAS/OpenMP thread each, networking disabled, same pinned image.

Candidates in tie-break order: −0.15, 0, +0.075, +0.15, +0.30, +0.60, +1.20,
+2.40. Positive values reward the exposure feature in exchange pricing only;
this does not change the actual risk or move valuation.

1. 24 preflight games: four stock/wrapper trace pairs at seed 730900000,
   indices 0–3; two independent-gain-check games per candidate at seed
   730900100 + candidate index, indices 0–1. Exclude from strength results.
2. 512 screening games: each candidate versus incumbent on the same 64
   independent boards, seed 730000000, indices 0–63. Rank by candidate wins;
   ties use registered order. Stop if none exceeds 32 wins.
3. 256 runoff games: top two candidates face each other on fresh seed
   730100000, indices 0–255. Select most wins; ties retain the screen leader.
   This is still selection data, not confirmation or proof of an optimum.
4. 800 fresh confirmation games: one selected finalist versus incumbent,
   seed 730200000, indices 0–799. Selection recorded before launch. Promote
   only if independent replay audit passes and the two-sided 95% Wilson
   lower bound exceeds 50%. Unfinished games count in the denominator.
5. At most eight reserved adoption trace games, keeping total <=1,600.
   No further tuning or re-selection after confirmation.

Preserve all configurations, source/runtime hashes, action/chance records,
activity observations, counts and selection decisions. Independently audit
all strength records and recompute selection/intervals. Also describe actual
cross-policy net card transfers and hand-threshold crossings from the replays,
to investigate behavior; these are diagnostics, not alternative adoption gates.
The shared board schedules are deliberate and not independent across screen
candidates. Infer confirmation strength only from its 800 fresh boards.
