# Direct-slider curve validation, frozen before screen

Question: does the current eight-eligible-turn slider track useful interpolation
positions, and preserve strength while avoiding manual profile selection?
Freeze production source bcb9b92, fingerprint
`d2df60dd0abeb72f1bd0ae4222716ce1b4246ac249c64cd2d76aeb0aa611b001`.
No curve, window, endpoint or criterion changes on these results.

Endpoints are current N (no-trade) and M (balanced trading), not legacy T.
Candidates: fixed alpha 0, .25, .5, .75, 1 and the production adaptive policy A.
All candidates use the T exchange evaluator, zero exchange expansion, floor0,
depth2, width6, 600 leaves, k1, standard win temperature and native public
history. All seats permit trades; controlled opponents may decline them.

| Regime | Opponent participation | Opponent move profiles |
| --- | --- | --- |
| none | 0,0,0 | N,N,N |
| moderate | .3,.3,.3 | M,M,M |
| heavy | .65,.65,.65 | M,M,M |
| full | 1,1,1 | M,M,M |
| mixed | .05,.45,.85 | N,M,T |
| rising | .1 to .75 at global turn40 | N,M,T |
| falling | .75 to .1 at global turn40 | N,M,T |

Participation uses stable private per-turn draws separately for acting and
responding roles, followed by the normal positive-gain gate. A sees only the
native public event payload. It cannot inspect these probabilities or profiles.
One focal candidate against three opponents, all four seats equally represented.
Same seeds/boards/seats/random streams across policies; independent boards,
antithetic=False. One30-worker pool at a time, with host contention reported.

Screen: 64 games/candidate/regime, 2,688 games, seeds724000000+10000*regime index.
Eight preflight games compare A versus fixed0 under no trading and fixed1
versus the registered balanced preset under full trading, two boards each.
Require exact full action digests, winners, points and whole-game trade histories.
Screen intervals are descriptive. Select the fixed position with most total
wins (global control G), and with most wins in each regime (conditional control
C). Ties prefer the position closest to .5, then the lower position.

Fresh confirmation uses seeds725000000+10000*regime index. Compare frozen A,
G, and C; run a duplicate control only once where G=C. Complete the fixed
sample before inspecting confirmatory inference. Within each regime, pair win
indicators by board and seat. Two co-primary contrasts are equal-regime means
A-G and A-C. Each has a one-sided97.5% lower normal confidence bound, using
sample variance of paired differences within each independent stratum.
Bonferroni controls family-wise error at5%. Non-inferiority margin is2 percentage
points: both lower bounds must exceed -.02. Failure to pass is inconclusive
unless evidence demonstrates a material loss. This tests aggregate performance
against selected controls; it does not certify a2-point bound in every regime.
Report each regime separately with descriptive95% intervals, so aggregate
results do not conceal uncertain or unfavorable conditions.

Choose confirmation games per policy per regime from screen variance only:
ceil to a multiple of64 of
  (z(.975)+z(.8))^2 * max(mean(var(A-G)),mean(var(A-C))) / (7 * .02^2).
Minimum256, maximum1024. This targets80% power under equal true strength;
the cap is a compute bound, not permission to relax the margin or extend the
sample after results. Report if the cap limits planned power. Outcome-dependent
curve tuning and repeated testing until success are out of scope.

Diagnostics: plot fixed-position win curves with adaptive win rates separately;
log activity and applied alpha at the first focal decision of each global turn.
Reconstruct every logged activity from public events. Report phase-binned alpha,
absolute step changes, reversals and adaptation after rising/falling transitions.
Compare with screen-selected fixed regions descriptively; the strongest point
on a noisy five-position screen is not a known continuous optimum. For a
non-inferiority success, describe one-policy value; do not require superiority.
For failure/inconclusive results, report which conditions or mapping need work.

Every game stores source/runner/plan hashes, full configuration, public events,
whole-game trades, correct candidate seat (Outcome.seating[0]), and action digest.
No Catanatron host or players. Preserve raw records and analysis separately.
