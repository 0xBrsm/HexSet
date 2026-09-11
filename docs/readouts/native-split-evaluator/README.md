# Separate move and trade evaluators: native screen and confirmation

The split using new no-trade move evaluation and trading-profile trade
valuation led the screen. Fresh confirmation retained a positive estimate,
but did not establish superiority over the all-trading baseline. No policy
default changed and no adaptive/sliding weight policy was adopted.

N denotes the new no-trade profile: road=0, expansion=.25, other no-trade
coefficients unchanged. T denotes trading weights with expansion=0. Each
candidate faces three fixed T/T opponents. All seats trade at floor zero.
Native HexSet public-history ledger, standard rules, depth 2, width 6,
600 leaves, k=1, win temperature 2.476644394795811, standard opening prior.
Both roles receive the same seat information; no hidden opponent cards.

## Exploratory four-way screen

128 games per cell, independent boards, seed 718000000, indices 0–127,
32 games in each candidate seat. The same board schedule is used for all
four cells; antithetic=False.

| Move evaluator | Trade evaluator | Wins / 128 | Win rate | 95% Wilson interval | Whole-game trades / game |
| --- | --- | ---: | ---: | ---: | ---: |
| N | N | 19 | 14.84% | 9.71–22.02% | 59.94 |
| N | T | 34 | 26.56% | 19.68–34.82% | 58.27 |
| T | N | 14 | 10.94% | 6.63–17.52% | 61.98 |
| T | T | 23 | 17.97% | 12.28–25.52% | 57.36 |

Holding N move evaluation fixed, switching its trade evaluator to T gained
11.72 percentage points on matched boards (exploratory paired 95% interval
+2.09 to +21.35). N/T versus T/T gained 8.59 points (-1.38 to +18.56).
These are unadjusted exploratory intervals, not validation after selection.
The all-identical T/T cell also landed below 25% in this small sample; its
interval includes fair share. Do not treat 17.97% as its underlying strength.

## Fresh fixed confirmation

N/T and T/T received 512 games each on fresh seed 719000000, indices 0–511,
128 games per candidate seat, shared board schedule, same fixed T/T opponents.
The primary contrast was registered before launch. Complete samples were
collected before analysis; screen games are excluded from confirmation.

| Move / trade evaluator | Wins / 512 | Win rate | 95% Wilson interval | Whole-game exchanges |
| --- | ---: | ---: | ---: | ---: |
| N / T | 137 | 26.76% | 23.11–30.76% | 29,371 |
| T / T | 121 | 23.63% | 20.16–27.50% | 29,748 |

Primary paired win-rate difference: **+3.125 percentage points**, 95% interval
**-1.98 to +8.23 points**, from 512 independent per-board outcome differences.
There were 178 discordant outcomes. The registered strictly-positive lower
bound criterion was not met. This is not proof of equivalence or inferiority;
the effect remains unresolved at this sample size. The direction agrees with
the earlier split-evaluator study, but these are different profiles and runs.

The result supports investigating the distinct roles of move and trade
valuation before assuming that one whole weight vector should slide with
observed trade frequency. It does not validate a new default or prove that
an adaptive policy cannot help. Further tests should distinguish evaluator
roles and opponent participation explicitly rather than infer an optimal
weight curve from two shared-profile settings.

## Execution, verification and corrected trade census

A benchmark wrapper routes choose to the move evaluator and gains_many to
the trade evaluator. Diagonal cells share the same bot object, preserving
ordinary behavior. Off-diagonal gates are separate objects with batch-scoped
caches and no search/RNG use. Eight old games exactly reproduced before the
screen. Every actual bot's profile and trading settings are asserted.

The trade-frequency discrepancy was a reporting bug: game.trades is reset
at end_turn, but the preceding runner read it only after game completion.
The corrected observer records every live trade_event result and excludes
hypothetical search. All 1,600 preceding games were replayed with identical
full action traces and outcomes. Floor-zero batches had 23,258 and 25,025
exchanges, not the previously reported final-turn totals of 601 and 580.
The standard-floor batches had zero exchanges over complete games as well.
See [the census correction](CENSUS-CORRECTION.md).

All runs used 30 workers and at most one pool at a time. The fresh 1,024-game
confirmation took 123.87 seconds. Independent checks validated completeness,
unique indices, balanced seating, exact identities, event counts, positive
gains on both sides and the paired confidence calculation. No game was dropped.

Source/runner hashes and full configurations are in the JSON summaries and
per-game identities. Plans and reproduction scripts are in this directory.
Raw archives: matrix-games.tar.gz (512 screen plus 8 preflight),
confirmation-games.tar.gz (1,024 fresh outcomes), and census-audit-games.tar.gz
(1,600 verification replays, never counted as new outcome samples). Matching
.sha256 files identify each archive. Original pre-audit records remain
immutable with their trade-count limitation explicitly documented.
