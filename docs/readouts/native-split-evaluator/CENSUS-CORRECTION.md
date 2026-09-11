# Whole-game trade census correction

The earlier fair-share runner counted `game.trades` after the game ended.
`hexset.game.end_turn` clears that list, so it describes only the final turn.
This was a reporting error in the research runner, not an engine defect or
policy change. Wins, settings, seeds and full action digests were unaffected.

The corrected observer wraps the existing `hexset.game.trade_event` binding,
records its executed trades only for the identical live Game object, and
preserves turn, participants, bundles and gains. Search copies are excluded.
It changes no decisions or random draws. All 1,600 old games were replayed;
every full action trace, winner, points, seats, turns, action count, gate-call
count and terminal-turn trade list matched the immutable original records.
Replays are audits, not additional statistical samples.

| Floor | Opponents | Wins / 400 (unchanged) | Whole-game trades | Games with trades | Candidate trades |
| --- | --- | ---: | ---: | ---: | ---: |
| .0197 | Old no-trade | 138 | 0 | 0 | 0 |
| .0197 | Trading | 129 | 0 | 0 | 0 |
| 0 | Old no-trade | 110 | 23,258 | 400 | 11,773 |
| 0 | Trading | 64 | 25,025 | 400 | 12,146 |

Floor-zero averages are 58.15 and 62.56 exchanges/game, consistent in scale
with the earlier trade-frequency study's 57.9. The reported 601 and 580 were
final-turn totals, not whole-game totals. The standard-floor zero-trade claim
is now verified over complete games rather than inferred from terminal data.

Original checkpoint archives, summaries and verification JSON stay unchanged
as historical artifacts. Their trade counts are superseded by
census-audit-summary.json and census-audit-games.tar.gz; source/runner identities
and all per-event records are retained there. Archive hashes are alongside.
The narrative floor-zero readout has been corrected to the complete counts.

Candidate participation counts were separately corrected using
`Outcome.seating[0]`, the entrant-to-seat map. The frozen runner incorrectly
used `seating.index(0)`. See ../trading-conditions/REPORTING-CORRECTION.md;
original archives remain unchanged, and candidate-seat-correction.json gives
the derived totals. Wins and whole-game exchange totals are unaffected.
