# Candidate-seat metadata correction

`arena.Outcome.seating` maps entrant index to physical seat. Candidate entrant
zero therefore occupies `seating[0]`. The frozen research runners used
`seating.index(0)`, the inverse map, for `candidate_seat` and `candidate_trades`.
With four rotating seats this labels the wrong seat on half the games.

This error affected descriptive candidate participation counts. It did not
affect the actual lineup, private participation controls, activity observer,
search, actions, completed trades, or wins. Winners are stored directly as
entrant IDs; candidate wins are `winner == 0`. Both the correct and inverted
rotations have equal seat counts, so the old balance assertion missed it.

Whole-game trade records contain physical participants and permit correction
without replaying games. `analyze.records` now derives the candidate's seat
and participation from those records. `audit_validation.py` independently
checks every fresh validation game and reproduces both primary win contrasts.
It also reproduces every activity value from the recorded public events.

Raw archives and frozen runner scripts remain immutable so source hashes and
reproduction commands keep their meaning. Their `candidate_seat` and
`candidate_trades` fields are superseded by the analyzer's derived values.
The same correction applies to the earlier native-split-evaluator archives;
its census narrative and candidate-seat-correction.json contain corrected
participation totals (11,773 versus old no-trade opponents; 12,146 versus
trading opponents, in the floor-zero 400-game batches).

This is separate from the earlier terminal-turn census error. All whole-game
trade totals in the corrected census remain valid: 23,258 and25,025. Neither
reporting correction changes a strength result or adds statistical samples.
