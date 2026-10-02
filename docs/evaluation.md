# Evaluating policies

Designing a comparison, reading the intervals the runners report, and
recording an experiment so its result can be identified later. Running the
engine is [guide.md](guide.md); reinforcement learning against it is
[training.md](training.md).

## Trading mechanism

Evaluate and fit under `trade_mode="round"`, the offer-and-answer round the
served table also runs. `compete`, `compete_batched`, `LaneEnv` and every
bench command default to it. `"auto"`, the exhaustive clearing house, deals
against counterparties that never hold out and is research only: a policy
fitted under it is fitted to that mechanism, and results from the two modes
are not comparable. Record the mode with every result. A custom driver used
for evaluation or fitting must match the served protocol: one trade event per
turn, the actor's own offer budget, and `observe_trade` notifications
(`hexset.game.publish_trade_event`).

## Design a comparison

Specify the engine version, board generation, seed, complete entrant
settings, player count, game type, trading mode, game count, action cap and
turn cap. Seed model sampling as well as the engine. Use complete seat
rotations and state how boards are paired. `compete` returns identical
results at any `workers`; a custom runtime keeps that property by keying its
randomness to the game, seat and decision, not to the worker or lane.

Record with every result the host engine, information model, source
revisions, dependency versions, acceleration mode and unfinished-game count.
"Self-play" or "vs the reference" names the opponents, not the engine or the
information model.

Report unfinished games alongside wins. A two-player duel and two copies of
each policy at a four-seat table answer different questions. A seat's win
rate and a side's combined wins have different denominators. Victory-point
margins add information but do not replace win rates.

An unfinished game is one the action cap stopped. A game that reaches the
turn cap with no winner is a defect, not a result: `compete` and
`compete_batched` raise `hexset.arena.Exhausted` on the first one, naming the
seed, index and seating to replay it with. Raise the cap (`turn_cap`,
`--turn-cap`) only for unstructured play such as random bots.

### Paired boards

`compete(antithetic=True)` (the default) plays every board twice, the second
game under the `complement` seating: entrant `e` takes the seat entrant
`complement[e]` had. The default complement, `half_turn`, maps entrant `e` to
`e + seats // 2`, which swaps the sides of `aabb` and `ab` only. The duel
builds the side swap for any geometry with equal sides. A complement that
moves a `retired` entrant onto a playing seat raises `ValueError`. With
retired seats the opening seat rotates among the playing entrants.

### Intervals

Paired games and several seats of one policy are dependent. The runners
report two kinds of interval:

| Keys | Unit | Method |
| --- | --- | --- |
| `wilson_low`, `wilson_high` (duel, `Verdict.metrics()`); `wilson_interval_95` (baselines) | game | Wilson score interval; assumes independent games |
| `board_win_rate_low`, `board_win_rate_high` (duel, `Verdict.metrics()`); `interval_95` (baselines) | board | Normal interval over per-board means of the paired games |
| `paired_vp`, `paired_vp_low`, `paired_vp_high` (duel, `Verdict.metrics()`) | board | Normal interval over per-board victory-point margins |

Both win-rate denominators include unfinished games (`win_rate_denominator`).
Board-level intervals are approximate and uninformative for very small or
constant samples. An unbounded estimate, such as a victory-point interval
over one board, serialises as `null`; it does not mean zero. Keep game-level
outcomes for further analysis.

Trade gains come from each bot's own evaluator, and their scales differ
across bot types and configurations. Report exchanges, resources and outcomes
independently of gains, and state any shared calibration before comparing
which side gained more.

### External reference

The `catanatron` entrant wraps Catanatron's depth-two alpha-beta player. Its
search runs in Catanatron; HexSet hosts the game, the legal actions and the
public history. At its default `worlds=0` it reads every seat's true hand,
so it is a reference baseline, not an information-set bot, and a comparison
against it says so; `catanatron:worlds=<n>` is the information-set read
([guide.md](guide.md#bundled-opponents)). A game hosted elsewhere, such as one
run inside Catanatron against its own ledger, is not comparable with natively
hosted results and cannot select or reject a policy.

## Save an experiment

`hexset.experiment.result_document` builds a JSON document with the entrant
settings, source and dependency provenance, checkpoint hashes and raw
per-game outcomes. Capture provenance before the run and pass the same
settings to the runner and the document builder:

```python
import json
from pathlib import Path

from hexset import experiment
from hexset.arena import Entrant, compete

entrants = [Entrant("control-a", kind="random"),
            Entrant("control-b", kind="random")]
settings = dict(seed=7, workers=1, action_cap=40, antithetic=True)
before = experiment.provenance(entrants)
result = compete(entrants, games=2, records=True, **settings)
document = experiment.result_document(
    result, entrants, run_provenance=before, **settings,
)
Path("result.json").write_text(json.dumps(document, indent=2, allow_nan=False))
```

This run exercises the workflow; it measures no strength.

| Field | Contents |
| --- | --- |
| `schema`, `schema_version` | `"hexset.experiment"`, `1` |
| `settings` | `entrants`, `games`, `seed`, `workers`, `action_cap`, `antithetic` |
| `provenance` | HexSet version, commit, `dirty`, Python, platform, dependency versions and VCS revisions, thread environment variables, checkpoints |
| `checkpoints`, `checkpoint_capture`, `checkpoints_changed` | SHA-256 and size per checkpoint path; whether they were captured `before_run` or `after_run`; whether they changed between the two captures |
| `summary` | `games`, `unfinished`, `seconds`, `standings` |
| `outcomes` | Per game: `index`, `board_index`, `winner`, `points`, `turns`, `seating`, and `roads`, `settlements`, `cities` |

`settings` does not hold `trade_mode`, `game_type` or `turn_cap`; record them
beside the document. Winner and point-vector indices refer to entrant order,
`seating[e]` is entrant `e`'s seat, and `board_index` is the board and random
stream two paired games share. The duel's JSON includes this document under
`experiment`.

Use immutable checkpoints. `checkpoints_changed` compares the pre-run capture
with the one taken at assembly, so a file changed and restored during the run
goes undetected. A hash identifies bytes without archiving them, and a
`dirty` commit does not capture the local edits: archive checkpoints and the
exact source when publishing. The document cannot recover unreported runtime
configuration, nondeterministic accelerator behaviour, or the code and
arguments of a `worker_initializer`; archive the driver too.
