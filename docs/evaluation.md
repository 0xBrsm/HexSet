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
rotations and state how boards are paired. Two runs on one seed pair game for
game: the same board, seating, development deck and dice turn for turn, so a
game they score differently is one where some seat decided differently.
`compete` returns identical results at any `workers`; a custom runtime keeps that property by keying its
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
turn cap with no winner is a defect to `compete`, which raises
`hexset.arena.Exhausted` on the first one, naming the seed, index and seating
to replay it with. `compete_batched` ends it there instead: a reading neither
side won, scored on the points it reached and counted in `Verdict.exhausted`.
Raise the cap (`turn_cap`,
`--turn-cap`) only for unstructured play such as random bots.

### The seed policy

A project that compares many runs can hold every run to a short list of
seeds, so that runs pair game for game (`hexset.seeds`). A
`seed-policy.json` names them:

```json
{"eval":  {"default": {"2": 2000, "*": 1000},
           "screen": [1000, 2000],
           "confirm": [3000, 4000]},
 "train": {"rl": [0, 1, 2], "bc": [10, 11, 12]},
 "analysis": [20, 21, 22]}
```

- `eval.default` is the seed a run takes when it names none, by the number
  of playing seats, `"*"` for any other count. Every default is a screen
  seed.
- `eval.screen` seeds may be named freely. For more games, play more indices
  on the same seed.
- `eval.confirm` seeds are for a test registered before its data. A run on
  one names its registration (`registration=`, `--registration PATH`), a
  file that names the seed, and the file's SHA-256 is kept with the run.
- `train` lists each kind of training's seeds. A training seed is never an
  evaluation seed. A run that plays games to train or fit on names its kind
  as its `seed_use` (`compete(seed_use="rl")`) and is held to that list.
- `analysis` lists the seeds of randomness that plays no game: resampling, a
  statistical null, a data split.

The policy is found once per process: the file `HEXSET_SEED_POLICY` names,
else the nearest `seed-policy.json` above the package, else the nearest one
above the working directory. A host package may install one
(`register_seed_policy`) or `require` one, so that a tree staged without its
policy file refuses to play. With no policy and no requirement nothing is
checked, and a run that names no seed plays seed 0.

`compete` holds its `seed` to the policy and raises `SeedRefused` for one
the policy does not allow. A reason given as `seed_bypass=` (on the command
line `--bypass-seed-policy REASON`; `HEXSET_SEED_BYPASS` for code that takes
neither) plays it anyway, prints a warning and keeps the reason in the
journal. How the seed passed (the policy's digest, `screen`, `confirm` with
its registration, or `bypassed` with the reason) is on the journal's source
line. Every bench command that plays games takes `--seed` (`--duel-seed`
for `duel`), whose default is the policy's, with `--bypass-seed-policy` and
`--registration`.
A tool that plays outside the arena, or seeds no game, holds its seed the
same way: `seeds.add_arguments(parser)`, then `seeds.hold(args, seats,
use=...)`, which returns the note to keep with its output. `seeds.suspend()`
turns the policy off for a test suite, whose games are fixtures rather than
runs, and for the processes it starts.

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
result = compete(entrants, games=2, journal="result.games.jsonl", **settings)
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

### The run journal

Every `compete` run keeps its games in its `journal` (`hexset.gamelog`), and
none runs without one: a header line with the run's settings, a source line
with its seed note, then one JSON line per game (its outcome, its record and
every seat's search decisions), written and fsynced as the game finishes, in
whatever order the workers finish them. A run that dies, is killed or raises
keeps every game it finished.

- `read_journal(path)` returns the header and every finished game by index,
  readable while the run is still going; `journal_tournament(path)` is the
  `Tournament` of the games so far.
- A journal that already holds a run is refused unless `resume=True`, which
  checks its header against this run's settings and plays only the indices
  it lacks. Every game is a function of the seed and its index, so the
  result is the run's own.
- A resumed run may name more `games` than it began with: it plays the
  indices after them on the same seed, an extension, which the journal's
  source line records (`extended_to`). Fewer games is a different run.
- A game that reaches the turn cap is written as an `exhausted` line before
  `Exhausted` is raised; `read_journal` leaves it out.

Every bench command that plays games keeps them in its `--journal`, and
all but `throughput` resume with `--resume` ([guide.md](guide.md#benchmark-commands)).
`compete_lanes` keeps a journal keyed by its own game keys, read by
`read_lane_journal`; `read_game_journal` reads any game journal, its
exhausted games included. A `LaneEnv` collector
writes each finished `Episode` as one line with
`hexset.gym.lanes.episode_entry`.

### Search decision logs

A seat that searches logs every decision it searched, beside the game's
result. `hexset.mcts.Search.decide` and `hexset.rollout.PairedRollouts.decide`
append each to the searcher's `decision_log`: the root's options, the
policy's prior and value, what each option scored (MCTS visits and means;
each rollout candidate's outcome in each world), the option chosen, the
turn, phase, seat and public points, and the CPU time spent. A rollout
decision also keeps the value head's read of each playout at later turns
(`hexset.rollout.MARKS`), so fewer worlds, fewer candidates or a shorter
horizon can be scored on the same playouts afterwards.

`hexset.arena.search_logs(bot)` finds the logs behind a seat, through the
wrappers it is dealt in (a trader, a veto, a dealer, a coalition, both
halves of a handoff). `compete` writes them into each game's journal line
under `searches`, and `read_journal` returns them on each `Outcome`, so
the run itself shows which decisions gained, and at what cost.
