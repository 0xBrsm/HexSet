# Research workflows

HexSet is the primary game engine and evaluation framework. Heximax is its
handcrafted baseline; Catanatron supplies external reference opponents. Run
self-play, ablations and both candidate-validation gates in HexSet. Seat the
`catanatron` entrant in the arena for an AB2 reference matchup. The random
policy is useful for engine throughput and control experiments. Learned
policies use the runtime-independent interfaces in [training.md](training.md).

Record the host engine, information model, source revisions, dependency
versions, effective bot settings, seeds, lineup, trading mode, bargaining
mechanism, acceleration mode and unfinished-game count with every
comparison. A label such as "vs shipped Heximax" identifies the opponents,
not the engine or information model. `hexset.catanatron.duel` runs a
separate external compatibility experiment; stock `DC:` Heximax there
receives a memoryless public ledger. Follow the [evaluation protocol and
recovery steps](evaluation.md) before using results to select or reject
native HexSet policies.

**The bargaining mechanism is part of what a result is about.** Games
default to one trade round a turn: the actor broadcasts one offer, every
other seat answers once, the actor picks (`Game.trade_mode="round"`;
`Game.max_trades` caps a turn's exchanges, `1` by default, `0` for none at
all, `-1` for no cap). Whether a seat trades at all is its own
gate's business, not a table setting. The exhaustive automatic clearing
house that used to run instead is `Game.trade_mode="auto"`, which
deals until nothing clears against a counterparty that never holds out and
never refuses a deal it merely dislikes -- so a policy fitted or trained
against it learns a game nobody plays. Every study recorded before this
changed (the fitted presets, the adaptive slider, the trading-condition
screens) is a clearing result; reproducing one means `trade_mode="auto"` **and**
`max_trades=-1` -- `"auto"` reads the same cap, so leaving it at the default
would cap the clearing house at one exchange a turn, which is not what those
runs recorded -- as well as running from its recorded source revision.

The standard `heximax` now adapts its weights automatically. Pin a testing
control at slider position 0 or 1 using `--pin-weights-a`/`--pin-weights-b`
or a `heximax:pin-weights=0` entrant spec. Weight pinning and trading
participation are independent. See [the API and migration guide](heximax.md).
Historical studies must run from their recorded source revision.

## Implement a bot

A Python bot implements `choose(game) -> Action`; no base class is required.
Use `hexset.actions.options_for(game)` for legal choices and
`game.state(hexset.game.to_move(game))` for the acting seat's information set.
The acting seat can differ from `game.current_player` during discards. Do not
mutate the live game while choosing. Search should operate on copies.

The [minimal example](../examples/custom_bot.py) registers a factory with the
arena, uses its supplied random generator, and checks replay of two bounded
games:

```sh
pip install -e .
python examples/custom_bot.py
```

`register_entrant_kind(name, factory)` supplies a constructor with signature
`factory(entrant, board, rng) -> bot`. An `Entrant` stores serializable settings
rather than a loaded bot. Pass a module-level registration function as `compete(worker_initializer=...)`
to install custom factories in every worker; optional `worker_initargs` pass
its arguments. With one worker it runs in the calling process.
`start_method="spawn"` provides explicit spawned-process execution, which
requires importable factories and initializers and a guarded driver entry
point (`if __name__ == "__main__":`). The example includes this structure. Player trading
is optional and has a separate [gate interface](bot-api.md#trading). A bot
without a gate declines player trades; bank and port trades remain actions.

The live `Game` exposes full state to Python callers. Information restrictions
are a policy contract, not a security boundary. A fair imperfect-information
policy uses its `View` and samples beliefs for search. Report any experiment
that deliberately uses opponents' true cards as such. The `# true state:`
source annotations help review access but do not prove that a policy respects
its information set; behavioral invariance tests are also required.

## Train a neural network

Training is a supported research workflow. `LaneEnv` collects concurrent
self-play or mixed-policy games, `Policy` connects batched action and value
inference, and shared adapters supply network trade evaluation and PUCT search.
The [training guide](training.md) defines the collection and runtime contracts.
Architecture, optimization, target construction and checkpoint export belong
to the training application; they need not use ONNX during collection.

A collector should store observations and legal masks before advancing the
live games, keep decisions associated with their seat and policy version, and
handle action-cap truncation separately from a terminal win. Use explicit
learner trade gates when training with player trading. Evaluate checkpoints
on a bounded set of games with a fixed opponent configuration.

## Design a comparison

Specify the engine, board generation, seed, complete entrant settings, player
count, trading mode, game count and action cap. Seed model sampling as well as
the engine. Use complete seat rotations and state how boards are paired.
Changing lane or worker count should affect throughput, not which experiment
is run; a custom runtime must preserve that property too.

Report unfinished games alongside wins. A two-player duel and two copies of
each policy in a four-player game answer different questions. A seat's win
rate and a side's combined wins also have different denominators. Victory-point
margins provide additional information, but do not replace win rates.
Paired games and multiple seats of one policy introduce dependence: ordinary
Wilson intervals describe the reported proportions without accounting for
all of that dependence. The duel and batched runner also report intervals computed across independent
board units, averaging the paired games before estimating uncertainty. These
are approximate normal intervals and can be uninformative for very small or
constant samples. Their win-rate denominator includes unfinished games. Retain
game-level outcomes for further analysis.

For trade analysis, gains come from each bot's evaluator. Their numerical
scales need not match across bot types or configurations. Report exchanges,
resources and outcomes independently of gain comparisons; explain any shared
calibration before interpreting which player benefited more.

## Save an experiment

`hexset.experiment.result_document` builds JSON data containing entrant settings, source
and dependency provenance, checkpoint hashes and raw per-game outcomes. Capture
provenance before execution and pass the same settings to the runner and the
artifact builder:

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

This short run tests the artifact workflow; it cannot establish playing
strength. Read it with `json.loads`; the `schema` and `schema_version` fields
identify the layout.
In each outcome, winner and point-vector indices refer to entrant order;
`seating[e]` is entrant `e`'s board seat. `board_index` identifies the shared
board/random-stream unit for paired games. The duel's JSON includes this
artifact under `experiment`.

Use immutable checkpoints for an experiment. `checkpoints_changed` flags
changes detected between the pre-run capture and artifact assembly; it cannot
detect a file changed and restored during execution. A hash identifies bytes but does
not bundle them, and a commit marked `dirty` does not capture local edits.
Archive checkpoints and the exact source separately when publishing results.
The artifact describes supplied settings; it cannot recover unreported custom
runtime configuration or nondeterministic accelerator behavior. Archive the
driver too: arbitrary worker-initializer code and arguments are not captured
automatically.

## Inspect behavior

Request records from the same evaluation that produces the score. Arena
`compete(..., records=True)` stores records in `Tournament.records`;
`compete_batched(..., episodes=True, records=True)` retains them on episodes.
Use `hexset.record.write(path, records)` and `read(path)` to persist them,
`replay(record)` for the final position, or `replay_to(record, ply)` for an
intermediate position. Recorded chance draws and exchanges reproduce the
trajectory without rerunning the policy.

Records capture game behavior, not model activations, search trees or training
examples. Store those separately when needed, keyed by game and decision
index. A private trade evaluation cannot be reconstructed from exchanged
resources alone.

## Benchmark commands

Commands are modules under `hexset.bench`; each provides `--help`.

| Module | Purpose |
| --- | --- |
| `duel` | Compare two sides with configurable seating, Wilson intervals and paired victory-point margins |
| `baselines` | Evaluate a lineup of named entrants |
| `throughput` | Measure random-policy engine throughput |
| `generate` | Write replayable arena games for datasets |
| `trade_census` | Summarize executed trades and optionally save the same games as records |
| `profile_heximax` | Profile a preset through the arena loop and save a cProfile artifact |
| `placement_policy` | Compare opening choices with the placement heuristic |
| `fit_weights` | Fit evaluation weights from records and measure held-out prediction loss |
| `fit_duel` | Compare a fitted weights/temperature pair with an incumbent |
| `ablate` | Zero each Heximax evaluation term; select `--pin-weights 0` or `1`; disable trading separately |
| `weight_sweep` | Evaluate candidate term weights in paired games |

`hexset.bench.versus.compete_batched` is the programmatic runner for batched
policies. It shares the arena's board and seating rules while scheduling
inference across live games. It remains separate because a synchronous
one-action-at-a-time loop cannot provide that batching.

Set `--workers` explicitly on shared machines. The duel uses the arena for
both one and multiple workers. `--geometry ab` means two players; the default
`aabb` seats two copies of each side at a four-player table. Games must complete
whole seat rotations. With antithetic pairing and an odd player count, the
arena requires a multiple of twice the player count to complete both pairing
and rotation. Batched antithetic evaluation requires an even game count.
These are individual players, not cooperating teams.

## Trade census interpretation

`summarize(result)` groups exchanges by base entrant label, preserving the
number of seats occupied by that label. Each trade contributes one side per
participant, including two when both participants have the same label.
`seat_games` counts occupied seats across games; `seat_game_turns` multiplies
that seat count by the sum of elapsed game turns. The participation rate
`trade_sides_per_seat_game_turn` uses that exposure, not the number of turns
on which the particular seat acted. Unfinished games contribute their observed
turns and trades, and labels with no trades remain in the summary.

`mean_net_cards` is received minus given card count per participation. It is
not economic value. `large_hand_share` describes participants whose hand held
at least eight cards at the start of the action step. That snapshot precedes
all trades cleared within the step, and may precede resource production;
it is not necessarily the hand immediately before each exchange.
Raw `gain_a` and `gain_b` retain private evaluator values. Cross-evaluator
`larger_gain` and bank-scaled `mean_value_swing` summaries were removed because
they implied a shared utility scale that the bot interfaces do not provide.

## Removed interfaces

The consolidation pass removes the standalone experiment reader/writer; use
standard JSON with `result_document`. It also removes `NetworkEvaluator`,
`evaluator_for` and `network_evaluator`, which adapted value heads to the
retired handcrafted search. Use `Policy.value_rows` for batched evaluation.
`NetworkBot` and `LeafEvaluator` no longer accept an unused `space` argument;
the policy owns its action space. The `bot_for` and `searcher_for` factory
interfaces and the `Checkpoint` contract remain unchanged.

The cleanup removes `search2`, `greedy`, the tiered evaluator, their presets,
and the `netsearch:`/`netgreedy:` checkpoint variants. Git history preserves
those implementations and their historical results. Existing code should
select Heximax, Catanatron, a direct policy or PUCT search explicitly; there
are no aliases that silently change the bot an experiment runs.

Shared interfaces have independent homes:

- `hexset.bots.base`: `Bot`, `TradeGate`, `RandomBot`.
- `hexset.bots.stances`: per-seat search objectives and fitted temperature.
- `hexset.bots.evaluate`: shared heuristic terms and weights.
- `hexset.actions.options_for`: legal actions, raising `Stuck` if none exist.

Result consumers should distinguish descriptive Wilson fields from the new
board-based intervals. Duel and `Verdict.metrics()` expose
`board_win_rate_low`/`board_win_rate_high`, `interval_unit`, and
`win_rate_denominator`; existing `wilson_*` fields remain available. Batched
`win_rate` now divides by every game, including unfinished games. JSON metric
endpoints use `null` for unbounded estimates, such as a victory-point interval
with only one independent board; consumers must not interpret these as zero.

The duel's external training-backend registration and training-only flags are
removed. Runtime drivers register policy and MCTS factories with
`hexset.clients.netbot.register_entrants`; batching uses `compete_batched`.

The trade census no longer accepts `--from-journals`. Its old parser expected
a different journal schema and substituted zero for missing hands and gains.
For journal conversion and replay, use `hexset.record.from_journal`. A replay
record does not recover the private evaluations needed for a trade census.

## Audit scope

The cleanup reviewed tracked engine, bot, benchmark, adapter, client, server,
environment, test, packaging and deployment files for retired dependencies
and duplicated infrastructure. It consolidates legal-action and random-policy
helpers, removes obsolete search tests, and retains information-set, replay,
trade, seat-pairing and runtime-contract checks. Packaging tests build a clean
wheel in a pytest-managed temporary directory. Optional ONNX dependencies no
longer prevent core and server test collection.

The benchmark modules above remain because each exposes a reusable measurement
or data workflow. The shared evaluator remains a dependency of Heximax.
Catanatron adapters, Gymnasium/PettingZoo wrappers and server journals serve
different external interfaces and are not interchangeable duplicates.
Docker installs the package and extras from `pyproject.toml` rather than
maintaining a second dependency list.

No new strength measurements were taken during this cleanup. See
[benchmarks.md](benchmarks.md) for historical results and their limitations.

## Contributing and verification

Install `.[test]` for core development and run `python -m pytest`. The default
suite omits `slow` tests. Run `python -m pytest -m ''` for the complete suite.
To exercise the optional integrations, install
`.[test,export,catanatron,gym]`; missing extras otherwise skip their tests.
CI checks the base installation on Python 3.11 and 3.13, and runs the complete
suite with optional integrations on Python 3.11. Packaging checks build a
wheel from a clean temporary copy, including the browser asset.

Prioritize behavioral tests for rules, legal actions, seat perspectives,
replay, deterministic collection, trade accounting and runtime contracts.
Keep short integration tests for collection through episode completion; use
bounded action caps to test truncation deliberately. Source inspections are
supplementary architecture checks and cannot substitute for these tests.
Avoid adding a full-game test when a constructed position can demonstrate the
same invariant. Strength sweeps are separate experiments, not CI tests.

When changing a training contract, update its implementation, contract tests
and documentation together. Preserve runtime-independent testing: core
rollout, masking and replay checks must work without a model framework.
