# Research tools and maintenance

Heximax and Catanatron are the supported handcrafted baselines. The random
policy is useful for engine throughput and control experiments. Learned
policies use the runtime-independent interfaces in [training.md](training.md).
Record the engine revision, dependency versions, bot settings, seed, lineup,
trading mode and unfinished-game count with every comparison. Results from
HexSet and Catanatron are separate experiments: their rules and adapters differ.

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
| `ablate` | Zero each Heximax evaluation term; select `--profile trading` or `notrade` |
| `weight_sweep` | Evaluate candidate term weights in paired games |

`hexset.bench.versus.compete_batched` is the programmatic runner for batched
policies. It shares the arena's board and seating rules while scheduling
inference across live games. It remains separate because a synchronous
one-action-at-a-time loop cannot provide that batching.

Set `--workers` explicitly on shared machines. The duel uses the arena for
both one and multiple workers. `--geometry ab` means two players; the default
`aabb` seats two copies of each side at a four-player table. Games must complete
whole seat rotations. These are individual players, not cooperating teams.

## Removed interfaces

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
