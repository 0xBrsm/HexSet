# Changelog

All notable changes to HexSet are recorded here — the `hexset` engine, its
bots (`hexset.bots`), `hexset.bench`, `hexset.server`, `hexset.clients` and
`hexset.gym`, shipped from `src/` as one distribution.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **An embedded ONNX seat now trades.** `hexset.clients.onnxbot.NetworkBot`
  gained `valuation`/`accepts`, both derived from the checkpoint's own value
  head with no new graph output: `valuation` is `tanh(delta_V_r /
  VALUE_SCALE)` per resource, from one batched forward over the seat's hand
  plus its five one-card imagined successors when the graph's declared batch
  dimension allows it; `accepts` is the head's strict preference for the
  concrete post-trade hand. `hexset.trading.VALUE_SCALE`, the pinned
  constant both cite.
- **Trading is one event a turn.** `hexset.trading.trade_event` clears deals
  for the current player after the roll and the robber and before any build
  is served: a one-for-one exchange executes when both seats' public
  valuation vectors say it helps them *and* both seats' private gates
  accept, best deal first, repeatedly, until nothing clears. No budget and
  no cap — the gate must be strictly positive and is re-asked after every
  exchange.
- `Game.valuations` — every seat's public vector, five floats in `[-1, 1]`,
  all-zero until something publishes; `Game.publish(seat, vector)` is the
  one way to set one, validated and recorded, nothing else; `Game.trades`
  and `Game.trades_made` for the turn's exchanges; `Game.max_trades` (`0`
  off, `None` unbounded); `Game.gates`, the per-seat objects `trade_event`
  asks for a private judgement. `Game.num_players`.
- `hexset.bots.Bot.valuation(view)` and
  `Bot.accepts(view, received, counterparty)`, both defaulting to "this seat
  never trades", so a bot written before the mechanic keeps working.
- `PUT /api/games/<CODE>/valuation` sets the calling seat's vector; every
  seat's `valuations` and the turn's `trades` ride in the game view and in
  `GET /api/record`. The browser client gains five per-resource toggles and
  a read-out of both.
- Presets `search2-notrade` and `greedy-notrade`.
- `hexset.gym` (`pip install -e ".[gym]"`): `HexSetAEC`, a PettingZoo
  `AECEnv` with one agent per seat and an honest `action_mask`; `HexSetEnv`,
  a single-agent Gymnasium `Env` (registered as `HexSet-v0`) with one
  learner seat and `hexset.arena` bots auto-played at the rest; `register()`.
  `import hexset` stays numpy-only — only `import hexset.gym` needs
  `pettingzoo`/`gymnasium`.
- `hexset.view.View` gained `__eq__`/`__hash__`, comparing `perspective`,
  `omniscient`, `num_players` and `signature()` rather than object identity.
- No lobby: `POST /api/games` deals a full game immediately and seats the
  creator at a random seat; `POST /api/join` or `GET /<id>` claims an open
  seat; an empty seat locks out after a grace window. `GET /api/table/<id>`
  serves a token-free observer view once no seat is left to claim.
- **The engine moved into this repo, under `engine/`.** `engine/hexset`
  (the rules engine, bots, ledger, arena and tuning) and `engine/heximax`
  (the honest handcrafted baseline, its own top-level package) were
  imported with full commit history from `0xBrsm/dev-HexNet`.
- **A single `pip install -e .` from the repo root now provides `hexset`,
  `heximax` and `hexset_ui`.** `hexset` is no longer installed from a
  separate checkout of `dev-HexNet`; see the README's "Running it"
  section.
- **`hexset.build_info()`**: version and git commit for a consumer (e.g.
  HexNet's run manifest) to stamp into its own provenance records.
- `benchmarks.duel` verdicts now carry `turns_mean`/`turns_median`/
  `turns_max` and `exhausted` (games that ran out `MAX_TURNS` without a
  winner, distinct from `unfinished`) on the arena path.
  `hexset.arena.Tournament` gained a `turns` tuple alongside
  `winners`/`points`.
- `Game.locked`: a per-seat setup lock / seat-retirement primitive. A locked
  seat is skipped by the setup snake and turn rotation, is never `to_move`,
  and drops out of any trade offer via the new `lock_seat(game, seat)`.
  `Game.locked: frozenset[int]` defaults to empty and is preserved by
  `imagine`. `start()` gained a `first=` keyword so the setup snake can begin
  at any seat.
- `hexset.catanatron`: an optional adapter to Catanatron's arena
  (`pip install -e "src[catanatron]"`), replacing the standalone
  `catan-bridge` repo. Translates a live `catanatron.Game` into a
  `hexset.Game`/`GameState`, maps board ids both ways, resolves `hexset`
  actions onto catanatron's `playable_actions`, and exposes a
  `catanatron.models.player.Player` (registered as `DC:<entrant>`) so any
  `hexset` bot can be seated in a Catanatron duel and vice versa.
  `python -m hexset.catanatron.duel` shards a duel across worker processes
  and pins catanatron to a fixed commit; the base `hexset` package stays
  catanatron-free.
- `hexset.tuning` can now fit heximax's weight profiles: `evaluator=
  "heximax-trading"` / `"heximax-notrade"` build `kind="heximax"` entrants.
  `heximax.heximax()` gained a `weights` keyword.
- The live trade offer is now part of the observation:
  `encoding.global_features` gains 18 features at four players — the
  standing offer's give/want bundles, the proposer's seat, and who has
  answered. The information-set record grows matching fields
  (`offer_give`, `offer_want`, `offer_proposer`, `offer_answered`).
  **The ONNX contract bumps to v3** (27 inputs, outputs unchanged); a
  consumer must fill these fields before deploying a v3 checkpoint.
- `hexset.migrate`: function-preserving checkpoint migration onto a wider
  observation. New `embed_global` columns are zeroed, so a migrated
  checkpoint plays exactly as its source until trained further. A
  checkpoint from before this change cannot be loaded without migrating it
  first.
- A public-knowledge ledger, `hexset.ledger`: a new `PublicLedger`, created
  with the `Game` and carried through `imagine()`, tracks each seat's
  reconstructed hand as `known[5]` (a certified per-resource lower bound)
  plus `unknown`. A robber/knight steal credits the thief's gain to
  `unknown` only, never revealing the true resource taken.
  `encoding.global_features` gains 18 more features at four players (each
  opponent's `known`/`unknown`, seat-relative); the information-set record
  grows `ledger_known` `(players, 5)` and `ledger_unknown` `(players,)`.
  **The ONNX contract bumps to v4** (29 inputs, outputs unchanged); a
  consumer must supply the ledger fields before deploying a v4 checkpoint.
  `hexset.migrate` zero-pads the tail growth automatically.
- `hexset.heximax`: an honest handcrafted bot that reads every opponent
  through the public ledger and public counts rather than the true hand
  (`search2` reads the true hand). Plugs into the existing `Bot`/`arena`
  API via a `Belief` model, an `HonestEvaluator`, and a max^n search with
  PIMC opponent determinization. Three presets: `heximax` (honest, three
  offers a turn), `heximax-omni` (the same bot reading every true hand, for
  measuring what honesty costs), and `heximax-notrade` (no-trade weights,
  offer budget zero). The PIMC determinization count is fixed at `k=1`; the
  `heximax-k2`/`-k4`/`-k8` ablation presets are not shipped.
  `heximax` also gained protocol-free trade valuation (`deficit`/`surplus`,
  `candidate_bundles`, `score_proposal`, `accept_rule`, `counter_of`,
  `rank_partners`) and a minimal adapter, `Heximax.propose_actions`, that
  replaces the engine's one-for-one trade sample with heximax's own scored
  candidates.

### Fixed

- `hexset.server.api.spawn_bot` imported `.onnxbot` from `hexset.server`,
  where the module no longer lives after the one-distribution restructure
  moved it to `hexset.clients.onnxbot`. Any `.onnx` model picked in the web
  UI returned HTTP 500 (`ModuleNotFoundError`); now it imports from
  `hexset.clients.onnxbot`.
- `heximax-omni` priced trades against a hand that did not exist:
  `_move_hand` folded a non-knower's hand into one all-one-resource total,
  which is exact for the honest bot but wrong once `omniscient` scores
  every hand verbatim. `_move_hand` now takes an `exact` flag, and
  `_partner_delta` passes `self.omniscient`. Only `heximax-omni`'s
  behaviour changes.
- `hexset.catanatron.state.translate` left `Game.valuations` at its empty
  default instead of one all-zero vector per seat, so `hexset.encoding`
  raised `IndexError` reading a bridged position; `tests/catanatron`'s
  three white-box suites read the upstream `catanatron.Game`'s `state`
  field as `_state` (a leftover from a project-wide sed that meant to
  touch only `hexset.game.Game`'s newly private field).

### Changed

- **The trade event reads published vectors instead of fetching them.**
  `hexset.trading.trade_event(game, gate)` drops its `valuation_of`
  parameter and reads `game.valuations` directly; a driver publishes a
  seat's vector once, right after that seat's own decision
  (`Game.publish(seat, vector)`, or `hexset.trading.publish_valuation(game,
  seat, trader)` for the common "ask the trader, then publish" case).
  `hexset.arena.play`, `hexset.record.record_game`, `hexset.bench.aivat`,
  `hexset.gym`'s auto-played opponents and the server's embedded bots all
  publish this way now. `CONTRACT_VERSION` stamps `"5"` (it stayed `"4"`
  after `RECORD_FIELDS` had already changed).
- ONNX record contract `"4"` → `"5"`: the four `offer_*` fields and
  `pair_mask` are gone, `valuations` (`players × 5` floats) is added, and a
  graph no longer needs a `pair_index` output. Contracts 2, 3 and 4 are
  refused by name.
- Flat action space 553 → 550, and `globals` 86 → 87 at four players.
- `Entrant.max_offers`, `SearchBot.max_offers`, `Heximax.max_offers` and the
  `max_offers` checkpoint metadata key are all `max_trades`; `network:<path>@0`
  replaces `@<offers>`. `hexset.server.web`'s `--max-offers` is `--no-trade`.
- `Record.offers` → `Record.trades`, with `hexset.record.steps`/`advance`
  replaying them; the server journal records a step's trades the same way.
- `hexset.server.rules` keeps only `options_for` and `is_legal`: with no
  trade action, `hexset.actions.legal_actions` is the honest list for every
  seat.

- **`Game.state` is now a method, not a field.** `game.state(seat, *,
  hidden=True)` is the access path: `hidden=True` (the default) returns
  `seat`'s information-set `View` (`hexset.view`, moved from
  `hexset.bots.heximax.belief.Belief` -- `Belief` is kept as an alias);
  `hidden=False` returns the true `GameState` (the same object every time,
  never a copy) and is the only sanctioned way to read it from outside the
  engine. The three sanctioned callers are `hexset.bots.search2`,
  heximax's own `omniscient` mode, and the Catanatron adapter when it hosts
  a Catanatron bot. `Game.set_state(state)` replaces the true state outright
  (the one write a determinizer or an undo needs) without exposing the
  now-private `Game._state` field.
- **HexSet is licensed GPL-3.0-only** (was AGPL-3.0). One licence for the
  whole distribution; third-party components are listed in `NOTICE.md`.
- **One distribution, `hexset`.** `engine/` and `src/hexset_ui/` are gone;
  everything now ships from `src/` as `hexset` (engine, bots, ledger),
  `hexset.bench`, `hexset.server`, `hexset.clients` and the sibling
  `heximax`, under one `pyproject.toml`. Update any import of `benchmarks.*`
  to `hexset.bench.*`, and of `hexset_ui.*` to `hexset.server.*` or
  `hexset.clients.*`; the PyPI/Docker distribution name changes from
  `hexset-ui` to `hexset`, and the MCP server's advertised name from
  `hexset-ui` to `hexset`. `onnxruntime` moves from a hard dependency to the
  `.[server]`/`.[clients]` extras. `hexset_ui/record.py`'s duplicate of
  `hexset.onnx_record` is deleted now that the latter no longer needs torch.
- **`hexset.bots` holds every heuristic bot.** `hexset/bots.py` and
  `hexset/evaluate.py` move into the package as `hexset.bots.search2` and
  `hexset.bots.evaluate`; `heximax` moves in alongside them as
  `hexset.bots.heximax`, split by concern (`belief`/`evaluate`/`search`/
  `trade`/`presets`). Public API unchanged: `from hexset.bots import
  SearchBot`, `from hexset.evaluate import Weights` and `import heximax`
  (now a deprecated shim) all still resolve the same names.
- The engine's own test suite now runs in place at `engine/tests` instead
  of a separate `dev-HexNet` checkout.
- **Package renamed `catan` → `hexset`**, ahead of release as its own
  public repo under GPL-3.0-only. `import catan` → `import hexset`
  throughout; `CatanNet` → `HexNet`; the `CATAN_EXPORT_COMMIT` env var →
  `HEXSET_EXPORT_COMMIT`. Every source file under `hexset/`, `benchmarks/`
  and `tests/` now carries an SPDX `GPL-3.0-only` header, and a `LICENSE`
  file is added.
- `hexset` split into `hexset` (engine, bots, ledger) and a sibling package
  `hexnet` (PPO/training research), ahead of HexSet and HexNet becoming
  separate repos. `collect`, `ddp`, `distill`, `distill_train`, `expert`,
  `export_onnx`, `league`, `migrate`, `model`, `netbot`, `policy`, `ppo`,
  `readout`, `rewards`, `schedule`, `selfplay`, `train`, `widen` and `run/`
  move to `hexnet` (e.g. `import hexset.train` → `import hexnet.train`);
  training-bound benchmarks move to `hexnet/benchmarks/`. `hexset` still
  declares only numpy and never imports `hexnet`: `hexset.arena` gained a
  registry (`register_entrant_kind`, `register_evaluator_provider`,
  `register_checkpoint_loader`, `register_leaf_evaluator_factory`) that
  `hexnet.netbot` populates on import. `relative_points` moved from
  `rewards` to `victory`; `NUM_PAIRS`/`pair_index`/`pair_mask` moved from
  `policy` to `actions` — both re-export their old names.
- `heximax` split out of `hexset` into its own top-level package
  (`hexset/heximax.py` → `heximax/__init__.py`). `hexset.arena` gained
  `register_preset` and `hexset.tuning` gained `register_heximax_evaluator`;
  `heximax` calls both on import to register its presets and evaluator
  names. `hexset` never imports `heximax`; consumers (`benchmarks.duel`,
  `hexset_ui`) import it explicitly.

### Removed

- **The offer protocol.** `Phase.TRADE_RESPOND`, `propose_trade`/
  `accept_trade`/`decline_trade`, `Offer`, `Game.offer`/`.pending_responders`/
  `.offers_made`/`.offered`, `MAX_OFFERS_PER_TURN`, `trading.responders`/
  `well_formed`/`can_propose`/`can_accept`, `actions._offer_actions`,
  `within_offer_budget`, `Action.give`/`.want`/`.ask`, `pair_index`/
  `pair_mask`/`NUM_PAIRS`, `server.rules.fair_legal_actions`/
  `proposable_options`, `SearchBot.partner_choice` and the `greedy-partner`,
  `greedy-offers1`/`2`/`3` and `search2-offers3` presets.
- The `heximax` top-level compatibility package (`import heximax` —
  use `hexset.bots.heximax`), the `hexset.evaluate` shim (use
  `hexset.bots.evaluate`), and the `Belief` alias for `hexset.view.View`.
- ONNX contract 1 is no longer served. `onnxbot` refuses a contract-1 or
  contract-unspecified checkpoint by name; the server serves contracts 2, 3
  and 4 only. `encoding_v1.py` and `OnnxPolicy` are deleted.

## 0.13.0

### Added

- `catan.widen`: function-preserving checkpoint widening (Net2WiderNet).
  Every trunk unit at width `d` is copied to fill width `D`; the wide net
  reproduces the narrow net's logits and values exactly. `--noise σ` adds
  Gaussian noise to the copies' incoming weights. `catan.train --resume`
  continues from a widened checkpoint with no new flag.
- `--mix` accepts a table entry, `table(a|b|c)=f`: in a share `f` of games
  the learner takes one seat and every other seat is an independent draw
  from the pool `a|b|c`, with replacement.
- `benchmarks.duel` records the seat geometry of every verdict — `blocked`
  (`[a, a, b, b]`) or `interleaved` (`[a, b, a, b]`) — as a new `geometry`
  field.
- `--geometry {blocked,interleaved}` on the arena path, default `blocked`
  (reproduces every prior verdict bit for bit). The versus path can only
  seat interleaved and refuses `--geometry blocked`.

### Changed

- An offer with no explicit `ask` is now put to the table in random order
  instead of clockwise from the proposer; `trading.responders` still uses
  clockwise order as the eligibility list.
- **The piece supply is now enforced: 15 roads, 5 settlements, 4 cities a
  player.** `state.can_place_settlement`, `can_upgrade_to_city` and
  `can_place_road` refuse a piece that is not in the box, so
  `legal_actions` stops offering builds that cannot be built. Previously
  unlimited.
- `benchmarks.duel` imports `catan.collect` and `catan.train` lazily, so
  the module loads on a box without torch.

## 0.12.0

### Added

- `benchmarks.aivat`: AIVAT variance reduction for duel verdicts —
  subtracts the chance-conditional expected value at every dice roll,
  dev-card draw, and robber steal from the observed outcome. `--check`
  replays a recorded verdict bit-identically to validate the estimator
  against it.

## 0.11.0

### Added

- `benchmarks.human_agreement`: scores the policy against recorded human
  decisions one decision at a time — **top-1 agreement** and **log-loss**,
  each against a matched null (uniform over the legal option set at that
  position). Decisions with a single legal action are excluded and counted
  separately. Results break down by `ActionType`, `Phase`, and game
  progress, and confidence intervals are clustered on the game rather than
  the position.

## 0.10.0

### Added

- `--mix` now accepts any arena entrant spec, not just two hardcoded
  names: `search2-offers3`, `mcts:<ckpt>@64`, `network:<ckpt>`, or any
  preset, resolved through `collect.named_opponent`. `collect.check_mix`
  refuses a mistyped entrant or missing checkpoint before the run starts.
- `collect.RESERVED_MIX` names `greedy` and `parent`, which keep resolving
  to the same bots as before.
- `benchmarks.mix_cost`: reports what a `--mix` costs a PPO iteration, per
  decision and per shard.

### Fixed

- The in-process collector's `--mix` fell through to the parent checkpoint
  for any name other than `greedy`. Both collectors now build opponents
  through the same function.

## 0.9.2

### Fixed

- **`benchmarks.rank` and `benchmarks.sibling` no longer freeze one chance
  outcome per sibling.** A chance child (a `MOVE_ROBBER`, `PLAY_KNIGHT` or
  `BUY_DEV_CARD` row) is now scored as the mean over `--chance-draws`
  independent draws (default 8), with the rollout budget for that child
  partitioned across its draws rather than duplicated. `--chance-draws 1`
  restores the old single-draw behaviour exactly. Every number either
  probe produced before this change was taken under the single-draw path.

### Added

- `catan.mcts.draws_hidden` and `catan.mcts.sampled_children`: the public
  predicate for which edges are chance edges, now shared by the tree and
  the probes.
- `benchmarks.rank.head_row`/`Row`, `.share`, `.lane_plan` and `.chance`
  (a payload reporting how many rows/children drew and the residual
  spread averaging leaves), plus `benchmarks.sibling.Spread.chance_children`
  and `.chance_spread`.

## 0.9.1

### Fixed

- `catan.mcts` no longer freezes the three chance edges (`MOVE_ROBBER`,
  `PLAY_KNIGHT`, `BUY_DEV_CARD`): each now uses a keyed `_Chance` slot so a
  repeated outcome reuses its child and the edge's `Q` becomes a true
  average over draws, instead of the first visit's single frozen outcome.
  Off-path (chance-free) search behaviour is unchanged.

### Added

- `catan.actions.victim_of` (was `_victim`) and `catan.mcts.HIDDEN_DRAW`
  (the three action types whose `apply` resolves a hidden card), both now
  public.

## 0.9.0

### Added

- A **quantile value head**: `ModelConfig(value_head="quantile")` widens
  the existing `"linear"` head to `players × quantiles` outputs
  (`ModelConfig.quantiles`, default 32); its forward still returns the
  `players`-vector mean, so `V`'s shape and meaning are unchanged
  everywhere else. The full tensor is exposed via `Prediction.quantiles`
  and `Evaluation.quantiles`.
- The matching quantile value loss (quantile Huber,
  `QUANTILE_HUBER_KAPPA = 1/30`) in `catan.ppo.minibatch_terms`;
  `quantile_levels` and `quantile_huber_loss` moved from
  `benchmarks.head_swap` into `catan.model`.
- `Stats.value_mse` / `Terms.value_mse`: the plain squared error of the
  mean, logged alongside `value_loss` and never differentiated.
- `catan.league --value-head` and `--quantiles`; asking for `quantile` off
  a scalar base warm-starts every level from the scalar head's own output
  (`catan.model.quantile_warm_start`).
- `catan.train --quantiles`, alongside the existing `--value-head`.

### Changed

- With any `value_head` other than `"quantile"`, a full training update
  remains bit-identical to `0.8.0`.

### Fixed

- `CatanNet._emit` now builds `logits`, `give`, `want` and the value read
  in a fixed, explicit order; the previous order-dependent gradient
  accumulation could make a default-config update diverge after one
  optimiser step.

## 0.8.0

### Added

- **Board-paired advantage baselines**: `Collector(pair_boards=True)`
  deals games `2k`/`2k+1` on the same board (independent dice), and
  `PPOConfig(pair_baseline=True)` makes each seat's policy-gradient
  terminal `r - (r + r')/2` against its paired game's same-seat reward.
  `catan.league --pair-boards` turns on both. With pairing off, dealing,
  casting, advantages and value targets are bit-identical to `0.7.2`.
- `benchmarks.noise_scale --paired`: runs the gradient-noise estimator on
  a board-paired cohort, reporting both the raw and pair-adjusted
  advantage streams from one batch.

## 0.7.2

### Fixed

- `run.manifest.freeze` read git provenance after creating the run
  directory, so a frozen run's `git_dirty` field was always `true`.
  Provenance is now read before anything is written.

## 0.7.1

### Added

- `catan.league --learner-order`: permutes the fixed table order the
  league rotation otherwise leaves invariant (`collect.league_caster`
  takes an `order`).

## 0.7.0

### Added

- `catan.league`: the table league — N learners share every game in one
  directory, each with its own `PPOConfig` overrides; a run is rated by
  its control arm, `learner0`.
- `catan.run`: a run is a directory with a frozen manifest
  (`freeze`/`load`) recording its parameters, resolved config and
  repository provenance before it starts.
- `catan.export_onnx`: `.pt` → `.onnx` conversion, behind a new `export`
  optional dependency; `torch` stays an unlisted hard dependency.
- A rolling checkpoint ring (`prune_recent`, keeps the newest N
  `recent-*.pt`) and blowout preservation (`preserve_blowout`, keeps the
  pre-update weights and offending batch when the training brake fires).
- Per-seat PPO overrides: `adam_eps` and an entropy-controller gain
  (`nudged`).
- `selfplay.owned` and a `learners` gate on `Collector`, so several
  learners can record from one game (`learners=(0,)` is the
  single-learner case).

### Fixed

- `catan.distill_train`'s manifest parameters no longer matched its
  parser; both are aligned again.

## 0.6.1

### Changed

- Duels are **antithetically paired by default** in `train.versus` and
  `arena.compete`: every board is played under both seat assignments and
  the two readings are averaged. Pass `antithetic=False` to reproduce the
  previous behaviour.
- Duels now draw a different seed per pair rather than sharing one
  `--duel-seed` across all comparisons.

### Fixed

- `benchmarks.duel` seeds a stochastic entrant from a hash of its **spec**
  rather than its argument position, so an entrant plays the same
  wherever it sits and a swapped duel measures the swap. Self-duels keep
  the positional tie-break.
- `benchmarks.duel` writes a verdict by default rather than only on
  request, and no longer writes `--json` output twice.
- `arena.wilson` no longer returns an upper bound above 1.0 at `p = 1`.
- A net is now rebuilt from the head shapes recorded in its own
  checkpoint (`catan.model.config_from_args`); `benchmarks.minibatch_iso_kl`
  and `benchmarks.noise_scale` previously could not load an
  `mlp`-policy-head checkpoint at all.

## 0.6.0

### Added

- `--critic {gae,none,aux}`: the value head's route into training as one
  flag; the head module is built in every mode so checkpoints stay
  loadable.
- `--kl-break`: a one-sided ceiling on a finished epoch's mean KL, with
  `epochs_taken` telemetry.
- `--fused`: the trunk's gather/scatter as dense row-normalised adjacency
  GEMMs; opt-in, since it only wins on GPU.
- A flat wire format for worker episodes, replacing the previous
  per-episode framing.
- `--rival`: every eval also duels a rival run's checkpoint at the
  matched iteration.
- `--detach-value` on `catan.train`.
- `benchmarks.generate --bot network:<path>`: records a trained
  checkpoint's self-play for `benchmarks.behaviour`.

### Changed

- Evaluation protocol: gates now decide on the matched-rival duel;
  `greedy` is demoted to a mix-exploitation canary; external anchors
  calibrate.

## 0.5.0

### Added

- `DistillConfig.contested_only` and `.hard_target`: train the policy
  only on rows where the search overruled it, toward its argmax rather
  than its visit distribution.
- `DistillConfig.anchor`: a cross-entropy toward the recorded prior on
  the rows `contested_only` zeroes out.
- `DistillConfig.stake_scale`: weight a contested row by what the
  correction is worth (the search's own Q-gap).
- `DistillConfig.buffer_iterations` and `.refresh_prior`: train on
  several iterations of collected rows, recomputing the filter and
  anchor against the live policy.
- `DistillConfig.pack_contested`: the policy loss term is trained on its
  own densely packed minibatches of contested rows.
- `ModelConfig.value_head` and `.policy_head`: ablatable readout shapes —
  `linear`, `mlp`, `pooled`, `mlp_pooled`, `attn` for the value head;
  `linear`, `mlp` for the policy head. Exposed as `--value-head` /
  `--policy-head` on both trainers and recorded in the checkpoint's
  `args`. Both default to `linear` and build identical modules, so every
  existing checkpoint stays loadable.
- `benchmarks.head_shape`: sweeps readout shapes by refitting a head on a
  frozen trunk.
- `catan.distill_train --collect-workers`: shards the searched collector
  across processes.
- Distillation statistics now split agreement by contested vs. settled
  rows, and report entropy, anchor loss, and contested-row counts.

### Changed

- `catan.ppo` and `catan.train` can cut the value loss off the trunk
  (`--detach-value`).
- `benchmarks.rank` gained a head learning-rate sweep.

### Fixed

- **`legal_actions` could re-enumerate a trade already declined this
  turn**, letting a seat spend its whole offer budget re-asking the same
  question. `Game.offered` now records the bundles put to the table this
  turn and the sample skips them; `imagine` copies the set. **This
  changes the action space every run measured before this change played
  under.**
- `--learning-rate` is now honoured on resume in the distillation
  trainer.
- The attention head's pooling query is now built off the head, not the
  trunk.
- The from-scratch benchmark arms deal 128 lanes rather than inheriting
  512.
- The zero-sum check now tolerates float32.
- Valued corpora are now collected from the parent checkpoint rather
  than the trained control.

## 0.4.0

### Added

- `catan.collect`: self-play collection sharded across worker processes
  with CPU inference.
- `catan.ddp`: the PPO update data-parallel across CPU worker processes,
  behind `--update-workers`.
- `catan.schedule`: the learning rate as a controller driven by
  `approx_kl` instead of a fixed schedule.
- `catan.selfplay.Collector.cohort` / `catan.train --collect-mode`: deal
  a fixed block of games and play every one to completion, instead of
  refilling a lane the moment its game ends. `--async-collect` now
  requires `--collect-mode stream`.
- TD(λ) value targets behind `--lam`.
- Opponent mixing in the collector, plus a frozen evaluation ladder
  including `search2-offers3` as a rung.
- `catan.ppo.Terms` carries `policy_term`, `value_term` and
  `entropy_term` (the loss summands, still attached to the graph) so a
  caller can differentiate one term at a time.
- `benchmarks.noise_scale`: the gradient noise scale (McCandlish et al.
  2018) from one collected batch, decomposed over the objective's terms.
- `benchmarks.duel`: two checkpoints head to head on identical boards,
  scored as paired terminal VP.
- `benchmarks.minibatch_iso_kl`: the learning rate that holds step
  length constant across minibatch sizes.
- `benchmarks.rank`: whether the value head orders siblings the way the
  truth does.
- `benchmarks.training_loop`: production-shape sync/async PPO timing.
- `catan.arena` takes `network:<path>` wherever a preset name is taken,
  and `netsearch:<path>` / `netgreedy:<path>` swap a checkpoint in as the
  leaf evaluation of the ordinary search. `network:<path>@<offers>`
  imposes an offer budget on the checkpoint. `arena.pooled` groups
  standings by base name.
- `catan.train` prints its effective device and worker counts and warns
  when running crippled (e.g. silently on CPU).

### Changed

- `catan.selfplay.Collector` encodes a worker tick as one vectorized
  NumPy batch laid out as the model's packed input.
- Board-template lookups are cached, and `catan.encoding._template`'s
  cache is raised to 4096 boards.
- Collection can overlap the GPU update behind `--async-collect`,
  training on a policy one iteration stale.
- `benchmarks.duel` defaults `--workers` to 26 unless both sides are
  bare networks.
- `lam` and `minibatch` are now columns in `log.jsonl`.
- The devcontainer now installs Python.

### Fixed

- **`--resume` was discarding `--learning-rate`**, along with:
  `--resume` with no checkpoint starting fresh in silence, RNG state
  restored as a CPU tensor on a CUDA resume, the KL gauge reporting a
  negative divergence on on-policy batches, and log scalars not being
  detached in `minibatch_terms`.
- `--workers` could not duel two checkpoints against each other (a
  naming collision shadowed the duel helper).
- A duel now sets an explicit torch thread count under a `--cpus` cap.
- The lambda sweep no longer oversubscribes the box with unneeded update
  workers.
- `summarise` is compatible with `distill_train` again.
- `tests/test_duel.py` no longer imports torch at module scope, so a
  torch-free box can collect the rest of the suite.

## 0.3.0

### Added

- `catan.placement`: a heuristic opening-placement prior over pip count,
  distinct resources reached, and whether a scarce resource is reached,
  fitted by conditional logit over four-player games.
- `benchmarks.placement_policy`: compares any arena entrant's setup picks
  against the prior's ranking of the same legal field.
- `Weights.scarce`: a new evaluation weight for reaching a scarce
  resource, converted via `evaluate.FITTED_SCARCE`.
- `board.scarce_resources`: resources with fewer hexes than the
  commonest.

### Changed

- Arena entrants can now be constructed by name.
- **Every arena number recorded before this release was measured with
  `scarce` at zero** — this release moves the baseline for all of them.

## 0.2.0

### Added

- `catan.distill`: distils the search's visit counts into the policy,
  with Dirichlet root noise and a bootstrapped value target.
- `catan.distill_train`: the expert-iteration training loop.
- `benchmarks.expert_scale`: how synchronized expert collection scales.
- `benchmarks.horizon`: what shortening the value horizon removes.
- `LeafEvaluator` supports fixed-shape leaf inference; the compiled
  search inference path is exposed.

### Changed

- Search leaves are batched across games rather than evaluated one at a
  time.
- Hidden deck shuffles are deferred until a draw actually needs them.
- Linear PUCT edge scores are cached, and the responder scan is no
  longer repeated per offer.

### Fixed

- A loaded network is now placed on the device it was asked for.
- Dirichlet root noise now defaults off (opt-in).

## 0.1.0

### Added

- `catan.board.coords`: cube hex coordinates, neighbours, distance, and
  hexagonal layout generation.
- `catan.board.topology`: vertices, edges and adjacency derived from any
  set of hex coordinates. Vertices are keyed by the three hexes touching
  them, so the key is canonical regardless of which hex reaches it.
  Disconnected and touching islands are supported.
- `catan.board.terrain`: resource and terrain types, including sea and
  gold for Seafarers.
- `catan.board.board`: terrain and number tokens, the official setup
  bags, and the variable-setup rule keeping 6 and 8 off adjacent hexes.
- `catan.board.maps`: base and mini layouts, plus multi-island layout
  construction.
- `catan.state`: occupancy, hands and bank stock; placement legality for
  settlements, cities and roads expressed as layout-agnostic graph
  queries; gross production; gold hex claim counts.
- `catan.economy`: build costs, affordability and payment, bank trades
  at the best rate the player's ports allow, and production payout
  applying the official bank shortage rule.
- `catan.board.ports`: coastlines derived from hex/edge adjacency, and
  the nine base-game ports.
- `catan.roads`: longest road as a longest trail, so loops count in
  full and an opponent's building breaks a route without invalidating
  the roads either side.
- `catan.cards`, `catan.devcards`: the 25-card deck, buying, and the
  four playable effects. Cards bought this turn are held aside until it
  ends.
- `catan.robber`: robber movement, stealing weighted by the victim's
  hand, and discarding on a seven.
- `catan.victory`: victory points, plus longest road and largest army
  with the rule that a challenger must beat the holder outright.
- `catan.game`: the turn and phase machine — snake-order setup,
  rolling, discarding, the robber, the main phase and win detection.
- `catan.actions`: a flat action space sized from the board, with
  legality masking.
- `catan.play`: a random player that plays full games end to end.
- `benchmarks.throughput`: games/sec measurement with environment
  recording.
- `catan.evaluate`: handcrafted position scoring, one score per seat
  rather than a scalar. Combines victory points, expected cards per
  turn, resource diversity, reachable production, progress towards the
  nearest purchase, roads, knights, hand size with a discard penalty,
  and port rates, as an ablatable weights dataclass.
- `catan.bots`: a `Bot` protocol the network will also satisfy, a random
  bot, and a max^n search over the evaluation (`greedy` is the one-ply
  case). Rolls are chance nodes weighted over all eleven outcomes rather
  than sampled; a beam bounds the main phase's branching.
- `catan.game.imagine`: a copy for hypothetical play, with its own
  random stream and a reshuffled deck so a search cannot read the real
  game's upcoming draw.
- `catan.game.to_move`: whose decision the legal actions are — not
  always the current player, e.g. while discarding on a seven.
- `catan.state.copy_state`, `catan.game.ROLL_ODDS`.
- `catan.arena`: head-to-head play with the lineup rotated so every
  entrant sits every seat equally, win rates reported with a Wilson
  interval, an action-per-game cap, and every seat's terminal victory
  points kept alongside the winner.
- `benchmarks.baselines`: runs a lineup and records the commit and
  environment with the result.
- `catan.tuning`: fits the evaluation weights by hill-climbing against
  the incumbent through the arena, with a `confirm` step that plays the
  fitted weights against the starting weights at a large budget.
- `benchmarks.tune`: runs the climb, reporting each duel as it resolves.
- `benchmarks.production_curve`: sweeps candidate `production` values
  against the intact weights to test whether the weight is identifiable
  from self-play at all.
- `catan.evaluate_tiered`: a second evaluation, selectable through the
  `greedy-tiered` and `search2-tiered` presets, kept as a comparison
  baseline rather than the default.
- `catan.encoding`: the heterogeneous graph the model reads. Seats are
  rotated so the player to move is always seat 0; only information the
  perspective player may legally know is encoded (own hand and cards
  exactly, opponents as counts). Board adjacency is cached per board.
- `catan.selfplay`: a vectorised rollout collector holding N games in
  flight and stepping them in lockstep, behind a `BatchPolicy` protocol.
  Trajectories are demultiplexed by seat; the decision-maker is
  `to_move` rather than `current_player`. Finished lanes are refilled in
  place, and an action cap truncates a game that will not end. No
  built-in reward — an `Outcome` reports the winner, terminal points,
  turns and truncation.
- `benchmarks.rollout`: ticks/sec and actions/sec for the collector
  under a trivial policy.
- `catan.rewards`: the scalarisation `catan.selfplay` leaves open —
  terminal victory points read against the mean of the other seats and
  scaled by the ten points a game is won on, zero-sum by construction
  with no discount factor. A truncated game is scored where it stopped.
- `catan.policy`: the torch `BatchPolicy`. One forward, one packed
  host-to-device copy and one concatenated read-back per tick.
  `PROPOSE_TRADE` is a single flat slot; the recorded `log_prob` is the
  joint over slot and offer.
- `catan.ppo`: GAE over per-seat trajectories, the clipped surrogate,
  value loss and entropy bonus. `GAMMA` is a module constant, not a
  config field. The value head is trained on terminal outcomes and never
  bootstrapped.
- `catan.train`: the runnable, resumable loop. Checkpoints are written
  to a temporary file and renamed; the game counter is saved with the
  weights so a resumed run replays its own training set. `--eval-at-start`
  duels the untrained network as a baseline, and duels fix their cohort
  of games in advance.
- `benchmarks.value_head`: what the value head explains, split by stage
  of the game.
- `catan.netbot`: a trained checkpoint as a `catan.bots.Bot`, so a
  network can enter the arena. The checkpoint is loaded once per process,
  keyed on topology and path; the offer budget defaults to the one the
  checkpoint recorded training under.
- `catan.mcts`: PUCT over a learned policy and value, with leaves
  gathered into waves and evaluated together. A node backs up a per-seat
  vector through `catan.bots.STANCES` instead of a scalar and sign flip;
  chance nodes are sampled rather than expanded eleven ways; nodes store
  their positions; terminal nodes take `catan.rewards.relative_points`
  directly. `simulations` counts descents that cross an edge.
- `catan.expert`: `SearchPolicy`, a `BatchPolicy` that runs one tree per
  decision, so expert-iteration games come out of the existing
  `Collector`. Visit counts ride to the transition on `Choice.aux` as a
  `Target`; the recorded value is the root's backed-up mean; actions are
  sampled, not argmaxed.

### Changed

- `catan.actions.Action` carries an `ask` order on `PROPOSE_TRADE`,
  naming who the proposer would rather have take the offer — an offer
  stops at the first player to accept. Enabled by
  `SearchBot(partner_choice=True)` and the `greedy-partner` preset;
  works only under the `paranoid` stance. Records carry the order, so a
  game with a choosing proposer replays.
- `catan.trading.responders` orders an offer round the table from the
  proposer rather than by ascending seat index.
- `relative` is now the default stance for `greedy` and `search2`;
  `greedy-own` and `search2-own` reproduce the old behaviour.
- `benchmarks.throughput.environment` reports whether the working tree
  was dirty; runners default `--workers` to every core.
- `catan.tuning` fits either evaluation, taking an evaluator name and
  resolving the matching `Weights` class and stance; `TUNABLE` becomes
  `tunable(weights)`.
- `benchmarks.ablate` takes an `--evaluator`, so the tiered evaluation's
  terms can be ablated the same way the default's are.
- `catan.bots.SearchBot` takes a `stance` saying how a seat turns the
  per-seat vector into the one number it maximises: `own` (plain max^n),
  `relative` (subtracts the mean of the other seats) or `paranoid`
  (subtracts the best of them). Selectable through the `greedy-relative`,
  `greedy-paranoid` and `search2-relative` presets.
- `catan.arena` entrants carry which evaluation to score with, so two
  evaluations can be played against each other directly.
- `catan.game.roll_dice` takes an optional explicit roll, so a search
  can enumerate the outcomes instead of sampling one.
- `catan.arena` entrants are now a frozen `Entrant` description rather
  than a bot-building closure (`FACTORIES` → `PRESETS` and `spawn`), so
  `compete` can fan out over a process pool and a lineup can go into a
  run manifest verbatim.

## Before 0.13 — hexset-ui 0.1.0

`hexset_ui`'s first release, before it consumed the `hexset` package (it
carried a private copy of the engine; see `docs/engine-divergence-2026-09-02.md`).

### Added

- A browser game of humans against bots: HTTP API with a lobby, MCP server,
  static web client, and ONNX Runtime inference for exported networks.
