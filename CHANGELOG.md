# Changelog

All notable changes to the `catan` package are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-22

An opening-placement prior, fitted by conditional logit over 40,803 recorded
four-player games, and the one evaluation term that fit produced.

### Added

- `catan.placement` — a heuristic opening prior over three terms: pip count,
  distinct resources reached, and whether a scarce resource is reached. Weights
  are expressed in pips and come from a conditional logit over the four seats of
  a game, which cancels the board because exactly one seat wins. Number
  diversity, port access, complementarity between the two settlements, pip
  balance and denial of a rival's best corner were all offered to the fit and
  were all null at fixed pips.
- `benchmarks.placement_policy` — walks any arena entrant through the eight
  setup picks and compares each choice against the prior's ranking of the same
  legal field. Three numbers separate "never learned to open", "matches the
  prior", and "diverges but still beats the field". A trained checkpoint already
  opens like the prior.
- `Weights.scarce` — how many scarce resources a seat reaches, the only weight
  not fitted against the engine. It comes from the opening fit at 0.91 pips,
  converted at `production / ROLLS` VP per pip by `evaluate.FITTED_SCARCE`, with
  a test pinning the default to it. Adopted untuned and it won anyway: 51.62%
  [50.75, 52.48] over 12,800 games against the same evaluation without it.
- `board.scarce_resources` — the resources with fewer hexes than the commonest,
  so brick and ore on the base map.

### Changed

- Arena entrants can be constructed by name, so a benchmark can ask any of them
  the same question.
- **Every arena number recorded before 2026-08-15 was measured with `scarce` at
  zero**, so this release moves the baseline for all of them.

## [0.2.0] - 2026-08-22

Search throughput and the first expert-iteration loop. Distillation closes
negative: it degraded the policy rather than improving it, and the benchmarks
here are what diagnose why.

### Added

- `catan.distill` — distils the search's visit counts into the policy, with
  Dirichlet root noise at the root to stop convention collapse and a
  bootstrapped value target.
- `catan.distill_train` — the expert-iteration training loop.
- `benchmarks.expert_scale` — how synchronized expert collection scales.
- `benchmarks.horizon` — what shortening the value horizon removes, including
  the statistic that separates a real target from a self-referential one.
- `LeafEvaluator` supports fixed-shape leaf inference, and the compiled search
  inference path is exposed.

### Changed

- Search leaves are batched across games rather than evaluated one at a time.
- Hidden deck shuffles are deferred until a draw actually needs them.
- Linear PUCT edge scores are cached, and the responder scan is no longer
  repeated per offer.
- The MCTS wave size is a named quantity rather than an inline constant.

### Fixed

- A loaded network is placed on the device it was asked for, instead of
  whichever one it was saved from.
- Dirichlet root noise defaults back off, so it is opt-in rather than silently
  applied to ordinary search.

## [0.1.0] - 2026-08-22

### Added

- `catan.board.coords` — cube hex coordinates, neighbours, distance, and hexagonal
  layout generation.
- `catan.board.topology` — vertices, edges and adjacency derived from any set of hex
  coordinates. Vertices are keyed by the three hex positions touching them, so the key
  is canonical regardless of which hex reaches it. Verified against the base board
  (19 hexes / 54 vertices / 72 edges), a mini board (7/24/30), and Euler's
  characteristic across radii 0-4. Disconnected and touching islands are supported.
- `catan.board.terrain` — resource and terrain types, including sea and gold for
  Seafarers.
- `catan.board.board` — terrain and number tokens, the official setup bags, and the
  variable-setup rule keeping 6 and 8 off adjacent hexes.
- `catan.board.maps` — base and mini layouts, plus multi-island layout construction.
- `catan.state` — occupancy, hands and bank stock; placement legality for settlements,
  cities and roads expressed as graph queries so it is layout-agnostic; gross
  production, and gold hex claim counts.
- `catan.economy` — build costs, affordability and payment, bank trades at the best
  rate the player's ports allow, and production payout applying the official bank
  shortage rule.
- `catan.board.ports` — coastlines derived from hex/edge adjacency, and the nine
  base-game ports spaced around the longest ring.
- `catan.roads` — longest road as a longest trail, so loops count in full and an
  opponent's building breaks a route without invalidating the roads either side.
- `catan.cards`, `catan.devcards` — the 25-card deck, buying, and the four playable
  effects. Cards bought this turn are held aside until it ends.
- `catan.robber` — robber movement, stealing weighted by the victim's hand, and
  discarding on a seven.
- `catan.victory` — victory points, plus longest road and largest army with the
  rule that a challenger must beat the holder outright.
- `catan.game` — the turn and phase machine: snake-order setup, rolling, discarding,
  the robber, the main phase and win detection.
- `catan.actions` — a flat action space sized from the board, with legality masking.
  Board-local actions get one slot per node.
- `catan.play` — a random player that plays full games end to end.
- `benchmarks.throughput` — games/sec measurement with environment recording.
- `catan.evaluate` — handcrafted position scoring, one score per seat rather than a
  scalar, matching the planned value head. Combines victory points, expected cards per
  turn, resource diversity, the production reachable within two roads discounted by
  distance, progress towards the nearest purchase, roads, knights, hand size with a
  discard penalty, and port rates. Weights are an ablatable dataclass.
  `pips` lives in `catan.board.board` so the encoder can share it.
- `catan.bots` — a `Bot` protocol the network will also satisfy, a random bot, and a
  max^n search over the evaluation. Depth counts decisions rather than turns; rolls are
  chance nodes weighted over all eleven outcomes rather than sampled; a beam bounds the
  main phase's branching. `greedy` is the one-ply case.
- `catan.game.imagine` — a copy for hypothetical play. Takes its own random stream so a
  search cannot disturb the real game's, and shuffles the copied deck so a search
  cannot read the card the real deck is about to deal.
- `catan.game.to_move` — whose decision the legal actions are, which is not the current
  player while discarding on a seven.
- `catan.state.copy_state`, `catan.game.ROLL_ODDS`.
- `catan.arena` — head-to-head play with the lineup rotated so every entrant sits every
  seat the same number of times, and win rates reported with a Wilson interval. Caps
  actions per game, which the engine's turn cap cannot do. Every seat's terminal victory
  points are kept alongside the winner, so a game says how close the losers came rather
  than only that they lost; `mean_interval` turns differences taken *within* a game into
  a paired estimate, which cancels board and dice variance instead of averaging it away.
- `benchmarks.baselines` — runs a lineup and records the commit and environment with the
  result.
- `catan.tuning` — fits the evaluation weights by hill climbing against the incumbent
  through the arena. The scale is pinned at `victory_point`, since scaling every weight
  alike cannot change the argmax; acceptance needs the win-rate interval's lower bound
  to clear half, with the strictness a knob because both extremes fail. `confirm` plays
  the fitted weights against the starting weights at a large budget, which is the only
  way to tell a real gain from an accumulation of accepted noise.
- `benchmarks.tune` — runs the climb, reporting each duel as it resolves.
- `benchmarks.production_curve` — maps candidate `production` values against the intact
  weights to ask whether the weight is identifiable from self-play at all, and reuses
  the same games to test terminal points as a proxy for wins. The answer over 18,000
  games is that it is not: only `3.0` is distinguishable once the eight alternatives are
  corrected for, a 58% cut, and `3.5` through `10` are indistinguishable. Terminal
  points are a sound proxy (Pearson 0.998 across the curve, 0.907 restricted to the
  plausible range, no significant sign conflicts). An evaluation can depend critically
  on a term at zero while being flat over every value an optimiser would ever try, and
  no search rule can recover a coefficient in that landscape.
- `catan.evaluate_tiered` — a second evaluation, reimplemented from the design
  catanatron's value function uses (described, not ported; it is GPLv3). Weights are
  magnitude tiers encoding a priority order rather than blended coefficients, own
  production is scored against the strongest opponent's, and reachable production keeps
  the player's own settled junctions in the set so building can never shrink it. Kept
  as a comparison baseline, not as the default. It played `catan.evaluate` to a dead
  heat (51.3%) under plain max^n before trading; it now loses 36.7% over 2000 games,
  and a refit under `relative` confirmed at 48.1%, so the gap is not a stale fit.
  Generates data at half the rate. Selectable through the `greedy-tiered` and
  `search2-tiered` presets.

- `catan.encoding` — the heterogeneous graph the model reads. Seats are rotated so
  the player to move is always seat 0, and only information the perspective player
  may legally know is encoded: own hand and cards exactly, opponents as counts.
  Board adjacency is cached per board, since it never changes during a game.
- `catan.selfplay` — a vectorised rollout collector. Holds N games in flight and steps
  them in lockstep so one batch of observations per tick serves every lane, which the
  network's ~1.5 ms fixed dispatch toll makes mandatory rather than preferable. The
  policy sits behind a `BatchPolicy` protocol — the batched analogue of `catan.bots.Bot`
  — so the collector imports no torch and is tested against a random policy on the
  development machine. Trajectories come out demultiplexed by seat, since a seat's next
  state is the next position that seat was asked about rather than the one following its
  action, and the decision-maker is `to_move` rather than `current_player`. Finished
  lanes are refilled on the spot so a long game never stalls the batch, and the action
  cap truncates a game that will not end. Reward is deliberately absent: an `Outcome`
  reports the winner, every seat's terminal points, turns and truncation, and the
  scalarisation is the caller's.
- `benchmarks.rollout` — ticks/sec and actions/sec for the collector under a trivial
  policy, so the cost of the plumbing is known separately from the cost of a network.
  Sweeping the lane count shows actions/sec roughly flat while ticks/sec falls with the
  lane count: lanes buy batch size, not throughput.
- `catan.rewards` — the scalarisation `catan.selfplay` deliberately left open, now
  settled: terminal victory points read against the mean of the other seats and scaled
  by the ten points a game is won on. Points rather than win/loss because a losing seat
  still says how close it came and the two are near-perfectly ranked on this engine;
  relative rather than absolute because Catan is a race, which is the argument that
  already decided the search's stance. Zero-sum by construction, which is what rules
  out a discount factor: about half of these rewards are negative, so a gamma below 1
  makes a late loss cheaper than an early one and pays a losing policy to stall. A
  truncated game is scored where it stopped, so stalling banks the deficit rather than
  escaping it.
- `catan.policy` — the torch `BatchPolicy`. One forward, one packed host-to-device
  copy and one concatenated read-back per tick. `PROPOSE_TRADE` is a single flat slot
  carrying ten numbers, so picking it means the policy has chosen to propose without
  yet saying what; it names the offer from the model's `give` and `want` heads, masked
  to the offers that were legal. The recorded `log_prob` is the joint over slot and
  offer, because PPO's ratio has to be taken against the distribution that generated
  the data — recording only the slot's is wrong by the offer's log-prob and wrong most
  where the policy is most confident. `evaluate` mirrors `act` and a test pins them
  together.
- `catan.ppo` — GAE over per-seat trajectories, the clipped surrogate, value loss and
  entropy bonus. `GAMMA` is a module constant rather than a config field and a test
  pins the absence of a `--gamma` flag. The value head is trained on terminal outcomes
  and never bootstrapped: with gamma 1 and a reward that is zero until the end, the
  Monte Carlo return is the terminal reward exactly. All of the head's per-seat outputs
  are trained from every position, which is what makes it backable-up by max^n later.
- `catan.train` — the runnable, resumable loop. Checkpoints are written to a temporary
  file and renamed, so a crash during the save cannot destroy the last good one, and
  the game counter is saved with the weights because a game is a pure function of the
  seed and its index — a resumed run that restarts it replays its own training set.
  Collect and update time are reported separately. `--eval-at-start` duels the
  untrained network so "did it learn" has a baseline, and duels fix their cohort of
  games in advance rather than taking the first to finish, which would select for short
  games.
- `benchmarks.value_head` — what the value head explains, split by stage of the game.
  A run's `explained_variance` is a ratio whose denominator moves: the target is
  terminal relative points, and four strong seats finish closer together than four weak
  ones, so the figure can fall while the head's error falls too. This reports the
  numerator and the denominator separately, off the predictions the collector already
  recorded rather than a second forward.
- `catan.netbot` — a trained checkpoint as a `catan.bots.Bot`, which is what lets a
  network enter the arena and be scored against the handcrafted baselines rather than
  only against uniform random. The adapter goes this way round because `catan.arena`
  already has seat rotation, Wilson intervals, a process pool and the offer budget.
  The checkpoint is loaded once per process and keyed on the topology as well as the
  path, since `arena.spawn` runs per game per worker and a `torch.load` there would be
  most of what a duel measured; `torch.set_num_threads(1)`, because thirty workers each
  taking a core's worth of intraop threads measures the thrash. Evaluation is greedy,
  and the offer budget defaults to the one the checkpoint recorded training under —
  scoring a three-offer policy at the engine's eight would measure it on a horizon it
  never saw.
- `catan.mcts` — PUCT over a learned policy and value, with leaves gathered into waves
  and evaluated together. Batching is the whole point: a forward costs a ~1.5 ms fixed
  dispatch toll plus ~25 us per position, so one leaf per call spends all of its time in
  dispatch, and virtual loss is what stops every simulation in a wave picking the same
  path. Four departures from the Go setting, each measured on this codebase rather than
  imported: a node backs up a per-seat vector read through `catan.bots.STANCES` instead
  of a scalar and a sign flip; chance nodes are sampled rather than expanded eleven ways,
  because under a fixed budget the frequencies approximate the same distribution; nodes
  store their positions, since replay costs ~19 us of engine per ply against ~25 us to
  evaluate; and terminal nodes take `catan.rewards.relative_points` directly, which is
  the same scale the value head is trained on. `simulations` counts descents that cross
  an edge, so root visit counts always sum to it and `visit_policy` means the same thing
  at any wave size. This does not replace `netsearch:<path>` on quality — that is a
  separate problem, and an unaddressed one — only on throughput.
- `catan.expert` — `SearchPolicy`, a `catan.selfplay.BatchPolicy` that runs one tree per
  decision, so expert-iteration games come out of the existing `Collector` rather than a
  second game loop: the lane bookkeeping, seat demultiplexing, action cap and seed/index
  replay contract are already there and none of them care what decided the action. The
  visit counts ride to the transition on `Choice.aux` as a `Target`, which keeps the
  options beside them because `PROPOSE_TRADE` is one slot standing for many offers and
  the split is not recoverable from the index. Counts are kept raw so a distillation step
  picks its own temperature. The recorded value is the root's backed-up mean rather than
  the network's own estimate of the root, which is the thing being improved on. Actions
  are sampled, not argmaxed: four identical greedy searches replay one game.

### Changed

- `catan.actions.Action` carries an `ask` order on `PROPOSE_TRADE`, naming who the
  proposer would rather have take the offer. An offer stops at the first player to
  accept, so the order is worth something, and choosing it is a tactic rather than a
  rule — `trading.responders` stays neutral and anyone unnamed keeps its place behind
  those named. Records carry the order, so a game with a choosing proposer replays.
  Enabled by `SearchBot(partner_choice=True)` and the `greedy-partner` preset. It works
  only under the `paranoid` stance: `relative` subtracts the mean of the other seats,
  and a trade hands the same value to whoever takes it, so the mean moves identically
  whichever opponent received it. Measured at 49.8% (95% CI [47.6%, 51.9%], 2000 games)
  against an identical bot that does not choose — it earns nothing, and is kept because
  it is what the real game does and puts the decision where a policy can learn it.
- `catan.trading.responders` orders an offer round the table from the proposer rather
  than by ascending seat index. First refusal is worth something, and seat order handed
  it permanently to the low seats: wins by seat ran 563/532/482/423 over 2000 games,
  chi-square 22.5 on three degrees of freedom, falling to 7.2 under rotation.
- `relative` is now the default stance for `greedy` and `search2`, since a baseline
  whose job is to be beaten should be the best one available. `greedy-own` and
  `search2-own` reproduce the old behaviour, so a recorded duel should name both sides
  explicitly rather than rely on the default. Refitting the weights for a stance was
  tried and is not needed: under `relative` the climb confirmed at 49.0% and under
  `paranoid` at 52.2%, and the paranoid fit carrying its own weights then lost to
  `relative` on the existing ones. Retaking the ablation under `relative` moved it a
  lot — seven of nine terms now earn their keep against four, and `search2` beats
  `greedy` 60.8% rather than 70.0% because `greedy` got stronger.
- `benchmarks.throughput.environment` reports whether the working tree was dirty, and
  asks git with `safe.directory` set on the command line, so runs inside the
  devcontainer stop recording their commit as "unknown". Runners default `--workers`
  to every core; defaulting to one meant a forgotten flag silently measured on a
  single core.
- `catan.tuning` fits either evaluation, taking an evaluator name and resolving the
  matching `Weights` class, and takes a stance. `TUNABLE` becomes `tunable(weights)`.
- `benchmarks.ablate` takes an `--evaluator`, so the tiered evaluation's terms can be
  ablated the same way the default's are. It read `catan.evaluate.Weights` directly and
  could only ever ablate the default nine.
- `catan.bots.SearchBot` takes a `stance` saying how a seat turns the per-seat vector
  into the one number it maximises. `own` is plain max^n and the previous behaviour;
  `relative` subtracts the mean of the other seats and `paranoid` the best of them.
  Catan has one winner, so a position is worth what it is worth compared to the table,
  and under `own` a responder took every trade that improved its own hand however much
  it improved the proposer's. `relative` beats `own` 55.0% (95% CI [52.9%, 57.2%],
  2000 games) while carrying weights fitted under `own`, and cuts the share of offers
  accepted from 47.5% to 29.1%. Selectable through the `greedy-relative`,
  `greedy-paranoid` and `search2-relative` presets.
- `catan.arena` entrants carry which evaluation to score with, so the two can be
  played against each other directly.
- `catan.game.roll_dice` takes an optional explicit roll, so a search can enumerate the
  outcomes instead of sampling one.
- `catan.arena` entrants are a frozen `Entrant` description rather than a bot-building
  closure, replacing `FACTORIES` with `PRESETS` and `spawn`. Closures cannot be pickled,
  so this is what lets `compete` fan out over a process pool — and it means a lineup can
  go into a run manifest verbatim. Results are identical at any worker count.

- `catan.encoding._template` is cached 4096 boards deep, twice raised. At 8 a
  sixteen-lane rollout missed on every call and rebuilt the board-static block the
  cache exists to avoid — 7.5k actions/sec against 8.5k, 1.13x, over three alternating
  runs. 64 was the same mistake one size up: PPO wants the largest batch the dispatch
  toll amortises over, measured at 512 lanes, and 512 boards miss a 64-entry cache
  every time. A/B'd in one process, alternating: 84.2 → 67.6 µs per position at 128
  lanes (1.25x), 118.5 → 101.1 at 512 (1.17x), 126.9 → 115.8 at 1024 (1.10x). The cost
  still climbs with lane count after the fix, so most of that rise is working set and
  raising the cache again will buy nothing.
- `catan.arena` takes a `kind="network"` entrant whose `weights` is a checkpoint path,
  named on a command line as `network:<path>` wherever a preset name is taken. It stays
  a picklable description, so a lineup with a network in it still goes into a manifest
  verbatim and still crosses a process. `arena.pooled` groups standings by base name,
  since a duel is two seats a side and the side's share of the games is the number worth
  quoting; `benchmarks.baselines` prints it under the per-seat standings. Setting
  `evaluator="network"` instead swaps the checkpoint in as the *leaf evaluation* of the
  ordinary search, which needs no new bot kind: `netsearch:<path>` is `search2` with
  learned leaves and `netgreedy:<path>` is one ply of the same. `search2-offers3` is the
  handcrafted search on the training horizon, so that comparison differs in the leaf
  evaluation and nothing else.
- `catan.selfplay.Request` carries the lane's `Game`. A searching policy needs positions
  to step and an observation is a lossy encoding of one; a policy that reads only the
  encoding can ignore it. Handed out rather than copied, because `catan.mcts` copies at
  its own root.
- `catan.selfplay.Collector` takes `deal`, bounding how many games are ever started,
  with `running` and `drain()` to play the bounded cohort out. Left unset the collector
  refills a freed lane immediately and runs forever, which is what training wants. An
  evaluation wants a fixed set of game indices, and the only way to get one before this
  was to keep dealing replacements and discard them — after playing each in full, which
  is where a 400-game duel went to spend over ten minutes. `catan.train.duel` and
  `benchmarks.value_head` now bound instead of filtering.
- `catan.train --keep-every` writes numbered checkpoints beside the `latest.pt` that
  gets overwritten, defaulting to every 25 iterations. The first run kept only `latest`,
  so "when did the policy stop improving" had no way to be asked afterwards.
- `catan.selfplay.Choice` and `Transition` carry an `aux` field, passed through
  untouched, and `Collector` takes `first_game` with `games_started()` to read it back.
  The first is where the torch policy stores the offer mask PPO must reuse, which
  depends on what opponents could cover and so is absent from the observation by
  design; the second is what makes a resumed run continue rather than repeat. Both
  default to the previous behaviour.
- The package declares `numpy>=1.26` as a dependency. The rules engine still has none;
  the encoder is what needs it. Torch is not declared, because the engine, the arena and
  the collector are all tested without it and only the learning layer imports it.
