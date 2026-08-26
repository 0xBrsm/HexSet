# Changelog

All notable changes to the `catan` package are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.1] - 2026-08-26

### Changed

- Duels are **antithetically paired by default**, in `train.versus` and in
  `arena.compete`. Every board is played under both seat assignments -- same
  board, same dice, same per-entrant stream, differing in who sits where and
  nothing else -- and the two readings are averaged per board. `alternating`
  keys the cast to the game index and the board is keyed to the index too, so a
  single cohort sampled each side on one seat-pair per board and never the
  other: the mean seat effect cancelled, the parity-correlated residual did not,
  and it was **55% of a single-order duel's variance** while being reported as
  ordinary noise. It is also why swapping a duel's arguments did not negate its
  result. A checkpoint duelled against itself read -0.771 VP over 48 games
  before, and exactly 0.000 after. Pass `antithetic=False` to reproduce an
  earlier reading; readings taken before this are valid measurements whose
  intervals were too narrow.
- Duels want a **different seed per pair**. Six comparisons sharing one
  `--duel-seed` share one draw of the seat residual, not six independent tests.

### Fixed

- `benchmarks.duel` seeds a stochastic entrant from a hash of its **spec**
  rather than from its argument position, so an entrant plays the same wherever
  it sits and a swapped duel measures the swap. Self-duels keep the positional
  tiebreak deliberately, since one stream for both sides would search in
  lockstep.
- `benchmarks.duel` **writes a verdict by default** rather than only on
  request. A 400-game mcts-against-its-own-policy result had been written up in
  prose and nowhere a tool could read it, so the ratings fit never saw it and
  placed that entrant half a VP wrong off a single unrelated duel.
- `benchmarks.duel --json` wrote the result twice: a second append survived the
  change that introduced the verdict default.
- `arena.wilson` returned an upper bound of 0.9999999999999999 at `p = 1`, an
  interval excluding its own point estimate.
- A net is rebuilt from the head shapes its checkpoint records, in one reader
  (`catan.model.config_from_args`). `benchmarks.minibatch_iso_kl` and
  `benchmarks.noise_scale` read only `width` and `rounds`, so neither could load
  a `--policy-head mlp` checkpoint at all — `heads.hexes.weight` against
  `heads.hexes.0.weight` — which is every checkpoint of the current lineage.
  `catan.netbot` was already correct and now shares the reader. The probe also
  hands those shapes to its collect workers, which build their own nets and
  would otherwise fail at the first weight sync, and both benchmarks mirror
  `catan.train`'s `detach_value` wiring so they reproduce the update they price.

## [0.6.0] - 2026-08-23

The critic decomposed, the pipeline made ~2.5x cheaper per experiment, and the
evaluation protocol rebuilt around matched opponents after the common-opponent
instrument deceived four separate readouts.

### The finding that reorganises training

The value head's ~2.5 VP contribution is the complete loop or nothing. Cutting
both of its wires (REINFORCE on the zero-sum return), only the pricing wire
(an auxiliary value loss nothing reads), or only the shaping wire (GAE over a
trunk-detached head) all land 2.3-2.9 VP below the control at matched
iterations — the wires are complements, not substitutes. Around it: reuse is
dead in both directions (2 << 4 ~= 8 epochs, measured), one iteration of
collection staleness costs 0.7-0.9 VP early, and the greedy lane opponent is
convicted of the accept-rate drift — its removal coincided with closing two
thirds of the gap to the previous campaign's ceiling, with the 100-extra-
iterations confound stated in the record rather than argued away.

### Added

- `--critic {gae,none,aux}` — the value head's two routes into training as one
  flag, with the head module built in every mode so checkpoints stay loadable.
- `--kl-break` — a one-sided ceiling on a finished epoch's mean KL, with
  `epochs_taken` telemetry; the damage band (0.018, 0.045) is measured, and its
  first live firing bounded a record blowout to two epochs.
- `--fused` — the trunk's gather/scatter means as dense row-normalised
  adjacency GEMMs with blockwise round MLPs; -18% on the GPU update, kept off
  the CPU path where it measured slower. Equivalence pinned across head shapes.
- A flat wire format for worker episodes: cohorts cross the pipes as a few
  large arrays and rebuild byte-identical, collect 15 -> 9 s and assemble
  3.1 -> 1.2 s an iteration, with `pack`'s gather path restored.
- `--rival` — every eval also duels a rival run's checkpoint at the *matched*
  iteration, so recipe-vs-recipe signal arrives in-run; matched or recorded as
  a miss, never nearest-neighbour.
- `--detach-value` on `catan.train`, plumbing the distillation trainer's
  existing gradient cut.
- `benchmarks.generate --bot network:<path>` records a trained checkpoint's
  self-play for `benchmarks.behaviour`.

### Changed

- Evaluation protocol: gates decide on the matched-rival duel; greedy is
  demoted to a mix-exploitation canary; external anchors calibrate. Four
  distinct ways the greedy ladder can deceive a gate are what demoted it.

## [0.5.0] - 2026-08-22

Distillation, made to work on the 5% of decisions where the search actually
overrules the policy — and a value-head shape study that says the head's
blindness to siblings is not a readout shape.

### The finding that reorganises the loss

The search moves the policy's argmax on ~4.8% of decisions, and the signed value
of those moves is +0.040 VP each, which over ~10-15 contested decisions a
seat-game reproduces the duel's +0.536 VP. The teacher's whole content is in
that 5%. The other 95% is the policy's own answer handed back — and handing back
the visit *distribution* is worse than handing back nothing, because its entropy
is set by the search's exploration settings rather than by the position. Every
distillation arm on record converged to the tree's stationary entropy
(~0.43-0.46 against the parent's 0.339) whatever else changed. Play reads the
argmax; training read the distribution.

### Added

- `DistillConfig.contested_only` and `.hard_target` — train the policy on the
  rows where the search overruled it, toward its argmax rather than its counts.
- `DistillConfig.anchor` — a cross-entropy toward the *recorded prior* on the
  rows `contested_only` zeroed. Filtering the policy loss cannot filter its
  effect: the trunk is shared, the value loss is an unweighted mean over every
  row, and a network has no per-position parameters. Unanchored, top-1 agreement
  with the search fell 0.941 to 0.788 while trade acceptance went 3.3% to 23.7%.
  Anchoring on the *visits* is what flattened the earlier arms; the prior's
  entropy is the policy's own, so it carries no flattening pressure. It is PPO's
  trust region spelled as a cross-entropy.
- `DistillConfig.stake_scale` — weight a contested row by what the correction is
  worth, `min(gap / stake_scale, 1)` over the search's own Q-gap, dropping rows
  whose gap is negative. 11% of contested rows are ones the search's own value
  disagrees with.
- `DistillConfig.buffer_iterations` and `.refresh_prior` — train on several
  iterations of collected rows, recomputing the filter and anchor against the
  live policy. Safe here where it is not safe for PPO: distillation is a
  supervised cross-entropy against a fixed label, with no ratio to correct, and
  a visit count is local to its position. The search costs ~700 s a corpus while
  the prior it is compared against is one forward pass, so refreshing turns the
  97% carrying no policy gradient from waste into not-yet-contested.
- `DistillConfig.pack_contested` — the policy term gets its own dense
  minibatches. At ~3% contested density a 1024-row minibatch carries ~31
  contested rows, so the policy loss was a 31-sample estimate taken ~419 times an
  iteration; packed it is ~3 steps of 1024 an epoch at a sixteenth of the
  per-step variance. Same shape PPG uses, for the same reason. The anchor rides
  the value pass deliberately, since that is the pass that moves the trunk.
- `ModelConfig.value_head` and `.policy_head` make the readout shapes ablatable:
  `linear`, `mlp`, `pooled`, `mlp_pooled` and `attn` for the value; `linear` and
  `mlp` for the policy. Exposed as `--value-head` / `--policy-head` on both
  trainers and recorded in the checkpoint's `args`, so `netbot.load`, the
  collector workers and the update workers all rebuild the shape a run trained
  with. Both default to `linear` and build identical modules under identical
  names, so every checkpoint already on disk stays loadable.
- `benchmarks.head_shape` — sweeps those shapes by refitting a head on a frozen
  trunk, which is minutes rather than the days a fresh PPO run per shape costs,
  and is explicit about what that cannot show.
- `catan.distill_train --collect-workers` — the searched collector sharded across
  processes, with the teacher synced every iteration. Collection is ~92% of an
  arm, so the searched games are now collected once and replayed.
- Distillation statistics split agreement by contested and settled rows, and
  report entropy, anchor loss and contested-row counts.

### Changed

- `catan.ppo` and `catan.train` can cut the value loss off the trunk, so the
  critic's two wires became one flag with a measured ceiling on epochs.
- `benchmarks.rank` gained a head learning-rate sweep. The sibling ratio orders
  exactly inversely to how well the head fits, which reverses within a single
  head — so the ratio measures underfit, not architecture.

### Fixed

- **`legal_actions` re-enumerated a trade the table had already declined this
  turn.** `offers_made` was a bare counter, so a seat could spend its whole offer
  budget asking one question three times; training logs showed 45-51 proposals
  per seat-game. `Game.offered` now records the bundles put to the table this
  turn and the sample skips them, and `imagine` copies the set so a search does
  not read every repeat as fresh. The rules do not move — `propose_trade` still
  performs a repeat and `MAX_OFFERS_PER_TURN` is still the cap. **This is
  unrelated to the distillation work and changes the action space every earlier
  run was measured on.**
- `--learning-rate` is honoured on resume in the distillation trainer.
- The attention head's pooling query is built off the head, not the trunk.
- The from-scratch benchmark arms deal 128 lanes rather than inheriting 512.
- The zero-sum check tolerates float32.
- Valued corpora are collected from the parent rather than from the trained
  control.

## [0.4.0] - 2026-08-22

The PPO campaign: the trained policy beats catanatron's own bot 33.8% over 2000
games, then a throughput arc, then the campaign's load-bearing defect — `--resume`
was silently discarding `--learning-rate`, so a whole run had been spent on the
recipe it was supposed to be varying.

### Added

- `catan.collect` — self-play collection sharded across worker processes with CPU
  inference, at 2.1x the iteration rate. Every part of a tick is Python compute
  holding the GIL, so the shard has to be a process rather than a thread.
- `catan.ddp` — the PPO update data-parallel across CPU worker processes, behind
  `--update-workers`. The GPU update sits at ~106 us a position-pass with compile,
  precision, minibatch size and CPU threads all ruled out; one CPU core runs the
  whole fwd+bwd at ~557 us, and at 159k parameters a gradient is a 640 KB vector.
  Measured at parity with this GPU, not the 5x first reported.
- `catan.schedule` — the learning rate as a controller driven by `approx_kl`
  rather than set once and never touched, following the rule the robotics PPO
  implementations settled on.
- `catan.selfplay.Collector.cohort` and `catan.train --collect-mode` — deal a block
  of games, play every one to completion, end with empty lanes. `collect` refills a
  lane the moment its game ends, so the batch was biased toward short games and
  stitched from several policy generations. On one iteration at the campaign's
  shape, streaming returned 128 games averaging 655 actions where the cohort's 128
  averaged 910, and `approx_kl_first_minibatch` — necessarily zero on an on-policy
  batch — went from 0.003 and climbing to float-noise zero. Lanes stay independent
  of the cohort size, so the inference batch can be chosen for throughput without
  deciding how many positions an update trains on. `--async-collect` now requires
  `stream`, since prefetching buys exactly the staleness a cohort removes.
- TD(lambda) value targets behind `--lam`, which beat every fixed horizon at matched
  noise. 0.95 is a local optimum; the sweep closed flat and the single-board control
  flips the tree's sign.
- Opponent mixing in the collector and a frozen evaluation ladder, including
  `search2-offers3` as a rung — which the original design called for and which no
  earlier run ever measured.
- `catan.ppo.Terms` carries `policy_term`, `value_term` and `entropy_term`, the
  summands of `loss` still attached to the graph, so a caller can differentiate one
  term at a time. Which head dominates the shared trunk is not answerable from the
  loss magnitudes, since `policy_loss` is ~0 at ratio 1 by construction.
- `benchmarks.noise_scale` — the gradient noise scale after McCandlish et al.
  (2018), from one collected batch with no training run, decomposed over the
  objective's terms. On `ppo4-585` the policy term measures order 10^5 positions
  against a `--minibatch` of 4096 (~14% signal) while the value term measures ~580
  (~94% signal), so the policy term is 99% of the gradient variance and a third of
  the signal.
- `benchmarks.duel` — two checkpoints head to head on identical boards in paired
  terminal VP. The in-loop ladder's 200-game rungs cannot resolve a slope: a
  150-iteration block carries a 95% half-width of ~9 points.
- `benchmarks.minibatch_iso_kl` — the learning rate that holds step length constant
  across minibatch sizes.
- `benchmarks.rank` — whether the value head orders siblings the way the truth
  does. It ranks them at +0.57, not at chance, which `benchmarks.sibling` could not
  see because a bias common to both children cancels in the comparison.
- `benchmarks.training_loop` — production-shape sync/async PPO timing.
- `catan.arena` takes `network:<path>` wherever a preset name is taken, and
  `netsearch:<path>` / `netgreedy:<path>` swap the checkpoint in as the *leaf
  evaluation* of the ordinary search. `arena.pooled` groups standings by base name,
  since a duel is two seats a side. `network:<path>@<offers>` self-imposes an offer
  budget, so a duel can price the policy's proposing behaviour — which costs
  nothing and earns nothing.
- `catan.train` prints its effective device and worker counts and warns when it is
  crippled, rather than silently running on CPU.

### Changed

- `catan.selfplay.Collector` encodes a worker tick as one vectorized NumPy batch
  laid out as the model's packed input, so CPU inference reuses it without
  restacking. Serialization deliberately drops the shared parent so one returned
  transition never carries the other lanes over a worker pipe; the canonical encoder
  remains the byte-identity oracle. 114.65 to 99.96 us/action (1.147x), plus 44.5 us
  per worker tick from the packed handoff.
- Board-template lookups hit an identity cache before Python recursively hashes an
  unchanged frozen board, and a batched encode resolves its shared topology graph
  once rather than once per lane. Recursive hashing was 0.759 s of an 8.196 s
  profile and drops out entirely: 160.65 to 117.55 us/action (1.367x).
- `catan.encoding._template` is cached 4096 boards deep, twice raised. PPO wants the
  largest batch the dispatch toll amortises over, and 512 lanes miss a 64-entry
  cache every time: 84.2 to 67.6 us per position at 128 lanes, 118.5 to 101.1 at
  512, 126.9 to 115.8 at 1024. The cost still climbs with lane count afterwards, so
  the rest is working set and raising it again will buy nothing.
- Collection overlaps the GPU update behind `--async-collect`, which trains on a
  policy one iteration stale.
- `benchmarks.duel` defaults `--workers` to 26 unless both sides are bare networks.
  `train.versus` batches inference across lanes in one process, which is ideal
  network-vs-network and catastrophic against a scripted bot whose search cannot
  batch and gets one core.
- `lam` and `minibatch` are columns in `log.jsonl`.
- The devcontainer installs Python. It previously had no Python at all, so it could
  not run this project's tests.

### Fixed

- **`--resume` was discarding `--learning-rate`**, along with four more PPO defects.
- `--resume` with no checkpoint started fresh in silence.
- The RNG state is restored as a CPU tensor on a cuda resume.
- The KL gauge reported a negative divergence on on-policy batches.
- Log scalars are detached in `minibatch_terms`.
- `--workers` could not duel two checkpoints against each other, and the extracted
  duel helper was shadowed by a local of the same name.
- A duel needs an explicit torch thread count under a `--cpus` cap.
- The lambda sweep oversubscribed the box with update workers it did not need.
- `summarise` stays compatible with `distill_train`.
- `tests/test_duel.py` imported torch at module scope, so a torch-free box failed
  collection for the whole suite instead of skipping one module.

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
