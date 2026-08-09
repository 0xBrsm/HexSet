# Changelog

All notable changes to the `catan` package are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  actions per game, which the engine's turn cap cannot do.
- `benchmarks.baselines` — runs a lineup and records the commit and environment with the
  result.
- `catan.tuning` — fits the evaluation weights by hill climbing against the incumbent
  through the arena. The scale is pinned at `victory_point`, since scaling every weight
  alike cannot change the argmax; acceptance needs the win-rate interval's lower bound
  to clear half, with the strictness a knob because both extremes fail. `confirm` plays
  the fitted weights against the starting weights at a large budget, which is the only
  way to tell a real gain from an accumulation of accepted noise.
- `benchmarks.tune` — runs the climb, reporting each duel as it resolves.

- `catan.evaluate_tiered` — a second evaluation, reimplemented from the design
  catanatron's value function uses (described, not ported; it is GPLv3). Weights are
  magnitude tiers encoding a priority order rather than blended coefficients, own
  production is scored against the strongest opponent's, and reachable production keeps
  the player's own settled junctions in the set so building can never shrink it. Kept
  as a comparison baseline, not as the default: tuned, it plays `catan.evaluate` to a
  dead heat (51.3%, 95% CI [48.2%, 54.4%]) while generating data at half the rate.
  Selectable through the `greedy-tiered` and `search2-tiered` presets.

### Changed

- `catan.arena` entrants carry which evaluation to score with, so the two can be
  played against each other directly.
- `catan.game.roll_dice` takes an optional explicit roll, so a search can enumerate the
  outcomes instead of sampling one.
- `catan.arena` entrants are a frozen `Entrant` description rather than a bot-building
  closure, replacing `FACTORIES` with `PRESETS` and `spawn`. Closures cannot be pickled,
  so this is what lets `compete` fan out over a process pool — and it means a lineup can
  go into a run manifest verbatim. Results are identical at any worker count.
- The devcontainer installs Python. It previously had codex, GitHub CLI and Docker
  features but no Python at all, so it could not run this project's tests.
