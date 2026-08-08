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
  turn, resource diversity, the settleable frontier the roads reach, roads, knights,
  hand size with a discard penalty, and port rates. Weights are an ablatable dataclass
  and are untuned. `pips` lives in `catan.board.board` so the encoder can share it.
