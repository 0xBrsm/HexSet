# Changelog

All notable changes to the `hexset` package are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`hexset.catanatron`, an optional adapter to Catanatron's arena**
  (`pip install -e "src[catanatron]"`). Collapses the standalone
  `catan-bridge` repo into this package: translates a live `catanatron.Game`
  into a `hexset.Game`/`GameState` every decision (`state.py`), maps board
  ids both ways (`board.py`), resolves a `hexset` `Action` back onto one of
  catanatron's `playable_actions` (`actions.py`), and exposes a
  `catanatron.models.player.Player` (`player.py`, registered as `DC:<entrant>`
  via catanatron's `--code` extension point) so any `hexset` bot can be
  seated in a Catanatron duel and any Catanatron bot can be seated against
  ours. `python -m hexset.catanatron.duel` shards a duel across worker
  processes and stamps catanatron's resolved commit, seed, worker count and
  `PYTHONHASHSEED` on every report. The extra pins catanatron to the exact
  commit the R-H1c gate measured against
  (`d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`); no catanatron import reaches
  the base package (`import hexset` stays catanatron-free), and the whole
  subpackage stays under `src/hexset/catanatron/` so it travels cleanly with
  the engine's extraction into its own repo. Ported with its full test suite
  (`tests/catanatron/`, `pytest.importorskip("catanatron.game")`-gated so the
  default suite stays torch- and catanatron-free) and its two determinism
  fixes: the honest-bot rng seeded from the game seed and seat rather than
  process state, and `main()` re-execing with `PYTHONHASHSEED=0` pinned
  before any game is played -- catanatron's own tie-breaks resolve via
  hash-order-sensitive set iteration
  (`agents/reference/heximax.md`, "R-H1c take 2").
- **`hexset.tuning` can fit heximax's two weight profiles** (honest, depth 2)
  — the P3 prerequisite (`agents/reference/heximax.md` §7, "Harness gap").
  `evaluator="heximax-trading"` / `"heximax-notrade"` build `kind="heximax"`
  entrants on both sides of the climb instead of silently falling back to
  `search2`; `heximax.heximax()` gained a `weights` keyword so a candidate
  vector actually reaches `HonestEvaluator`.
- **The live trade offer is observed** (trading design part 1,
  `agents/reference/trading-design.md` §3.1). `encoding.global_features`
  gains 18 features at four players, appended at the tail of the globals
  vector: the standing offer's give and want bundles (hand-scaled), the
  proposer's seat (relative one-hot), and who has answered — the last
  visible only from the proposer's perspective, approximating the
  simultaneous responses part 3 introduces. A responder in `TRADE_RESPOND`
  now sees the terms it is deciding on; every recorded checkpoint decided
  blind. The information-set record grows the same four fields
  (`offer_give`, `offer_want`, `offer_proposer`, `offer_answered`), so the
  ONNX contract bumps to **v3** (27 inputs, unchanged outputs) and
  hexset-ui must fill them before any v3 deployment.
- **`hexset.migrate`**: function-preserving checkpoint migration onto the
  wider observation — the new `embed_global` columns are zero, so a migrated
  checkpoint plays exactly as its source until trained further (asserted on
  real observations before anything is written, as `hexset.widen` does).
  Checkpoints from before this change cannot be loaded without it.
- **The public-knowledge ledger** (`hexset.ledger`, trading-design §7.2
  — D1 read flat because the observation carried opponents' hand *totals*
  only, no composition, so ΔV could not price a partner even in principle).
  A new `PublicLedger`, created with the `Game` and carried through
  `imagine()`, tracks each seat's reconstructed hand composition —
  `known[5]` (a certified per-resource lower bound) plus `unknown` (cards
  whose type the public log cannot pin) — updated incrementally wherever
  `hexset.game` mutates a hand: production, distribution, the second-round
  settlement grant, builds, dev-card buys, bank and player trades, discards,
  monopoly and year of plenty are all public and update `known` exactly; a
  robber or knight steal moves one hidden card, so the thief's gain is
  credited to `unknown` only and the victim's side is resolved
  identity-independently: `ledger.PublicLedger.steal` floors *every*
  `known[r]` by one (never below zero) and re-solves `unknown` from the
  seat's own previously tracked total, never reading which resource was
  actually taken (a convention that reads the true identity to decide
  which `known[r]` to touch leaks it straight back out through which entry
  visibly drops — an earlier draft of this did exactly that; see `steal`'s
  docstring for the proof the floor rule stays safe regardless). **v1
  simplification, documented in the module: this is the common-knowledge
  view** — the thief/victim's own sharper knowledge of a steal is
  deliberately not modelled; uncertainty can balloon by up to
  `NUM_RESOURCES - 1` cards in one steal, the honest price of losing one
  bit of information. `encoding.global_features`
  gains 18 features at four players, appended at the globals tail after the
  live-offer block: each opponent's `known[5]`/`unknown` (hand-scaled),
  seat-relative, own seat excluded (own hand is already exact) — 68 → 86.
  The information-set record grows `ledger_known` `(players, 5)` and
  `ledger_unknown` `(players,)`, board-seat order like every other field, so
  the ONNX contract bumps to **v4** (29 inputs, unchanged outputs) and
  hexset-ui must supply the ledger record fields before any v4 deployment.
  `hexset.migrate` needs no logic change — it zero-pads any `embed_global`
  tail growth, single- or double-widening alike. **Registration owed before
  any run trains on this.**
- **`hexset.heximax`: the honest handcrafted baseline** (design note
  `heximax.md`, P1 of its §8). A new bot in one file, plugging into the
  `Bot`/`arena` API exactly as `search2` does, that reads every opponent
  through the public ledger and the public counts and never through
  `state.hands[opponent]` or `state.dev_cards[opponent]` — information-set
  honest by default, where `search2` reads every true hand. Four sections:
  a `Belief` (certified counts, untyped counts, and the residual pool the
  hidden cards are drawn from; `expected_hand`, `p_holds`, and `sample`, a
  determinized world consistent with everything public), an
  `HonestEvaluator` (`evaluate.Evaluator`'s term set read through the
  belief, progress zeroed toward a piece whose supply is exhausted, and two
  weight profiles — today's trading fit and the pre-trading fit recovered
  from `87d9095`), a max^n search with a per-move leaf budget and iterative
  deepening, opponents expanded from `k` determinized worlds (PIMC) and every
  hidden draw — steal or dev-card buy — valued as the probability-weighted
  expectation rather than one frozen sample, and the trade *valuation* layer
  (marginal gains and losses, bundle deltas with a named counterparty).
  Three presets: `heximax` (honest, three offers a turn, the placement prior
  composed in), `heximax-omni` (the same bot reading every true hand, to
  measure what honesty costs) and `heximax-notrade` (the no-trade weights, an
  offer budget of zero, declines everything). P1 scope only: trading is the
  engine's one-for-one sample valued by the search; the bundle-offer
  generator and protocol adapter (P2), the `k` ablation (P1½) and the refit
  (P3) are pending and nothing here is registered.
  **P1½** settled `k`: every honest arm beat the omniscient `search2` at 400
  games, but no `k > 1` beat `k = 1` beyond the instrument's resolution
  (paired VP the registered tiebreak), so `k = 1` ships and the ablation
  arms (`heximax-k2/-k4/-k8`) are removed from `arena.PRESETS`.
  **P2** adds the protocol-free valuation the design's `trade` section
  promised — `deficit`/`surplus` (marginal gain of receiving, loss of
  giving, per resource), `candidate_bundles` (1-2 cards a side from
  deficit × surplus, a 2-for-1 where a port makes it rational, every
  candidate `well_formed` and `can_propose`), `score_proposal`
  (`ΔEval_me` at the best counterparty, weighted by each opponent's
  `p_holds` and a crisp `willing` read on their own row of the vector —
  never their true hand), `accept_rule`, `counter_of`, and `rank_partners`
  (opponents ranked `paranoid`, first to whoever a trade helps least) —
  and the minimal, mechanical, disposable protocol-P0 adapter over it:
  `Heximax.propose_actions` replaces the engine's one-for-one
  `PROPOSE_TRADE` sample among the root options with heximax's own top-`n`
  scored candidates, and `_options_in` gates every `TRADE_RESPOND` node's
  `ACCEPT_TRADE` — root or simulated — with `accept_rule`. Cost: with the
  adapter active heximax measures 3.46x `search2-offers3` per move (P1's
  own search alone still costs 1.66x), over the design's 2x ceiling — the
  crisp `willing` gate, read under `relative`, proposes far more
  selectively than the engine's blind sample, and that trade-volume drop
  alone accounts for most of the overrun; flagged for the PI rather than
  quietly absorbed by loosening the gate (module docstring, "Cost"). A later,
  behaviour-preserving optimization pass — gated by a byte-identical choice
  census over 30 seeded games (`test_heximax.py`) — memoized `Evaluator.survey`
  per decision and shared `propose_actions`' "before" reading across its own
  calls, taking the adapter-active cost from 3.46x down to 2.87x
  `search2-offers3` per move, still over the ceiling for the same
  trade-volume reason, not a compute one. A second exact, census-preserving
  pass memoized `Belief` construction (`HonestEvaluator.belief_for`) and
  `HonestEvaluator.evaluate`'s per-seat vector within one decision, cutting
  `Belief.from_game` calls from 37.2 to 3.9/decision and moving the cost to
  2.69x on the same protocol, still over the ceiling for the unchanged
  trade-volume reason. A third exact pass precomputed `progress_toward`'s
  cost table, cached the vertex walk `progress` needs beside the survey it
  is a function of, counted roads with `list.count`, built each `Belief`'s
  cache signature once, and dropped a doubled legality check in
  `candidate_bundles` — **-12.3% ms/move and -13.0% function calls** against
  the previous code, paired in one process; **2.69x -> 2.37x**. Three
  behaviour-changing steps were measured and none landed: a transposition
  table across the iterative-deepening passes hits 0.046% of `_value` calls,
  a vectorised evaluator would need a breadth-first rewrite of `_value` to
  beat a loop that is 2.0% of runtime, and sampling the ply-1 roll (worth
  -11.9%) missed its pre-stated no-trade strength floor by 0.6 of a game and
  was reverted. Read the ratio beside its phase-neutral form (**2.08x**):
  `search2` books 2481 cheap `TRADE_RESPOND` decisions to heximax's 126 over
  the same nine games, and those sit in the mirror table's denominator.

### Fixed

- **`heximax-omni` priced trades against a hand that did not exist.**
  `Heximax._partner_delta` builds the post-trade position with `_move_hand`,
  which folds a non-knower's hand into one all-one-resource total rather than
  moving it per resource. That is exact for the honest bot -- an honest
  evaluation reaches a non-knower only through `Belief.expected_hand`, and the
  one thing it takes from `state.hands` is the size, which the fold preserves
  while a per-resource move (clamped at zero where the seat cannot cover what
  it gives) would not. Under `omniscient` every row is scored on `state.hands`
  verbatim, so the fold replaced the counterparty's real cards with a fiction:
  `score_proposal`'s `willing` gate saw every trade as ~10x more ruinous for
  the partner than it was and stopped firing, while `accept_rule`, reading a
  proposer it had just impoverished under the `relative` stance, cleared far
  too easily. `_move_hand` now takes `exact`, and `_partner_delta` passes
  `self.omniscient`. `heximax` and `heximax-notrade` are byte-identical
  through the change; only `heximax-omni` moves.

### Changed

- **Package renamed `catan` → `hexset`**, in prep for release under GPL-3.0-only
  as its own public repo. `import catan` → `import hexset` throughout; the only
  renamed identifiers are the package itself, `CatanNet` → `HexNet`, and the
  `CATAN_EXPORT_COMMIT` env var → `HEXSET_EXPORT_COMMIT`. Every source file
  under `hexset/`, `benchmarks/` and `tests/` now carries an
  `SPDX-License-Identifier: GPL-3.0-only` header, and a `LICENSE` file (GPL-3.0,
  full text) is added at this directory's root. Entries above this one describe
  the package as it was named at the time; see the repo's top-level README for
  the trademark note this rename exists to satisfy.
- **`hexset` split into `hexset` (engine, bots, ledger) and a new sibling
  package `hexnet` (PPO/training research)**, in prep for HexSet and HexNet
  becoming separate repos. `collect`, `ddp`, `distill`, `distill_train`,
  `expert`, `export_onnx`, `league`, `migrate`, `model`, `netbot`, `policy`,
  `ppo`, `readout`, `rewards`, `schedule`, `selfplay`, `train`, `widen` and
  `run/` move to `hexnet` (`import hexset.train` → `import hexnet.train`,
  etc.); the training-bound benchmarks move to `hexnet/benchmarks/` so
  `python -m benchmarks.duel` and the rest of the hexset-side tools are
  unaffected. `hexset` still declares only numpy and never imports `hexnet`:
  `hexset.arena` gained a small registry (`register_entrant_kind`,
  `register_evaluator_provider`, `register_checkpoint_loader`,
  `register_leaf_evaluator_factory`) that `hexnet.netbot` populates at
  import, replacing five direct `hexset.X` imports of what is now `hexnet.X`
  (`arena`↔`netbot`, `mcts`↔`rewards`, `onnx_record`↔`policy`,
  `duel`↔`collect`/`train`, `aivat`/`human_agreement`↔`netbot`) plus a
  sixth found during the split: `onnx_record` imported torch unconditionally
  for `RecordEncoder`, which now lives in `hexnet.export_onnx` — the
  torch-free half (`RECORD_FIELDS`, `record_from_game`, `record_batch`,
  `CONTRACT_VERSION`, `record_shapes`) stays in `hexset.onnx_record`, and
  `record_from_game` gained an optional `options` parameter so a caller that
  already computed the legal-option set does not pay for a second one.
  `relative_points` moved from `rewards` to `victory` (both `hexset.mcts`
  and `hexnet.rewards` need it); `NUM_PAIRS`/`pair_index`/`pair_mask` moved
  from `policy` to `actions` for the same reason — both modules re-export
  their old names, so no other call site changed. Training-only tests moved
  to `src/tests/hexnet/` alongside their modules, so `pytest src/tests -k`
  still runs everything and the engine+bots slice runs torch-free on its
  own. `pyproject.toml`'s `packages.find` now includes `hexnet*` alongside
  `hexset*`, one distribution for now.

## [0.13.0] - 2026-08-31

### Changed

- **An offer is put to the table in a random order** when the proposer gives
  no `ask`, drawn from the game's own RNG, instead of clockwise from the
  proposer. An offer stops at the first taker, so whoever is asked first has
  first refusal; clockwise handed that to the next seat in turn order, which
  in a 2v2 duel is a copy of yourself half the time when the copies sit
  together and never when they alternate — the entire "seat geometry" effect
  (+0.08 vs +0.43 VP for the same pair) was this one line. Random order hands
  the advantage to nobody in either seating. Still not the rulebook, where the
  proposer chooses among the acceptors; that design is written down in the
  trading design note and deferred. `trading.responders`
  keeps its clockwise order as the eligibility list. Every seeded game's RNG
  sequence differs from before, so no duel reproduces bit-for-bit across this
  change.

- **The piece supply is enforced: 15 roads, 5 settlements, 4 cities a
  player.** `state.can_place_settlement`, `can_upgrade_to_city` and
  `can_place_road` refuse a piece that is not in the box, so bought pieces,
  the road-building card's free roads and initial placement all read one
  rule and `legal_actions` stops offering what cannot be built. This engine
  had no supply at all until now — every recorded run, duel and ladder was
  played with unlimited pieces — while the deployment's engine (hexset-ui,
  the rules reference) and catanatron both capped. Ported verbatim from
  hexset-ui `state.py`. Measured before the fix, the cap would have bound in
  3.1% of the frontier network's player-games (a sixth settlement; never a
  fifth city or sixteenth road) and 10.4% of `greedy`'s (roads to 26). Every
  internal number recorded before this entry is an uncapped number; the
  bridge's were always capped.

### Added

- `catan.widen`: a **function-preserving widening** of a trained checkpoint
  (Net2WiderNet). Every trunk unit at width `d` is copied to fill width `D`,
  copies keep their source's incoming weights and every consumer divides its
  weight on a copy by the copy count, so the wide net emits the narrow net's
  logits and values to float precision — asserted on real observations, both
  forward paths, before anything is written (max relative |Δ| 3e-7 on
  `lam095-805`, 64 → 128). `--noise σ` adds σ × row-RMS Gaussian noise to the
  copies' incoming weights, because identical copies get identical gradients
  forever; the mean policy KL it costs is reported. The attention value head's
  query is rescaled by √(D/d) over the copy count so its softmax is unchanged.
  The output checkpoint keeps the parent's `iteration`, `games_started` and RNG
  state, carries `args` with the new width (what every loader rebuilds the
  shape from), a fresh Adam state, and a `widen` block naming the source by
  sha256. `catan.train --resume` continues from it unchanged; no new flag.

- `--mix` accepts a **table entry**, `table(a|b|c)=f`: in a share `f` of
  games the learner takes one seat, drawn per index, and every other seat is an
  independent draw from the pool `a|b|c` — with replacement, so three copies
  of one bot and a fully heterogeneous table are both dealt. This is the
  seating the deployment, the external bridge and every real game put the
  network in, and no run had ever collected in it: a plain entry gives its
  opponent 2 of 4 seats on alternating parity, so the learner always had a
  twin at the table. `collect.mix_caster` handles both entry kinds and is
  `mixed_caster` to the cast when no table entry is present (pinned by test
  over 1000 indices); `collect.mix_names` is the one place mix names become
  caster ids, shared by `mix_opponents`, `check_mix` and the caster. No new
  flag, so every frozen config on disk still loads.

- `benchmarks.duel` records the seat geometry of every verdict. A 2v2 can seat
  each copy beside its twin (`[a, a, b, b]`, **blocked**) or between two
  opponents (`[a, b, a, b]`, **interleaved**), and until now the worker count
  chose silently: `--workers 1` plays through `collect.alternating`, which is
  interleaved, and `--workers >1` hardcoded the blocked lineup. On identical
  boards and dice the seating alone moves `lam095-805` vs `ppo4-585` from
  +0.08 to +0.43 VP, replicated at two seeds. Every verdict now carries a
  `geometry` field, and the resolved geometry is printed beside the resolved
  worker count.
- `--geometry {blocked,interleaved}` on the arena path. The default is
  `blocked`, so a default invocation reproduces every recorded arena verdict
  bit for bit; `interleaved` seats `[a, b, a, b]` with side A on slots
  `[0, 2]`, the lineup the seat-geometry probe used. `sides()` labels the two
  sides by slot rather than by position, so either lineup pools correctly. The
  versus path can only seat interleaved and refuses `--geometry blocked`
  rather than playing one seating under the other's name.

### Changed

- `benchmarks.duel` imports `catan.collect` and `catan.train` where they are
  used, so the module -- and its lineup and arena-path tests -- load on a box
  without torch.

## [0.12.0] - 2026-08-28

### Added

- `benchmarks.aivat`: AIVAT's chance-correction term (Burch, Schmid, Moravčík,
  Morrill & Bowling, AAAI 2018), measured on duels this project has already
  recorded. Subtracts `V(observed outcome) - E_p[V(outcome)]` at every chance
  event, which has conditional expectation zero given the history before the
  draw, so the estimator is unbiased for **any** value function -- the argument
  is a martingale difference on the chance filtration and never mentions the
  player count, the payoff's zero-sumness, or `V`'s accuracy.
- All four chance transitions are enumerated exactly: 2d6 from
  `game.ROLL_ODDS`, the dev-card draw as the remaining deck's multiset, and the
  robber steal as the victim's hand normalised. The dice and the steal are drawn
  at the event, so their law is exact under the full history; the deck is
  shuffled once at `new_game`, so the estimator must forget the unrevealed order
  -- which is sound only because neither `catan.encoding` nor any bot can read
  it, and `game.imagine(..., randomize_deck=True)` exists to keep that true.
- `instrumented` is a twin of `arena._play_one` that enumerates outcomes on
  `imagine` copies fed a separate generator, so an instrumented replay of a
  recorded cell is bit-identical to it. `--check` asserts that against the
  recorded verdict, and it holds on four 800-game cells to the digit.
- Reports the reduction AIVAT's unit coefficient actually delivers *and* the
  ceiling over every coefficient, `1 - sqrt(1 - rho^2)`. The second is the
  number that settles the question, because no tuning beats it. Measured here:
  the unit coefficient **raises** the paired-VP SD by 11-17% and the ceiling is
  2.2-5.7%, against the paper's 68% for full AIVAT in HUNL (and 33.8% for its
  chance-only term on Leduc).

## [0.11.0] - 2026-08-28

### Added

- `benchmarks.human_agreement`: our policy scored against *recorded* decisions,
  one decision at a time, rather than against an opponent one game at a time.
  Takes `catan.record.Record`s from any source and reports **top-1 agreement**
  and **log-loss** at every decision point, each against **its matched null** --
  uniform over the legal option set *at that position*, so the baseline is
  `mean(1/n)` and `mean(log n)` and never a global constant derived from the
  mean option count. The distribution scored is the one a search acts on:
  `netbot.LeafEvaluator.evaluate` on a `mcts.Leaf`, which is what keeps the
  trade slot's mass split across the legal offers by the pair distribution
  instead of being credited whole to one arbitrary offer.
- Decisions with a single legal action are excluded -- agreement there is 1.0 by
  construction -- and the excluded count is reported, along with actions the
  enumerated option set does not contain. `actions._offer_actions` is a sample
  and not the whole legal set, so a recorded multi-for-one offer, or one the
  offer budget forbids, is counted per `ActionType` rather than silently scored.
- Everything is broken down per `ActionType`, per `Phase` and by game progress,
  because `PROPOSE_TRADE`-family rows dominate an unstratified mean. Each
  breakdown partitions the decisions and pools back to the aggregate exactly;
  `summarise` rounds nothing so that identity is exact.
- Intervals are clustered on the game, not taken over positions. Consecutive
  decisions in one game differ by a single build, which is the same reason
  `dataset.split_by_game` exists; the position-level Wilson interval is still
  reported, labelled as the understatement it is.

## [0.10.0] - 2026-08-28

### Added

- `--mix` accepts **any arena entrant spec**, not the two hardcoded names. A
  training lane opponent can now be `search2-offers3`, `mcts:<ckpt>@64`,
  `network:<ckpt>`, or any preset, resolved through `collect.named_opponent` and
  therefore through `catan.arena.spawn` -- so a run trains against *literally*
  the entrant the arena scores it on. `collect.mix_opponents` is the one place
  that turns names into lane opponents, shared by the sharded and in-process
  collectors; `collect.check_mix` is the pre-flight, and it refuses a mistyped
  entrant or a missing checkpoint before the manifest is frozen rather than as a
  traceback out of a worker subprocess.
- `collect.RESERVED_MIX` names what may not be routed. `greedy` and `parent`
  keep resolving to exactly the bots they resolved to before: `--mix greedy`
  takes the run's own `--max-offers`, so what 34 recorded runs played is
  `greedy-offers3`, and the arena's `greedy` preset -- `max_offers=None`, the
  engine's whole eight-offer budget -- is a different bot at a different
  strength. `test_collect` pins the equivalence twice, once field-for-field
  including the tie-break rng state, and once by playing two cohorts and
  requiring identical action streams.
- `benchmarks.mix_cost`: what a mix costs a PPO iteration, per decision and per
  shard, with a stopwatch on the learner and on every opponent. Interpolates
  rather than models -- `mixed_caster` draws per game and cost is additive over
  games, so `S(f) = (1-f) S(0) + f S(1)` is exact and only the endpoints need
  measuring.

### Fixed

- The in-process collector's `--mix` construction fell through to the parent
  checkpoint for any name that was not `greedy`, which only the two-name
  validation kept unreachable. Both collectors now build their opponents through
  the same function.

## [0.9.2] - 2026-08-28

### Fixed

- **The sibling-ranking probes no longer freeze one chance outcome per child.**
  `benchmarks.rank` and `benchmarks.sibling` both built each sibling with
  `imagine` then `apply`, so a `BUY_DEV_CARD`, a `PLAY_KNIGHT` or any
  `Phase.ROBBER` row's `MOVE_ROBBER` embedded **one sampled outcome per child**
  and the head-versus-truth comparison across siblings was partly a comparison
  of decks rather than of decisions. This is `0.9.1`'s own listed exception and
  the second half of the afterstate audit; the tree half was `0.9.1`.

  **A chance child is now scored as the mean over `--chance-draws` independent
  draws, default 8.** Averaging is chosen over borrowing the tree's keying
  because averaging is what the search *experiences*: after `0.9.1` a chance
  edge resamples on every visit, so its `Q` converges on the mean over outcomes
  and PUCT orders actions rather than realised children. That is the quantity
  this metric exists to predict. The tree's keying is its *implementation* of
  the same average and does not transfer — a tree may reuse a repeated
  outcome's held child because the deck order beneath the top card is
  unobservable to it, while a probe rolls its children out for hundreds of plies
  where that order decides real draws.

  **Both columns are averaged, and the rollout budget is partitioned rather than
  multiplied.** Under one draw the head and the truth at a chance child were at
  least consistent — both conditioned on the same realised outcome, a shock they
  shared, which quietly inflated their agreement there. Averaging only the head
  would have measured a mismatch instead of an ordering. `--rollouts` is
  therefore split across a child's draws by `share`, and `lane_plan` walks the
  same consecutive stream offsets whatever the draw count, so total games rolled
  out is unchanged and draw `d` lane `k` still shares its deck and its sampling
  stream with draw `d` lane `k` of every sibling. Cost is 1.35x on the head
  column, which is under 1% of a probe's wall clock, and nothing on the rollout
  column.

  **Incidence, measured on this engine rather than assumed:** 7-9% of probeable
  rows hold at least one chance child and 3.5-5.0% of children are chance
  children, over 12 games of both `RandomPolicy` and `greedy` self-play. Every
  `Phase.ROBBER` row is affected.

  **Off-path behaviour is unchanged, proven rather than asserted.** Draw one of
  every child comes off the shared stream and a deterministic action takes only
  that draw, so a row resolving no hidden information is bit-identical and the
  shared stream ends in the same state — a chance row cannot move a chance-free
  row after it. Anchored over 1,269 chance-free rows from two games of `greedy`
  self-play, exact float equality on all three statistics plus an exact
  post-sweep rng draw, with all 187 chance rows in the same sweep moving.

  **Every number either probe produced before 2026-08-28 was taken under the
  single-draw path.** `--chance-draws 1` restores it exactly.

### Added

- `catan.mcts.draws_hidden` and `catan.mcts.sampled_children`, public because
  the probes need the tree's chance semantics without building a tree.
  `Search._draws_hidden` now delegates to the first, so the predicate deciding
  which edges are chance edges and the predicate deciding which children get
  averaged cannot drift apart — pinned by a test over every action of a robber
  position.
- `benchmarks.rank.head_row` and `benchmarks.rank.Row`, the row construction
  lifted out of `main`. It was inline in a two-hundred-line function, which is
  where the defect survived being read: nothing could reach the object the whole
  metric is computed from. Now torch-free testable.
- `benchmarks.rank.share` and `benchmarks.rank.lane_plan`, the rollout-budget
  partition and its stream offsets.
- `benchmarks.rank.chance`, a payload block reporting how many rows and children
  drew, the measured spread of the head's read *within* a chance child, and the
  residual `spread / sqrt(draws)` that averaging leaves. Reported so the draw
  count can be checked against the run rather than against an assumption; it
  reads `None` rather than `0.0` at one draw, since a single draw measures no
  spread and zero would read as "no contamination".
- `benchmarks.sibling.Spread.chance_children` and `.chance_spread`, the same two
  quantities per probed row.

## [0.9.1] - 2026-08-28

### Fixed

- **`catan.mcts` no longer freezes the three chance edges it does not roll.**
  `MOVE_ROBBER`, `PLAY_KNIGHT` and `BUY_DEV_CARD` children were created once and
  cached for the life of the tree, so each such edge's `Q` was one frozen steal
  or one frozen card draw rather than an expectation over them, and the first
  visit decided the edge for every later one. They now use the `_Chance` slot
  that `ROLL` already used, keyed by the outcome: `_drawn` reads the card back
  off the child that produced it — `apply` returns nothing — and a repeat of an
  outcome the slot already holds reuses that child, so repeated visits
  accumulate in one subtree and the edge's `Q` becomes an average. Found by the
  afterstate audit, which called it the only unambiguous chance-handling
  defect in the tree.

  Discarding the duplicate child is exact rather than approximate: a steal's
  child is `imagine`d without reshuffling, so two steals of one resource give
  identical positions, and two purchases of one card differ only in the deck
  order beneath the top card, which the encoder cannot see and a later
  `BUY_DEV_CARD` reshuffles before it draws.

  **Off-path behaviour is unchanged, and the search stays a pure function of
  its seed.** Every draw comes from the search's own `rng` in descent order. An
  edge that resolves nothing — a robber or knight naming no victim — keeps the
  plain cached child rather than a slot, so it is bit-identical as well as
  cheaper. A full-stack A/B over 408 searches from real positions off
  `lam095/latest.pt` reproduced all 182 chance-free searches byte-for-byte,
  visit counts *and* the position of the rng stream afterwards; the 226 that
  touched a chance edge moved on 86% of them. Both halves of that are pinned by
  tests: exact visit counts and an exact post-run rng draw, anchored to
  `33c6032`.

### Added

- `catan.actions.victim_of`, which was `_victim`. The rule it holds — a slot at
  or past `num_players` means nobody — decides whether a robber or knight edge
  draws a hidden card at all, so `catan.mcts` needs it and a second copy would
  drift.
- `catan.mcts.HIDDEN_DRAW`, the three action types whose `apply` resolves a
  hidden card.

### Changed

- `test_packing_reports_the_policy_loss_over_contested_rows_only` now thins the
  batch to three contested rows before comparing. It asserts that packing raises
  the reported policy loss because unpacked minibatches holding nothing
  contested report zero — but `losses` divides by the weight, so any minibatch
  with a contested row in it already reports the mean over those rows, and at
  the fixture's natural 33% density all seven of them did. The assertion was
  therefore comparing two numbers that agreed to a tenth of a percent, and it
  changed sign when the search fix perturbed the batch. Thinned, the six empty
  minibatches the docstring describes actually occur, and the test passes
  against `33c6032`'s search as well as this one.

### Unchanged

- `ROLL` edges, whose sampled expectimax over `ROLL_ODDS` was already correct
  and is untouched. Chance is still sampled rather than expanded over all
  outcomes, for the budget reason in the module docstring.
- `benchmarks.rank`'s head-vs-truth column, by construction: at `--simulations
  0` it never builds a `Search`, and above zero the search draws from its own
  generator. Its own `imagine`-then-`apply` sibling construction still embeds
  one sampled outcome per chance child, which is the audit's second finding and
  is not addressed here.

## [0.9.0] - 2026-08-27

### Added

- **A `quantile` value head** (the variance screen's candidate 3, Gate B).
  `ModelConfig(value_head="quantile")` is `"linear"`'s head widened to
  `players x quantiles` outputs — the same input (`g` alone), the same depth,
  the same initialisation. **Its forward returns the `players`-vector mean of
  its quantiles, so `V` keeps its shape and its meaning everywhere**: GAE,
  `lambda_returns`, the zero-sum projection, `catan.mcts`, `catan.bots` and
  every `benchmarks.*` reader are untouched. The full `(B, players, Q)` tensor
  is exposed separately, through `Prediction.quantiles` and
  `Evaluation.quantiles`, and has exactly one consumer: the value loss.
  `ModelConfig.quantiles` (default 32) reaches a rebuild through the
  checkpoint's `args`, the way `value_head` already does; the levels are
  derived from it and are not state_dict keys.
- **The quantile value loss in `catan.ppo.minibatch_terms`.** Under the
  quantile head the differentiated value term is the per-seat quantile Huber
  loss against the *same* `value_target` vector, at midpoint levels
  `(i + 0.5) / Q` and Huber width `QUANTILE_HUBER_KAPPA = 1/30` — one lattice
  step of the return, because at this project's ~0.2 residual scale QR-DQN's
  kappa=1 would fit expectiles rather than quantiles. `quantile_levels` and
  `quantile_huber_loss` moved from `benchmarks.head_swap` into `catan.model`
  and are imported back, so Gate A2's arithmetic and the heat's are one
  implementation. The advantage path is untouched — GAE reads the mean, exactly
  as before — and `value_target` construction is unchanged.
- `Stats.value_mse` / `Terms.value_mse`: the plain squared error of the mean,
  logged beside `value_loss` and never differentiated. The two arms of a heat
  differentiate losses on different scales, so a curve comparison needs one
  column both compute identically; under every scalar head it is `value_loss`
  by the same expression. `catan.league` now logs both per learner.
- `catan.league --value-head` and `--quantiles`. Empty (the default) keeps the
  base checkpoint's own shape, so every heat on record replays unchanged.
  Asking for `quantile` off a scalar base warm-starts it through
  `catan.model.quantile_warm_start`: every level begins at the scalar head's
  own output, so the treatment arm of a heat opens on the same policy *and* the
  same critic as its control (equal to one float32 rounding of a Q-term mean,
  measured under 2.4e-7 — four orders below the label's 1/30 lattice). The
  shape the seats carry, not the base's, is what the heat's checkpoints record.
- `catan.train --quantiles`, alongside the existing `--value-head`, so the
  shape reaches a checkpoint's `args` from every trainer that writes one.

### Fixed

- `CatanNet._emit` now builds `logits`, `give`, `want` and the value read in
  that order explicitly. `g` feeds four heads, so its gradient is a sum of four
  terms accumulated in forward-creation order; hoisting the value read above
  the two trade heads reassociates that sum, leaving the loss bit-identical
  while moving the gradient in its last bits — enough to make a default-config
  update diverge from the previous build after one optimiser step. Caught by a
  full-stack A/B, and pinned by a test on `grad_fn` creation order rather than
  on loss equality, which does not see it.

### Unchanged

- With `value_head` anything but `"quantile"`, a full update is **bit-identical
  to 0.8.0**: verified by an A/B against `6150836` over the whole cross product
  of the five value shapes, both policy shapes, `detach_value` on and off, all
  three `critic` modes and `value_lam` 1.0/0.9 — same assembled batch, same
  post-update weights, same logged losses, to the byte.

## [0.8.0] - 2026-08-27

### Added

- **Board-paired advantage baselines** (the variance screen's candidate 1),
  one flag because the registration treats it as one treatment.
  `Collector(pair_boards=True)` deals games `2k` and `2k+1` on the board keyed
  `f"{seed}:{2k}:board"` while each keeps its own game rng — same geometry,
  independent dice and play — and refuses a fixed `board=` alongside it.
  `collect.paired_caster` repeats each cast for both halves, so within a pair
  the same policy holds the same seat; seat shares then balance over any
  `2*learners`-game window instead of `learners`. `PPOConfig(pair_baseline=
  True)` makes each seat's policy-gradient terminal `r - (r + r')/2`, `r'` the
  same seat's reward in the mate game `index ^ 1` — a valid control variate
  given (board, seat, policy), leaving the gradient unbiased and the value
  target on the raw terminal. Adjusted per-game vectors stay exactly zero-sum,
  the two halves are bit-exact negatives, and a pair with identical outcomes
  pays exactly zero; a batch missing a mate refuses rather than baselining
  against nothing. `catan.league --pair-boards` turns on all three wires.
  With pairing off, dealing, casting, advantages and value targets are
  bit-identical to 0.7.2.
- `benchmarks.noise_scale --paired` — Gate A of the screen: one board-paired
  cohort, the estimator run twice on the same batch (raw vs pair-adjusted
  advantage stream, same positions, weights and shuffles) in one JSON, plus
  the pair correlations `rho = corr(r, r')` and `rho_v` on the head-residuals
  at each seat's last decision, per seat and pooled.

## [0.7.2] - 2026-08-26

### Fixed

- `run.manifest.freeze` read git provenance *after* creating the run directory,
  so `git_dirty` was true for every frozen run: the directory is untracked at
  the moment it is created and `git status --porcelain` counts untracked paths.
  Harmless while `.gitignore` carried a blanket `/runs/` rule and the new
  directory was ignored; a constant once run records became tracked. Provenance
  is now read before anything is written, so the field can once again mean
  "this result cannot be cited".

## [0.7.1] - 2026-08-26

### Added

- `collect.league_caster` takes an optional `order`, a permutation of learner
  ids applied before its rotation, exposed as `catan.league --learner-order`.
  Rotation balances every learner over every board seat but leaves the cyclic
  order round the table invariant, so learner *k*'s turn-order successor is
  learner *k+1* in every game played -- a fixed structure the league applies
  without recording it. Permuting the order varies table adjacency while
  leaving seat shares balanced, which is what makes the two-tight-pairs
  structure in the noise heats testable.

## [0.7.0] - 2026-08-26

### Added

- `catan.league` — **the table league: N learners share every game.** One
  directory holds several learners that play each other rather than a frozen
  opponent, each with its own `PPOConfig` overrides parsed from a per-seat spec,
  and `standings` reads the order out of play instead of out of a separate
  evaluation. A league run is rated by its `learner0`, the control arm, not by
  its best arm — four learners share one directory, so one number cannot stand
  for the run.
- `catan.run` — **a run is a directory with a frozen manifest, and the manifest
  is the input.** `freeze` records the parameters, the resolved config and the
  repository provenance (`run.json`) before a run starts; `load` reconstructs
  the invocation from it. A run's configuration used to exist only in a script
  under gitignored `tmp/`, which is why results could not be regenerated from
  the repository.
- `catan.export_onnx` — `.pt` to `.onnx` conversion, with `onnx`/`onnxruntime`
  behind a new `export` optional dependency. `torch` stays unlisted as a hard
  dependency, so the torch-free engine remains installable without it.
- A **rolling checkpoint ring** (`prune_recent`) that keeps the newest N
  `recent-*.pt` and never touches a kept `iter-*.pt`, and **blowout
  preservation** (`preserve_blowout`), which writes the pre-update weights and
  the offending batch when the brake fires — the evidence used to be discarded
  by the recovery it triggered.
- Per-seat PPO overrides: `adam_eps`, and a `gain` on the entropy controller
  (`nudged`), so a league seat can vary one knob against its own control.
- `selfplay.owned` and a `learners` gate on `Collector`, so several learners can
  record from one game. `learners=(0,)` is the pre-league behaviour: one
  learner, opponents as scenery.

### Fixed

- `catan.distill_train` had been unbuildable for two days: the manifest
  parameters it read no longer matched the parser it read them with.

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
  demoted to a mix-exploitation canary; external anchors calibrate. A common
  opponent read four separate improvements that matched duels did not confirm.

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
