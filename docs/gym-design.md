# `hexset.gym`: a PettingZoo AEC environment, and a Gymnasium wrapper on top

Draft for the PI to correct; nothing here is decided. HexSet's README already
promises this ("A Gymnasium-style environment wrapper (`hexset.gym`) ... is
planned but does not exist yet") and nothing under `src/hexset/gym/` exists
today — this is greenfield.

The engine as it stands: `start(board, num_players, rng)` deals a game
(`game.py:108`), `apply(game, action)` executes one (`actions.py:418`),
`legal_actions(game)`/`legal_mask(game, space)` give the options
(`actions.py:325`, `:410`), `to_move(game)` says whose decision it is
(`game.py:320`), `is_over(game)` says the game ended (`game.py:586`), and the
arena's loop is `while not is_over(game): apply(game, bots[to_move(game)]
.choose(game))` (`arena.py:453-465`). External agents today go through the
HTTP API or MCP (`server/api.py`). None of that is PettingZoo- or
Gymnasium-shaped; this proposes the two thin layers that make it so.

## 1. Env ids and entry points

| name | type | id |
|---|---|---|
| `hexset.gym.HexSetAEC` | `pettingzoo.utils.env.AECEnv` | not registered with `pettingzoo.env()` — PettingZoo has no global registry; imported directly |
| `hexset.gym.HexSetEnv` | `gymnasium.Env` | `gymnasium.register(id="HexSet-v0", entry_point="hexset.gym:HexSetEnv")` |

Both live behind a new optional extra, mirroring the `catanatron` extra
already in `pyproject.toml:41-45`:

```toml
gym = ["pettingzoo>=1.27", "gymnasium>=1.3"]
```

`import hexset` stays numpy-only (`pyproject.toml:19-20`); `import hexset.gym`
is the only thing that requires the extra, exactly as `hexset.server`/
`hexset.clients` gate `onnxruntime` behind their own extras
(`pyproject.toml:32-35`). Versions above are what `pip index versions` showed
current at this read (pettingzoo 1.27.0, gymnasium 1.3.0, 2026-09-02).

## 2. AEC semantics

Agents are seats: `agent_selection = f"seat_{to_move(game)}"`, four agents
for the standard board (`possible_agents = ["seat_0".."seat_3"]`). `to_move`
already resolves to a single seat in every phase, including the two that
hand the decision to someone other than the current player — discard-on-seven
and `TRADE_RESPOND` (`game.py:320-334`) — so AEC's one-agent-active
abstraction needs no special-casing for those phases; the engine already
serializes them.

**`observe(agent)`** returns
`{"observation": {"hexes", "vertices", "edges", "globals"}, "action_mask": ...}`,
the same `{"observation", "action_mask"}` shape PettingZoo's own classic
environments (chess, connect_four) use. The four arrays are exactly
`encoding.encode(game, perspective=seat).{hexes,vertices,edges,globals}`
(`encoding.py:608-627`), seat-relative and information-set correct by
construction (§4). At 4 players (`encoding.py:111-121`, computed against the
standard 54-vertex/72-edge/19-hex board):

| array | shape |
|---|---|
| `hexes` | `(19, 11)` |
| `vertices` | `(54, 14)` |
| `edges` | `(72, 5)` |
| `globals` | `(86,)` |

`action_mask` is a flat `(553,)` boolean vector at the same board size
(`actions.build_space(54, 72, 19, 4).size == 553`, `actions.py:145-183`). It
must come from the table's honest sample, not `actions.legal_actions` raw —
see §4, this is a real, previously-shipped bug class.

**`step(action)`** decodes the flat index with `space.decode(index)`
(`actions.py:117-131`), calls `apply(game, action)`, then advances
`agent_selection` to the new `to_move(game)`. Catan ends for the whole table
at once, so on the step that ends the game every agent's `terminations`/
`truncations`/`rewards` entry is set together, not just the acting agent's —
`terminations[a] = True` for all `a` when `game.won_by is not None`
(`victory.winner`, `game.py:374-379`), `truncations[a] = True` for all `a`
when `is_over(game)` fires with `won_by is None` (turn-limit exhaustion,
`game.py:568-583`, `MAX_TURNS = 1000`, `game.py:37`).

**Rewards**, default terminal-only: `+1` to the winning seat, `0` to
everyone else and on every non-terminal step. An opt-in alternative reads
`victory.relative_points` (`victory.py:86-118`) — each seat's terminal points
less the mean of the others, over 10, exactly zero-sum — for a denser
per-seat value signal instead of win/loss.

**`reset(seed)`** → `start(random_base_board(random.Random(seed)), 4, rng)`,
mirroring the pattern `arena._play_one` already uses to seed a board
(`arena.py:501`).

**`TRADE_RESPOND`**: because `game.py`'s `propose_trade`/`accept_trade`/
`decline_trade` (`game.py:485-566`) already pop `pending_responders` one seat
at a time, each responder's ACCEPT/DECLINE is simply the next AEC agent step
— no offer is ever answered by more than one agent per `step()` call. When
the one-event trading mechanic lands (`agents/reference/trading-design.md`
§8, not registered), `PROPOSE_TRADE`/`ACCEPT_TRADE`/`DECLINE_TRADE` disappear
and the public valuation vectors enter `globals` instead; AEC's per-seat-turn
model survives that change regardless of what the actions become.

## 3. Gymnasium wrapper

`HexSetEnv` owns a `HexSetAEC` instance internally. Configuration:

- `learner_seat: int | Literal["rotate"] = "rotate"` — a fixed seat, or a
  new seat drawn each `reset()`. Default `"rotate"` because seat is not
  neutral: the seat-geometry effect measured in duels (+0.08 vs +0.43 VP for
  the same pair depending on who sits next to whom,
  `agents/reference/trading-design.md` §1) is a property of the same
  responder-ordering machinery this env drives, and an eval loop that always
  learns from seat 0 inherits that bias silently.
- `opponents: Sequence[str] = ("heximax", "heximax", "search2")` — names
  resolved through `arena.entrant_from_name`/`arena.PRESETS`
  (`arena.py:599-644`, `:240-284`) and spawned with `arena.spawn`
  (`arena.py:290`). Default is `heximax` at its default `mode="honest"`
  (`arena.Entrant.mode` field default, `arena.py`), the live-deployment
  referent, not `heximax-omni` (the perfect-information evaluation ceiling)
  — a learner training against opponents that can see hidden cards would be
  training against a threat model it will never face at the actual table.
  `search2` fills the last seat as the shipped tree-search baseline
  (`arena.py:243`).

`step(action)` applies the learner's action, then auto-plays every
non-learner turn in a loop that is `arena.play`'s loop
(`arena.py:453-465`) restricted to non-learner seats — the same pattern as
`catanatron.gym.envs.CatanatronEnv._advance_until_p0_decision`, which spins
the other `Player`s' turns inside `step()`/`reset()` until its own seat
(`Color.BLUE`) is back to move (pinned catanatron `d3f4ad05bb7`,
`catanatron/gym/envs/catanatron_env.py`). Observation and `action_mask` are
the identical dict `HexSetAEC.observe` returns, always for `learner_seat`;
`info` carries the raw information-set view once available (§4's dependency)
so a caller that wants more than the encoder's arrays — the certified
`known`/`unknown` ledger counts, `sample(rng)` — is not blocked from it.

Catanatron's `CatanatronEnv` is the shape users already know
(`Discrete(action_space_size)`, `get_valid_actions()`, `action_masks()` for
`sb3-contrib`'s `MaskablePPO`, an `enemies` config of fixed `Player`s
auto-played inside `step`, `reward_function` config point, `simple_reward` =
+1/−1/0 terminal-only by default). `HexSetEnv` mirrors it point for point —
`Discrete(553)`, `action_masks()`, an opponents list, seeded `reset`,
terminal reward by default — plus the honest observation (§4); reward is
+1/0 rather than +1/−1/0 since the other three seats are not this env's
"loss," only "not this seat's win."

## 4. Honesty

Every observation this design proposes is built from the seat's own
information set, never from `game._state` (today: `game.state`) directly for
an opponent's hidden cards.

**Today**, `encoding.encode(game, perspective=seat)` already is that
boundary: own hand and own development cards are exact, everyone else
contributes a count and `hexset.ledger`'s public-knowledge reconstruction of
their composition (`known`/`unknown` per opponent), never a hidden card
(`encoding.py:1-18`, `_offer_parts` at `:322-357`, `_ledger_parts` at
`:359-377`). This is already what the gym should call.

**When P0 lands** (`hexset.game.state(seat, hidden=True)`, branch
`feat/state-view` — checked 2026-09-02: no commits ahead of `main` yet, no
PR open; registered in `agents/reference/trading-design.md:697`, "the seat's
information-set view ... known and unknown counts per seat, expected hands,
hold probabilities, `sample(rng)`"), `hexset.gym`'s observation-building
function should call that view directly rather than re-deriving honesty
against `Game` fields itself, so there is exactly one honest-view boundary
in the codebase rather than two that must be kept in sync by hand. This is a
soft dependency: the gym can ship against `encoding.encode` today and switch
its internals to the `View` surface as a follow-up once P0 merges, since
`encode`'s output shape does not need to change either way.

**The mask leak this design must not repeat**: `actions.legal_actions`
enumerates `PROPOSE_TRADE` options by reading which specific opponents could
*cover* an offer — i.e., their true hands — which is exactly the
hand-composition leak `server/rules.py`'s module docstring documents and
`fair_legal_actions` exists to close (`server/rules.py:1-20`, `:71-80`; PR #2
review flagged the same defect in an embedded bot's mask, per
`hexset-pr2-review` notes). `HexSetAEC`'s `action_mask` must be built the
same way `server/rules.py.fair_legal_actions` is — widen `PROPOSE_TRADE`'s
sample to "proposing is available," never to "these particular opponents can
cover it" — not from `actions.legal_mask` raw.

**Test**: mirror `tests/catanatron/test_catanatron_information_set.py`'s
audit (`:142-166`, plus its control at `:173`, "the audit can fail"), adapted
to a native game instead of a foreign one: play a random game to some tick,
capture `observe(agent)` for a perspective seat, redeal the *other* seats'
`GameState.hands` and shuffle `GameState.deck` (composition-preserving —
same per-seat totals, same bank, same global multiset) and shuffle
`pending_responders`' declined-order bookkeeping where applicable, then
assert the observation dict is byte-identical; a second test perturbs the
perspective seat's own hand and asserts the observation *does* move, so a
test that permutes nothing reachable cannot pass by accident.

## 5. Tests to write

- PettingZoo's own `pettingzoo.test.api_test` and `seed_test` against
  `HexSetAEC` (four seats, random policy, several thousand cycles).
- `gymnasium.utils.env_checker.check_env` against `HexSetEnv` (with
  `skip_render_check=True` — no renderer proposed here).
- Mask correctness: for every `(observation, action_mask)` pair collected
  over a random-agent episode, replay `actions.legal_actions(game)`-minus-
  `PROPOSE_TRADE`-plus-the-honest-sample against the same game and assert the
  two masks agree bit-for-bit.
- Determinism: two `reset(seed=N)` runs, same random-agent action sequence,
  byte-identical observation sequence.
- The honesty permutation test and its control (§4).
- One full 4-seat random-agent AEC episode reaches `is_over` (either a
  winner or `MAX_TURNS` exhaustion) inside a generous step budget, with no
  exception and no seat ever asked to act with an empty `action_mask`.

## 6. Not in scope

- The one-event trading mechanic (`agents/reference/trading-design.md` §8) —
  registered separately; this design's action/observation shapes are the
  current engine's.
- Human or LLM interfaces — the existing HTTP API and MCP server already
  serve those; this is a training-loop-facing surface only.
- Vectorised/parallel envs (`gymnasium.vector`, PettingZoo's `ParallelEnv`)
  — AEC is the primary API per the owner's decision; a vectorised wrapper is
  a possible future layer, not proposed here.
- Reward shaping beyond §2's two options (terminal win/loss,
  `relative_points`) — no potential-based shaping, no auxiliary rewards.

## 7. Open questions for the PI

- Observation as one flat vector (what most off-the-shelf single-agent RL
  code expects, and what Catanatron's default `"vector"` representation
  does) vs. the dict-of-arrays shape proposed above (what a graph model
  reads, and what `hexnet` already consumes) — or both, gated by a
  `representation` config key the way `CatanatronEnv` does.
- Whether the observation should be the encoder's `(hexes, vertices, edges,
  globals)` arrays at all, or the `onnx_record` fields
  (`onnx_record.RECORD_FIELDS`, `onnx_record.py:83-107`) instead — the
  latter is torch-free and rules-native rather than model-native, and is
  already what gets served over the wire (`server/api.py:788-812`).
- Reward for non-winners: flat `0` (proposed default) vs. `-1/(players-1)`
  (zero-sum against the winner) vs. always offering `relative_points` as the
  only reward rather than an opt-in.
- Truncation semantics precision: is reaching `MAX_TURNS` a `truncation` (as
  proposed) or should some exhausted games instead score as a loss for
  everyone, given the arena already treats "exhausted" as its own outcome
  category distinct from "unfinished" (`CHANGELOG.md`, "Unreleased" section)?

## Cost estimate

Two to three days: `HexSetAEC` plus its observation/mask plumbing off
`encoding.encode` and `fair_legal_actions`'s pattern (~1 day, mostly
adapting existing, tested code rather than writing new rules logic);
`HexSetEnv` on top, including opponent auto-play and seat rotation (~0.5
day); the test suite in §5, including the honesty permutation test adapted
from the catanatron adapter's (~0.5–1 day, most of it copy-and-adapt); docs
and the `pyproject.toml` extra (~0.25 day). Excludes any work triggered by
P0 landing under it (§4) or by the one-event trading mechanic changing the
action space (§6) — both are separate, unscheduled efforts this design
explicitly declines to anticipate.

## PI ratification (2026-09-02, Fable)

Decisions on §7, so implementation can start:

1. **Observation form.** The AEC env returns a dict of the encoder's arrays
   (`hexes (19,11)`, `vertices (54,14)`, `edges (72,5)`, `globals (86,)`) plus
   `action_mask`. The Gymnasium wrapper flattens by default (`flatten=True`,
   Catanatron's shape, what `MaskablePPO` users expect) and can return the dict
   with `flatten=False`.
2. **Source.** Observations are the encoder's arrays, never the ONNX record —
   the record is a deployment contract for `hexset.server`/`clients`, not a
   learning interface.
3. **Reward.** Terminal only by default: `+1` to the winner, `0` to everyone
   else (Catanatron parity). `reward="relative_points"` is the one alternative
   (per-seat VP relative to the leader at the terminal step). No shaping.
4. **Exhaustion.** Hitting `MAX_TURNS` is a `truncation` for every agent with
   reward `0`, mirroring the arena's `exhausted` (distinct from `unfinished`).

Two requirements the draft flagged become hard rules: the `action_mask` is
built the honest way (never from the raw `legal_actions` sampler that reads
opponents' hands), and every observation is built from
`game.state(seat, hidden=True)` once P0 lands — implementation therefore
starts **after** `feat/state-view` merges, not against today's boundary.
The honesty permutation test (§4) is a merge requirement, not optional.
Sequence: P0 → `hexset.gym` → P1 (one-event trading; the gym's observation
gains the public valuation block and loses the trade actions then).
