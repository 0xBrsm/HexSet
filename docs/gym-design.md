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

Agents are seats: four agents for the standard board (`possible_agents =
["seat_0".."seat_3"]`), and `agent_selection` names the one seat entitled to
act next. In every phase but one that is `f"seat_{to_move(game)}"`, including
`TRADE_RESPOND`, which hands the decision to someone other than the current
player but still to exactly one of them (`game.py:320-334`). The exception is
discard-on-seven, where the engine's single-seat answer is a serialization
the rules do not ask for and the environment does not impose one either — see
"`Phase.DISCARD`: any owing seat, in any order" below.

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
(`actions.py:117-131`), calls `apply(game, action, seat)` for the seat
`agent_selection` names, then advances `agent_selection` to whoever is
entitled next (`_next_agent`, below). The seat is dispatched with the action
rather than recovered from the position because `DISCARD` is the one action
whose actor the position does not fix; every other action ignores it
(`actions.apply`'s own docstring). Catan ends for the whole table
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

**`Phase.DISCARD`: any owing seat, in any order.** Discarding on a seven is
not a turn: every seat over the limit discards at the same instant, bounded
only by its own hand and its own `discard_quota` entry, and
`hexset.game.to_move`'s answer during that phase — the lowest-indexed seat
still owing — is a serialization the rules do not ask for. The live server
stopped imposing it (`hexset.game.may_act`, true for *every* owing seat, plus
the optional `seat` on `hexset.actions.legal_actions`/`legal_mask`/`apply`);
this environment does not impose it either. Live play and the gym now permit
the same set of orders.

This section used to argue the opposite — that since a discard round is
order-invariant, serializing it in the gym "costs nothing". The
order-invariance is real and is what makes any of this safe: a discard reads
and writes exactly one seat's hand, that seat's own quota and the bank; no
seat's discard can make another's legal or illegal (quotas are fixed at the
roll, `game.py`'s `roll_dice`, and do not shrink as hands do); and a chosen
discard draws no chance event, so every interleaving of a round reaches the
same position (`tests/test_actions.py::test_a_discard_round_is_order_invariant`).
What it does *not* make free is always picking the same seat first. Under
ascending order seat 0 never observes another seat's completed discard before
choosing its own and the last owing seat always observes all of them — an
asymmetry the seats do not have at a table, taught to a policy by every
self-play corpus this environment produces. Serializing was also the one
concrete thing standing between `hexset.record` and replaying a real table's
discard order (§2's sibling change, `Record.actors`).

The reconciliation with AEC, whose contract is exactly one active agent per
`step()`: **keep the contract, drop the hardcoded choice.** `agent_selection`
still names one agent and `step()` still resolves one action, so nothing
about the PettingZoo API changes — what changes is that the seat it names
during a discard round is chosen among the owing seats rather than fixed at
`min`. Three ways, in precedence order:

| | who acts next during a discard round |
|---|---|
| `select_agent(agent)` | the caller says, for any seat `may_act` allows; `ValueError` otherwise. This is the AEC-shaped hook for a caller that owns the order — a replay of a recorded table (`hexset.record.moves`), a test, a training loop sampling orders itself. |
| `discard_order="random"` (default) | drawn uniformly from `players_owing_discards(game)` each time a seat is named, from a stream seeded off `reset(seed)` alone (`random.Random(f"discard-order:{seed}")`) and never shared with the game's rng — so a fixed seed still deals a fixed board *and* a fixed discard order, and a self-play corpus carries no seat-order bias. |
| `discard_order="seat"` | the ascending order this environment used to hardcode, which is `to_move`'s own answer, for a caller that wants the old stream back. |

`observe(agent)` keeps the PettingZoo convention — the mask is all zeros for
every agent but `agent_selection` — and now builds the selected agent's mask
from `legal_actions(game, seat)` rather than the bare call, since during a
round the two answer for different hands. The flat action space is untouched:
`ActionType.DISCARD` still carries only a resource, every index in the
`(553,)` mask means what it always did, no checkpoint is invalidated. Nothing
is added to the observation either — the seat about to act already reads its
own quota and hand.

`HexSetEnv` (§3) inherits all of this and adds one wrinkle of its own, in
`_auto_play_opponents`: the learner is handed control the moment it *owes*
cards (`may_act`), not after every lower-numbered bot seat has cleared its
quota, so the learner is never queued behind a seat index. The bot seats are
then driven in `to_move` order — not this wrapper asserting an order, but the
`hexset.bots.Bot` protocol, whose `choose(game)` takes the position and
nothing else and therefore answers for the seat bare `legal_actions(game)`
answers for. Asking seat 3's bot while `to_move` names seat 0 would hand back
a card out of seat 0's hand. A caller wanting arbitrary order among *bot*
seats needs a seat-aware `Bot`; inventing one is not this environment's job,
and the round is order-invariant, so the bots lose nothing by going second.

Information leakage was checked rather than assumed: `hexset.encoding` gives
another seat only its hand *total* and `hexset.ledger`'s public-knowledge
`known`/`unknown` block, and a discard is a public event by the rules (the
cards are named to the table and the hand size is visible), so a seat's
completed discard becoming visible to a seat that has not yet discarded
reveals nothing that was ever hidden — only that it happened first, which
hand sizes give away in any implementation. The ledger is therefore updated
per discard as before; deferring it would break the invariant `sum(known) +
unknown == the seat's true hand size` that `hexset.ledger` holds at every
step. The ordering *is* suppressed where it would otherwise be asserted as
fact: `hexset.server.webplay.render_log` holds a round's discard lines back
until every owing seat has cleared its quota.

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
- `discard_order: str = "random"` — passed straight through to the wrapped
  `HexSetAEC` (§2's table). Same reasoning as `learner_seat="rotate"`: an
  ordering that is not in the rules should not be baked into what the learner
  sees.

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
- Discard order (§2): a round driven through `HexSetAEC` with
  `select_agent` in ascending order and in reverse order reaches the same
  hands, bank, quotas and phase; `select_agent` refuses a seat owing
  nothing; a `HexSetEnv` learner owing cards is handed control without
  waiting for a lower-numbered bot seat; `discard_order="seat"` reproduces
  `to_move` exactly and `"random"` is reproducible from `reset(seed)` while
  not always naming the lowest owing seat.
  (`tests/gym/test_discard_order.py`.)

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

## Implemented (2026-09-02)

`src/hexset/gym/` (`HexSetAEC`, `HexSetEnv`, `register()`), the `gym` extra,
and `tests/gym/` landed against this ratification, on `feat/state-view`'s
merged `game.state(seat, hidden=True)`. Two deviations from the letter of
this design, both mechanical necessities rather than design changes:

- **`PROPOSE_TRADE` through the flat `Discrete` space.** §2 says `step`
  decodes the index and applies it; `ActionSpace.decode` returns
  `PROPOSE_TRADE` with empty `give`/`want` (its own documented limit — an
  offer is ten numbers), which `trading.well_formed` rejects outright. A bare
  `Discrete` sample choosing that slot now has `HexSetAEC` draw a concrete,
  honest offer from the game's own seeded `rng` among
  `fair_legal_actions`'s one-for-one pairs; a caller with its own give/want
  heads can still pass a full `Action` to `step` directly. See
  `hexset/gym/aec.py`'s module docstring.
- **A pre-existing gap in `encoding._offer_parts`, found by §4's permutation
  test.** The "who has declined" block re-derives eligibility
  (`trading.responders`) from the *live* true state rather than a fact
  frozen at proposal time, which is invisible in real single-timeline play
  but moves under the audit's counterfactual redeal whenever the perspective
  seat is a standing offer's proposer. `tests/gym/test_honesty.py` documents
  and skips exactly that precondition rather than weakening the audit or
  patching `hexset.trading`/`hexset.game`'s schema unreviewed; flagged for
  the PI to triage separately.

## Amended by the one-event trade mechanic (2026-09-03)

Both deviations above are gone, and neither needed fixing: the thing each
worked around no longer exists. `hexset.trading` replaced the offer protocol
with one engine event a turn, so there are no trade actions at all.

- **The flat-`Discrete` gap is closed.** Every action now decodes exactly
  from its index, so `HexSetAEC.step` is the plain "decode the index, apply
  the action" §2 asked for. The interim random-offer fill is deleted.
- **The `_offer_parts` leak is closed by deletion.** The offer block, whose
  "who has answered" part re-derived responder eligibility from the live
  true state, is replaced by every seat's public valuation vector — public by
  construction, and a function of nothing hidden. `tests/gym/test_honesty.py`
  no longer skips anything, which was the point of recording the gap rather
  than weakening the audit.
- **One mask, not two.** §4's "build the mask the honest way, never from the
  raw `legal_actions` sampler" was about the offer sample, the only branch
  that read opponents' hands. `action_mask` is now `legal_actions` itself,
  and `tests/gym/test_aec_mask.py` pins the property directly: permuting
  every opponent's hidden cards does not move the mover's option list.
- **Observations gain the valuation block**, as this design anticipated:
  `globals` loses 18 offer features and a phase (`TRADE_RESPOND`) and gains
  `players * NUM_RESOURCES`, 86 → 87 at four players.
- **The learner does not trade.** `HexSetEnv` supplies its opponents'
  `valuation`/`accepts` to the engine; the learner seat publishes nothing, so
  it never trades. Giving a learner a way to publish is the deferred half of
  the mechanic's interface (the trading design defers the human/LLM surface),
  not an omission here.
