# Training and runtime interfaces

Reinforcement learning against HexSet. The engine provides game execution,
legal actions, information sets, encoding, seating, board pairing, trade
evaluation and replay. A training application supplies its model runtime,
learning algorithm and training targets; architecture, optimisation, target
construction and checkpoint export are its own, and none of them needs ONNX
during collection. Running the engine itself is [guide.md](guide.md).

## Batched games

`hexset.gym.LaneEnv` holds several games in flight. `requests()` returns one
decision per live lane, and `step()` advances each lane by one action. It
needs only the base installation.

```python
import random
from hexset.gym import LaneEnv

rng = random.Random(0)
env = LaneEnv(players=4, seed=0, lanes=4, deal=8, action_cap=40, records=True)
episodes = []
while env.running:
    requests = env.requests()
    actions = [rng.choice(request.options) for request in requests]
    episodes.extend(env.step(actions))
```

This bounded example exercises collection and truncation, not learning. An
undecided game is not a terminal loss. `episode.outcome.winner` is `None` in
two cases: `outcome.truncated` (the action cap stopped a game still in play)
and `outcome.exhausted` (the game reached `turn_cap` turns, default
`MAX_TURNS`).

Each `Request` holds the lane, game index, seat, policy ID, action step,
legal `options` and the live `Game`; `request.view` is the seat's
information set. The `Game` is mutable and exposes full state: `View` is the
information contract, not an access-control mechanism. Encode the
observation and legal mask before calling `step()`, because the request
refers to the live game, and never mutate it; simulations use copies.

`step()` takes actions in request order or a mapping from lane IDs to
actions. A missing answer (`None`) is played by the seat's configured bot;
an unanswered seat without a bot raises `ValueError`, as does an illegal
action, before any lane advances. Finished games return as `Episode` objects
with per-seat decision histories, every cleared exchange and an `Outcome`.
With `records=True`, `episode.record` holds a `hexset.record.Record` for
replay.

`deal` bounds the number of games started; without it, finished lanes refill
indefinitely. For evaluation, bound the games dealt so the result does not
favour the games that finish first. `cohort(games)` re-arms a bounded
environment with more games, keeping the ones in flight. `first_game` and
`stride` set the game indices for resumed or sharded runs (worker `w` of `K`:
`first_game=w, stride=K`), and `games_started()` is the `first_game` to
resume from.

A `caster(game_index)` maps seats to policy IDs. `bots` maps IDs to bot
factories taking the board; `gates` maps IDs to trade-gate factories taking
`(game, seat)`. A bot-played seat is its own gate. Any other seat trades only
through an explicit gate; without one it never trades. A gate whose
`TradeParams.max_offers` is `0` opens no trade event on its turn.

`board` takes a fixed board or a function from game index to board (`None`
for the default deal). A bot serves every lane on its board, so with a fixed
board one bot serves every lane; before each action the environment calls
`seat_at(game)` on every gate in that lane. `hexset.casting` provides
rotation, pairing and policy-ID swapping casters.

### Collector practice

- Store observations and legal masks before advancing the live games.
- Keep each decision with its seat, step and policy version.
- Treat action-cap truncation and turn-cap exhaustion apart from a terminal
  loss.
- Give the learner an explicit trade gate to train with player trading.
- Evaluate checkpoints on a bounded set of games against a fixed opponent
  configuration.

## Batched evaluation

`hexset.bench.versus.compete_batched` evaluates policies over lanes with the
arena's boards and seating:

```python
import random
from hexset.bench.versus import BotPolicy, compete_batched
from hexset.bots import RandomBot

policies = {0: BotPolicy(lambda board: RandomBot(random.Random(0))),
            1: BotPolicy(lambda board: RandomBot(random.Random(1)))}
verdict = compete_batched(
    policies, games=8, players=4, seed=0, lanes=4, action_cap=40,
)
print(verdict.metrics())
```

This is an interface example, not a strength comparison. The runner seed
controls game generation, not sampling inside a runtime. `BotPolicy` builds
one bot per board and policy ID and keeps it, with its random generator, in
an LRU cache of 256 boards (`capacity`); a bot shared across games makes its
draws depend on lane count and request order. For scheduling-independent
sampling, key randomness by game index, seat and decision step.

A batch policy implements `act(requests)` and returns one action per
request; an optional `gate(game, seat)` returns that seat's trade gate.
`BotPolicy` adapts a bot factory and is its own gate. `PolicyPolicy` adapts
the model `Policy` protocol below; given a checkpoint as well it installs
the checkpoint's network trade gate, and without one the seat never trades.

`antithetic=True` (the default) needs exactly two policies and an even
`games`, and plays each board under both seatings. The `Verdict` holds
standings, Wilson intervals, board-level win-rate and victory-point
intervals, and the count of truncated games; `metrics()` flattens it
([evaluation.md](evaluation.md#intervals)). `episodes=True, records=True`
keeps every episode, with its record, in `verdict.episodes`.

## Model runtimes

[policy.py](../hexset/clients/policy.py) defines the structural `Policy` and
`Checkpoint` protocols. A policy exposes its `space` and implements batched
`act_rows`, `value_rows` and `score_rows` over live positions and seat
perspectives: actions, per-seat values in board-seat order, and priors
aligned with each row's supplied options. An ONNX graph's dense action prior
is converted to that order by its adapter.

A checkpoint carries `policy`, `space` and `players`. How it bargains is its
`trade_params` (a `hexset.trading.TradeParams`), or the loose `trade_floor`
and `gate_plies`; declaring none reads `hexset.trading.UNLIMITED`. An
optional `trader` names a bot that answers its trades.
[netbot.py](../hexset/clients/netbot.py) builds bots from these fields:
`bot_for(checkpoint)` one forward per decision, `searcher_for(checkpoint,
simulations=128, wave=16, k=1)` PUCT search, both with the checkpoint's own
trade gate. `checkpoint.policy.value_rows([(game, seat), ...])` evaluates
positions directly.

`bot_for(rng=...)` seeds the gate's protocol and its sampled worlds;
`searcher_for(rng=...)` seeds the search and a separate gate stream derived
from it. Neither seeds sampling inside the model runtime, which the training
application controls. The ONNX implementation is
`hexset.clients.onnxbot.V2Policy`, loaded by `onnxbot.load(path, topology)`;
the [ONNX contract](onnx.md) gives the record tensors and output
requirements.

A driver registers an ONNX loader for the `network:` and `mcts:` entrants
explicitly:

```python
from hexset.clients.netbot import register_entrants
from hexset.clients.onnxbot import load

def configure_runtime():
    register_entrants(load)

configure_runtime()
```

This requires the `clients` extra. Importing a module does not register a
loader; `register_entrants(loader)` takes any `loader(path, topology) ->
Checkpoint`, so a custom runtime registers its own the same way. For an arena
run, pass `worker_initializer=configure_runtime` to `compete` so every worker
registers it, `start_method="spawn"` included; define the function at module
scope in an importable driver with a guarded entry point. A module that
calls `register_entrants` at import is a runtime, and `--runtime <module>`
loads it on the command line.

### The determinized root

A search decision is not made on the real position. `hexset.mcts.Search`
draws `k` samples from the mover's own `View`, folds them to the distinct
worlds among them (keyed by `hexset.bots.determinized.holdings_signature`,
which ignores development-deck order, since a `BUY_DEV_CARD` edge reshuffles
the deck) and searches one tree per world. Every unseen development card is
dealt alike unless the search is built with `hold=` (a
`hexset.view.HoldReading`). Inside a world the tree resolves steals,
development-card draws and terminal points against that world's hands, which
are sampled, so no opponent's actual card reaches the answer. The roots are
combined by summing visit counts, priors and values weighted by each world's
share of the draws; the chosen action, the visit distribution
(`hexset.mcts.visit_policy`, the expert-iteration target) and the root value
come from that combined root, and `Node.worlds` lists the per-world roots.

`simulations` is one world's budget, not the decision's: combined visit
counts still sum to it, and a `k>1` decision costs one tree per distinct
world. A root with one legal move is evaluated but not searched; its edge
holds all `simulations` visits at the root's value. A descent in flight
counts against its edge as a lost visit, valued at the stance's reading of
a game another seat won (`hexset.mcts.lost_value`: -1/3 for `relative` at
four seats), so a wave spreads over the edges whatever the sign of their
means. `k` defaults to 1
and reaches `searcher_for` and `onnxbot.searcher` as a keyword and the arena
as `mcts:<path>@<simulations>w<wave>:k=<n>`. `Search(hidden=False)` roots on
the true state instead: the omniscient search, an analysis tool that nothing
seating a bot passes.

### The network trade gate

`hexset.clients.netbot.NetworkBot` is the trade gate for any policy runtime,
and `searcher_for`'s `GatedSearch` delegates to it. Its `trade_floor` and
`gate_plies` come from the checkpoint, defaulting to `0.0` and `0`.

At `gate_plies = 0`, every candidate is priced by the value head in one
batched forward alongside the live position. The post-trade position is
built in the seat's own frame: its hand exactly, plus the bundle; the
counterparty's certified cards moved by the bundle and its hand size moved
by the bundle's total, never its true hand. The evaluating seat's row gives
the gain and the counterparty's row the estimate of the other side's gain. A
candidate the seat cannot cover, or one the counterparty's public hand
cannot cover (`hexset.trading.has_room`), is answered `-1.0`.

A checkpoint declaring `gate_plies = N` advances the mover's own policy
(`act_rows`) up to `N` plies from each candidate before the value forward,
stopping at the mover's `END_TURN`. Because the rollout acts, it runs on one
world sampled from the seat's belief (`View.sample`), certified to cover
every candidate, so every candidate is priced on the same draw.

A gate installed separately from the action bot is a fresh `NetworkBot` per
seat, with `bot.seat = seat` and `bot.seat_at(game)` called in the gate
factory: the collector never calls that gate's `choose()`, which would
otherwise bind the game, and `seat` refuses requests for another seat.
`PolicyPolicy` does this when given a checkpoint.

## Third-party environment adapters

`HexSetAEC` and `HexSetEnv` wrap the engine in the PettingZoo and Gymnasium
APIs. Both need the `gym` extra; `LaneEnv` does not, and is the interface for
a collector that drives the loop itself.

```sh
pip install -e ".[gym]"
```

`HexSetAEC(num_players=4, reward="terminal", discard_order="random",
game_type=STANDARD_GAME, turn_cap=MAX_TURNS)` provides one PettingZoo agent
per seat, `seat_0` to `seat_{n-1}`. Each observation is a dict:
`observation` holds the encoder's `hexes`, `vertices`, `edges` and `globals`
arrays, and `action_mask` is the engine's legal actions for the agent about
to act (all zeros for the others). `step` raises `ValueError` on an action
outside the mask, before anything is applied.

- `reward="terminal"` gives 1 to the winner on the terminal step and 0
  elsewhere; `"relative_points"` gives terminal points less the others'
  mean, divided by the points the game is played to
  (`hexset.victory.relative_points`, zero-sum).
- A game that reaches `turn_cap` turns without a winner sets `truncations`,
  not `terminations`, with reward 0 in either mode; random play needs
  `turn_cap=hexset.game.UNSTRUCTURED_TURN_CAP`.
- During a discard round several seats owe at once; `discard_order="random"`
  picks the next one from a stream seeded by `reset(seed)`, `"seat"` in
  ascending order, and `select_agent(agent)` overrides either.
- `reset()` without a seed draws one from `np_random`, so resets after a
  seeded one are reproducible.

```python
from hexset.gym import HexSetAEC

env = HexSetAEC(num_players=4)
env.reset(seed=0)
for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    action = None if terminated or truncated else env.action_space(agent).sample(
        observation["action_mask"]
    )
    env.step(action)
env.close()
```

`HexSetEnv`, registered as `HexSet-v0` on `import hexset.gym`, has one
learner seat and plays the others automatically. `opponents` (default three
`random`) takes one entrant name per opponent seat, so the table has
`len(opponents) + 1` seats; `runtime=("mybots",)` imports modules first
(`hexset.arena.load_runtime`), in the constructor, which every worker of a
vectorised environment also runs. `retired` is refused: a smaller table is
fewer opponents. `learner_seat` is a seat index or `"rotate"` (default), a
seat drawn at each reset. `reward`, `discard_order` and `turn_cap` are
`HexSetAEC`'s.

```python
import gymnasium
import hexset.gym  # registers HexSet-v0

env = gymnasium.make("HexSet-v0")
observation, info = env.reset(seed=0)
for _ in range(1000):
    action = env.action_space.sample(mask=info["action_mask"])
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

`HexSetEnv` returns a flat `Box` observation by default; `flatten=False`
returns the four arrays as a dict. The action mask is in
`info["action_mask"]`, and `action_masks()` is the `sb3-contrib` masking
hook. An action outside the mask is a no-op with reward 0 that leaves the
game untouched. `info["view"]` is the learner's `View`.

Player trading is not in either action space; bank and port trades are.
`HexSetAEC` seats no trade gates, so its agents never trade. In `HexSetEnv`
the opponents' bots are their own gates and trade with each other where they
implement it; the learner has no gate and takes no part. A learner that
trades uses `LaneEnv` with a gate.
