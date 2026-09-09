# Training and runtime interfaces

HexSet provides game execution, legal actions, information sets, encoding,
seating, board pairing, trade evaluation, and replay. A training application
supplies its model runtime, learning algorithm, and training targets.

The PettingZoo and Gymnasium examples are in the
[README](../README.md#training-environments). This guide covers batched games
and model runtimes that use the same engine without those API adapters.

## Batched games

`hexset.gym.LaneEnv` holds multiple games in flight. `requests()` returns one
decision per live lane, and `step()` advances each lane by one action. It
requires only the base HexSet installation.

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

This bounded example exercises collection and truncation, not full-game
learning. For training, choose an action cap that fits your task and inspect
`episode.outcome.truncated`: an unfinished game is not a terminal loss.

Each request contains the lane, game index, seat, policy ID, legal actions,
and live `Game`. `request.view` provides the seat's information set. The `Game` remains mutable and exposes full state; `View` is the information
contract, not an access-control mechanism. Encode or copy the observation and
legal mask before calling `step()`, because the request refers to a live game.
Do not mutate that game; simulations should use copies.

`step()` accepts either actions in request order or a mapping from lane IDs
to actions. Missing answers use a configured bot for that seat; an unanswered
seat without a bot is an error. Finished games return as `Episode` objects,
with per-seat decision histories and an outcome. With `records=True`,
`episode.record` also contains the chance stream and recorded trades for
replay through `hexset.record`.

`deal` limits the number of games started. Without it, completed lanes refill
indefinitely. For evaluation, bound the games dealt so the result does not
select only the games that finish first. `cohort()` extends a bounded environment with another group of games,
retaining any games still in flight.

A `caster(game_index)` maps seats to policy IDs. `bots` maps those IDs to bot
factories; `gates` maps them to trade-gate factories taking `(game, seat)`.
A configured bot supplies its own gate. Other seats need an explicit gate to
participate in player trading. Unlike the PettingZoo and Gymnasium learner
interfaces, a lane learner can therefore trade. `max_trades=0` disables the
automatic trade event.

`board` accepts a fixed board or a function from game index to board.
`hexset.casting` provides rotation, pairing, and policy-ID swapping helpers.
`first_game` and `stride` control game indices for resumed or sharded runs.

## Batched evaluation

`hexset.bench.versus.compete_batched` evaluates policies over lanes with the
arena's board and seating rules:

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

This is another bounded interface example, not a strength comparison. A
runner seed controls game generation; it does not seed arbitrary runtime
sampling. `BotPolicy` shares a bot per board and policy ID, including its
mutable random generator, so stochastic decisions can depend on lane count
and request order. For scheduling-independent sampling, a runtime should key
its randomness by game index, seat and decision step.

A batch policy implements `act(requests)` and returns one action per request.
`BotPolicy` adapts a regular bot factory. `PolicyPolicy` adapts the model
`Policy` protocol below; pass its checkpoint as well to install the network
trade gate. Without a checkpoint, it has no network trade gate.

The default antithetic evaluation plays boards under complementary seat
assignments. Results include standings, Wilson win-rate intervals, and
paired terminal victory-point margins. Set `episodes=True, records=True` to
retain replayable episode records in the verdict.

## Model runtimes

[policy.py](../src/hexset/clients/policy.py) defines structural `Policy` and
`Checkpoint` protocols. A policy exposes its `space` and implements batched `act_rows`, `value_rows`,
and `score_rows` methods over live positions and seat perspectives. It returns
actions, per-seat values, and legal-option priors in board-seat order.
`score_rows` priors align with each row's supplied options; an ONNX graph's
dense action prior is converted to that order by its adapter.

A checkpoint carries `policy`, `space`, `players`, and `max_trades`.
[netbot.py](../src/hexset/clients/netbot.py) provides `bot_for` and
`searcher_for` constructors using these fields. Evaluate positions directly
with `checkpoint.policy.value_rows([(game, seat), ...])`. The bot,
continuation-based trade gate, and search are shared across runtimes.
Pass `rng=random.Random(seed)` to `bot_for` to seed trade-gate belief sampling.
A supplied `rng` in `searcher_for` seeds both MCTS and a separate trade-gate
stream. These generators do not seed sampling inside the model runtime; the
training application must control that source of randomness separately.
The ONNX implementation is `hexset.clients.onnxbot.V2Policy`; see the
[ONNX contract](bot-api.md) for record tensors and output requirements.

A driver can register an ONNX loader for arena entrants explicitly:

```python
from hexset.clients.netbot import register_entrants
from hexset.clients.onnxbot import load

def configure_runtime():
    register_entrants(load)

configure_runtime()
```

This requires the `clients` extra. Registration assigns the process's network
and MCTS entrant factories. Imports alone do not register a runtime. A custom
runtime can register its own loader with the same interface. For an arena
run, pass `worker_initializer=configure_runtime` to `compete` to register in
each worker process, including under `start_method="spawn"`. Define this
function at module scope in an importable driver and guard its entry point;
with one worker the initializer runs in the calling process. Optional
`worker_initargs` supplies initializer arguments.

For a gate installed separately from the action bot, construct a fresh
`NetworkBot` per seat, set `bot.seat = seat`, and call `bot.seat_at(game)` in
the gate factory. The collector does not call that gate's `choose()` method,
so the factory must bind the game explicitly. The `seat` field guards against
requests for another seat. `PolicyPolicy` supplies this
wiring when given a checkpoint.
