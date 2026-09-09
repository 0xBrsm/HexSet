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
env = LaneEnv(players=4, seed=0, lanes=4, deal=8, records=True)
episodes = []
while env.running:
    requests = env.requests()
    actions = [rng.choice(request.options) for request in requests]
    episodes.extend(env.step(actions))
```

Each request contains the lane, game index, seat, policy ID, legal actions,
and live `Game`. `request.view` provides the seat's information set. Do not
mutate the live game; simulations should use copies.

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
from hexset.bench.versus import BotPolicy, compete_batched
from hexset.bots import RandomBot

policies = {0: BotPolicy(lambda board: RandomBot()),
            1: BotPolicy(lambda board: RandomBot())}
verdict = compete_batched(policies, games=8, players=4, seed=0, lanes=4)
print(verdict.metrics())
```

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
[netbot.py](../src/hexset/clients/netbot.py) provides `bot_for`,
`evaluator_for`, and `searcher_for` constructors using these fields. The bot,
continuation-based trade gate, and search are shared across runtimes.
The ONNX implementation is `hexset.clients.onnxbot.V2Policy`; see the
[ONNX contract](bot-api.md) for record tensors and output requirements.

A driver can register an ONNX loader for arena entrants explicitly:

```python
from hexset.clients.netbot import register_entrants
from hexset.clients.onnxbot import load

register_entrants(load)
```

This requires the `clients` extra. Registration assigns the process's network
and MCTS entrant factories. Imports alone do not register a runtime. A custom
runtime can register its own loader with the same interface. Register it in
each worker process that spawns entrants.

For a gate installed separately from the action bot, `NetworkBot.seat_at(game)`
sets the game it evaluates without choosing an action. Its optional `seat`
field guards against requests for another seat. `PolicyPolicy` supplies this
wiring when given a checkpoint.
