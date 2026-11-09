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
env = LaneEnv(players=4, seed=0, lanes=4, deal=8, action_cap=40)
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
Every game is recorded: `episode.record` holds its `hexset.record.Record`
for replay.

During a seven's discards several seats owe at once. `discard_order="seat"`
(the default) asks the lowest owing seat first; `discard_order="random"`
draws the next seat among those owing, from a stream of the game's own that
shifts none of its dice or deck, and lets it finish its whole discard first.
A discard taken out of seat order is filed in the record's `actors`.

`decisions` names policy IDs asked for whole decisions (`hexset.decisions`):
their requests carry `request.decision`, the decision's tensors
(`decisions.tensors`), and empty `options`, and are answered with an action
or a `Compound` among the decision's options. `step()` applies a compound's
parts one at a time, each its own step of the action stream and the record,
so a record replays as if the seat had acted a move at a time. Bot seats
always act a move at a time. The snake seat's double setup is asked as two
setup turns, the second on the board after the first.

Every request names the run of options it belongs to: `request.decision_id`
(counted from 0 per game, every seat's together) and `request.part` (0 for
the run's first). A seat's run lasts from its first option to the first that
shows something or hands the table on (`decisions.closes`: the roll, a dev
card bought, a robber's steal, a Monopoly, a trade event the seat may offer
in, its discard done, the end of its turn, another seat to act), so a
learner can credit the run as one decision while each part stays what one
forward chooses among. Each `Decision` of an episode carries its run's
`decision_id`.

`deal` bounds the number of games started; without it, finished lanes refill
indefinitely. For evaluation, bound the games dealt so the result does not
favour the games that finish first. `cohort(games)` re-arms a bounded
environment with more games, keeping the ones in flight. `first_game` and
`stride` set the game indices for resumed or sharded runs (worker `w` of `K`:
`first_game=w, stride=K`), and `games_started()` is the `first_game` to
resume from. `indices` deals exactly the games it lists, in that order, and
nothing else, such as a resumed cohort's missing games; each is the same
game it is under any other deal. `hexset.gym.lanes.episode_entry(episode)`
is a finished `Episode` as one journal line (`hexset.gamelog.append`): its
index and seed, the policy ID in each seat, how it ended, its cleared trades
and its record.

A `caster(game_index)` maps seats to policy IDs. `bots` maps IDs to bot
factories taking the board, or to a `BoardBots`; `gates` maps IDs to
trade-gate factories taking `(game, seat)`. A bot-played seat is its own
gate. Any other seat trades only through an explicit gate; without one it
never trades. A gate whose `TradeParams.max_offers` is `0` opens no trade
event on its turn.

Every bot seat gets a bot of its own for its game (`BoardBots.fresh`): built
by `seat_spawn(board, rng)` off `random.Random(seat_key(seed, index, seat))`
where the `BoardBots` has one, else by `spawn(board)`. A game therefore
draws the same numbers whatever the lane count or worker split. A bot whose
class defines `seen` is shown every move of its lane
([guide.md](guide.md#implement-a-bot)). A bot class with `choose_many`
(`hexset.bots.ChoosesMany`) is asked once a tick for every lane where one of
its bots is to move, so one batched forward serves them all; the network bot
(`NetworkBot.choose_many`) asks its policy once per policy per tick.

`board` takes a fixed board or a function from game index to board (`None`
for the default deal). A caller's gate may serve several lanes, so before
each action the environment calls `seat_at(game)` on every gate in that
lane. `hexset.casting` provides rotation, pairing and policy-ID swapping
casters.

### One loop over Python and native lanes

`hexset.gym.driver.Driver(env, seats, keep=...)` plays a `LaneEnv`'s games
with the caller's seats answering their policy IDs. Each tick it collects one
request per live lane, asks each seat's `choose_many(requests)` once for all
of its rows, in lane order, and applies the answers. One seat object may
answer several policy IDs; it is asked once for all of their rows. An answer is an object
with `action` (an action, or a `Compound` for a decision) and `decision`
(the `(DecisionKind, index)` it chose, or `None` for an atomic answer). Bot
seats are played inside the environment. Every finished game goes to
`keep(episode)` before its lane is dealt again, and `keep` must return only
once the game is retained; `on_decision(request, answer)` sees each answered
decision; `play()` ticks until every dealt game is kept.

```python
import random

from hexset import gamelog
from hexset.gym import LaneEnv
from hexset.gym.driver import Driver
from hexset.gym.lanes import episode_entry


class Answer:
    def __init__(self, action):
        self.action, self.decision = action, None


class Uniform:
    def choose_many(self, requests):
        return [Answer(random.Random(r.draw).choice(r.options)) for r in requests]


env = LaneEnv(players=4, seed=0, lanes=4, deal=8, action_cap=40)
driver = Driver(env, {0: Uniform()},
                keep=lambda episode: gamelog.append("driver.games.jsonl", episode_entry(episode)),
                lanes="auto")
episodes = driver.play()
```

`request.draw` is the decision's sampling key (`hexset.gym.driver.draw_key`),
a function of the seed, game index, seat and step, so a policy that samples
from it alone plays each game the same however games are batched or
sharded. `Driver.targets()` holds the referee's labels for the tick's rows
(every seat's true points, the turn count), never part of a request.

`lanes` is `"python"` (the default), `"native"` or `"auto"`. Native lanes
import each dealt game into the optional C++ core ([native.md](native.md))
and refill from the same dealer in the same order, so a game is the same
function of `(seed, index)` on either kind and keeps the same record. They
need the native extension, a bounded deal, and no bot, watcher or trade
gate. A seat asked for whole decisions (`decisions=`) must read one
contract-9 observation natively (the seat's `native_contract`, the same for
every such seat); a seat asked atomic moves must read native positions (its
`native_atomic`, the contracts its policies read, each one of
`hexset.gym.driver.NATIVE_READS`: 8, 8.1, the contract-9 ones, and
"record", the information-set record an ONNX record graph reads). One seat
object answers whole decisions or atomic moves, not both. `"auto"` takes
native lanes where that holds and the Python lanes otherwise, as in the
example above, whose seat reads no observation. A whole-decision request on
a native lane carries no `Game`; `hexset.gym.driver.observations(requests,
contract)` reads each row's observation from the native buffer or encodes
it from the row's position. An atomic request on a native lane carries the
same legal actions, in the same order, as on a Python lane, and as its
`game` a read-only view of the lane's position
(`hexset.native.playout.NativeCopy.viewing`), which a policy reads as it
reads a `Game`; it is valid until the lane moves. Under
`discard_order="random"` the seat drawn to discard keeps discarding until
its quota is met, on either kind, so an atomic seat discards a card at a
time as on a Python lane.

### Policy seats and labels

`hexset.gym.seats.PolicySeat(policies, plays, record=False)` is a driver seat
for atomic requests, on Python lanes or native ones. Each tick it calls every policy in
`policies` once (`score_rows`) over all of its rows that have more than one
option; a row with one option is played without a forward. `plays` maps
each policy ID it answers to a `Play`: the product of some of `policies`
(`members`, a committee; one member is that policy alone) or a policy of its
own (`policy`, read on that ID's rows only), at a `temperature`. At 0 it
plays the option with the highest mean log prior over the members, as
`hexset.clients.committee.Committee` does; above 0 it draws from the
renormalised product raised to `1 / temperature`, reading only the request's
`draw` key. With `record=True` each answer to a multi-option row carries a
`Label`: the seat, the option played, and every policy's prior and
per-seat values as float32. `Labels` is an `on_decision` hook that keeps a
game's labels until `take(index)`, normally in `keep`.

```python
from hexset.gym.seats import Labels, Play, PolicySeat

seat = PolicySeat([net_a, net_b, net_c], {0: Play(members=(0, 1, 2), temperature=0.5),
                                          1: Play(members=(1,), temperature=0.5)}, record=True)
labels = Labels()
driver = Driver(env, {0: seat, 1: seat}, on_decision=labels,
                keep=lambda episode: store(episode, labels.take(episode.index)))
```

`PolicySeat.native_atomic` is the set of contracts its policies read
(each policy's `native_contract`), or None where a policy reads only a
`Game` or the seat reads through a `scorer`; the driver plays the seat on
native lanes only where it is set. There it plays the same games and records
the same labels, bit for bit, as on Python lanes: each tick's rows are the
same rows in the same order, and a native position encodes to the same bytes.

`PolicySeat(..., scorer=fn)` reads every policy through `fn(rows)`, which
returns one `score_rows` answer per policy in order: policies served by
another process then cost one round trip a tick for all of them.
`PolicySeat(..., batch=False)` asks each policy one row at a time. It plays
the same games where the policies' forwards do not depend on their batch,
and is the reference the batched seat is checked against.

`LaneEnv(no_trade=ids)` puts every seat of those policy IDs under a trade
embargo with every other seat, as the arena closes an `Entrant.no_trade`
seat.

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

policies = {0: BotPolicy(lambda board: RandomBot(random.Random(0)),
                         seat_spawn=lambda board, rng: RandomBot(rng)),
            1: BotPolicy(lambda board: RandomBot(random.Random(1)),
                         seat_spawn=lambda board, rng: RandomBot(rng))}
verdict = compete_batched(
    policies, games=8, players=4, seed=0, lanes=4, action_cap=40,
    journal="batched.games.jsonl",
)
print(verdict.metrics())
```

This is an interface example, not a strength comparison. The runner seed
controls game generation, not sampling inside a runtime. The lanes build
each seat a bot of its own for the game, off `seat_spawn(board, rng)` with
the stream `hexset.arena.seat_key` names for that game and seat, or off
`spawn(board)` without one, so a bot that draws from `rng` plays the same
moves at any lane count. `BotPolicy`'s own `act` and `gate`, for a caller
with no lanes, share one bot per board in an LRU cache of 256 boards
(`capacity`).

A batch policy implements `act(requests)` and returns one action per
request; an optional `gate(game, seat)` returns that seat's trade gate.
`BotPolicy` adapts a bot factory and is its own gate. `PolicyPolicy` adapts
the model `Policy` protocol below; given a checkpoint as well it installs
the checkpoint's network trade gate, and without one the seat never trades.

`antithetic=True` (the default) needs exactly two policies and an even
`games`, and plays each board under both seatings. The `Verdict` holds
standings, Wilson intervals, board-level win-rate and victory-point
intervals, and the count of truncated games; `metrics()` flattens it
([evaluation.md](evaluation.md#intervals)). `episodes=True` keeps every
episode, with its record, in `verdict.episodes`.

## Model runtimes

[policy.py](../hexset/clients/policy.py) defines the structural `Policy` and
`Checkpoint` protocols. A policy exposes its `space` and implements batched
`act_rows`, `value_rows` and `score_rows` over live positions and seat
perspectives: actions, per-seat values in board-seat order, and priors
aligned with each row's supplied options. An ONNX graph's dense action prior
is converted to that order by its adapter. A policy that chooses whole
decisions may also keep `plans` (`hexset.decisions.Plans`), the rest of each
decision it chose, played a move at a time, and list a decision's whole
options with their probabilities (`option_rows`, `has_options`), which
paired rollouts search over.

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
application controls. The ONNX implementations are
`hexset.clients.onnxbot.V2Policy` for a record graph and `DecisionPolicy` for
a decision graph, both loaded by `onnxbot.load(path, topology)`; the
[ONNX contract](onnx.md) gives their inputs and outputs.

A driver registers an ONNX loader for the `network:`, `mcts:` and `rollout:`
entrants explicitly:

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
means. Values read over the seats still playing (`hexset.mcts.live_seats`),
so a duel dealt four seats with two retired reads `relative` over two: 0 is
an even game and a loss is -1. `k` defaults to 1
and reaches `searcher_for` and `onnxbot.searcher` as a keyword and the arena
as `mcts:<path>@<simulations>w<wave>:k=<n>`. `Search(hidden=False)` roots on
the true state instead: the omniscient search, an analysis tool that nothing
seating a bot passes.

`Search(decisive=p)` plays a root without searching it where the prior on
one of its moves is at least `p`: that move is credited with the whole
budget at the root's own value, as a root with one legal move is
(`mcts:<path>@128:decisive=<p>`). `Search.decide(game)` searches one
position and appends a `hexset.mcts.Decision` to `decision_log`: the
options, prior, root value, visits and mean values, the choice, whether it
searched, and the CPU time spent; `choose` is `decide`'s most-visited move.

### Searching by paired rollouts

`hexset.rollout.PairedRollouts(policy, worlds=32, top=3)` decides a position
by playing its candidate moves out on shared chance. The candidates are the
policy's own move (`act_rows`), then its other moves by prior, `top` in all.
Each is played out in `worlds` worlds, each world sampled from the mover's
own view as an MCTS world is, with its development deck in an order of its
own and its dice and steal streams split so the n-th roll and the n-th steal
are the same in every candidate's copy. After the candidate, the policy plays
every seat, one forward per decision, to the game's end or `horizon` turns
on. A playout that ends scores 1 where the mover won and 0 otherwise; one
stopped short is scored by the value head's chance for the mover, read from
the mover's own frame. The candidate with the highest mean score is played,
the earlier candidate on a tie, so the policy's own move first. Where a
move does not change how a world plays out, the shared chance cancels, so a
few dozen worlds separate moves that independent playouts would not.

- A root with one legal move, or under `decisive` one whose prior on one move
  is at least `decisive`, plays the policy's own move with no playout.
  `full_setup=True` (`rollout:...:setup=full`) exempts every opening root
  from `decisive`.
- A policy that lists whole options (`option_rows`) is searched over whole
  decisions at a root that opens one (a setup turn, the double turn, a Road
  Building card, a discard): each candidate is the whole decision, applied
  in every world before the playout, and the one chosen is played a move at
  a time with no further search. A decision graph that answers
  `hexset.decisions.OPTION_OUTPUTS` lists them ([onnx.md](onnx.md#outputs)).
- Where the search plays another move than the policy's own, the policy's
  plan for the rest of its decision is dropped (`Plans.drop`).
- `decision_log` keeps every `hexset.rollout.Decision`: each candidate's
  outcome in each world, the value head's read of each playout at the
  mover's turn starts `MARKS` turns on, the cost, and the choice. Reading
  the marks draws nothing and changes no game.
- `mix=<temperature>` (`rollout:...:mix=`, metadata `mix`) plays a mixed
  strategy instead of the best mean score: the candidate is drawn with
  probability proportional to `prior(a) * exp(score(a) / mix)`
  (`hexset.rollout.mixed_strategy`), the policy that maximises the expected
  score less `mix` times its KL divergence from the prior. `mix` is in the
  scores' units, a win chance: candidates within about one `mix` of the
  best are mixed in proportion to the prior, one trailing it by many is all
  but never played, and the expected shortfall from the best score is at
  most `mix * ln(1 / prior(best))`. The draws come from a stream seeded off
  the searcher's `rng` that draws nothing else, so the worlds and scores are
  the greedy search's and a game stays a function of its seeds.
  `anchor=<exponent>` (metadata `anchor`) raises the prior first:
  `prior(a) ** anchor`, so 1 anchors to the prior, 0 to every candidate
  alike (a softmax over the scores), and the bound reads the raised prior,
  renormalised over the candidates. The log
  keeps each candidate's probability (`mix`) and the candidate the best
  score would have played (`greedy`). A root decided without playouts draws
  nothing.
- `det=True` (`rollout:...:det=1`, metadata `det`) seeds each searched
  root from what the deciding seat can see instead of the searcher's
  running `rng`: `decision_key` is a sha256 of `salt` (0 by default) and
  `hexset.game.observed_digest(game, seat)`, the canonical form of
  `observe(game, seat)` (the board, every public field, the seat's own
  hand and development cards, every other pile and the deck by size, and
  the ledger as the seat knows it; the table's turn cap, trade embargoes
  and trade mode, which a record does not keep, are left out). The worlds, their playout seeds and the
  `mix` draw come from streams on that key. The same visible position then
  searches the same worlds and plays the same move on any searcher and in
  any process, two positions that differ only in what the seat cannot see
  search alike, and a position rebuilt from a record is searched as it was.
  The searcher's own `rng` is not drawn from, so a run stays a function of
  the seeds that dealt its games. The log keeps the key (`det_key`) and the
  `salt`.
- `pool=RolloutPool(load_policy, workers)` plays the worlds out in worker
  processes beside the searching one, each loading its own copy of the
  policy; the worlds and their seeds are still drawn in the searching
  process, so the decision is the one made without the pool.
  `SharedPools` keeps one pool per file for every bot spawned from it
  (`onnxbot.spawn(..., workers=, pools=)`). A worker that dies breaks the
  pool, and later decisions play its worlds in the searching process.
- `hexset.rollout.RolloutBatch` holds playout copies that can be advanced
  to later horizons (`advance(horizon, indices)`) without replaying their
  prefix, for comparing horizons on the same copies.
- Where the native core is built and the policy reads a contract-9
  observation natively, each world's copy is held natively from the
  candidate on ([native.md](native.md#playout-copies)).

`hexset.clients.netbot.rollouts_for(checkpoint, worlds=, top=, decisive=,
horizon=, full_setup=, mix=, anchor=, det=, salt=)` builds one with the checkpoint's own trade gate
(`GatedRollouts`); the arena's `rollout:` entrant and a file's
`search=rollout` metadata do the same.

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
  not `terminations`: reward 0 under `"terminal"`, the points at the cap
  under `"relative_points"`. Random play needs
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

## Decision observations

`hexset.contract9.encode9(game, seat, contract)` encodes a position from one
seat for a network that reads every seat the same way: each seat as a token
(the perspective first, the others in turn order), the board as hexes and
vertices, a building's or road's owner as a pointer to its seat's token.
`flatten` lays it out as one float32 row (`layout`), and `observation_row`
does both. Five contracts are served (`contract9.CONTRACTS`):

| Contract | Seats | Seat token | Table | Hex |
| --- | --- | --- | --- | --- |
| `9` | any | 39 | 20 | 7 |
| `9.1` | any | 40 | 35 | 8 |
| `9d` | 2 | 22 | 20 | 8 |
| `9d.1` | 2 | 32 | 20 | 8 |
| `11d` | 2 | 35 and 27 | 3 | 17 |

`hexset.contract9.encode9(game, seat, contract="9.1")` and
`layout(players, hexes, vertices, edges, contract="9.1")` provide 40-feature
seat tokens and a 35-feature table token. Contract `9` is the default (39
and 20 features). The appended seat feature is `live`, false for a
retired slot. The appended table features are `winning_points / 10`,
`discard_limit / 10`, `friendly_robber`, `balanced_dice`, and the next roll's
probabilities for sums 2 through 12, including zeros for depleted sums.
`SEAT_COLUMNS_91` and `TABLE_COLUMNS_91` name the slices. Hex tokens append
their number token divided by 12 (0 for desert), giving eight features;
pips alone cannot distinguish two sums with different conditional odds.
Under `9.1` a seat's public points are divided by its game's victory target. Building
points on vertices retain their unit scale. Board mode is not an input:
the observation already states the dealt board. ONNX decision files select
the encoding with their `contract` metadata.

`9d` is the two-seat observation, defined for a table of two and refused at
any other size (`contract9.DUEL_CONTRACTS`). Across two seats every card flow
is public except a steal, which shows its card to the one opponent, so each
hand is exact and only the development deck is hidden. A seat token holds
what both seats see; what only the perspective knows (its development cards
by kind and which it can play, and the cards it has not seen) is on the
table token. Each hex carries the chance its number is the next roll, and
the table the seven's. `9d.1` appends each seat's production by resource
(`seat_production`: the cards a roll yields it at the dice's long-run odds,
a city twice, nothing from the robber's hex) and its bank rate by resource
over 4. `11d` is two-seat too, with the facts grouped by what they describe: my token
(35 wide) holds the public seat columns 9d.1 reads plus my development
cards, those I can play and the cards I have not seen; the opponent's
token (27 wide) holds the same public columns plus its development cards
held and their ages; the table is a board block of the turn count, the
next roll's chance of a seven and whether the dice have been thrown; and a
hex reads its number one-hot over 2-6 and 8-12 (`NUMBERS_11D`). The module
docstring of `hexset.contract9` gives every column.

A two-seat network sits at a duel dealt four seats with two retired. It
reads the table over the live seats alone (`hexset.encoding.SeatFrame`,
`seat_frame`): the frame orders the live seats from the perspective, and
`hexset.decisions.frame_for` gives a decision's frame, so its robber
victims and value rows count the live seats only.

`LaneEnv(deal_law=law)` asks `law(index)` for a
`hexset.gym.distribution.Deal(game_type, locked, board_mode)` before setup.
The engine checks the number of live seats and records the rules and
retired slots for replay. `Distribution(specification, players, seed)`
builds such a law from JSON or a mapping:

```python
from hexset.gym.distribution import Distribution

law = Distribution({
    "random": {
        "weight": 1,
        "players": [[2, 1], [3, 1], [4, 1]],
        "winning_points": [[10, 1], [15, 1]],
        "discard_limit": [[7, 1], [9, 1]],
        "friendly_robber": [[False, 1], [True, 1]],
        "balanced_dice": [[False, 1], [True, 1]],
        "board_mode": [["random", 1], ["spiral", 1]],
    },
}, players=4, seed=0)
```

Every random field lists `[value, weight]` pairs; fields are drawn
independently. `presets` is a list of fixed values for those same six
fields, each with its own `weight`. Component weights determine their
shares. `exclude` is a list of partial field mappings to hold out from the
random component, conditioning that component on the remaining formats.
Fewer live players retire a uniformly drawn subset of the dealt slots.
Draws depend only on `(seed, index)` and remain stable across worker counts.
