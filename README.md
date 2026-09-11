<img src="docs/logo.svg" width="96" alt="HexSet logo">

# HexSet

HexSet implements a hex-tile trading and building game based on the rules of
*Settlers of Catan*. It includes a NumPy rules engine, heuristic bots, ONNX
model inference, a browser interface, HTTP and MCP interfaces, Gymnasium and
PettingZoo adapters, and batched environments for neural-network training and
evaluation.

The engine supports resource production, construction, development cards,
the robber, victory conditions, and player trading. Trading uses two
protocols: automatic exchanges for engine simulations and offer–response
rounds for server games. Neural-network training and checkpoint export are
maintained outside this repository in the sibling HexN project. HexSet owns
the rules, information sets, encoding, game loops, seating and board pairing,
records and replay, and trade evaluation. Training projects supply model
runtimes and learning algorithms through those interfaces.

**HexSet is the primary engine and evaluation framework. Catanatron is an
external reference opponent.** Run Heximax self-play, ablations and candidate
validation in HexSet, including matches against Catanatron's AB2 bot. See the
[evaluation protocol and recovery steps](docs/evaluation.md).

## Research workflows

- **Implement a bot:** start with the [Python bot example](examples/custom_bot.py)
  and [extension guide](docs/research.md#implement-a-bot).
- **Train a neural network:** collect batches, attach policy/value runtimes and
  evaluate checkpoints through the [training interfaces](docs/training.md).
- **Compare policies:** use the [research tools](docs/research.md) for seating,
  reproducible experiments and interpretation of results.
- **Inspect a game:** retain [replay records](docs/research.md#inspect-behavior)
  alongside aggregate results.

## Bots and Catanatron integration

- **`heximax`** is a handcrafted **expectimax/maxⁿ search bot**. Decision
  nodes maximize the acting player's score, and chance nodes average possible
  outcomes by probability. It samples opponents' holdings from public
  information (Perfect Information Monte Carlo, or PIMC), with one sampled
  world by default. It uses iterative deepening, a leaf budget and heuristic
  evaluation, and requires no trained model. `heximax-notrade` disables player
  trading and uses a separate fitted weight profile.
- **Catanatron** supplies an **external reference opponent**. The `catanatron`
  entrant runs its depth-two alpha-beta player (AB2) at a HexSet table and
  requires the optional `catanatron` extra. AB2's hypothetical search uses
  Catanatron internally; HexSet runs the real game. The reverse adapter,
  which runs Heximax inside Catanatron, is for explicitly labeled external
  compatibility experiments.

Heximax knows its own cards and estimates opponents' cards from the public
resource ledger. Its default objective converts per-seat heuristic scores to
an estimated win probability. Catanatron's adapter and search have different
information assumptions; comparisons must identify which engine hosted the
games and whether trading was enabled.

### Recorded Catanatron result

In the archived September 7, 2026 benchmark, `heximax-notrade` won **47.3%**
of 1,000 recorded four-player games against three Catanatron `AB:2` bots.
For context, an equal share of wins at a four-player table is 25%.
The games ran in Catanatron with player trading disabled and Heximax's
no-trade evaluation weights. The stock `DC:` adapter supplied a memoryless
public hand ledger, so these results do not establish the strength of Heximax
with native HexSet public history. This is a historical result, not a
measurement of the current revision. See the [benchmark record](docs/benchmarks.md) for
run settings and limitations.

## Installation

Requires Python 3.11 or later. From the repository root:

```sh
pip install -e .
```

The base installation requires only NumPy. Install extras for the interfaces
you use:

| Extra | Provides |
| --- | --- |
| `.[server]` | ONNX Runtime for embedded model opponents; Heximax needs only the base install |
| `.[clients]` | ONNX Runtime for standalone model clients |
| `.[gym]` | Gymnasium and PettingZoo environments |
| `.[catanatron]` | Catanatron integration, pinned to a specific Git commit |
| `.[export]` | ONNX and ONNX Runtime libraries; no training or export command is included |
| `.[test]` | pytest |

Extras can be combined, for example `pip install -e ".[server,gym,test]"`.

## Play in a browser

```sh
python -m hexset.server.web
```

The server opens a browser at `http://127.0.0.1:8770`. Share a game's URL to
invite other players. Each new game starts with its creator seated and the
remaining seats open. Fill them with people or bots, or close unused seats
with the picker’s `none` option. Play waits until every seat is filled or
closed. Closed seats can be reopened before the first move; the participating
seats are fixed once play starts.

The opponent picker lists `heximax`, `catanatron` when its extra is installed,
and models found in `models/`. To add a model opponent, place a compatible `.onnx` file in `models/`; the next model
listing includes it under its filename stem. See the
[ONNX model contract](docs/bot-api.md) for requirements.

The trade modal supports bank and port trades, offers to other players,
and responses to their offers. Player trades exchange 1–3 cards per side.
A bot makes at most one broadcast per turn and waits for manual seats to
answer before continuing. A seat with no resource cards passes automatically.

Games have a public spectator view that reveals all hands, development
cards, and victory points. Anyone with the game URL can access it, including
players at that table. Seat-specific responses filter hidden information,
but the public view means a server game does not enforce secrecy between
participants.

See [Server operation and client interfaces](docs/server.md) for Docker,
configuration, saved games, HTTP routes, and MCP tools.

## Evaluate bots

`heximax-adaptive` uses one policy that slides between the current best no-trade
and trading move profiles. Recent public exchange activity sets the slider;
both endpoints are exact. Its exchange evaluator remains the trading profile,
with floor zero. The fixed presets remain benchmark controls. See the
[slider implementation](docs/readouts/trading-conditions/SLIDER.md) and
[preceding validation](docs/readouts/trading-conditions/README.md).

```sh
python -m hexset.bench.duel heximax heximax-notrade --games 400 --workers 4
```

The arena runner reports win rates with Wilson confidence intervals and
balances seats. Game counts must complete seat rotations and paired boards;
for an odd number of players under antithetic pairing, use a multiple of twice
the seat count.
Use `--geometry ab` for a two-player game; the default is four-player
`aabb`. One and multiple workers use the same arena implementation.
Checkpoint entrants require a registered runtime loader; installing the
`clients` extra alone does not register one. A driver can call
`hexset.clients.netbot.register_entrants` to use its loader with the shared bot
and search implementations. See the [research tools guide](docs/research.md)
for the benchmark commands and removed legacy interfaces.

With the `catanatron` extra installed, this small wiring check seats one
Heximax against three external AB2 reference opponents **in HexSet**:

```sh
python -m hexset.bench.duel heximax-notrade catanatron \
  --geometry abbb --games 8 --workers 1 --duel-seed 700000000 \
  --records runs/preflight/ab2.jsonl
```

Eight games exercise seat rotation; they are not a strength estimate or a
completed validation of the recovery protocol. This CLI uses the reference
AB2 search without the optional fast patches. See the
[evaluation protocol](docs/evaluation.md) for the checks required before
another campaign. The separate `hexset.catanatron.duel` command hosts games
in Catanatron and must not be used as the HexSet ablation runner.

For batched policies, `hexset.bench.versus.compete_batched` evaluates multiple
games per tick using the arena's board and seat-pairing rules. It reports win
rates, descriptive Wilson intervals, and board-based intervals for win rates
and victory-point margins. See
[Training and runtime interfaces](docs/training.md).

## Training environments

`hexset.gym.LaneEnv` runs multiple games in lockstep and accepts a batch of
actions per tick. It uses the base installation, supports learner trade gates,
and can return replayable episode records. See the
[training guide](docs/training.md) for an example and runtime integration.

The two third-party environment adapters require the `gym` extra:

```sh
pip install -e ".[gym]"
```

`HexSetAEC` provides one PettingZoo agent per seat. Observations contain
`hexes`, `vertices`, `edges`, `globals`, and an `action_mask` derived from the
engine's legal actions.

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

`HexSetEnv`, registered as `HexSet-v0`, provides one learner seat and plays
the opponent seats automatically. Its default opponents are three `heximax`
bots. The learner seat is sampled at each reset unless a fixed seat is given.

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

The single-agent environment returns a flat observation by default;
`flatten=False` returns the four feature arrays as a dictionary. Its action
mask is in `info["action_mask"]`, and `action_masks()` provides a masking
hook. `info["view"]` contains the learner's `View` object.

In all three environments, player trading is separate from the action space. `HexSetAEC` supplies no
trade gates, so its agents do not trade. In `HexSetEnv`, opponent bots have
trade gates and may trade with each other; the learner has no gate and does
not participate. Bank and port trades remain available as actions in both
environments.

## Engine interfaces

`game.state(seat)` returns a `View`: the seat's own cards, public information,
and estimates of opponents' hands derived from the resource ledger.
`game.state(seat, hidden=False)` returns the true `GameState`. Code that needs
true state outside the engine should explain its purpose with a
`# true state: <reason>` comment.

For simulations with trade gates installed, the engine evaluates coverable
bundles when the game enters MAIN after a roll or robber resolution. Builds,
purchases, and bank trades do not trigger another automatic event. Both sides
must gain more than their own gate's `trade_floor`, and each side may
exchange at most three cards. Heximax's floor is `0.0197`; the network gate
uses `0.0`. There is no engine-wide default floor. The default `egalitarian` rule
selects the trade with the largest minimum gain; `nash` and `actor` are
alternative ranking rules. Evaluation repeats after each exchange until no
trade clears, a position is revisited, or a configured positive `max_trades`
limit is reached.
`max_trades=0` disables the automatic event. Server games use
the separate [trade-round protocol](docs/bot-api.md#trading).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/hexset/` | Rules, board topology, state, ledger, encoding, records, search, and arena |
| `src/hexset/bots/` | Heximax, bot protocols, random policy and shared evaluation |
| `src/hexset/bench/` | Duels, throughput measurements, record generation, and weight fitting |
| `src/hexset/catanatron/` | Adapters for running bots in either engine |
| `src/hexset/server/` | HTTP and MCP server, sessions, journals, and static browser UI |
| `src/hexset/clients/` | Runtime-independent policy, trade, and search interfaces; ONNX inference; bot clients |
| `src/hexset/gym/` | Lane, PettingZoo, and Gymnasium environments |
| `tests/` | Engine and integration tests |
| `models/` | Local ONNX opponents |

## Tests

```sh
pip install -e ".[test,server,export,catanatron,gym]"
pytest
pytest -m slow
```

The default run excludes tests marked `slow`. Run both commands to cover the
regular and slow suites, or use `pytest -m ""` to run all markers together.
Optional integration tests may skip when their dependencies are unavailable.

## License and attribution

HexSet is licensed under GPL-3.0-only. See [LICENSE](LICENSE) and
[third-party notices](NOTICE.md). Development history is in
[CHANGELOG.md](CHANGELOG.md).

CATAN and SETTLERS OF CATAN are trademarks of Catan GmbH and Catan Studio.
HexSet is not affiliated with, endorsed by, or sponsored by either company.
The names identify the game whose rules this project implements.
