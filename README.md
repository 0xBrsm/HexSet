<img src="docs/logo.svg" width="96" alt="HexSet logo">

# HexSet

HexSet implements a hex-tile trading and building game based on the rules of
*Settlers of Catan*. It includes a NumPy rules engine, heuristic bots, ONNX
model inference, a browser interface, HTTP and MCP interfaces, Gymnasium and
PettingZoo adapters, and batched environments for neural-network training and
evaluation.

The engine supports resource production, construction, development cards,
the robber, victory conditions, and player trading. Trading uses two
protocols: offer–response rounds for served games, evaluation and fitting;
automatic clearing is retained for research only. Neural-network training and checkpoint export are
maintained outside this repository in the sibling HexN project. HexSet owns
the rules, information sets, encoding, game loops, seating and board pairing,
records and replay, and trade evaluation. Training projects supply model
runtimes and learning algorithms through those interfaces.

**HexSet is the primary engine and evaluation framework. Catanatron is an
external reference opponent.** Run Heximax self-play, ablations and candidate
validation in HexSet using the served offer–response protocol, including
matches against Catanatron's AB2 bot. The arena now defaults to engine-driven
rounds; served equivalence still requires checking the driver and notifications. See the
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

## Heximax

**Heximax** is HexSet's built-in search bot. It needs no trained model, uses
public resource history to estimate opponents' cards, and adapts its weights
to trading activity. Select `heximax` to play it; [testing options](docs/heximax.md)
can pin its weights to either endpoint without disabling trading.

In the historical automatic-arena benchmark against Catanatron's depth-two
alpha-beta player (**AB2**), Heximax
achieved these win rates in the **HexSet engine**:

| Matchup | Heximax wins | Win rate (95% interval) |
| --- | --- | --- |
| 1v1: Heximax vs AB2 | 590 / 800 | **73.75%** (70.6–76.7%) |
| Four players: one Heximax vs three AB2 | 450 / 800 | **56.25%** (52.8–59.6%) |

These use the default adaptive configuration, standard 10-VP rules, fresh
boards and balanced seats. Trading is enabled; AB2 declines exchanges, so
Heximax's slider remains at zero. These recorded arena results do not
validate the served evaluation driver or trading behavior. See the [current benchmark readout](docs/readouts/current-heximax-ab2/README.md)
for the exact revisions, settings and all 1,600 audited game records.

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

Use native HexSet with the **served offer–response protocol** for evaluation
and fitting. Automatic clearing is research-only. The stock arena and its
trading-enabled duel/ablation/fitting runners default to engine-driven rounds.
Before a campaign, verify their budgets, proposal policy and activity
notifications against the intended served driver.

See the [evaluation contract](docs/evaluation.md) and
[protocol correction](docs/readouts/trading-protocol-correction.md).
[Heximax configuration](docs/heximax.md) documents coefficient pins and
independent trading controls. The separate `hexset.catanatron.duel` runner
hosts games in Catanatron and is reserved for external compatibility work.

Model callbacks can use [cached world voting](docs/determinized-worlds.md).
For batched model evaluation and training environments, see
[training interfaces](docs/training.md).

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
exchange at most three cards. Heximax's floor is `0`; the network gate
uses `0.0`. There is no engine-wide default floor.

By default, `trade_mode="round"` runs the [trade-round protocol](docs/bot-api.md#trading):
the actor broadcasts an offer, opponents accept, counter or pass, and the
actor picks a response. `max_trades=1` permits one broadcast per turn;
`0` disables the engine's trade driver and `-1` allows further distinct offers.
For exhaustive clearing, use `trade_mode="auto"`: the `egalitarian` rule
selects the trade with the largest minimum gain, with `nash` and `actor` as
alternative rankings. In this mode, `max_trades` caps completed exchanges.
Reproducing the old uncapped behavior requires `trade_mode="auto"` and
`max_trades=-1`. Server games use `trade_mode="external"` and manage rounds
across requests, with limits owned by the session.

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
