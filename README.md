# HexSet

HexSet implements a hex-tile trading and building game based on the rules of
*Settlers of Catan*: a NumPy rules engine, ONNX model inference, a browser
interface, HTTP and MCP interfaces, Gymnasium and PettingZoo adapters, and
batched environments for training and evaluation.

The engine covers resource production, construction, development cards, the
robber, victory conditions and player trading. It owns the rules, information
sets, encoding, game loops, seating and board pairing, records and replay, and
trade evaluation. A training project supplies the model runtime and the
learning algorithm through those interfaces; neural-network training and
checkpoint export live in the sibling HexN project.

HexSet ships no playing bots. Its entrants are `random` (uniform over legal
actions), `retired` (an empty seat), the `network:` and `mcts:` checkpoint
kinds, which need a model runtime, and `catanatron`, Catanatron's alpha-beta
player seated at a HexSet table as an external reference (optional extra).
Every other bot is registered in process or joins over the API.

## Documentation

Start with [guide.md](https://github.com/0xBrsm/HexSet/blob/main/docs/guide.md); the rest are listed by name.

| Document | Contents |
| --- | --- |
| [api.md](https://github.com/0xBrsm/HexSet/blob/main/docs/api.md) | JSON routes, seat tokens, client identity and reclaim |
| [evaluation.md](https://github.com/0xBrsm/HexSet/blob/main/docs/evaluation.md) | Designing a comparison, the intervals the runners report, experiment records |
| [guide.md](https://github.com/0xBrsm/HexSet/blob/main/docs/guide.md) | Running games, trade rounds, implementing and registering a bot, records, bench commands, the public interface |
| [hosted.md](https://github.com/0xBrsm/HexSet/blob/main/docs/hosted.md) | A bot seat at a table hosted elsewhere: the observed game it plays forward |
| [mcp.md](https://github.com/0xBrsm/HexSet/blob/main/docs/mcp.md) | MCP tools and their semantics |
| [onnx.md](https://github.com/0xBrsm/HexSet/blob/main/docs/onnx.md) | The ONNX model contract, and checking a file against it |
| [server.md](https://github.com/0xBrsm/HexSet/blob/main/docs/server.md) | Running the server: configuration, Docker, saved games |
| [testing.md](https://github.com/0xBrsm/HexSet/blob/main/docs/testing.md) | Markers, optional extras, a complete check |
| [trading.md](https://github.com/0xBrsm/HexSet/blob/main/docs/trading.md) | Trading's three parts: the table's rules, a gate's own `TradeParams`, the shared protocol |
| [training.md](https://github.com/0xBrsm/HexSet/blob/main/docs/training.md) | Batched collection, model runtimes, PettingZoo and Gymnasium adapters |
| [worlds.md](https://github.com/0xBrsm/HexSet/blob/main/docs/worlds.md) | Cached votes over sampled worlds |

## Your own bots

A bot is anything with `choose(game) -> Action`; trading is an optional gate
(`hexset.bots.Bot`, `TradeGate`). It reaches a table in process or over the
API.

**In process.** A module registers how to build it when imported:

```python
from hexset.arena import Entrant, register_entrant_kind, register_preset

register_entrant_kind("mybot", lambda entrant, board, rng: MyBot(board, rng))
register_preset("mybot", Entrant("mybot", kind="mybot"))
```

Every command that names bots takes `--runtime <module>` (repeatable) and
imports it first, in each worker process too:

```sh
python -m hexset.bench.duel mybot random --runtime mybots --games 400
python -m hexset.server.web --runtime mybots
```

`HexSetEnv(opponents=("mybot",) * 3, runtime=("mybots",))` seats it in the
Gymnasium wrapper. `hexset.arena.load_runtime("mybots")` is the same import in
code; `register_spec` adds a parsed spelling (`mybot:depth=3`) beside a
preset. A name means one bot: registering a different one under a taken name
raises `ValueError`, HexSet's own names (`random`, `retired`, `catanatron`)
and prefixes (`network:`, `mcts:`, `catanatron:`) are refused, and
`unregister_*` frees a name ([docs/guide.md](https://github.com/0xBrsm/HexSet/blob/main/docs/guide.md#implement-a-bot)).

**Over the API.** A bot in any language joins a served table as a player
(`POST /api/join`), reads `/api/state` and acts through `/api/action`, as a
browser does ([docs/api.md](https://github.com/0xBrsm/HexSet/blob/main/docs/api.md)). `python -m
hexset.clients.botclient` is an example of such a client. A Python bot
that keeps an engine-side game in step with a table hosted elsewhere uses
`hexset.seat.Seat` ([docs/hosted.md](https://github.com/0xBrsm/HexSet/blob/main/docs/hosted.md)).

## Installation

Requires Python 3.11 or later. From the repository root:

```sh
pip install -e .
```

The base installation requires only NumPy. Install extras for the interfaces
you use:

| Extra | Provides |
| --- | --- |
| `.[server]` | ONNX Runtime for embedded model opponents |
| `.[clients]` | ONNX Runtime for standalone model clients |
| `.[gym]` | Gymnasium and PettingZoo environments |
| `.[catanatron]` | Catanatron integration, pinned to a specific Git commit |
| `.[export]` | ONNX and ONNX Runtime libraries; no training or export command is included |
| `.[browser]` | Playwright, for the browser tests; the browser itself is `playwright install chromium` |
| `.[test]` | pytest |

Extras can be combined, for example `pip install -e ".[server,gym,test]"`.

## Play in a browser

```sh
python -m hexset.server.web
```

The server listens on `http://127.0.0.1:8770` (`--host`, `--port`) and opens
that URL in a browser unless `--no-browser` is given. Share a game's URL to
invite other players. A new game seats its creator and leaves the other seats
open. Fill them with people or bots, or close a seat with the picker's `none`
option. Play starts once every seat is filled or closed. Until the first move
a seat can be closed, reopened or given a bot; after it the seats are fixed.

The opponent picker lists every preset a `--runtime` registered, `catanatron`
when its extra is installed, and every `*.onnx` file in `models/` (or
`$HEXSET_UI_MODELS_DIR`) under its filename stem. The directory is rescanned
on each listing, so a file dropped there appears without a restart; it must
meet the [ONNX model contract](https://github.com/0xBrsm/HexSet/blob/main/docs/onnx.md).

The trade modal makes bank and port trades, offers to other players, and
answers to their offers. A bot seat makes as many offers a turn as its own
gate declares and waits for manual seats to answer. A manual seat has no card
limit beyond its hand, and one holding no resource cards is passed on every
offer.

`GET /api/table/<code>` needs no seat token and returns the whole game: every
hand, development card and victory point. Anyone with the game URL can read
it, players at that table included, so a served game does not keep hands
secret between participants. Reads with a seat token filter hidden
information.

Configuration, Docker, journals and recovery are in
[docs/server.md](https://github.com/0xBrsm/HexSet/blob/main/docs/server.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `hexset/` | Rules, board topology, state, ledger, encoding, records, search, and arena |
| `hexset/bots/` | The bot and trade-gate protocols, `TradesBy`, the random policy, sampled worlds and search stances |
| `hexset/bench/` | Duels, baselines, trade censuses, throughput measurements and record generation |
| `hexset/catanatron/` | Seats a Catanatron `Player` as a bot at a HexSet table: board, state and action translation |
| `hexset/server/` | HTTP and MCP server, sessions, journals, and static browser UI |
| `hexset/clients/` | Runtime-independent policy, trade, and search interfaces; ONNX inference; bot clients |
| `hexset/gym/` | Lane, PettingZoo, and Gymnasium environments |
| `tests/` | Engine and integration tests |
| `models/` | ONNX opponents the server's picker lists; empty as shipped |
| `docs/` | The documents indexed above |

## Tests

```sh
pip install -e ".[test,server,export,catanatron,gym,browser]"
playwright install chromium
pytest
pytest -m slow
```

The default run excludes tests marked `slow`; `pytest -m ""` runs every
marker. Tests for an optional dependency skip themselves when it is absent, so
the installed extras decide what a green run covered.
[docs/testing.md](https://github.com/0xBrsm/HexSet/blob/main/docs/testing.md) maps each extra to its tests, including the
browser tests, which need a Chromium download beyond their extra.

## License and attribution

HexSet is licensed under GPL-3.0-only. See [LICENSE](https://github.com/0xBrsm/HexSet/blob/main/LICENSE). Development
history is in [CHANGELOG.md](https://github.com/0xBrsm/HexSet/blob/main/CHANGELOG.md); each release is also one commit
here, whose message is that version's entry.

Dependencies are installed from PyPI or from git, never vendored into this
repository, and each carries its own licence; the set and their extras are
declared in [pyproject.toml](https://github.com/0xBrsm/HexSet/blob/main/pyproject.toml). Catanatron is GPL-3.0. The
browser interface in
`hexset/server/static/index.html` loads no third-party scripts, stylesheets or
web fonts; its system font stack names fonts on the user's device rather than
bundling them.

CATAN and SETTLERS OF CATAN are trademarks of Catan GmbH and Catan Studio.
HexSet is not affiliated with, endorsed by, or sponsored by either company.
The names identify the game whose rules this project implements.
