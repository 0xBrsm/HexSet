<img src="docs/logo.svg" width="96" alt="HexSet logo">

# HexSet

HexSet is a gym for a hex-tile trading and building game: a numpy-only rules
engine with an information-set-honest public ledger and bundle trading, a
sample handcrafted bot, an adapter into the Catanatron benchmark suite, and a
UI layer — HTTP API, MCP server, and browser client — that seats bots, LLMs,
and humans at the same table.

The rules implemented here are those of the classic hex-tile trading game
published as *Settlers of Catan*. That name is used only to say what game
this plays; see [Trademarks](#trademarks) below.

## What's here

One distribution, `hexset`, ships two top-level packages from `src/`:

- **`hexset`** (`src/hexset`) — the rules engine: actions, board, trading,
  the development deck, victory conditions, a seat-balanced arena for
  measuring one bot against another, and the ledger of public knowledge a
  policy may honestly read instead of the true hidden state. No seat, human
  or bot, is ever shown what it could not legally know. Depends on nothing
  but numpy.
  - **`hexset.catanatron`** (`src/hexset/catanatron`) — an adapter that
    seats `hexset` bots as players in [Catanatron](https://github.com/bcollazo/catanatron),
    for sharded duels against Catanatron's own shipped bots.
  - **`hexset.bench`** (`src/hexset/bench`) — the duel, throughput, and
    tuning scripts the engine is measured with.
  - **`hexset.server`** (`src/hexset/server`) — the gym's server half: a
    dependency-free `http.server` HTTP API, an MCP server, a single-file
    vanilla-JS browser client, and a game journal.
  - **`hexset.clients`** (`src/hexset/clients`) — the gym's client half: a
    bot as a peer client of the API (embedded or external) and the ONNX
    Runtime model boundary. No PyTorch, no GPU required to play.
- **`heximax`** (`src/heximax`) — the sample bot, a sibling top-level
  package rather than a subpackage of `hexset`: a handcrafted
  perfect-information-Monte-Carlo player that reads its own belief through
  the ledger rather than the true state, registered as an `hexset.arena`
  entrant on import.

Training — self-play, PPO, expert iteration — is not part of this repo. It
lives in HexNet, the sibling package this gym plays exported checkpoints
from; see [Adding an opponent](#adding-an-opponent) below.

## Install

```
pip install -e .
```

Provides `hexset` (and its `hexset.bench`/`hexset.server`/`hexset.clients`/
`hexset.catanatron` subpackages) and `heximax` from one editable install.
Extras:

- `.[test]` — pytest, for this repo's own test suite.
- `.[server]` — onnxruntime, to run `hexset.server` (it embeds a bot via
  `hexset.clients`, which needs it).
- `.[clients]` — onnxruntime, to run `hexset.clients` standalone (an
  external bot process with no server of its own).
- `.[export]` — onnx + onnxruntime, for building `.onnx` checkpoints.
- `.[catanatron]` — pulls in Catanatron itself, for `hexset.catanatron` duels.

`pip install -e ".[server,catanatron,test]"` covers everything below.

## Run a duel

```
python -m hexset.bench.duel heximax search2 --games 400
```

`a`/`b` are checkpoint paths or `hexset.arena` entrant names (`heximax`,
`heximax-omni`, `search2`, `search2-offers3`, ...). Reports a Wilson interval,
not a raw win count. See `python -m hexset.bench.duel --help` for the full
flag set (workers, board/duel seeds, geometry).

To duel against Catanatron's own bots instead:

```
python -m hexset.catanatron.duel --players=DC:search2-offers3,AB:2,AB:2,AB:2 --num=400 --workers=8
```

## Run the server

```
python -m hexset.server.web
```

Or via Docker:

```
cp compose.example.yaml compose.yaml
docker compose up -d --build
```

`compose.yaml` is gitignored, so that copy is yours to edit and a `git pull`
on a deployment will never collide with it. The image only carries
`numpy`/`onnxruntime` — `src/` and `models/` are bind-mounted read-only, so a
code change is a `git pull` + `docker compose restart`, not a rebuild. It
runs unprivileged on a read-only filesystem with no Linux capabilities.

Then open the printed URL (or the mapped port, `8770` by default under
compose). Opponents come from `model_options()` in
`src/hexset/server/api.py`: `heximax` and `search2` (handcrafted, no
checkpoint needed) plus one entry per `*.onnx` file found in the models
directory.

Tests are `pip install -e ".[test,server,clients,catanatron]" && pytest`.

## Adding an opponent

Drop a `.onnx` file into `models/` (or wherever `HEXSET_UI_MODELS_DIR`
points) and it shows up in the in-game picker — no restart, no code change.
The filename's stem (minus `.onnx`) is what's shown in the dropdown.

`.onnx` files aren't built here. HexNet's `export_onnx` converts a trained
`.pt` checkpoint:

```
# from HexNet's src/, with torch + onnx + onnxruntime installed
python -m export_onnx --checkpoint runs/some-run/latest.pt --out latest.onnx
```

Copy the resulting file into this repo's `models/` directory.

### A checkpoint configures itself

How an opponent plays is declared in the `.onnx` file, not here. `export_onnx` writes ONNX `metadata_props`, and `src/hexset/server/modelmeta.py` reads them. See [`docs/bot-api.md`](docs/bot-api.md) for the complete interface — metadata plus the graph's own inputs/outputs — that any `.onnx` file, from any source, must satisfy to plug in; a checkpoint author never needs this repo's source, only that document.

| key | meaning | default |
| --- | --- | --- |
| `players` | table size the graph was traced for | required |
| `num_hexes` / `num_vertices` / `num_edges` | board-shape fingerprint, so a mismatched board fails loudly | required |
| `max_offers` | trade-offer budget the run trained under | engine's cap |
| `search` | `mcts` to search over the model's own priors; anything else plays one forward pass | none |
| `simulations` | descents per decision, when `search=mcts` | 128 |
| `wave` | leaves batched per expansion, when `search=mcts` | 16 |

So a checkpoint exported with `search=mcts` and `simulations=256` is just `mcts256.onnx` in `models/` — there is no spec grammar and no flag. `simulations` and `wave` are clamped on read (`models/` is a drop directory and a bot is built inside a request, so a file asking for ten million simulations would hang the seat rather than play it).

Inference device is **not** read from metadata — it's a property of the host, not the checkpoint, so it stays on `--device`.

## Playing without a browser

The human seat can also be driven by a script or an LLM, over either interface, as a peer to the browser rather than a replacement for it — both still go through the same `apply_human_action`/`legal_actions` path the browser does, so nothing sent this way skips validation.

- **HTTP**: the same `/api/*` endpoints the frontend calls (`GET /api/state`, `POST /api/action`, etc. — see `web.py`). `POST /api/register {"name": "..."}` names the human side, in the journal header for a fresh game and immediately in `GET /api/state`'s `player_name` for one already in progress; optional, and works before a game is dealt or mid-game.
- **MCP**: `python -m hexset.server.mcp`, run alongside an already-running `web.py` (`HEXSET_UI_BASE_URL`, default `http://127.0.0.1:8770`). It's a thin stdio client of that same HTTP API — one MCP connection is one `hexset_id` identity, same as one browser tab — exposing `register`, `models`, `new_game`, `board`, `state`, `act`, and `undo` as tools. `act` takes an index into `state()`'s `legal_actions` and settles the whole bot cascade before returning, so one tool call is one full human turn, not one click. Hand-rolled against the MCP stdio wire format rather than built on the official SDK, which pulls in a compiled dependency (`pydantic`) this project otherwise has none of.

Any number of these seats — browser, HTTP script, MCP-connected LLM, or an embedded `.onnx` bot — can sit at the same table; the server does not distinguish who or what is behind a seat beyond the interface it came in on.

## Layout

- **The engine lives in this repo, under `src/hexset/`** (`actions`, `game`, `ledger`, `board`, `mcts`, `bots`, `arena`, `tuning`, `catanatron`, `bench`, and the rest; `src/heximax`: the sample bot, a sibling top-level package rather than part of `hexset` — see [`docs/engine-divergence-2026-09-02.md`](docs/engine-divergence-2026-09-02.md) for how it was imported, with history, from the training repo, and for what this repo used to carry as its own copy before that). `hexset`, `hexset.bench`, `hexset.server`, `hexset.clients` and `heximax` are all one distribution (`hexset`) and one `pyproject.toml` now; see the CHANGELOG's "one distribution" entry for what was renamed to get there.
- `src/hexset/server/api.py` — tables, seats, join codes, seat tokens, the `/api/*` surface. `web.py` is the HTTP transport over it, `mcp.py` a stdio MCP client of the same routes, `webplay.py` the session: what a seat may see, the human-readable log, undo, and the wire encoding of an action.
- `src/hexset/server/rules.py` — the one legality authority every seat shares. `fair_legal_actions` is the honest trade sample: no seat, human or bot, is shown which specific opponents could cover an offer.
- `src/hexset/server/seating.py` — the setup snake starting at whoever created the game, and retiring a seat nobody claimed.
- `src/hexset/clients/onnxbot.py` — the entire model boundary: the record contract, action-space indexing, masking, sampling, and search all live behind it, and `spawn(path, board)` is the only entry point anything else uses. Builds its record with `hexset.onnx_record.record_from_game` directly — the torch-free split that used to block that (`docs/engine-divergence-2026-09-02.md`, R1) has landed, so this package no longer carries its own copy. `botclient.py` is the other half: a bot plays its seat as a peer client of the API, embedded or external, never as a privileged writer. Only the record contracts (`2`, `3`, `4`) are served — contract 1 was dropped 2026-09-02, see the divergence audit.
- `src/hexset/server/static/index.html` — the entire frontend: inline CSS, inline SVG icons, vanilla JS. No build step.
- `models/` — drop `.onnx` files here.
- `games/` — where every game is journalled: one JSON lines file per game, written as it is played, with nothing hidden (the dice, the deck order, every card drawn or stolen, every seat's hand after every action — see `src/hexset/server/journal.py`). On by default; `HEXSET_UI_GAMES_DIR` moves it, and setting that empty turns it off.

  These files are also what a game is resumed from. Sessions live in memory, so a restart or a long enough silence used to lose whatever was in flight; now a browser returning to a game it never finished has it replayed from its own journal instead of being dealt a new one. Pressing New Game is what ends a game short of winning it — that writes a closing line, and a closed game is never handed back. Turning journalling off turns resuming off with it.
- `docker/Dockerfile` — a small CPU-only image (deps only) for deploying this without a GPU.
- `compose.example.yaml` — copy to `compose.yaml` (gitignored) and edit. Bind-mounts `src/` and `models/` into the image rather than baking them in.

## Roadmap

A Gymnasium-style environment wrapper (`hexset.gym`) around the engine is
planned but does not exist yet.

## License

GPL-3.0-only — see [LICENSE](LICENSE). Third-party components used or
bundled by this distribution are listed in [NOTICE.md](NOTICE.md).

## Trademarks

CATAN and SETTLERS OF CATAN are trademarks of Catan GmbH and Catan Studio. This project is not affiliated with, endorsed by, or sponsored by either, and it ships no Catan artwork, text, or other content. Those names appear here only to identify which game's rules this implements — nominative use, not a claim on the marks. HexSet is the name of this software.
