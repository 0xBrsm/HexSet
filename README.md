<img src="docs/logo.svg" width="96" alt="HexSet logo">

# HexSet-UI

HexSet is a human-vs-bot hex-tile trading and building game you play in a browser. A dependency-free `http.server` backend, a single-file vanilla-JS frontend, and ONNX Runtime for bot inference — no PyTorch, no GPU required.

The rules implemented here are those of the classic hex-tile trading game published as *Settlers of Catan*. That name is used only to say what game this plays; see [Trademarks](#trademarks) below.

This repo is the UI half, and the engine's home: `engine/hexset` and `engine/heximax` carry the rules engine and its sample bot with their own history (imported from the training repo, `dev-hexset`/HexNet, which remains the repo of record for training and everything torch-dependent). `src/hexset_ui` is the gym around them — the HTTP API, the MCP server, the browser client, the journal, and the ONNX-runtime serving layer.

What it deliberately does *not* carry is the training pipeline — no self-play collector, no tournament harness, no reward shaping. The game rules live here because you cannot play without them. Everything that knows a neural network exists lives behind `src/hexset_ui/onnxbot.py`, and the rest of the package talks to a bot through one method: `choose(game) -> Action`.

## Running it

Locally — the engine lives in this repo now (`engine/`), so one install
covers `hexset`, `heximax` and `hexset_ui`:
```
pip install -e .
python -m hexset_ui.web
```

Or via Docker:
```
cp compose.example.yaml compose.yaml
docker compose up -d --build
```
`compose.yaml` is gitignored, so that copy is yours to edit and a `git pull` on a deployment will never collide with it. The image only carries `numpy`/`onnxruntime` — `src/`, `engine/` and `models/` are bind-mounted read-only, so a code change is a `git pull` + `docker compose restart`, not a rebuild; only a dependency bump touches the image. It runs unprivileged on a read-only filesystem with no Linux capabilities, with a reason next to each line.

Then open the printed URL (or the mapped port, `8770` by default under compose). Opponents come from `model_options()` in `src/hexset_ui/api.py`: `heximax` and `search2` (handcrafted, no checkpoint needed — both are `hexset.arena` presets, so the server seats the same bot the training repo duels) plus one entry per `*.onnx` file found in the models directory.

Tests are `pip install -e ".[test]" && pytest` for this package's own suite (173 passed/4 skipped), and `pytest engine/tests` for the engine's torch-free slice (609 passed/6 skipped: 534 hexset, 73 heximax, 2 for `hexset.build_info`) — see `engine/README.md`. `tests/test_packaging.py` is the odd one out in this package's own suite, building a real wheel from a clean copy of the tree to check that an installed copy still has a frontend in it — every other entry point here reads `src/` directly and would not notice a wheel that did not.

## Adding an opponent

Drop a `.onnx` file into `models/` (or wherever `HEXSET_UI_MODELS_DIR` points) and it shows up in the in-game picker — no restart, no code change. The filename's stem (minus `.onnx`) is what's shown in the dropdown.

`.onnx` files aren't built here. The training repo's `export_onnx` converts a trained `.pt` checkpoint:

```
# from the training repo's src/, with torch + onnx + onnxruntime installed
python -m export_onnx --checkpoint runs/some-run/latest.pt --out latest.onnx
```

Copy the resulting file into this repo's `models/` directory.

### A checkpoint configures itself

How an opponent plays is declared in the `.onnx` file, not here. `export_onnx` writes ONNX `metadata_props`, and `src/hexset_ui/modelmeta.py` reads them. See [`docs/bot-api.md`](docs/bot-api.md) for the complete interface — metadata plus the graph's own inputs/outputs — that any `.onnx` file, from any source, must satisfy to plug in; a checkpoint author never needs this repo's source, only that document.

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
- **MCP**: `python -m hexset_ui.mcp`, run alongside an already-running `web.py` (`HEXSET_UI_BASE_URL`, default `http://127.0.0.1:8770`). It's a thin stdio client of that same HTTP API — one MCP connection is one `hexset_id` identity, same as one browser tab — exposing `register`, `models`, `new_game`, `board`, `state`, `act`, and `undo` as tools. `act` takes an index into `state()`'s `legal_actions` and settles the whole bot cascade before returning, so one tool call is one full human turn, not one click. Hand-rolled against the MCP stdio wire format rather than built on the official SDK, which pulls in a compiled dependency (`pydantic`) this project otherwise has none of.

## Layout

- **The engine lives in this repo, under `engine/`** (`engine/hexset`: `actions`, `game`, `ledger`, `board`, `mcts`, `bots`, `arena`, `tuning`, and the rest; `engine/heximax`: the honest handcrafted baseline, a sibling package rather than part of `hexset` — see `engine/README.md` for how it was imported, with history, from the training repo). What lives under `src/` is the gym around it. See [`docs/engine-divergence-2026-09-02.md`](docs/engine-divergence-2026-09-02.md) for what this repo used to carry as its own copy before the import, and why one file still does.
- `src/hexset_ui/api.py` — tables, seats, join codes, seat tokens, the `/api/*` surface. `web.py` is the HTTP transport over it, `mcp.py` a stdio MCP client of the same routes, `webplay.py` the session: what a seat may see, the human-readable log, undo, and the wire encoding of an action.
- `src/hexset_ui/rules.py` — the one legality authority every seat shares. `fair_legal_actions` is the honest trade sample: no seat, human or bot, is shown which specific opponents could cover an offer.
- `src/hexset_ui/seating.py` — the setup snake starting at whoever created the game, and retiring a seat nobody claimed.
- `src/hexset_ui/onnxbot.py` — the entire model boundary: the record contract, action-space indexing, masking, sampling, and search all live behind it, and `spawn(path, board)` is the only entry point anything else uses. `botclient.py` is the other half: a bot plays its seat as a peer client of the API, embedded or external, never as a privileged writer. Only the record contracts (`2`, `3`, `4`) are served — contract 1 was dropped 2026-09-02, see the divergence audit.
- `src/hexset_ui/record.py` — the one engine-adjacent file that stays, and not by preference. It mirrors `hexset.onnx_record`, which cannot be imported here because it needs torch. Documented in the divergence audit.
- `src/hexset_ui/static/index.html` — the entire frontend: inline CSS, inline SVG icons, vanilla JS. No build step.
- `models/` — drop `.onnx` files here.
- `games/` — where every game is journalled: one JSON lines file per game, written as it is played, with nothing hidden (the dice, the deck order, every card drawn or stolen, every seat's hand after every action — see `src/hexset_ui/journal.py`). On by default; `HEXSET_UI_GAMES_DIR` moves it, and setting that empty turns it off.

  These files are also what a game is resumed from. Sessions live in memory, so a restart or a long enough silence used to lose whatever was in flight; now a browser returning to a game it never finished has it replayed from its own journal instead of being dealt a new one. Pressing New Game is what ends a game short of winning it — that writes a closing line, and a closed game is never handed back. Turning journalling off turns resuming off with it.
- `docker/Dockerfile` — a small CPU-only image (deps only) for deploying this without a GPU.
- `compose.example.yaml` — copy to `compose.yaml` (gitignored) and edit. Bind-mounts `src/` and `models/` into the image rather than baking them in.

## License

AGPL-3.0 — see [LICENSE](LICENSE). Running a modified copy of this
project as a network service carries the same source-disclosure obligation
as distributing it: your users are entitled to the corresponding source of
what they're playing against.

## Trademarks

CATAN and SETTLERS OF CATAN are trademarks of Catan GmbH and Catan Studio. This project is not affiliated with, endorsed by, or sponsored by either, and it ships no Catan artwork, text, or other content. Those names appear here only to identify which game's rules this implements — nominative use, not a claim on the marks. HexSet is the name of this software.
