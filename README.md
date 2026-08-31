# HexSet-UI

HexSet is a human-vs-bot hex-tile trading and building game you play in a browser. A dependency-free `http.server` backend, a single-file vanilla-JS frontend, and ONNX Runtime for bot inference — no PyTorch, no GPU required.

The rules implemented here are those of the classic hex-tile trading game published as *Settlers of Catan*. That name is used only to say what game this plays; see [Trademarks](#trademarks) below.

This repo is the UI half. It carries a one-time copy of the torch-free engine modules it needs to run a game and serve the board, taken from the separate training repo that remains the repo of record for the engine, the training pipeline and everything else; it does not track that repo's ongoing engine changes automatically.

What it deliberately does *not* carry is the training pipeline — no self-play collector, no tournament harness, no reward shaping. The game rules live here because you cannot play without them. Everything that knows a neural network exists lives behind `src/hexset_ui/onnxbot.py`, and the rest of the package talks to a bot through one method: `choose(game) -> Action`.

## Running it

Locally:
```
pip install -e .
python -m hexset_ui.webserver
```

Or via Docker:
```
cp compose.example.yaml compose.yaml
docker compose up -d --build
```
`compose.yaml` is gitignored, so that copy is yours to edit and a `git pull` on a deployment will never collide with it. The image only carries `numpy`/`onnxruntime` — `src/` and `models/` are bind-mounted read-only, so a code change is a `git pull` + `docker compose restart`, not a rebuild; only a dependency bump touches the image. It runs unprivileged on a read-only filesystem with no Linux capabilities, with a reason next to each line.

Then open the printed URL (or the mapped port, `8770` by default under compose). Opponents come from `model_options()` in `src/hexset_ui/webserver.py`: `search2` (a handcrafted bot, no checkpoint needed) plus one entry per `*.onnx` file found in the models directory.

Tests are `pip install -e ".[test]" && pytest`. Most of them play the engine; `tests/test_packaging.py` is the odd one out, building a real wheel from a clean copy of the tree to check that an installed copy still has a frontend in it — every other entry point here reads `src/` directly and would not notice a wheel that did not.

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

## Layout

- `src/hexset_ui/` — the game engine (torch-free, copied from the training repo) plus `webserver.py`/`webplay.py` (the HTTP server and session logic).
- `src/hexset_ui/onnxbot.py` — the entire model boundary: encoding, action-space indexing, masking, sampling, and search all live behind it, and `spawn(path, board)` is the only entry point anything else uses. `mcts.py`, `encoding.py` and `modelmeta.py` are imported by this module and nothing else.
- `src/hexset_ui/search2.py` — the handcrafted opponent, whole: its fitted evaluation and the max^n search that reads it. Needs no checkpoint, which is why an empty `models/` still gives you something to play.
- `src/hexset_ui/bots.py` — the ~60 lines the two have in common: what a `Bot` is, how to ask the engine for legal options, and how a seat reads a per-seat vector.
- `src/hexset_ui/web/index.html` — the entire frontend: inline CSS, inline SVG icons, vanilla JS. No build step.
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
