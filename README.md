# hexset-ui

HexSet is a human-vs-bot hex-tile trading and building game you play in a browser. A dependency-free `http.server` backend, a single-file vanilla-JS frontend, and ONNX Runtime for bot inference — no PyTorch, no GPU required.

The rules implemented here are those of the classic hex-tile trading game published as *Settlers of Catan*. That name is used only to say what game this plays; see [Trademarks](#trademarks) below.

This repo is the UI half. It carries a one-time copy of the torch-free engine modules it needs to run a game and serve the board, taken from the separate training repo that remains the repo of record for the engine, the training pipeline and everything else; it does not track that repo's ongoing engine changes automatically.

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

## Layout

- `src/hexset_ui/` — the game engine (torch-free, copied from the training repo) plus `webserver.py`/`webplay.py` (the HTTP server and session logic) and `onnxbot.py` (ONNX Runtime inference, this repo's own).
- `src/hexset_ui/web/index.html` — the entire frontend: inline CSS, inline SVG icons, vanilla JS. No build step.
- `models/` — drop `.onnx` files here.
- `games/` — where every game is journalled: one JSON lines file per game, written as it is played, with nothing hidden (the dice, the deck order, every card drawn or stolen, every seat's hand after every action — see `src/hexset_ui/journal.py`). On by default; `HEXSET_UI_GAMES_DIR` moves it, and setting that empty turns it off.

  These files are also what a game is resumed from. Sessions live in memory, so a restart or a long enough silence used to lose whatever was in flight; now a browser returning to a game it never finished has it replayed from its own journal instead of being dealt a new one. Pressing New Game is what ends a game short of winning it — that writes a closing line, and a closed game is never handed back. Turning journalling off turns resuming off with it.
- `docker/Dockerfile` — a small CPU-only image (deps only) for deploying this without a GPU.
- `compose.example.yaml` — copy to `compose.yaml` (gitignored) and edit. Bind-mounts `src/` and `models/` into the image rather than baking them in.

## License

MIT — see [LICENSE](LICENSE).

## Trademarks

CATAN and SETTLERS OF CATAN are trademarks of Catan GmbH and Catan Studio. This project is not affiliated with, endorsed by, or sponsored by either, and it ships no Catan artwork, text, or other content. Those names appear here only to identify which game's rules this implements — nominative use, not a claim on the marks. HexSet is the name of this software.
