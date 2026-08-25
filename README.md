# catan-web

A human-vs-bot [Settlers of Catan](https://en.wikipedia.org/wiki/Catan) web demo. A dependency-free `http.server` backend, a single-file vanilla-JS frontend, and ONNX Runtime for bot inference — no PyTorch, no GPU required.

This is a standalone split of the web demo from [dev-catan](https://github.com/0xBrsm/dev-catan), which remains the repo of record for the game engine, training pipeline, and everything else. This repo carries a one-time copy of the torch-free engine modules it needs to run a game and serve the board; it does not track dev-catan's ongoing engine changes automatically.

## Running it

Locally:
```
pip install -e .
python -m catan.webserver
```

Or via Docker:
```
cp compose.example.yaml compose.yaml
docker compose up -d --build
```
`compose.yaml` is gitignored, so that copy is yours to edit and a `git pull` on a deployment will never collide with it. The image only carries `numpy`/`onnxruntime` — `src/` and `models/` are bind-mounted read-only, so a code change is a `git pull` + `docker compose restart`, not a rebuild; only a dependency bump touches the image. It runs unprivileged on a read-only filesystem with no Linux capabilities, with a reason next to each line.

Then open the printed URL (or the mapped port, `8770` by default under compose). Opponents come from `model_options()` in `src/catan/webserver.py`: `search2` (a handcrafted bot, no checkpoint needed) plus one entry per `*.onnx` file found in the models directory.

Tests are `pip install -e ".[test]" && pytest`. Most of them play the engine; `tests/test_packaging.py` is the odd one out, building a real wheel from a clean copy of the tree to check that an installed copy still has a frontend in it — every other entry point here reads `src/` directly and would not notice a wheel that did not.

## Adding an opponent

Drop a `.onnx` file into `models/` (or wherever `CATAN_WEB_MODELS_DIR` points) and it shows up in the in-game picker — no restart, no code change. The filename's stem (minus `.onnx`) is what's shown in the dropdown.

`.onnx` files aren't built here. dev-catan's `catan.export_onnx` converts a trained `.pt` checkpoint:

```
# from dev-catan's src/, with torch + onnx + onnxruntime installed
python -m catan.export_onnx --checkpoint runs/some-run/latest.pt --out latest.onnx
```

Copy the resulting file into this repo's `models/` directory.

## Layout

- `src/catan/` — the game engine (torch-free, copied from dev-catan) plus `webserver.py`/`webplay.py` (the HTTP server and session logic) and `onnxbot.py` (ONNX Runtime inference, this repo's own).
- `src/catan/web/index.html` — the entire frontend: inline CSS, inline SVG icons, vanilla JS. No build step.
- `models/` — drop `.onnx` files here.
- `docker/Dockerfile` — a small CPU-only image (deps only) for deploying this without a GPU.
- `compose.example.yaml` — copy to `compose.yaml` (gitignored) and edit. Bind-mounts `src/` and `models/` into the image rather than baking them in.
