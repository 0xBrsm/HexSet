# catan-web

A human-vs-bot [Settlers of Catan](https://en.wikipedia.org/wiki/Catan) web demo. A dependency-free `http.server` backend, a single-file vanilla-JS frontend, and ONNX Runtime for bot inference — no PyTorch, no GPU required.

This is a standalone split of the web demo from [dev-catan](https://github.com/0xBrsm/dev-catan), which remains the repo of record for the game engine, training pipeline, and everything else. This repo carries a one-time copy of the torch-free engine modules it needs to run a game and serve the board; it does not track dev-catan's ongoing engine changes automatically.

## Running it

```
pip install -e .
python -m catan.webserver
```

Then open the printed URL. Opponents come from `model_options()` in `src/catan/webserver.py`: `search2` (a handcrafted bot, no checkpoint needed) plus one entry per `*.onnx` file found in the models directory.

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
- `docker/` — a small CPU-only image for deploying this without a GPU.
