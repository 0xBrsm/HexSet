# Running the tests

The engine suite needs only `numpy` and `pytest`: no GPU, no browser, no
Catanatron.

```sh
pip install -e ".[test]"
pytest
```

## Markers

`pyproject.toml` sets `addopts = -m "not slow"`, so the default run skips
tests marked `slow`: full games through the Catanatron adapter, about twenty
seconds or more each.

```sh
pytest -m slow     # only the slow ones
pytest -m ''       # everything
```

## Optional extras

A test for an optional dependency skips itself when the dependency is
absent, so a bare `pytest` passes without the extras and covers less.

| Extra | Covers |
| --- | --- |
| `browser` | The page tests, `tests/server/test_page_*.py`, which drive Chromium through Playwright |
| `server`, `clients` | ONNX inference (`onnxruntime`): the model tests in `tests/clients/` and `tests/server/test_onnx_spawn.py` |
| `export` | The tests that build ONNX graphs with `onnx` |
| `catanatron` | The reference-opponent adapter, `tests/catanatron/`, at the pinned revision |
| `gym` | The PettingZoo and Gymnasium environments, `tests/gym/` |

`tests/test_packaging.py` builds a wheel offline and skips without
`setuptools`.

The page tests skip when the `playwright` package is missing, and when its
Chromium binary is (`tests/server/_page_server.launch`). The binary is
a separate download:

```sh
pip install -e ".[test,browser]"
playwright install chromium
pytest tests/server -q
```

## A complete check

This tree carries no CI configuration. A run that covers everything:

- the engine suite on Python 3.11 and 3.13, the oldest and newest supported;
- the page tests with Chromium installed, as a step whose skips count as
  failures;
- one environment with every extra installed running `pytest -m ''`.
