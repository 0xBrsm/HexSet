# Third-Party Notices

HexSet itself is licensed GPL-3.0-only (see [LICENSE](LICENSE)). This file
lists every third-party component the distribution depends on, how it
enters the distribution, and under what licence.

## Runtime dependencies

| Component | License | How it enters |
|---|---|---|
| [numpy](https://numpy.org/) | BSD-3-Clause | Hard dependency (`[project].dependencies` in `pyproject.toml`). Imported by `hexset` itself; not vendored, installed from PyPI. |
| [onnxruntime](https://onnxruntime.ai/) | MIT | Optional dependency, `server` and `clients` extras (`.[server]`, `.[clients]`). Used by `hexset.clients.onnxbot`/`botclient` to run `.onnx` checkpoints, and transitively by `hexset.server.api`/`hexset.server.web`. Not vendored, installed from PyPI. |
| [onnx](https://onnx.ai/) | Apache-2.0 | Optional dependency, `export` extra (`.[export]`). Not vendored, installed from PyPI. |
| [catanatron](https://github.com/bcollazo/catanatron) | GPL-3.0 | Optional dependency, `catanatron` extra (`.[catanatron]`). Pinned to a specific commit (`bcollazo/catanatron@d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`, see `[project.optional-dependencies]` in `pyproject.toml`) rather than a floating branch, for reproducibility. Installed directly from that git commit at install time; **never vendored** into this repository. |

## Test-only dependencies

| Component | License | How it enters |
|---|---|---|
| [pytest](https://pytest.org/) | MIT | Optional dependency, `test` extra (`.[test]`). Used only to run `tests/`; not required to run the game itself, and not imported by any shipped module. Not vendored, installed from PyPI. |

## Build backend

| Component | License | How it enters |
|---|---|---|
| [setuptools](https://github.com/pypa/setuptools) | MIT | `[build-system].requires` in `pyproject.toml`. Used only to build the distribution; not imported at runtime, not vendored. |

## Bundled frontend (`src/hexset/server/static/`)

`src/hexset/server/static/index.html` is the entire frontend: one static
HTML file containing its own inline `<style>` and inline `<script>`, served
as-is by `hexset.server.web`. It was inspected for third-party code, CDN
`<script src=…>` includes, external stylesheets, web fonts, and icons:

- No `<script src=…>` or `<link rel="stylesheet" …>` to any external or
  bundled third-party file.
- No web-font (`@font-face` / Google Fonts / similar) includes — the page's
  `font-family` stack is the system font stack
  (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`),
  which names but does not embed operating-system fonts.
- The favicon is a `data:image/svg+xml` URI authored for this project
  (a hex-grid mark), not a third-party icon.

**Nothing third-party is bundled in `src/hexset/server/static/`.** The
frontend is 100% original code under this project's own GPL-3.0-only
licence; no third-party licence text is required here.
