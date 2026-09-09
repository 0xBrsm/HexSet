# Third-party notices

HexSet is licensed under GPL-3.0-only; see [LICENSE](LICENSE). This notice
covers direct dependencies declared in [pyproject.toml](pyproject.toml) and
the bundled browser interface. Dependencies are installed separately rather
than vendored. Their distributions include their own license notices and
may introduce additional dependencies.

## Direct dependencies

| Component | License | Use |
| --- | --- | --- |
| NumPy | BSD-3-Clause | Required engine dependency |
| ONNX Runtime | MIT | Model inference; `server`, `clients`, and `export` extras |
| ONNX | Apache-2.0 | ONNX tooling; `export` extra |
| Catanatron | GPL-3.0 | Optional game adapter; `catanatron` extra |
| Gymnasium | MIT | Single-agent environment; `gym` extra |
| PettingZoo | MIT | Multi-agent environment; `gym` extra |
| pytest | MIT | Test runner; `test` extra |
| setuptools | MIT | Build backend |

Catanatron is installed from Git commit
`d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`. The same revision is specified
in the package configuration and Dockerfile.

## Browser interface

[src/hexset/server/static/index.html](src/hexset/server/static/index.html)
contains the browser interface, including its CSS, JavaScript, and SVG
icons. It loads no third-party scripts, stylesheets, or web fonts. Its system
font stack references fonts installed on the user's device; it does not
bundle those fonts. The interface is distributed under HexSet's license.
