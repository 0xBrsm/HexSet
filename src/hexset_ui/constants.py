"""The handful of names shared across process boundaries.

Nothing here may import anything else in this package: `mcp.py` is a
standard-library-only HTTP client that has to run on a machine with no
ONNX Runtime (see its own docstring), so a constant both it and `web.py`
need has to live somewhere neither pulls the other's dependencies in to
reach.
"""

from __future__ import annotations

# The header a seat's token travels on, between the browser (or an MCP
# client) and the API. One definition, so renaming it is one edit rather than
# three independently-drifting literals.
TOKEN_HEADER = "X-HexSet-Token"
