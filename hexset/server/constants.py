"""Names shared across process boundaries.

Nothing here may import anything else in this package: `botclient.py`'s
`HttpTransport` is a standard-library-only HTTP client, so constants it and
`web.py` both need must not pull either one's dependencies in.
"""

from __future__ import annotations

__all__ = [
    "TOKEN_HEADER",
    "RECORD_CONTRACTS",
]


# The header a seat's token travels on, between the browser (or an MCP client)
# and the API.
TOKEN_HEADER = "X-HexSet-Token"


# Which graph shape an ONNX checkpoint's `contract` metadata value names; the
# numbers are `hexset.onnx_record.CONTRACT_VERSION`'s, and only its. A number
# names exactly one graph shape: read it here, never assign it here.
#
# Contract 6 is "record in, decision out": the graph masks, normalises,
# argmaxes and un-rotates, and the caller only states the position
# (`hexset.onnx_record.record_from_game`) and reads the answer back. Every
# other value, and a file with no `contract` key at all, is refused by name at
# load.
RECORD_CONTRACTS = frozenset({"6"})
