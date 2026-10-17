"""Names shared across process boundaries.

Nothing here may import anything else in this package: `botclient.py`'s
`HttpTransport` is a standard-library-only HTTP client, so constants it and
`web.py` both need must not pull either one's dependencies in.
"""

from __future__ import annotations

__all__ = [
    "TOKEN_HEADER",
    "KEY_HEADER",
    "RECORD_CONTRACTS",
]


# The header a seat's token travels on, between the browser (or an MCP client)
# and the API.
TOKEN_HEADER = "X-HexSet-Token"

# The header an API key travels on, from a client the owner issued one to.
KEY_HEADER = "X-HexSet-Key"


# Which graph shape an ONNX checkpoint's `contract` metadata value names; the
# numbers are `hexset.onnx_record.CONTRACT_VERSION`'s, and only its. A number
# names exactly one graph shape: read it here, never assign it here.
#
# Contract 6 is "record in, decision out": the graph masks, normalises,
# argmaxes and un-rotates, and the caller only states the position
# (`hexset.onnx_record.record_from_game`) and reads the answer back. Contract 7
# is the same graph shape over a record with two more fields (`roads_to_take`,
# `to_move`), so a contract-6 graph's declared inputs are a subset of today's
# record and it is still served: the loader feeds a graph only the inputs it
# declares. Contract 8 adds eight more record fields and keeps every earlier
# one, so contracts 6 and 7 are still served beside it. Contract 8.1 adds the
# table's rules and the next roll's odds, and keeps every contract-8 field.
# Every other value, and a file with no `contract` key at all, is refused by
# name at load.
RECORD_CONTRACTS = frozenset({"6", "7", "8", "8.1"})
