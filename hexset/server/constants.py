"""Names shared across process boundaries.

Nothing here may import anything else in this package: `botclient.py`'s
`HttpTransport` is a standard-library-only HTTP client, so constants it and
`web.py` both need must not pull either one's dependencies in.
"""

from __future__ import annotations

import re

__all__ = [
    "TOKEN_HEADER",
    "KEY_HEADER",
    "PROFILE_FIELDS",
    "PROFILE_LENGTHS",
    "profile_value",
    "RECORD_CONTRACTS",
]


# The header a seat's token travels on, between the browser (or an MCP client)
# and the API.
TOKEN_HEADER = "X-HexSet-Token"

# The header an API key travels on, from a client the owner issued one to.
KEY_HEADER = "X-HexSet-Key"

# What a client may say about itself beside its id, on `POST /api/games` and
# `/api/join` (`api.parse_client`): the latest a client id sent is its profile
# in the rankings (`rankings.Rankings.seen`).
PROFILE_FIELDS = ("name", "version", "owner", "url", "description")

# Each profile field's longest, in characters: a name, version or owner
# longer is cut, a url or description longer is refused. A `description` is
# one sentence: what the client is.
PROFILE_LENGTHS = {"name": 40, "version": 40, "owner": 40, "url": 200, "description": 100}

_WEB_URL = re.compile(r"^https?://[^\s/]+\S*$", re.IGNORECASE)


def profile_value(field: str, value: object) -> str | None:
    """`value` as a profile's `field` holds it: trimmed and cut to its
    `PROFILE_LENGTHS`, `None` when nothing is left. Raises `ValueError` for
    anything but a string, a `url` that is not an http(s) address or is longer
    than its length, which cut would point elsewhere, and a `description`
    longer than its length, which cut would stop mid-sentence."""
    if not isinstance(value, str):
        raise ValueError(f"{field} is a string")
    value = value.strip()
    longest = PROFILE_LENGTHS[field]
    if field == "url" and value and (not _WEB_URL.match(value) or len(value) > longest):
        raise ValueError(f"url is an http(s) address of at most {longest} characters")
    if field == "description" and len(value) > longest:
        raise ValueError(f"description is at most {longest} characters: one sentence saying what it is")
    return value[:longest] or None


# Which graph shape an ONNX checkpoint's `contract` metadata value names; the
# numbers are `hexset.onnx_record.CONTRACT_VERSION`'s and
# `hexset.contract9.CONTRACT`'s, and only theirs. A number names exactly one
# graph shape: read it here, never assign it here.
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
#
# Contract 9: the contract-9 observation and decision candidates in, decision
# out (`hexset.clients.onnxbot.DecisionPolicy`).
#
# Every other value, and a file with no `contract` key at all, is refused by
# name at load.
RECORD_CONTRACTS = frozenset({"6", "7", "8", "8.1", "9"})
