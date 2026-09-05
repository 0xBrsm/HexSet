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


# Which graph shape an ONNX checkpoint's `contract` metadata value names.
# Contract 6 is "record in, decision out" -- the graph masks, normalises,
# argmaxes and un-rotates, and the caller only states the position
# (`hexset.onnx_record.record_from_game`) and reads the answer back.
#
# 2, 3 and 4 are no longer served. They are the offer protocol's contracts:
# 3 added the four live-offer fields, 4 the two public-ledger ones, and all
# three declare a `pair_mask` input and a `pair_index` output for the
# one-for-one give/want heads. Trading is now one engine event with no
# actions at all (`hexset.trading`), so those graphs describe a game this
# engine does not play -- there is no honest way to feed them, and refusing
# by name beats guessing.
#
# 5 is no longer served either: the knight two-step fix shrinks the flat
# `ActionSpace` (`PLAY_KNIGHT` dropped its operands), so a contract-5
# checkpoint's `action_mask`/`prior` are the wrong width for this engine's
# action space now (`hexset.onnx_record.CONTRACT_VERSION`'s own comment).
#
# The numbers are `hexset.export_onnx._CONTRACT_VERSION`'s, and only its. PR #2
# re-stamped the 29-field record as `"2"` in this repo while dev-HexNet was
# exporting it as `"4"`, so one number named two different graphs and no real
# export could load. One number, one meaning, defined there.
#
# Contract 1 ("observation in, raw logits/give/want/value out", masked and
# softmaxed in Python against the frozen `encoding_v1` feature layout) went
# the same way earlier: the owner decided on 2026-09-02 that legacy
# checkpoints are not worth carrying `encoding_v1.py` for
# (`docs/engine-divergence-2026-09-02.md`, B5). A file with no `contract`
# key at all is refused by name at load, same as any other unknown
# contract.
RECORD_CONTRACTS = frozenset({"6"})
