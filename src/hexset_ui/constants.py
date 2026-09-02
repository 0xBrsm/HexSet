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
# Contracts 2, 3 and 4 are "record in, decision out" -- the graph masks,
# normalises, argmaxes and un-rotates, and the caller only states the
# position (`record.build_record`) and reads the answer back. They differ
# from each other only in how many record fields the graph declares -- 23,
# then +4 live-offer fields, then +2 public-ledger fields -- which every
# reader here works out from `session.get_inputs()` rather than from the
# number itself.
#
# The numbers are `hexset.export_onnx._CONTRACT_VERSION`'s, and only its. PR #2
# re-stamped the 29-field record as `"2"` in this repo while dev-HexNet was
# exporting it as `"4"`, so one number named two different graphs and no real
# export could load. One number, one meaning, defined there.
#
# Contract 1 ("observation in, raw logits/give/want/value out", masked and
# softmaxed in Python against the frozen `encoding_v1` feature layout) is no
# longer served: the owner decided on 2026-09-02 that legacy checkpoints are
# not worth carrying `encoding_v1.py` for (`docs/engine-divergence-2026-09-02.md`,
# B5). A `contract=1` file, or one with no `contract` key at all, is refused
# by name at load, same as any other unknown contract.
RECORD_CONTRACTS = frozenset({"2", "3", "4"})
