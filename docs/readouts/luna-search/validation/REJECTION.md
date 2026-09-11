> Evaluation-scope correction — 2026-09-11
>
> The search confirmation harness uses `DC:` for candidates and shipped opponents. Its memoryless ledger differs from native HexSet public history. These historical rejections apply to that adapter protocol; the search settings are not thereby rejected for native HexSet.
> Raw results and historical verdicts are preserved. See [scope audit](../../ledger-scope-audit/README.md).

# Confirmation rejection

The three fresh 1,024-game confirmations all missed the AB2 point target and were rejected before holdout inference:

- nodes2400-width12: 441/1024 AB2 (43.1%), 269/1024 shipped (26.3%)
- nodes2400-k4: 464/1024 AB2 (45.3%), 272/1024 shipped (26.6%)
- nodes600-k4: 479/1024 AB2 (46.8%), 259/1024 shipped (25.3%)

An obsolete AB2 holdout had started before the tightened rejection gate was read by the running shell. It was stopped and any partial artifact is excluded from inference.
