# Scope after the trade-round default changed

The archived confirmation remains a result of source `36c62009d206f5a04c1db7102a643fb125ef8c96`
and the source hash in its manifest. It compared exchange robber coefficients
+0.075 and -0.30 through production GameSession rounds with both move sliders
held at zero. It supports the selected coefficient under that configuration;
it does not establish an optimum or validate adaptive movement.

PR #151 changed current games to `trade_mode="round"` with one broadcast per
turn, and served sessions to `trade_mode="external"`. The arena therefore no
longer uses automatic clearing by default. Engine-driven rounds notify trade
observers, while the served session still does not deliver completed exchanges
to Heximax's adaptive observer. A fresh served bot consequently keeps its
zero-initialized activity. Driver equivalence must include those notifications,
not only the shared offer/response primitives.

During this PR review, all 19 original artifact checksums passed. The original
audit was rerun over the saved records using its pinned baseline checkout
`4757b28`, with output written outside the archive. All 392 games, 102,089 legal
actions and 25,429 recorded notifications passed; 272 candidate wins and the
reported confidence interval were reproduced. No new strength games were run.

The original manifest, driver, records, audit and readout remain unchanged.
References there to the "current" server describe the recorded revision.
