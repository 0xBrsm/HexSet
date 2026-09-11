# Quick served robber revalidation

User requested quickly revalidating the robber result after rejecting automatic
clearing for evaluation/fitting. Confirm only the prior finalist +0.075 against
the restored −0.30 baseline; do not rerun the grid or reselect on these results.
Both use spare_card 0.15, floor zero, stock search and current served behavior.

Drive production GameSession.submit/begin_round/_broadcast/_resolve directly.
One initial broadcast per turn, three cards per side for offers/counters,
production default_offer/default_respond/default_pick and referee candidate
enumeration. This is the embedded server's current policy, not the historical
Clio custom view-safe proposer. Automatic clearing is disabled and its search
function raises if entered. Bots retain enabled trade gates.

Important existing behavior: game.max_trades=0 holds the unpinned Heximax move
slider at zero. Automatic-off notifications contain no trades, and the served
round does not update that observer with its executed exchanges. This experiment
explicitly reproduces that production behavior on both sides; it does not fix
or validate adaptive movement. Only the exchange robber coefficient differs.

Frozen baseline source commit 36c62009d206f5a04c1db7102a643fb125ef8c96,
SHA256 2e7b8d58e7a020723e180374397f3235de213b11bd83aaf4ac799724ba802d85.
Native standard 10-VP games; two candidates and two controls, cyclic A,A,B,B
seating. Each plays individually. Fresh seed 733000000, indices 0–399, no
antithetic pairing. Equal-share baseline 50%. 30 workers, pinned image,
network disabled, one BLAS/OpenMP thread per worker.

Budget: 408 attempted games and 900 seconds cumulative evaluation wall time.
No retries, replacements, new candidates or extensions. Preflight six games:
stock/control-explicit full-trace pairs at seed 733900000 indices 0–1 (four
games), plus two candidate games at 733900100. Require exact action/chance and
served-round-note matches, zero automatic exchanges and exact independent
server-journal conversion. Every game preserves native records and journals.
Then 400 confirmation games; all scheduled games count. Exceptions fail-stop.
Promote only if the two-sided 95% Wilson lower bound exceeds 50%, independent
full-record/protocol audit passes, and at most two reserved adopted-default
trace replays match. Otherwise retain −0.30 provisionally and stop.
