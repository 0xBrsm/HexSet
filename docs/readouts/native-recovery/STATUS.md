# Recovery status

Goal: fresh holdout lower bounds above 50% versus three AB2 references and
25% versus three frozen shipped Heximax opponents, both in native HexSet.
See PLAN.md for protocol, fixed holdout rules and repeated-attempt correction.

Current source: `7c7ce2f` on `research/native-heximax-recovery`.
Initial candidates: control, prior g6000-p10, road-zero (manifest-v1.json).
Local checks: 47 ledger/view tests and four harness tests passed. Four native
control-versus-shipped games completed with per-decision ledger assertions.
Those local games are preflight only and excluded from strength evidence.

Remote immutable source: `/home/bsm/tmp/hexset-native-recovery-7c7ce2f`.
Remote outputs: `/home/bsm/tmp/hexset-native-recovery-results`.
Container: `hexset-native-recovery-initial`, ID
`30d252ddfed57a463e3ff3a9374737407bdcd1fc763fba4a893e88da64c58019`.
Image: `58c092875440`; 16 CPUs, network disabled, user 1000:1000, hash seed 0,
BLAS/OpenMP thread counts one.

The orchestrator first compares eight audited native games at one worker,
four workers and unpatched AB2 (24 preflight plays). It launches the initial
32-per-gate three-policy screen only if all repeats preserve complete action
traces, outcomes and observed public-history evidence. No later stage has
been launched yet. Inspect container logs/status and saved summaries; do not
repeat completed checkpoints or reuse local Python 3.14 preflight as remote
Python 3.12 results.

Preflight passed on Wintermute: 24 plays, all 16 repeat comparisons exact,
with native ledger knowledge observed. Initial 32/game-per-gate screen:
control 11 AB2 / 5 shipped wins; p10 11 / 9; road-zero 12 / 8.
These small exploratory counts do not meet the endpoints. The p10 self-play
signal warrants follow-up, but the reference gate remains uncertain.

Extended the same three arms to 128 games/gate, reusing all 192 checkpoints.
Active container `hexset-native-recovery-screen128`, immutable source unchanged,
outputs `/out/screen-v1`, seed 711000000, 16 workers. This adds 576 games.

Manifest-v2 preregisters two cheap follow-ons (32 games/gate each): p10-temp1
changes softmax temperature to 1; p10-searched-opening disables the fixed
opening prior and uses the existing search for setup. Both retain 600 leaves,
width 6, depth 2 and k=1. They will run only after the current pool finishes;
no additional opponent, host or information-model changes are introduced.
