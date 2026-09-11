# Recovery status

Goal: fresh holdout lower bounds above 50% versus three AB2 references and
25% versus three frozen shipped Heximax opponents, both in native HexSet.
See PLAN.md for protocol and fixed holdout rules. No confirmation or holdout
has been launched; every score below is exploratory.

Engine source: `7c7ce2f`, source SHA256
`ce4f1c02424216b525f9ce27728525216ae4872850bb64f444cda35b1ee0eddf`.
Remote source: `/home/bsm/tmp/hexset-native-recovery-7c7ce2f` (read-only).
Remote results: `/home/bsm/tmp/hexset-native-recovery-results`.
Image: `58c092875440`; 16 CPUs, network disabled, user 1000:1000,
Python 3.12, hash seed 0, BLAS/OpenMP threads one. Only one pool runs at a time.

Validation: 47 ledger/view tests, four harness tests and CI passed. Remote
preflight played eight native games three times (one worker, four workers,
and unpatched AB2): all 16 repeat comparisons matched exact action traces,
seating, outcomes and public-knowledge evidence. Ledger assertions passed.
Preflight scores are excluded from strength evidence.

Screen seed 711000000; independent boards, balanced seating. The same board
and game schedules are reused across arms; each completed checkpoint has
its own immutable identity. Initial 32/game arms were extended to 128 with
no repeated games. Counts are wins against three opponents:

| Candidate | AB2 /128 | Shipped /128 |
|---|---:|---:|
| Control | 47 | 25 |
| Prior p10 | 55 | 33 |
| Road zero | 56 | 32 |

The differences from control are encouraging but their paired 95% intervals
still include zero. Neither candidate has established either endpoint.

Bounded follow-ons (32 games per gate, same seed):

| Candidate | AB2 /32 | Shipped /32 |
|---|---:|---:|
| Prior p10 baseline | 11 | 9 |
| p10 temperature 1 | 11 | 9 |
| p10 searched opening | 10 | 6 |
| p10 width 12, 2400 leaves | 11 | 10 |
| p10 depth 3, 2400 leaves | 7 | 5 |
| p10 production 6 | 9 | 7 |
| p10 knight .32 | 11 | 7 |
| p10 own-score | 13 | 6 |

Temperature 1 changes full game traces (only 12 AB2 and 6 shipped traces
match p10), despite identical aggregate wins. Neither it nor the searched
opening warrants expansion now. Depth 3 regresses in this small screen;
wider search offers little evidence to pay for more search yet.

Manifest-v4 registers three cheap follow-ons: p10 production 6, knight .32,
and own-score stance. Each receives 32 games/gate before further allocation.
All earlier attempted configurations remain in their immutable manifests.

Containers: `hexset-native-recovery-initial`, `...-screen128`,
`...-followon32`, `...-search32`, `...-policy32` completed.
Manifest-v5 diagnostic block is running in `...-diagnostic64`; see
stage5-plan.md for the finite allocation and selection rule.


Diagnostic block completed: 1,152 additional games, 433.5 seconds on 16
workers. None of the ten arms beat the registered p10 joint target margin
(-7.03 percentage points). Best margins were -7.81 points (road-zero-vp4
and p10-spare005), so no arm receives a 256-game extension from this block.
See diagnostic64-summary.json for all arms, including depth-one regression.

Optional development value implemented in `cb6200b`; default remains zero.
Source SHA256 `2630ead823e3fc4072b836f40cbbc80b8de38a6117417145e8d37370e82d936d`.
Eight audited native default-control games exactly matched the old source's
full traces and outcomes. Focused tests and CI passed. Two manifest-v6 arms
(.15 and .30 per unused non-VP card) are screening at 64 games/gate in
`hexset-native-recovery-development64`, output root `screen-development`.
This screen uses the changed source and never reuses old-source checkpoints.


Development .15 extension completed: 108/256 AB2 (42.19%, 95% interval
36.30–48.31%) and 68/256 shipped (26.56%, 21.53–32.29%). The AB2 upper bound
is below the endpoint; this arm is stopped, without fresh confirmation.

Opening-variety screen completed: premium 0 gave 19/64 AB2 and 11/64 shipped;
premium 4 gave 23/64 and 15/64; premium 8 gave 23/64 and 12/64. None earns
extension. Frozen defaults reproduced all eight prior traces after the
opening parameter was added. The shipped premium remains 1.19.

Bounded expansion option is now screening two coefficients (.25, .50) at
64/gate, manifest v8 and source `d3d4618`, in
`hexset-native-recovery-expansion64`. It first verifies eight frozen-control
traces. Three new feature tests pass; evaluator/Heximax tests also pass.
All feature changes remain optional, default zero; no candidate is adopted.


Expansion .25 extension completed: 135/256 AB2 (52.73%, interval 46.62–58.76%)
and 91/256 shipped (35.55%, 29.94–41.58%). It is the first candidate with a
promising joint signal at 256/gate. Fresh confirmation registered at seed
712000000, 512 games/gate, unchanged source d3d4618. Advance only with at
least 269 AB2 and 144 shipped wins. No final holdout has been launched.
The probability-backup experiment is implemented/tested but deferred while
this candidate receives confirmation; it is not part of this candidate.
