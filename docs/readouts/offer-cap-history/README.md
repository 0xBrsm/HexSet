# Historical initial-offer card-cap study

Recovered on 2026-09-11 from wintermute's HexSet-Clio checkout in response to
the user's question about a two-card cap beating three. **Cap two led
numerically, but its final interval included equal share.** No current
Heximax setting is changed by this historical readout.

| Candidate vs cap-three control | Candidate wins | Rate | Board-clustered 95% CI |
| --- | ---: | ---: | ---: |
| Cap three (calibration) | 507/1,024 | 49.51% | 47.66–51.36% |
| Cap two | 538/1,024 | 52.54% | 49.81–55.27% |
| Cap one | 535/1,024 | 52.25% | 49.55–54.94% |

Each cell used 512 boards with two rotations per board, seed 310000. These
are the archived analysis's clustered intervals, not game-IID Wilson
intervals. Its simultaneous 98.333% intervals also include 50%.

The cap restricted cards **on each side of the initial broadcast offer**.
Counteroffers retained the engine cap of three. Both candidate and control
used floor zero, minimum proposal gain zero and one broadcast per turn.
The Clio `HonestOfferPolicy` enumerated initial offers from the actor's hand
and unconstrained possible received resources, and used HexSet's served
`GameSession.open_round_for` protocol. This differs from the automatic
exchange process in the current native arena sweeps. It also predates the
current unified adaptive move policy and exchange-weight updates.

The result is a reason to consider a fresh current-policy test, not a proven
improvement or a measured speed claim. No such new cap experiment is launched
as part of the spare-card sweep.

## Provenance

Archived [analysis](direct-analysis.json) and [source receipt](source-verification.json)
were copied unchanged from:
`wintermute:/home/bsm/code/HexSet-Clio/data/heximax-cap-study310000-recovery-v3-0451-20260909/`.

The receipt identifies base commit `9b156974773b05a6848d895382c7b4f65e32ba15`
and frozen source at
`/home/bsm/code/HexSet-Clio/data/ablation-source-cap310000-v2-0451-20260909`.
The archived `clio/heximax_offer_policy.py` and preregistration
`data/analysis/heximax-cap-study-310000.json` were inspected to verify cap
semantics. Full journals and result rows remain at the original remote
location; this retrieval is not a new independent full-record audit.
