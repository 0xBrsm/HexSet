> Evaluation-scope correction — 2026-09-11
>
> Both retained confirmation JSON files explicitly use `DC:` lineups. The stock adapter's memoryless public ledger differs from native HexSet. Preserve REJECTED_AT_CONFIRMATION as the historical protocol verdict, but do not treat it as rejection of g6000-p10 for native HexSet without revalidation.
> Raw results and historical verdicts are preserved. See [scope audit](../ledger-scope-audit/README.md).

# Round6000 confirmation final

The strict continuation for selected `g6000-p10` completed on Wintermute
using the certified source and pinned image, 30 workers, and disjoint seeds.
Both Docker containers exited 0. Runtime receipts verified the image digest,
read-only certified `/study` mount, writable output mount, `network=none`,
30-CPU cap, user `1000:1000`, one-thread numeric environment, exact command
line, gate, seed, and worker count.

| Gate | Wins | Rate | Wilson 95% lower bound | Target |
|---|---:|---:|---:|---:|
| vs AB2 | 448/1000 | 44.8% | 41.74% | >50% |
| vs shipped | 305/1000 | 30.5% | 27.73% | >25% |

The artifact identity validator passed the schema and runtime-independent
checks, then wrote an atomic `REJECTED_AT_CONFIRMATION` verdict because the
AB2 rate missed its strict target. The shipped rate cleared its point target,
but does not rescue the two-gate result. Artifact SHA-256 values are
`e4a87492248c0177dcdf820c36cbb534d58c7de0e8c2e9541fb42cd90f496e70` (AB2)
and `a84a9b26e148e571299a9df7b205e4bf15a48c33606c69a82e213b6482784320`
(shipped). The remote queue marker is
`/home/bsm/tmp/luna-round6000-confirmation/round6000-confirmation-COMPLETE.json`.

Local evidence is under `/data/data/com.termux/files/usr/tmp/luna-round6000-recovery/`:
the two artifacts, Docker inspect receipts, sidecar, artifact verdict, and
runtime verdict. The local validator is
`scripts/round6000_confirm_validate.py`; its regression tests pass 2/2.
