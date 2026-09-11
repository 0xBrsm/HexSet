# DCP public-production ledger oracle

The pinned Docker run completed 30 one-game checks: 15 DCP vs AB2 and 15 DCP vs frozen Heximax no-trade. The test-only oracle asserted, at every DCP decision, typed lower bounds against native private hands and `sum(known)+unknown` against each public hand total. All 30 games passed with zero bridge fallbacks. Nonzero opponent certainty occurred in every game (42–142 decisions); maximum typed opponent certainty was 3–7 cards per game.

Executed image: `58c092875440` (Catanatron 3.3.0, commit `d3f4ad05bb78`). Container metadata and mount information are in `container-inspect.json`. `public_ledger.py` is the exact mounted source used by the successful run; its SHA256 is recorded in `source.sha256`. The newer local API reset patch is intentionally separate from this certified snapshot.

## Scope and limitations

DCL tracks public maritime trades and named build or development costs. DCP adds production only for the latest non-seven roll, using the unchanged board visible at that decision and requiring aggregate post-roll bank supply to cover all visible demand for each resource. Unsupported hidden events clear typed certainty; their payloads are never read. This is not a full historical production replay: older rolls and ambiguous shortage histories are intentionally skipped. The 30-game oracle passed on the exact `f4b09...` snapshot; subsequent local API reset improvements are separate and were not used for that certification.
