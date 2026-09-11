> Evaluation-scope correction — 2026-09-11
>
> The campaign's AB2 and shipped-Heximax gates both use the stock `DC:` Catanatron adapter and its memoryless public hand ledger. Candidate rankings and rejection decisions require revalidation in native HexSet before use as conclusions about its normal public-history policy.
> Raw results and historical verdicts are preserved. See [scope audit](../ledger-scope-audit/README.md).

# Luna four-player ablation campaign

This campaign searches complete Heximax-notrade policies against three frozen
Catanatron `AB:2` opponents and, as a second gate, against three frozen shipped
Heximax-notrade copies. The latter is the incumbent self-play strength gate:
literal four-way self-play of one candidate is symmetric and cannot establish
an edge over fair share.

The screen contains the eight single-term drops from the original evaluator
ablation lineage, plus a small local ring around the shipped production,
buy-progress, robber-risk, road, and spare-card weights. Each candidate gets
64 fresh games on each gate. The four candidates with the best minimum margin
over the two thresholds receive a fresh 256-game confirmation screen; the
shipped control is excluded. One finalist then receives a fresh 2,048-game
holdout on each gate. A supported pass requires the AB2 holdout lower bound to
exceed 50% and the shipped-Heximax holdout lower bound to exceed 25%; the
screen and confirmation are selection data only.

The production preset is unchanged. The exact source snapshot, seed, worker
count, image digest, per-candidate JSON, selector ranking, and confirmation
outputs are retained with the Wintermute run.

The historical generic ablation also included `victory_point` and `scarce`. This campaign leaves `victory_point=1` as the evaluator scale anchor and keeps `scarce` tied to production, so their isolated drops would change calibration rather than test a distinct policy preference; they are recorded exclusions, not silently forgotten features.

## Completed result

The valid run completed all 19 screen candidates, four 256-game confirmation
candidates, and one finalist on two fresh 2,048-game holdout invocations
(14,800 games executed; each holdout invocation also emits the other gate, while
the verdict uses the designated fresh 2,048-game row for each gate).
The screen leader was `progress-141`; the confirmation finalist was
`road-zero`. The finalist did not meet either target: it won 964/2048 (47.1%,
Wilson 95% [44.9%, 49.2%]) against AB2, and 537/2048 (26.2%, [24.4%, 28.2%])
against shipped Heximax-notrade. The complete JSON artifacts are in `results/`;
the machine verdict is `results/holdout-verdict.json`.
