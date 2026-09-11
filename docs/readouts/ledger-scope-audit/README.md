# Evaluation information-model scope audit

September 11, 2026. This is an initial impact audit of the locally retained
campaigns, not a claim that every historical run and immutable source snapshot
has been reverified. No new strength games were launched during this audit.

## Finding and consequence

The stock Catanatron `DC:` adapter constructs a fresh HexSet game for each
Heximax decision with a memoryless public hand ledger. Own-hand information
remains available, but accumulated public knowledge about other hands does not.
The original workspace's optional `_translation_ledger` hook also defaults to
`None`; the hook does not fix the stock `DC:` entrant. The main repository has
the same memoryless behavior without that optional hook.

Native HexSet games maintain a public resource ledger across actions. Heximax
uses the resulting beliefs in its search and evaluation. Equal weights, search
settings, seeds and code therefore do not make the two host protocols equivalent.

The affected experiment records remain observations of the protocol actually
run. They cannot, without revalidation, select or reject weights/hyperparameters
for the normal native HexSet public-history policy. Giving both candidates and
incumbents the same restricted ledger controls that experiment, but does not
establish that rankings transfer to the intended information model. The sign
and magnitude of the effect on each candidate remain unmeasured.

## Identified impact

| Work | Evidence inspected | Status for native HexSet strength |
| --- | --- | --- |
| Main 19-candidate weight-ablation campaign; 14,800 reported executed games | `luna-ablation/RESULT.md`, both designated road-zero holdout JSON files; each JSON records `DC:` candidate and `DC:` shipped opponents | Both gates affected; rankings and native-policy rejection conclusions provisional |
| Search depth/width/k confirmations | `searchcfg.py` constructs both gates with `DC:`; `validation/REJECTION.md` and a retained confirmation JSON | Revalidate before rejecting native search settings; bind historical source when reusing artifacts |
| Round6000 g6000-p10 confirmations | Both retained 1,000-game JSON files explicitly record `DC:` lineups; `evolve.py` uses native `play_batch` for both gates | Both gates affected; historical protocol rejection does not settle native candidate strength |
| Bank/fair-budget and other stock-DC follow-on screens | Follow-on readout and stock adapter default; campaigns have separate source identities | Treat native-policy conclusions as provisional; verify each immutable source before reuse |
| DCP public-ledger experiments | Certified oracle snapshot and campaign readouts identify a distinct opt-in partial public ledger | Separate protocol; not stock memoryless DC, and not demonstrated equivalent to the full native ledger |
| Native `hexset.bench.ablate` / `arena.compete` runs | Native harness imports `compete` and does not route through Catanatron | Outside this specific adapter defect, conditional on recorded run provenance |
| Catanatron AB2 versus ValueFunction | No Heximax/`DC:` entrant | Outside this specific adapter defect |
| Copy/evaluator acceleration trace and leaf-equivalence checks | Comparisons within a pinned implementation; no assertion of native Heximax strength | Remain evidence for the tested implementation and workload; retain original limitations |

Do not sum counts across campaign reports: some totals already include screen,
confirmation, ancillary cross-gate and repeated games. No total hours-lost
estimate or exhaustive affected-game count is asserted here.

## Evidence locations

- `src/hexset/catanatron/player.py`: default `_translation_ledger` returns None.
- `src/hexset/catanatron/state.py`: translation creates the memoryless ledger.
- `src/hexset/catanatron/ablation.py`: both gate lineups use `DC:`.
- `src/hexset/catanatron/searchcfg.py`: both gate lineups use `DC:`.
- `src/hexset/catanatron/evolve.py`: `_play_one` uses native `play_batch` and `DC:`.
- `docs/readouts/luna-ablation/results/results/holdout-vs-ab2-road-zero.json`:
  designated AB2 row 964/2048, candidate `DC:heximax-ablate-road-zero`.
- `docs/readouts/luna-ablation/results/results/holdout-vs-shipped-road-zero.json`:
  designated shipped row 537/2048, all four entrants through `DC:`.
- `docs/readouts/luna-search/round6000/round6000-confirmation-vs-ab2-620000000.json`:
  selected candidate 448/1000 with `DC:` candidate.
- `docs/readouts/luna-search/round6000/round6000-confirmation-vs-shipped-620100000.json`:
  selected candidate 305/1000 with all four entrants through `DC:`.
- `docs/readouts/luna-search/dcp-oracle/`: separate ledger source and invariant
  audit; lower-bound correctness is not full information-model equivalence.

Five related readouts now have scope-correction notes. Their before/after
hashes are in `annotations.json`. Raw JSON results, machine verdicts, source
snapshots and campaign selection history were not rewritten. Existing
independent exclusions (seeding, interrupted runs, provenance, fallbacks or
selection defects) still apply; this audit does not rehabilitate those data.

## Recovery requirements

Use native HexSet as the common real-game host for both AB2 and shipped-Heximax
gates. Verify that the same public event sequence yields the same ledger and
belief inputs for every candidate and incumbent, including production, public
spending, hidden steals/discards and reset/new-game boundaries. Record host,
information model, source hashes, rule set, effective phenotype, seeds and
adapter/fallback behavior in the result identity.

Revalidate a preregistered set of retained candidates and a matched shipped
control on fresh seeds after those checks pass. Reconsider earlier rejections
without treating old selection data as fresh confirmation. Use that result to
prioritize further reruns; do not launch a blanket replay of all historical
campaigns before validating the evaluation path.
