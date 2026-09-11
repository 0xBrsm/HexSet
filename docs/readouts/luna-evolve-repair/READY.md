# Evolve recovery repair READY

The reviewable repair is isolated at [`evolve.py`](/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py).

- Final repair SHA256: `3b39bf32966aeacf6960019c95257fbb19b16179954c59fdca5f4c5b6cba5c13`
- Ops baseline SHA256: `cc06ac8ca398e64f39dd8c41b8eac87e21c7937df8d0b7abce457dd3c244a107`
- Production controller diff scope: `src/hexset/catanatron/evolve.py` only; all common non-controller files match the ops snapshot.
- Strict recovery suite: 5 passed.
- Ops fixture: 2 passed.
- Worker identity: 1 passed.
- Spawn alias probe: native candidate alias resolved as width 6, 600 nodes, k1, `notrade`, `max_trades=0`.
- `py_compile` and CLI `--help`: passed.

## Actual archive evidence

Input archive root: `/data/data/com.termux/files/usr/tmp/luna-repair-archive-11f2ae/evolve`

- Original checkpoint SHA256: `e0c5a13a41e166c7919d60c9cac7b7ac00e2eee0ce3efa9359acf344f7a67716`
- Historical source hash: `11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb`
- Raw files: 3,462.
- Raw concatenated digest: `bbbcffd22c4df2b40feced22df3da7797c3d4013f197af54d9c01903f0be279d`
- Authoritative migration manifest: `/data/data/com.termux/files/usr/tmp/luna-repair-archive-11f2ae/migration-manifest.json`, SHA256 `06f62ae431fd9170415aea118e075e6d74508ce30ff9bf1c1b72a764b11d4bb2`.
- Repaired-controller raw-record manifest used in dry recovery: SHA256 `b7715936657bc591b5fd1adc34925f2a666431b263439532b42bc542dd0ca1c2`.

A copied archive was resumed with a mock game dispatcher. It requested exactly 1,440 confirmation jobs for `g00-p06`, `g00-p02`, `g00-p04`, and `g00-p03`, and zero discovery jobs. Four candidates had complete 300-game rows. Existing generation-01 raw files were retained under an archive record with `excluded_from_selection: true`. A second resume requested zero jobs and produced a semantically identical checkpoint.

## Production resume

Before resuming, create and review a sidecar manifest over every existing `games/**/*.json`, with this exact identity:

```json
{"schema":1,"controller":"evolve.py","source_hash":"11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb","files":{}}
```

The `files` object must contain each relative raw path and its SHA256. Store the manifest digest in the checkpoint as `raw_manifest_sha256`, and its path as `raw_manifest`. Do not alter raw game JSON.

After the repaired controller is installed in the production source snapshot, run:

```sh
export PYTHONPATH=/followon/src
python3 -m hexset.catanatron.evolve \
  --checkpoint /study/evolve/checkpoint.json \
  --source-hash 11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb \
  --count 12 --games 120 --total-games 300 \
  --promote-top 4 --generations 6 --seed 10000000 --run
```

The explicit historical source hash is required for the archived raw records. The repaired executable accepts legacy records without per-record provenance only after the immutable manifest authenticates their complete bytes; any present forged controller/source fields are rejected. New records include the repaired controller hash. Do not upload or launch the production run until root completes ops review and user authorization for that payload.

## Fresh validation handoff

`evolve_validation.main` is selector-only: it writes a selection JSON and does not execute fresh games. The executable handoff is [`evolve_recovery_pipeline.py`](/data/data/com.termux/files/home/code/HexSet/scripts/evolve_recovery_pipeline.py). It dynamically selects the strongest promoted candidate whose two rows are complete 300-game rows, verifies the legacy checkpoint source and repaired controller hashes, then invokes the existing `evolve_validation.run_fresh()` API. It cannot treat a discovery-only row as a finalist or a non-PASS fresh result as a goal pass.

Wrapper SHA256: `6e67177389f444850a233ae627d509c9ac95f5d4aa3c887aa21c62650b291e0f`.

The wrapper has ten strict tests covering promoted full-row selection, discovery/incomplete fail-closed behavior, phenotype stance/temperature binding, excluded-generation handling, six-generation-before-fresh ordering, fresh PASS and rejection terminal persistence, missing or stale phenotype/source manifests, and worker binding. Against the recovered archive it selects `g00-p06` from the actual complete rows; that ID is data-derived and is not hardcoded.

After the repaired controller is installed, run this single chain; `--run-evolution` resumes or completes the six-generation controller first, then invokes fresh confirmation and holdouts:

```sh
export PYTHONPATH=/followon/src
python3 /followon/scripts/evolve_recovery_pipeline.py \
  --checkpoint /study/evolve/checkpoint.json \
  --controller /followon/src/hexset/catanatron/evolve.py \
  --out /study/evolve-validation \
  --run-evolution
```

The wrapper binds checkpoint source `11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb` and repaired controller SHA `3b39bf32966aeacf6960019c95257fbb19b16179954c59fdca5f4c5b6cba5c13` before any fresh evaluator subprocess. `run_fresh()` then executes its registered evolution validation family: confirmation 1,024 at seed `50,000,000`, followed by cumulative holdout blocks `4,096 + 4,096 + 8,192` at gate seeds `60,000,000` and `70,000,000`, using exact binomial continuation decisions. Every terminal outcome is written atomically to `pipeline-verdict.json`: `PASS`, `NEEDS_FRESH_VALIDATION`, `REJECTED_AT_LOOK`, `UNRESOLVED_AT_CAP`, and `ERROR`. The process exits successfully only for `PASS`; a rejection or error remains durable for later continuation and cannot be mistaken for a goal pass.

The existing `post_stance.run_controller()` auto-chain is not used: its evolve subprocess spelling currently uses `--promote-top4`, while the repaired evolve CLI requires `--promote-top 4`.

## Legacy effective-phenotype binding (2026-09-10)

The immutable sidecar [`legacy-effective-phenotypes.json`](legacy-effective-phenotypes.json)
records the declared and effective phenotype for every `g00-p00` through
`g00-p11` candidate. The pristine generation-zero archive contains exactly
2,880 discovery records (1,440 per gate). The 582 generation-one records are
ancillary and excluded from selection.

The legacy controller registers every candidate with candidate-specific fields
except `scarce`; the evaluator therefore uses the baseline no-trade scarcity
`0.07039861111111112` for all twelve candidates. The sidecar binds this fact to
the legacy source hash, checkpoint, migration manifest, repaired controller,
and recovery wrapper. The planned 1,440 missing generation-zero confirmation
games must use the same baseline-scarcity effective phenotype. They are not a
coupled scarcity fit.

Any approved upload or continuation must include this sidecar and its SHA-256
`b5ef8063c36eaffd46dec47899ef815fccd593cd2374590d1320d77365001ff0`. A future
experiment that applies candidate-specific scarcity must use a separate source,
protocol, and disjoint seed families and must rerun its own discovery and
confirmation.

The exact checkpoint population-loading branches and the absence of a supported
explicit candidate-vector input are recorded in
[`evolve-input-audit-20260910.md`](evolve-input-audit-20260910.md). A clean
independent population centered on `g00-p06` therefore needs a separately
reviewed controller/protocol or CLI extension with its own candidate manifest;
using a crafted `next_candidates` checkpoint branch would not be a strict
controlled input path.

The legacy effective-phenotype sidecar is scoped to `g00-*`. It must not be
applied to the current `g3000-p06`; that candidate is bound by the certified
`expected-phenotypes.json` manifest and its own declared vector. The distinction
and complete current ID/stance/temperature table are recorded in
[`evolve-input-audit-20260910.md`](evolve-input-audit-20260910.md).

The unchanged evolve controller can technically consume an explicit population
through an incomplete predecessor's `next_candidates` field when invoked at a
new generation offset. That path requires an external strict input/post-run
validator and a new candidate manifest; top-level `state["candidates"]` is
ignored. The current target for any such future pass is the separate
`g3000-p06` vector, with its effective legacy scarcity recorded explicitly.
