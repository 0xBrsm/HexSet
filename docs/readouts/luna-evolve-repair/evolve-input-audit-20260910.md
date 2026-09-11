# Evolve input and legacy phenotype audit

Date: 2026-09-10

This is a local read-only audit. No evolve package files, archived raw records,
or checkpoint data were changed; no campaign was launched.

## Legacy binding

The immutable metadata sidecar
[`legacy-effective-phenotypes.json`](legacy-effective-phenotypes.json) binds all
twelve pristine generation-zero candidates to both their declared weights and
their effective weights. The effective scarcity is the baseline no-trade value
`0.07039861111111112` for every candidate. It covers the pristine 2,880
generation-zero records and marks the 582 generation-one records as excluded
ancillary data. The planned missing 1,440 generation-zero confirmations must
carry this same sidecar binding.

## Exact controller branches

Checkpoint loading validates only the top-level source, protocol, and config,
then retains the parsed state:

```python
if a.checkpoint.exists():
    old = json.loads(a.checkpoint.read_text())
    config = old.get("config", {})
    expected = {"count": a.count, "games": a.games,
                "total_games": a.total_games, "promote_top": a.promote_top,
                "generations": a.generations, "seed": a.seed}
    if (old.get("source_hash") != a.source_hash or old.get("protocol") != 1
            or any(config.get(k) != v for k, v in expected.items())):
        p.error("checkpoint source/protocol/config mismatch; refusing resume")
    state = old
```

For a new generation, the population is selected by this branch:

```python
completed = {g["generation"]: g for g in state.get("generations", [])
             if g.get("complete")}
for generation in range(a.generation, a.generations):
    if generation in completed:
        ...  # reuse entry["candidates"] and archived records
        continue
    previous = completed.get(generation - 1)
    if generation == a.generation and not previous and not state.get("generations"):
        candidates = proposals(generation, a.count, a.seed)
    elif previous:
        candidates = next_generation(previous["summary"], generation,
                                     a.count, a.seed)
    else:
        prior = state.get("generations", [])[-1]
        candidates = prior["next_candidates"]
```

There is no branch that loads an explicit population from
`state["candidates"]`; in a clean checkpoint that field is ignored and
`proposals()` creates the population. However, the final `else` deliberately
consumes `prior["next_candidates"]`. A data-only input checkpoint can therefore
carry an explicit population by containing an incomplete predecessor record at
`generation=N-1` with those candidates, then invoking the unchanged controller
at `--generation N --generations N+1`. This is viable, but the controller's
protocol-1 checks do not validate that input population, its effective weights,
or its provenance.

A strict local prevalidator can make that data-only path reviewable: validate
exact candidate IDs/vectors/search fields, bind them to a new input-manifest
hash and the runnable source hash, require an incomplete predecessor only,
require no existing raw records for generation N, and compute every expected
paired seed from N before launch. Post-run validation must require every
candidate/gate/index record, exact seed, source, phenotype, and complete count.
A new protocol number cannot be used with this unchanged controller because it
rejects `protocol != 1`; the new manifest and external prevalidator are the
boundary unless a separately reviewed controller extension is approved.

For a one-generation retest centered on the current `g3000-p06`, use a new
candidate ID and copy its exact declared vector, with its legacy effective
scarcity binding made explicit. Choose a generation offset above all existing
ranges (for example N=5000 gives discovery seeds 510,000,000+ and shipped
510,050,000+, pending a global overlap audit). This keeps it separate from the
legacy 10/20/30M families and the existing 300M/400M-series campaigns.

## Effective worker registration

The old ops controller and repaired controller both register candidates with:

```python
replace(NO_TRADE_WEIGHTS,
        **{k: v for k, v in c["weights"].items() if k != "scarce"})
```

The repaired controller invokes that helper before reused discovery and before
new confirmation jobs. Thus the planned confirmation continuation remains the
legacy baseline-scarcity phenotype. Candidate-specific scarcity belongs only in
a separately sourced coupled experiment.

## Current generation-3000 candidate is a different input

The legacy sidecar is scoped to pristine `g00-*` only. It must not be used to
label the current `g3000-p06` candidate. The certified current manifest is
`/data/data/com.termux/files/usr/tmp/luna-discovery-3000-pristine/expected-phenotypes.json`,
SHA-256 `41ec094177f72efc9dee3a52469e0bcb5353fdaa6b62f4603a9d308508d2a2f8`,
with source hash `11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb`.
Its source and seed plan is documented at
`docs/readouts/luna-search/luna-discovery-3000-PLAN.md`; the pristine raw
archive is `/data/data/com.termux/files/usr/tmp/luna-discovery-3000-pristine`
with 5,760 records.

Every current generation-3000 candidate has the same search phenotype,
`stance=win`, and `temperature=2.476644394795811`:

| ID | stance | temperature |
|---|---|---:|
| g3000-p00 | win | 2.476644394795811 |
| g3000-p01 | win | 2.476644394795811 |
| g3000-p02 | win | 2.476644394795811 |
| g3000-p03 | win | 2.476644394795811 |
| g3000-p04 | win | 2.476644394795811 |
| g3000-p05 | win | 2.476644394795811 |
| g3000-p06 | win | 2.476644394795811 |
| g3000-p07 | win | 2.476644394795811 |
| g3000-p08 | win | 2.476644394795811 |
| g3000-p09 | win | 2.476644394795811 |
| g3000-p10 | win | 2.476644394795811 |
| g3000-p11 | win | 2.476644394795811 |

The current target is therefore `g3000-p06`'s own manifest vector. Its
production/scarce pair is `3.8803981040675644 / 0.0980878409639301`; this is
not the old `g00-p06` pair `3.185349226263347 / 0.08051854988610127`. The
original controller's registration still omits candidate-specific scarcity,
so the effective retest vector should record `scarce=0.07039861111111112`
while preserving the full declared vector in the input manifest.
