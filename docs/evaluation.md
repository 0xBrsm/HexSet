# HexSet evaluation protocol

**HexSet is the primary engine and evaluation framework. Catanatron is an
external reference opponent.** Heximax self-play, ablations, weight/search
tuning and candidate validation run in HexSet. Use `hexset.arena` and its
`hexset.bench` runners for both the incumbent and external-reference gates.

## Engine and opponent are separate choices

| Experiment | Real-game host | Opponents | Heximax information |
| --- | --- | --- | --- |
| Native candidate versus shipped Heximax | HexSet | Frozen Heximax entrants | Native public-history ledger |
| Native candidate versus AB2 reference | HexSet | `catanatron` entrants | Native public-history ledger |
| Explicit external compatibility study | Catanatron | `DC:` Heximax and/or native Catanatron bots | Stock `DC:` uses a memoryless public ledger |

The arena's `catanatron` entrant wraps Catanatron's depth-two alpha-beta player.
AB2's hypothetical search remains in Catanatron, while HexSet controls the real
game, legal actions and Heximax's public history. An opponent's implementation
does not choose the host engine.

The reverse `DC:` adapter and `hexset.catanatron.duel` exist for explicitly
labeled external compatibility studies. They are not the HexSet ablation
framework. A lineup of four `DC:heximax` players still runs in Catanatron.
"Self-play" or "vs shipped" alone is never enough to identify an experiment.

## Native reference-matchup example

Install the optional reference dependency with `pip install -e '.[catanatron]'`.
This command runs one Heximax against three AB2 players in HexSet:

```sh
python -m hexset.bench.duel heximax catanatron \
  --geometry abbb --games 8 --workers 1 --duel-seed 700000000 \
  --records runs/preflight/ab2.jsonl
```

`abbb` specifies one candidate seat and three reference seats. The arena
rotates the lineup and pairs boards. Eight games are a small wiring check,
not a strength estimate or proof that the full protocol has been validated.
For a candidate-versus-incumbent gate, register and freeze both Heximax
entrants and use the same HexSet runner and geometry. Literal identical-policy
self-play is a control, not evidence that a candidate beats an incumbent.

The native duel CLI does not currently expose `--catanatron-speedups`; this
example uses the unpatched reference search. The existing fast-mode native
host benchmarks install `catanatron_speedups('fast')` inside each worker.
Make the mode explicit and record it when adding acceleration to the native
campaign runner. Do not switch to the external-host command to obtain speedups.
See [external-reference acceleration](catanatron-speedups.md) for the measured
patches and their limits.

## Result identity and historical scope

Every new campaign must record host engine, public-information model, source
and dependency revisions, effective candidate/incumbent settings, rule set,
trading mode, acceleration mode, lineup and seat schedule, seeds, workers,
unfinished games and adapter exceptions/fallbacks. Use the arena's result
metadata and replay records; supplement fields that the current CLI does not
yet emit. This is the required experiment contract, not a claim that all its
fields are already enforced by the runner.

The stock external `DC:` adapter rebuilds a memoryless public hand ledger at
each decision. It retains own-hand information but loses accumulated public
resource history about opponents. Heximax uses that history to form beliefs,
so matching weights and search settings does not make the hosted policies
equivalent. Giving both candidate and incumbent the restricted ledger does
not establish that their ranking transfers to native HexSet.

Historical `DC:` strength results remain records of the protocol actually run.
Their selections and rejections must be revalidated before informing the native
policy. Preserve raw artifacts, source snapshots and historical verdicts;
annotate their scope instead of rewriting outcomes. Separate experimental
ledger adapters require their own information-model audit.

Pure HexSet experiments, native Catanatron AB2-versus-ValueFunction comparisons,
and validated copy/evaluator equivalence checks are outside this particular
ledger defect. Performance measurements remain specific to the measured
implementation and workload. See [recorded benchmarks](benchmarks.md) for
corrected historical interpretation.

## Recovery sequence

These are the next steps; documenting them does not mean they have passed.

1. **Verify the native evaluation path.** Check both gates use HexSet and pass
   the same native ledger/belief inputs to each Heximax entrant. Cover public
   production/spending, hidden steals/discards and new-game resets. Verify
   legal-action handling and surface adapter failures. Confirm worker count
   and optional acceleration do not silently change the intended protocol.
2. **Establish a small recorded baseline.** Freeze the shipped policy and
   external AB2 revision, use predetermined fresh seeds with complete seat
   rotations, and retain replay records and the full experiment identity.
   This stage checks wiring, reproducibility and cost; it does not select a
   better policy from a handful of games.
3. **Revalidate a bounded candidate set.** Preregister representative prior
   finalists, previously rejected candidates and a matched shipped control.
   Evaluate both gates in HexSet on fresh seeds with fixed budgets and a
   declared selection/confirmation rule. Old selection data are not fresh
   confirmation evidence, and previous adapter rejections do not exclude a
   candidate from this reassessment.
4. **Expand only from that evidence.** Decide which earlier screens need
   rerunning after the native comparison is trustworthy. Do not replay every
   historical campaign before verifying the evaluation path, and do not infer
   optimal weights or hyperparameters from failed finite searches.

The verified speed improvements remain useful. The immediate priority is a
trustworthy native baseline before another broad parameter search.
