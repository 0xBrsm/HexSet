# HexSet evaluation protocol

**HexSet is the primary engine; Catanatron is an external reference opponent.
Evaluation and fitting must use HexSet's served offer/response/counter/pick
protocol. Automatic clearing is research-only and must never be used for
evaluation, fitting, weight selection or policy adoption.**

The stock `hexset.arena` loop currently uses automatic clearing. Its engine
identity and replay checks do not make it an approved evaluation path. Do not
launch its trading-enabled `duel`, `ablate`, `weight_sweep`, `fit_duel` or
dataset-generation paths for evaluation/fitting. A served driver must first
be verified against the actual served implementation, including proposal
policy, responses, execution, offer budgets and public activity observations.
See the [protocol correction](readouts/trading-protocol-correction.md).

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

## Evaluation launch prerequisite

No stock automatic-arena command is approved here as an evaluation example.
The served driver must freeze native board/chance seeds and seating while
using the production offer/response protocol. The manifest must explicitly
identify that protocol; a generic `trading=true` field is insufficient.
Automatic clearing must be off at the game level while bot trade gates remain
on. A game-level off switch must not silently disable the bot's adaptive policy.

Verify a small recorded trace before a campaign. This applies to AB2 reference
matches too, even though AB2 declines player exchanges. Historical zero-exchange
records remain useful evidence of their recorded configurations, but do not
establish served-driver equivalence by themselves.

A [bounded served robber confirmation](readouts/served-robber-confirmation/README.md)
now provides a verified direct GameSession driver for the current embedded
server policy, with full journal/record checks and no automatic exchanges.
Its scope explicitly includes the current served move slider staying at zero;
it is not a validation of adaptive served movement. The stock automatic arena
remains prohibited for evaluation and fitting.

## Result identity and historical scope

Every new campaign must record host engine, public-information model, source
and dependency revisions, effective candidate/incumbent settings, rule set,
trading protocol, proposal policy, offer budget, per-side card cap, trading mode,
acceleration mode, lineup and seat schedule, seeds, workers,
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

1. **Verify the native served evaluation path.** Confirm automatic clearing
   never executes and offers/responses/picks match served games. Check both gates use HexSet and pass
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
   Evaluate both gates through the verified served driver on fresh seeds with fixed budgets and a
   declared selection/confirmation rule. Old selection data are not fresh
   confirmation evidence, and previous adapter rejections do not exclude a
   candidate from this reassessment.
4. **Expand only from that evidence.** Decide which earlier screens need
   rerunning after the native comparison is trustworthy. Do not replay every
   historical campaign before verifying the evaluation path, and do not infer
   optimal weights or hyperparameters from failed finite searches.

The verified speed improvements remain useful. The immediate priority is a
trustworthy native baseline before another broad parameter search.
