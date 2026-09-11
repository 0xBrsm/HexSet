# Trading protocol correction — September 11, 2026

> Historical correction: the arena statements below describe the source at
> the time of this correction. PR #151 subsequently made engine-driven rounds
> the default. Automatic clearing remains research-only; current driver and
> notification limits are documented in [current scope](served-robber-confirmation/CURRENT-SCOPE.md).

**Automatic clearing is research-only. Never use it for evaluation, fitting,
coefficient selection or adoption, or claims about served Heximax behavior.**
The required trading protocol is the served offer/response/counter/pick flow
in native HexSet. Native engine identity alone does not satisfy this contract.

At the correction, PR #149 was returned to draft and its exchange robber
coefficient adoption was withdrawn. The code restored main's preceding −0.30
provisionally, not because that value had passed served-protocol validation. The tested −0.225
and +0.075 remain research candidates only. Move weights are not retuned.

## Affected evidence

The following used the automatic arena for trading comparisons and cannot
support served-game evaluation or fitting conclusions:

- `native-trading-fair-share/` (including `floor0/`)
- `native-split-evaluator/`
- `trading-conditions/`, `slider-curve/`, `slider-shape/` trading-enabled cells
- `adaptive-exchange-2v2/`
- `exchange-weight-sweep/`, `exchange-robber-ceiling/`, `exchange-spare-ceiling/`
- `current-trade-behavior/`

Preserve frozen plans, sources, records, selections and audit receipts. Their
statistical calculations and replay checks describe what ran; they do not
validate the required trading protocol. Historical `eligible_for_adoption`
fields and adoption wording are superseded by this correction. The affected
readouts carry a scope notice. No new sweep is launched to replace them.

In particular, the 400-game behavior run's 38.97 exchanges/game, 39.66%
three-card-side exchanges and 62.835 turns/game are automatic-clearing
research measurements. They are not the current served Heximax baseline.
The reported roughly sevenfold contrast with human completed exchanges is
also not a served-policy comparison. Human census measurements themselves
are unchanged.

No-domestic-trading expansion results and observed zero-exchange AB2 traces
must be distinguished from trading-enabled studies. The latter remain
historical native-arena AB2 measurements; no served-driver equivalence has
been established here. They do not validate served trading, and new evals
must use the required protocol even when the opponent declines trades.
The old Clio cap study did use served initial offers, but predates current
weights and its cap-two numerical lead was inconclusive.

## Required next checkpoint

Before any further evaluation or fitting, verify the native served driver:

1. Automatic clearing is disabled in the real game; bots still have their
   trade gates enabled. These are separate switches.
2. Initial offers, recipient responses/counters, proposer choice and execution
   use the same policy and protocol as served games. Any offer budget or cap
   is explicit in the manifest; no silently substituted clearing loop.
3. Instrument actual offers, responses, executions, bundle sizes and the
   resulting public activity observations, with explicit denominators.
4. Validate a small fixed trace against the served implementation, then freeze
   the source and budget before measuring current behavior or selecting weights.

The current `hexset.arena` loop still uses automatic clearing. Documentation
now identifies that limitation; this correction does not pretend to have
implemented or validated a replacement runner. Do not launch its stock
trading-enabled duel, ablation, weight-sweep, fitting or dataset-generation
paths as evaluation/fitting work.

## Subsequent served confirmation

The user requested a quick revalidation. The
[production-served confirmation](served-robber-confirmation/README.md) subsequently
found +0.075 beat −0.30 in 272/392 fresh games (69.39%, 95% CI 64.66–73.74%).
Its direct GameSession driver passed stock/instrumented trace checks and
independent full journal/record audit. The +0.075 coefficient is restored on
that new evidence only; no old automatic-clearing conclusion is relabeled.
Current served move-slider behavior is explicitly zero on both sides and
remains a separate wiring issue. Stock automatic-arena eval/fitting remains
prohibited. The old automatic-clearing behavior baseline remains invalid for
served behavior.
