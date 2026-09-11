# Native Heximax recovery campaign

Authorized September 11, 2026: iterate frugally until a candidate clears both
established endpoints: more than 50% against three pinned AB2 reference bots,
and more than 25% against three frozen shipped Heximax-notrade bots, supported
by fresh holdout lower confidence bounds. Both gates run in native HexSet.

Protocol: standard rules; no player trading; native public-history ledger;
independent generated boards with complete four-seat rotations, explicitly
`antithetic=False`. Candidates share board/game/entrant seed schedules within
a stage. Confidence intervals are calculated over independent game boards,
not falsely independent halves of paired boards. Unfinished games block reuse
as completed checkpoints and must be investigated. AB2 uses the pinned
implementation with explicit fast mode after trace-equivalence preflight.

Frozen incumbent: shipped main at `9ca8280`, NO_TRADE_WEIGHTS, depth 2, width 6,
600 leaves, k=1, win stance, default temperature, placement enabled. Factory
checks verify the effective candidate parameters, and a regression checks the
frozen control against the normal shipped factory. The initial manifest
recovers the prior g6000-p10 phenotype exactly and adds road-zero plus control.
Old Catanatron-hosted scores are hypothesis sources only.

Initial budget: preflight, then 32 games per policy per gate for the three
manifest-v1 policies (192 screen games). Reuse only exact source/protocol/
phenotype/seed/index-matching checkpoints. Increase screen budgets only after
reviewing completed blocks; screens remain exploratory. New candidate IDs and
immutable manifests identify later adaptations. Record every attempted arm.

A promising candidate receives a preregistered fresh 512-game confirmation
per gate, then a fixed 2,048-game holdout per gate if justified. No optional
stopping of a holdout for a success claim. Both ordinary 95% Wilson lower
bounds must clear their thresholds. If multiple final candidates reach fresh
holdouts, additionally spend familywise alpha across attempts as
`alpha_j = .05 / (j * (j + 1))`, requiring the corresponding stricter two-sided
bounds at attempt j. This prevents repeated attempts from turning chance
success into endpoint completion. A new attempt always uses fresh seeds.

Preflight seed 710000000. Initial screen seed 711000000. These are separate
from historical campaigns. Later screen, confirmation and holdout seed blocks
must be recorded before their games launch. Native runner checkpoints contain
host, information model, reference/source identity, effective policies, action
trace digest, seating, outcomes and audit evidence. Public ledger audits read
truth only to assert invariants; assertions never supply search inputs.
