# Development-card hypothesis

The shipped evaluator gives unused non-VP development cards no direct value.
Buying a knight cannot reach its played-knight or army reward within the
usual shallow horizon because newly purchased cards cannot be played yet.
This can suppress buying useful cards. This is a hypothesis, not a proven
cause of losses.

Add an optional development_value (default zero), multiplying unused non-VP
holdings. Count own VP cards exactly and opponents' VP cards only in
expectation from the unseen pool; do not inspect opponents' card identities.
The existing VP term remains unchanged. The score_many trade path receives
the identical bonus. Positive values could discourage playing a held card;
therefore both .15 and .30 are tested, with all attempted results retained.

Tests cover fresh cards, exclusion of VP cards and played cards, hidden-card
information invariance, and scalar/batch agreement. Native full-game traces
for the zero-default frozen control must match the prior source before any
strength results with the new source are interpreted. Use separate output
roots for the changed source fingerprint. Frozen opponents retain zero.

Manifest v6 registers p10-dev015 and p10-dev03: 64 games per gate initially,
seed 711000000. Expand only if the joint target margin is promising. This
candidate screen follows the current diagnostic pool, with no overlap.
