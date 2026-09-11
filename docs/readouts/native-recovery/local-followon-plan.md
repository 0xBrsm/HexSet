# Registered conditional continuation

Registered before the fresh local 64/gate screen finishes. To reduce agent
orchestration cost, run_local_followon.py performs only these bounded steps:

1. Apply expansion-local-plan.md's existing selection rule. Extend at most
   two changed arms and the matched parent to 256/gate, reusing all records.
2. A changed arm may confirm only with at least 131 AB2 and 72 shipped wins
   out of 256, and a better joint margin than the matched parent. Select one
   by largest joint margin, candidate ID breaking ties.
3. Fresh confirmation: seed 715000000, 512/gate. Advance only with at least
   269 AB2 and 144 shipped wins, as in the preceding confirmation protocol.
4. Only if qualified: fixed holdout seed 716000000, 2048/gate, attempt 1.
   This would be the campaign's first final holdout; earlier confirmation
   failure did not consume a holdout attempt. Require both ordinary 95%
   lower bounds and alpha_1=.025 adjusted lower bounds above .50/.25.

Stop automatically if any advancement rule fails. Never change a threshold,
seed, candidate, sample size or alpha after seeing its stage's results. Do
not stop a holdout early for success. Record stage transitions atomically.
No concurrent pool, no unrelated source changes, no automatic default-policy
adoption. All dependent stages use immutable source d3d4618 and manifest v10.
