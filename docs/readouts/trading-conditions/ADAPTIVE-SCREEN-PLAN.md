# Observable adaptive screen

The initial 64-game grid does not identify a monotonic optimum. Across the
five interior regimes N won 104/320, midpoint 107/320, T 85/320. These are
exploratory selection observations, not a frozen-policy validation.

Test six move policies, always using T trade valuation: fixed N, midpoint,
T, and H (N structural/hand terms plus T production and correspondingly
scaled scarcity); adaptive A blends the complete N/T move profiles;
adaptive P blends only production and scaled scarcity, retaining N's useful
expansion .25 and other terms. H is P's high-production fixed control.
All adaptive values are frozen throughout each search decision.

Both adaptive policies use the same public activity estimate: a Beta(1,3)
prior plus exponentially decayed binary observations with an eight-eligible-
turn half-life. One observation per real turn with actor hand size >=2 and
some partner hand size >=2. Success means at least one exchange completed.
Repeated deals in a turn count once; ineligible turns and duplicate observations
do not add failure evidence. This measures realized exchange availability,
not private willingness. No offered gains, hidden hands, gate objects or
scenario parameters enter the estimator. Changing policy can affect observed
history, so predictive value must be validated rather than assumed causal.

All six policies receive the same 64 independent-board schedules per seven
regimes as PLAN.md, on fresh seeds 721000000 + 10000*regime_index. Total 2,688
games, 30 workers. The five interior regimes are equally weighted for primary
screen ranking; none/full remain boundary checks. Replay four initial fixed
N controls with exactly matching full actions/trades before launch. Observer
unit checks cover repeated deals, no-opportunity turns and changing conditions.
This is still exploratory. Select at most one adaptive policy and the strongest
fixed alternatives for fresh validation; do not adopt based on this screen.
