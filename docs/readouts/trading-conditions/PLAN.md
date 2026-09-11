# Principled play across trading conditions

User authorization: iterate overnight toward a principled approach to adjusting
to trading conditions, emphasizing the space between all-trade and no-trade.
Native HexSet, public history, floor zero, one 30-worker pool at a time.

Endpoint: a documented implementable policy with observable inputs, strong
fixed comparators, and fresh fixed-sample validation across moderate,
heterogeneous and changing conditions. Adaptive weights are a hypothesis,
not a required outcome: retain a robust fixed split if adaptation adds no
supported benefit. A recommendation must distinguish what is established
from uncertainty and preserve a fixed fallback. Do not claim optimality.

Initial screen: fixed N/T, midpoint/T and T/T. N is new no-trade including
expansion .25; T is trading including expansion 0; midpoint interpolates all
move coefficients and expansion. Gate always T. Opponents use T/T but
participate with private probabilities per turn and active/responding role:
none 0, quiet .15, moderate .4, active .75, full 1, mixed [0,.3,.9], rising
.1 until global turn 48 and .8 afterward. These are participation controls,
not forced acceptance: every executed deal still needs strictly positive
bilateral gain. Draws are stable within a turn/role and independent of cards,
offers, policy RNG and candidate identity. No policy may inspect these draws.

64 independent boards per cell, balanced candidate seats, 7 regimes x 3
profiles = 1,344 games. Seeds 720000000 + 10000*regime_index (listed order),
indices 0–63. Shared board schedules within regime, antithetic=False. Native
source 0254eb3dbe0d14bea1b2d9eb42ec18a553d38c2f9100dd3477edb62347778c3e.
Every result records source/runner hashes, settings, policies, action traces
and complete live trade-event census. No default production changes.

Moderate-population comparisons weight quiet/moderate/active/mixed/rising
equally; none/full are boundary checks, not the primary population. The
screen is exploratory and may guide one narrow follow-up. Any adaptive policy
must consume only observable trade history, avoid interpreting every rejected
trade as unwillingness, and be tested against the strongest fixed alternative.
Before fresh validation, freeze selected candidates, evaluation population,
seeds, sample size and contrasts. Do not repeatedly test until a desired win.
