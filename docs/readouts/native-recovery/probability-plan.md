# Expected win utility at chance nodes

Current search averages heuristic score vectors over chance outcomes and
then applies softmax to the average. This differs from averaging each
outcome's estimated win probabilities. In particular, the large terminal
score can make a modest chance of winning look almost certain after the
nonlinear conversion. A deterministic diagnostic illustrates the mismatch:
a 50/50 mixture of [100,0,0,0] and [0,10,10,10] outranks a certain [5,0,0,0]
under the current backup, but loses under expected leaf win probability.

Test optional probability_backups=True, restricted to win stance. Convert
scores to per-seat probabilities at leaves, use one-hot actual game winners
at terminal leaves, and maximize/average those utility vectors throughout
the search. Trading and discard still use the existing scalar win valuation.
This changes search semantics and is a strength hypothesis, not a claim that
the previous policy or data did not exist. Defaults remain unchanged.

Register p10-probability and p10-exp025-probability at 64/gate, seed 711000000,
only after the current expansion extension and default-control trace replay.
At most two may extend to 256 if their joint target margins beat p10's
128-game screen. The fixed fresh confirmation and holdout rules remain.
