# Bounded expansion option

The road-count term and yes/no settlement-availability gate do not directly
value the quality of a future site. Test a bonus for the best legal site
reachable now (full credit) or after one additional legal road (half credit).
Use unblocked production pips/15 capped at one, multiplied by a coefficient
of .25 or .50 VP. Take a maximum, never sum incompatible alternatives.

Unlike the historical harmful reachable-production term, losing this entire
bonus when a site is occupied cannot offset the one actual VP earned by
settling it: its maximum is below one VP. It still may encourage bad roads;
this is a testable hypothesis, not an adopted improvement. The bonus is zero
without settlement supply and uses public board fields only. Search and
batch trade scores agree; source changes retain a zero default for opponents.

Manifest v8, p10-exp025 and p10-exp05, 64 games/gate on seed 711000000.
Run only after the opening screen and eight default-control trace replays.
At most two candidates may extend to 256 if their joint target margin beats
the p10 128-game screen. Fresh confirmation/holdout rules in PLAN.md apply.


Completed screen: exp025 won 32/64 AB2 and 20/64 shipped, joint margin zero;
exp05 won 25/64 and 22/64, joint margin -10.94 points. Extend only exp025 to
256/gate on the same seed, reusing all 128 records and adding 384 games.
This decision is recorded before launch. No fresh confirmation yet.
