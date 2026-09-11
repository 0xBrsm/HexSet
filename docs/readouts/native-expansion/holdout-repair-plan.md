# Holdout continuation after a previously fatal adapter state

The first holdout stopped after 1,978 completed AB2 games. Original index
1935, seed 716000000, deterministically failed at turn 120: acting seat 0 had
15 roads, one free-road credit, no legal road location, and exactly END_TURN
as its native action. The mirror remained in road-building mode and generated
an empty offer. No shipped games had completed. No interim win totals were
consulted to choose this repair or to change the holdout.

The repair changes only catanatron/state.py: after generating an empty offer,
if road-building mode is active and HexSet has no pending legal free roads,
clear the mirror's unspendable credit and generate the normal turn actions.
It does not mutate the real game. MAIN and pre-roll cases have regression
tests; 43 relevant adapter/action tests passed.

Compatibility argument: the new branch runs only when the original mirror
had an empty playable_actions list. The original CatanatronBot._offer must
raise on that list; therefore no completed original game can have traversed
this branch at any reference decision. Everything else in src is byte-for-
byte the evaluated d3d4618 source. Old records remain immutable with their
old source fingerprint; new records identify the patched source and repair.
This is an explicitly documented compatibility exception, not general source-
hash relaxation in native_search.py.

Before continuation, replay original completed AB2 indices 0, 511, 1023 and
1535 under the patch and require identical full action traces, outcomes,
points, seats, turns and action counts. Then complete the original failing
index 1935 and all remaining fixed indices. Do not discard or replace any
game. Retain the original 2048 games/gate, candidate, seed, alpha, attempt
number and opponent configuration. Do not inspect a success verdict until
all 4096 records exist. A dedicated continuation/verdict script accepts only
the exact original identity or the exact patched identity, with all fields
other than the declared source/repair label equal. Save a hash manifest of
all original records to demonstrate that none were rewritten.

Original source SHA256:
`efaecd276604bc23c7d16a23a996cc3815064353dd2d453e52f146fc21d347b1`.
Patched source SHA256:
`8c677b8f7475ec45fe02ecd5e1ba1d9e18415956b0f594666d7da4a3de8ddafe`.
