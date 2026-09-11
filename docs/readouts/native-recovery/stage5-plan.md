# Bounded diagnostic screen

Manifest v5 adds eight specified policies. Each receives 64 games per gate
on seed 711000000 before ranking. Include existing p10-own and p10-wide2400
in this 64-game block, reusing their first 32 games. Rank on the smaller
margin above the two target rates; extend at most three arms to 256/gate
if they improve that joint margin over the 128-game p10 baseline. Treat
all selection data as exploratory. No holdout is authorized by this screen
alone: PLAN.md still requires fresh confirmation before fixed holdouts.

This is a finite diagnostic set, not an open-ended random population:
immediate-points priority, diversity, spare-card/discard incentives,
search depth, and hidden-world averaging. The 64-game block adds 1152 games
(1024 new-arm games and 128 extensions), with one 16-worker pool.
