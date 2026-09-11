# Local search around bounded expansion

The fresh 512/gate confirmation supports self-play improvement but not the
AB2 target. Probability backups did not improve the 64/gate screen (23/16
wins alone, 28/20 with expansion, versus expansion's 32/20), so defer that
semantic change rather than pay for expansion of a weaker parent comparison.

Manifest v10 tests four local changes: expansion .125 and .375, the shipped
road-zero vector plus expansion .25, and p10 with production reduced to
2.785 plus expansion .25. Add p10-exp025 as a matched reference arm.

Fresh screen seed 714000000 avoids continuing to optimize the original
64 boards. Run 64 games/gate for all five arms (640 new games), on the exact
expansion source d3d4618, output screen-expansion-local. A changed arm may
extend to 256/gate only if it improves the matched parent's joint margin
and records at least 32 AB2 and 18 shipped wins; extend at most two, ranked
by the smaller margin above .50/.25. If extending, include the parent up to
256 to measure the local change. No confirmation or holdout uses this seed.
