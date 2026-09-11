# Trading-condition study status

Active user goal: principled handling of moderate, heterogeneous and changing
trading conditions, not a forced adaptive-policy win. PLAN.md defines scope.

The fixed screen completed 1,344 native games across seven conditions. The
five-interior equal mixture gave N/T 104/320, midpoint/T 107/320, T/T 85/320.
No clean monotonic curve was established in 64-game cells. Full raw results
are in fixed-screen-games.tar.gz. Four prior full traces/trade histories matched.

The adaptive screen completed 2,688 games. Its five-interior ranking was
M130, A117, H110, P106, N99, T88 wins out of320 each. These screen results
selected M (fixed midpoint), A (public-activity full interpolation), and T
for fresh validation. Raw records, summary and analysis are checked in.

Fresh validation is running in container hexset-trading-validation-722,
source commit54ac458 exported to /home/bsm/tmp/hexset-trading-validation-source,
output /home/bsm/tmp/hexset-trading-conditions-results/validation. One30-worker
pool, 5,632 games. VALIDATION-PLAN.md and validate.py froze the design before
launch: six interior regimes, new probabilities, mixed opponent move profiles,
rising and falling activity, and two boundary checks. Two primary contrasts
use multiplicity-adjusted intervals. No default policy changes before results.

Working branch research/trading-conditions lives in
/data/data/com.termux/files/usr/tmp/hexset-trading-conditions. Original phone
checkout remains dirty and untouched. Prior branch research/native-trading-fair-share
holds the split screen, inconclusive fresh split confirmation, and 1,600 exact
census replays. The earlier apparent low trade rate was a terminal-turn-only
reporting bug, corrected there; floor-zero counts were about58–63/game.
