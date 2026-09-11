# Trading-condition study status

Active user goal: principled handling of moderate, heterogeneous and changing
trading conditions, not a forced adaptive-policy win. PLAN.md defines scope.

The fixed screen completed 1,344 native games across seven conditions. The
five-interior equal mixture gave N/T 104/320, midpoint/T 107/320, T/T 85/320.
No clean monotonic curve was established in 64-game cells. Full raw results
are in fixed-screen-games.tar.gz. Four prior full traces/trade histories matched.

Adaptive screen is running in Wintermute container hexset-trading-adaptive-721,
source export /home/bsm/tmp/hexset-trading-adaptive-source, output
/home/bsm/tmp/hexset-trading-conditions-results/adaptive-screen. One pool of
30 workers, no other evaluation pool running. Fixed N/M/T/H and adaptive A/P,
7 regimes x64 =2,688 games. See ADAPTIVE-SCREEN-PLAN.md. Five observer tests
passed. Four fixed observer-control games must match previous complete traces
before the screen starts; they passed. Source policy defaults are unchanged.

Next: archive and analyze the completed adaptive screen, ranking equal-weight
quiet/moderate/active/mixed/rising results. None/full remain boundary checks.
Select at most one adaptive policy and strongest fixed alternatives; preregister
fresh fixed-sample validation, including held-out intermediate participation,
heterogeneity and a falling-activity regime. If adaptation is selected, include
a stage-only signal control if feasible to distinguish observed conditions
from generic time-in-game effects. No fresh validation has been launched yet.

Working branch research/trading-conditions lives in
/data/data/com.termux/files/usr/tmp/hexset-trading-conditions. Original phone
checkout remains dirty and untouched. Prior branch research/native-trading-fair-share
holds the split screen, inconclusive fresh split confirmation, and 1,600 exact
census replays. The earlier apparent low trade rate was a terminal-turn-only
reporting bug, corrected there; floor-zero counts were about58–63/game.
