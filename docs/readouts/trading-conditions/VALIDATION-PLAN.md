# Fresh condition validation, frozen before launch

The adaptive screen's five-interior mixture selected fixed midpoint M/T
(130/320 wins) and adaptive full-profile A/T (117/320). Fixed H scored110,
adaptive P106, fixed N99 and shared T88. All are exploratory observations;
only M, A and T advance. No reselecting on validation outcomes.

Fresh regime probabilities: quiet .1, moderate .3, active .65; mixed
[.05,.45,.85]; rising .1 to .75 at turn40; falling .75 to .1 at turn40.
All three opponent move profiles are T for quiet/moderate/active, but mixed
uses N/M/T, rising N/T/M, falling M/N/T. All trade evaluators remain T and all
trade floors zero. Adaptive policy observes only the same public activity
signal and cannot inspect these settings. Falling activity and these exact
probabilities/profile mixtures were not in either screen.

Six primary regimes, equal weights, 256 independent boards per candidate per
regime, M/A/T =4,608 games. Boundary checks: none with old O move opponents,
full with T opponents, 128 games each for M/A/T/N =1,024 games. Total5,632.
Seeds 722000000+10000*regime_index in validate.py order, complete contiguous
indices, balanced candidate seats, antithetic=False. Thirty workers, one pool.
Complete every game before inspecting inference. All original profile and
observer settings stay frozen. No Catanatron host or opponents.

Two primary paired contrasts, each with a 97.5% two-sided stratified normal
interval (Bonferroni family-wise95%): M minus T and A minus M. Within a regime,
use sample variance of matched per-board win differences; combine the six
independent strata equally. Report per-regime Wilson intervals descriptively.
Do not let boundary extremes determine the primary verdict.

If M beats T but A does not beat M, prefer the validated fixed compromise and
retain adaptation as experimental, stating its uncertainty. If A beats M,
it still needs a frozen stage-only control on new data to establish that its
benefit comes from trading observations rather than a generic time-in-game
schedule. If neither contrast passes, do not adopt; use findings to identify
one bounded follow-up rather than repeat this test until it passes. Boundary
results limit scope; explicit no-trade mode remains a supported separate case.
