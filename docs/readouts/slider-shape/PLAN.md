# Adaptive shape comparison — frozen before screen

Question: does changing only the activity-to-weight mapping improve the current
linear adaptive policy? Production source remains bcb9b92, fingerprint
`d2df60dd0abeb72f1bd0ae4222716ce1b4246ac249c64cd2d76aeb0aa611b001`.
Experimental mappings live in this runner; production is unchanged.

Candidates: square (alpha = activity * activity), linear (alpha = activity),
and sqrt (alpha = sqrt(activity)). The linear arm uses production directly.
All use the same eight-eligible-public-turn window with eight startup zeros,
exact N/M endpoints, expansion interpolation .25 to .125, T exchange evaluator
with zero expansion and floor zero, depth2/width6/600 leaves/k1, standard win
temperature, native public history and native HexSet host and opponents.
The nonlinear arms transform the existing observer's mean immediately before
production AdaptiveHeximax applies its interpolation. Raw activity is logged
separately. No change to trade observation, search, endpoints or opponent gates.

Seven conditions and opponent move profiles are identical to the preceding
slider-curve study: none q=0 (N,N,N); moderate q=.3, heavy q=.65 and full q=1
(M,M,M); mixed q=(.05,.45,.85), rising q=.1 to .75 and falling q=.75 to .1 at
global turn40 (N,M,T). These probabilities privately gate participation before
the normal positive-gain decision. They are never visible to the focal policy.
Each turn has separate stable private acting/responding draws. All seats permit
trades. Across conditions some opponent move profiles differ, so the experiment
is a robustness comparison, not an isolated causal estimate of trade frequency.

Screen: 64 independent matched boards/candidate/condition, all seats balanced,
antithetic=False: 1,344 games. Seeds726000000 + 10000*condition index. Select the
nonlinear candidate with more total wins; ties prefer square. It advances only
if its equal-condition paired difference from linear is at least +.01 and its
one-sided80% normal lower bound exceeds zero. These are futility/selection rules,
not a confirmatory claim; if the selected candidate fails, stop and retain linear. There
is no further shape search or expansion of the screen based on its outcomes.

Before screening, ten behavior controls: two no-trade boards for all three
curves (six games), and two fully participating boards comparing an experimental
exponent-one observer with the unmodified production linear arm (four games).
Require exact whole-game action digests, winners, seats, points and trade traces.
Seeds726900000. Synthetic observations check raw/mapped means at all nine
possible window fractions, endpoints, expiration and opportunities. All 64
screen no-trade pairs must also match full action digests across the curves.

Fresh confirmation, only for the selected qualifying challenger: same seven
conditions with seeds727000000 + 10000*condition index, versus linear. One
primary contrast is the equally weighted mean of within-condition paired win
indicator differences. Require its one-sided95% normal lower bound > +.01:
evidence of more than one percentage point aggregate improvement. Selection is
independent of fresh outcomes; only one challenger is tested, so no multiplicity
adjustment is needed for this one confirmatory test. Per-condition two-sided95%
intervals are descriptive; an aggregate pass does not certify every condition.
No claim of universal or optimal shape follows either outcome.

Choose confirmation n per policy per condition from selected screen variance:
round up to a multiple of64 of
  (z(.95)+z(.8))^2 * mean(var(challenger-linear)) / (7 * (.03-.01)^2).
Minimum256, maximum768. This targets80% power for a true +3pp gain against the
+1pp margin, not power for a true +1pp gain. Report cap-limited power if relevant.
Complete the fixed sample before inference; no interim outcome inspection,
optional extension, margin changes or confirmation-driven tuning. If it fails,
retain linear and report uncertainty. A pass warrants considering a production
change, subject to the reported condition results; this study itself is not a
silent change to the production preset.

Use one30-worker/30-CPU pool, fixed Python3.12.14 image and NumPy2.5.2,
PYTHONHASHSEED0 and BLAS/OpenMP threads1; report contention. Preserve raw records,
source/runner/plan hashes, correct candidate seat (Outcome.seating[0]), full-game
trade census, public events and first-focal-decision-per-turn activity/alpha.
Independently audit complete records, mappings and paired inference. Diagnostics
include mean raw activity and alpha by curve/condition plus response over game
phases; trajectories are descriptive and may differ because decisions affect
later trading opportunities. No Catanatron host or opponent is involved.
