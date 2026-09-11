# Final status

The direct linear slider passed fresh aggregate non-inferiority confirmation;
no nonlinear challenger qualified for confirmation. See
[the curve results](../slider-curve/README.md) and
[shape results](../slider-shape/README.md). It is now the single `heximax`
policy, with numeric endpoint pins for testing. Historical notes below refer
to the frozen study and its original API and runtime locations.

# Earlier slider update

The user selected direct interpolation between the current best no-trade and
trading settings. heximax-adaptive now interpolates N to the validated M move
profile, keeping M's T exchange evaluator and floor zero. An eight-eligible-turn
public activity window replaces the old persistent-prior estimator. Both
endpoints are exact; see SLIDER.md. The preceding study below tested a different
adaptive rule and does not establish this revised slider's win rate.

# Preceding trading-condition study

Research endpoint reached. Fresh validation confirms fixed midpoint move
weights with a separate trading-profile exchange evaluator. M wins523/1536
versus T396/1536: paired +8.27pp, adjusted interval[4.93,11.61]. Adaptive A
wins520/1536; A minus M=-.20pp, interval[-2.87,2.48], so adaptation is not
adopted. See README.md, the frozen VALIDATION-PLAN.md, and validation-analysis.json.

Selectable heximax-balanced implements the confirmed fixed compromise with
floor0, expansion.125 and unchanged search settings. heximax-notrade remains
the explicit no-trade choice. Historical heximax defaults are retained for
reference. Forty full traces through the new preset/API match validation
exactly. All5,632 validation public activity traces independently reproduce.
Whole-game records, source identities, analyses and hashes are archived here.

No evaluation pool remains running. Wintermute source exports and completed
containers are retained for provenance. Working branch research/trading-conditions
is in /data/data/com.termux/files/usr/tmp/hexset-trading-conditions. Original
phone checkout remains untouched. No upstream Catanatron changes were made.
