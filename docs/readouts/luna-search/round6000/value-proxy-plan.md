# Round 6000 ValueFunction calibration plan

The pinned Wintermute image was inspected read-only. Native `F` resolves to
`catanatron.players.value.ValueFunctionPlayer`, which evaluates each legal
action once with `base_fn(DEFAULT_WEIGHTS)`. Native `AB:2` also uses
`base_fn(DEFAULT_WEIGHTS)`, but expands a depth-two tree with chance handling.
The evaluator identity is therefore verified; policy or strength equivalence
is unproven.

The bounded calibration is preregistered for `g6000-p00`, `g6000-p04`,
`g6000-p05`, and `g6000-p10`, with 240 games per candidate, seeds
`610000000..610000239`, and lineup `DC:heximax-evolve-{candidate},F,F,F`.
Archived round6000 AB:2 records provide the same candidate/index comparison
series. Same-seed outcome differences are descriptive because the opponent
lineup changes; they are not causal matched-game estimates.

The plan and report code do not execute games. When authorized, the runner
must preserve one JSON record per job and include `wall_seconds`; the report
checks identity, seed, candidate-win encoding, completeness, paired outcome
deltas, win-rate differences, rank order, and per-game wall time. A useful
proxy can guide selection speed only after review of those results. Final
AB:2 and self-play gates remain mandatory, and no equivalence claim is made.

Plan artifact: `value-proxy-plan.json`.
