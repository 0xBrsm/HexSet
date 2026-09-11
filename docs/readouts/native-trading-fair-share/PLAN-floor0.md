# Corrected request: zero trade floor

The user clarified that both requested 400-game matchups must use floor 0.
Rerun the same fixed 800 games, seed 717000000, indices 0–399 per matchup,
with trade_floor=0.0 on every actual bot. All other policies, game rules,
search budgets, source, board schedule and seating remain identical to the
previous run. The new profile includes road=0 and expansion=.25. All seats
have max_trades=None. Thirty workers. Preserve previous .0197-floor records
separately; they do not count toward this corrected request.

run_floor0.py records the zero floor in every identity and asserts it on each
spawned bot. Save corrected records under /out/floor0. Complete both 400-game
samples, check all four candidate seats occur 100 times per matchup, verify
trade gate calls and report actual exchanges, wins and 95% Wilson intervals.
No source policy defaults change as part of this evaluation.
