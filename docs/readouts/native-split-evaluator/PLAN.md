# Split move/trade evaluation screen and census correction

Authorized: test four combinations and investigate the historical trade-count
discrepancy. N is new no-trade weights plus expansion .25; T is trading weights
with expansion 0. Cross N/T move and gate profiles, versus three fixed T/T
opponents. All seats trade at floor zero. 128 independent games per cell,
seed 718000000, balanced candidate seats, antithetic=False, identical boards
across cells. Native HexSet public ledger, unchanged source from PR141.
Thirty workers. Complete 512 games before interpreting the exploratory screen.
Report paired policy contrasts against T/T; do not claim a validated winner
from the screen or automatically adopt a profile. No sliding weights tested.

A wrapper routes choose to the move bot and gains_many to the trade bot. The
same object is used on diagonal cells to preserve ordinary policy exactly.
Separate gates clear their local caches before each live batch; they never
run search or inspect opponents' hidden cards. Both configurations are explicit
in every record. All search settings, opening prior and trade mechanism stay
fixed. Eight previous games must replay with exact full traces before launch.

Census correction: game.trades accumulates within a turn and is cleared by
end_turn. The preceding fair-share runner incorrectly labeled the terminal
turn's list as whole-game trades. Win outcomes are unaffected; whole-game
activity claims require correction. Observe the existing engine.trade_event
return value on the identical real Game object and accumulate across turns.
Do not count hypothetical search copies. Replay all 1,600 earlier games at
both floors and require exact actions/outcomes plus the original final-turn
trade list. Preserve earlier records unchanged and save corrected counts as
separate audit records. This is verification, not additional outcome samples.
