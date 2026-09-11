# Next Heximax search hypotheses

Date: 2026-09-10

This is a local, read-only search review. No native or remote games were run
for this readout, and no private source was exported.

## What the local counterexample check established

The source has no explicit terminal-action reservation, but that is not proof
that the stock search excludes a winning action. At depth two, root depth-one
values are computed before the root width beam is applied. A terminal leaf is
recognized by `is_over(game)` and the evaluator adds `WIN_SCORE` when the
winner reaches `WINNING_POINTS` (`src/hexset/bots/heximax/search.py:770-793`,
`src/hexset/bots/heximax/evaluate.py:328-340`). In the default `win` stance,
that makes an evaluated terminal action's rank approximately one and normally
puts it inside the six-action root beam.

I constructed a local state from a real seeded setup, performed a legal city
build, put seat 0 at nine points, and gave it one remaining legal city that
wins. With the stock no-trade parameters (`depth=2,width=6,max_nodes=600,k=1`)
the bot evaluated 26 leaves and selected that city. The first-ply ranks were:

```text
1.000000000 BUILD_CITY 16 0
0.863698498 END_TURN 0 0
0.857560731 BANK_TRADE 3 1
0.856242507 BANK_TRADE 3 0
0.856242507 BANK_TRADE 3 2
0.856242507 BANK_TRADE 3 4
```

I also constructed a nine-point state where seat 0 needed a bank trade before
its city could win. With hand `[4,4,4,1,3]`, no dev-card win available, and 18
legal actions, the stock bot selected `BANK_TRADE(0,3)`, whose next legal
`BUILD_CITY(16)` won. It used 104 leaves and reached depth two. The relevant
first-ply ranks put the three wheat-producing trades at the top:

```text
0.864348 BANK_TRADE 0 3
0.864348 BANK_TRADE 1 3
0.864348 BANK_TRADE 2 3
0.863632 END_TURN 0 0
0.862599 BANK_TRADE 0 1
0.862599 BANK_TRADE 2 1
```

These are reachable action positions built on the public engine's setup and
legal build transitions; the test normalizes the point/award ledger to make a
small deterministic search position. They are counterexamples to any claim
that the current stock search *necessarily* misses immediate or bank-trade
wins. I did not find a stock `d2/w6/n600/k1` miss, so exclusion remains
unproven.

The remaining mechanism is still real: `_search` retains only
`ranked[:self.width]` after root depth one (`search.py:304-330`), and interior
nodes rank one-ply actions before applying their width beam (`search.py:770-793`).
If an evaluated action ties or loses its one-ply score, it can be discarded;
that requires an affected state or telemetry to establish practical impact.
Budget exhaustion is also a possible partial-result path: `_root_values`
appends completed candidates until `_leaf` raises `_Exhausted`, and `_search`
then selects from `partial` (`search.py:346-362`, `664-667`). No local
reproducer showed a winning action omitted by that fallback.

## Reproducer core

The local check used this public-engine construction (the two award fields and
hidden VP count are set only to normalize a compact nine-point fixture; the
candidate actions and their application are native engine operations):

```python
from hexset.actions import ActionType, apply, legal_actions
from hexset.bots.heximax.evaluate import HonestEvaluator, NO_TRADE_WEIGHTS
from hexset.bots.heximax.search import Heximax
from hexset.cards import DevCard
from hexset.game import (Phase, build_city, imagine, legal_initial_roads,
                         place_initial_road, place_initial_settlement, start)
from hexset.state import Building, can_place_settlement

board_rng = random.Random(0)
game = start(random_base_board(board_rng), 4, random.Random(0))
while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
    if game.phase is Phase.SETUP_SETTLEMENT:
        vertex = next(v for v in range(game._state.board.topology.num_vertices)
                      if can_place_settlement(game._state, game.current_player,
                                              v, connected=False))
        place_initial_settlement(game, vertex)
    else:
        place_initial_road(game, legal_initial_roads(game)[0])
game.phase = Phase.MAIN
game.current_player = 0
vertex = next(v for v, owner in enumerate(game._state.vertex_owner)
              if owner == 0 and game._state.vertex_building[v] == Building.SETTLEMENT)
game._state.hands[0] = [0, 0, 0, 2, 3]
build_city(game, vertex)
game._state.longest_road_holder = 0
game._state.largest_army_holder = 0
game._state.dev_cards[0][DevCard.VICTORY_POINT] = 2
game._state.deck = [DevCard.KNIGHT] * 25
game._state.hands[0] = [4, 4, 4, 1, 3]  # trade first, then BUILD_CITY can win

bot = Heximax(HonestEvaluator(game._state.board, NO_TRADE_WEIGHTS),
              depth=2, width=6, max_nodes=600, k=1,
              rng=random.Random(7), max_trades=0, mode="notrade")
chosen = bot.choose(game)
```

For the immediate-city fixture, replace the final hand with `[0, 0, 0, 2, 3]`.
The observed choices were respectively `BUILD_CITY(16)` and
`BANK_TRADE(0,3)`; applying the latter and then `BUILD_CITY(16)` sets
`won_by == 0`.

## Bank-trade quiescence is already tested

The existing opt-in `BankTradeQuiescence` extends a horizon `BANK_TRADE` by
one post-trade action only (`src/hexset/bots/heximax/bank_quiescence.py:32-50`).
That is the exact trade→build hypothesis; it is not a new untested idea. The
archived 1,000-game result for `bankext-wide` was 458/1000 against AB2 and
259/1000 against frozen self (`docs/readouts/luna-search/CAMPAIGN-STATUS.md:103`).
A new “protected terminal” experiment would differ by preserving candidates
*before* root/interior beam selection, rather than extending a bank-trade
horizon. The local evidence above does not justify prioritizing that change
without first instrumenting ties, dropped action classes, and exhaustion.

## Candidate follow-ups

1. Add read-only instrumentation to count root/interior candidates dropped by
   width, their one-ply rank ties, and whether they are terminal or
   `BANK_TRADE`. Run a small local fixture sweep first. Only if it finds
   affected positions should an opt-in protection variant receive a game
   screen; the existing CLI cannot express protection.

2. Fit the evaluator for depth-two search. The shipped `Weights` documentation
   explicitly says the values were fitted for one-ply play and deeper search
   has not been refitted (`src/hexset/bots/evaluate.py:131-140`). This remains
   a separate hypothesis from beam retention.

3. Re-screen the corrected port-aware evaluator after native source
   fingerprinting. The archived port-aware result has no source fingerprint
   and cannot certify the corrected implementation; details are in
   `docs/readouts/luna-search/port-aware-correctness-20260910.md`.

## Historical search context

The archived screens show no monotonic “more search is better” pattern: the
AB2/shipped win counts were 52/37 for nodes1200-width8, 69/30 for
nodes2400-width12, 49/21 for nodes2400-full, 53/26 for nodes600-k4, 54/36 for
nodes2400-k4, 45/22 for depth3-width6, 28/21 for depth3-width6-2400, and
43/22 for depth3-width6-4800. These controls do not isolate beam retention.

## Provenance

```text
0a9a4b5df187153484c3de15da14995a1938c891328ac3e9a705f2fd55220dce  src/hexset/bots/heximax/search.py
5e1a8415e27c66d2235d81aee3c2297c055707ed49560a565ed77558ea191dad  src/hexset/bots/heximax/evaluate.py
```
