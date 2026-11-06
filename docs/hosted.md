# A seat at a table hosted elsewhere

The engine's own tables (the arena, the served web table) hold the true game
and every seat's gate. A bot playing at a table somebody else hosts has
neither: it sees its own cards and the public record, the other players act
on the host's schedule, and the host moves the cards. This document is the
engine's support for that seat.

## The observed game

`hexset.game.observe(game, seat)` returns a copy of `game` as `seat` sees
it: every other seat's hand and development cards, and the deck, are hidden
piles of the same size (`state.Hidden`, as `state.observed_by` makes them),
and everything public (the board, the public ledger, whose turn it is and
every flag of it) is carried over.

It is played forward by the ordinary `actions.apply`, every seat's actions
alike, as the table's events arrive:

- A hidden hand takes every public change by name. A production, a payment,
  a bank or player trade, a discard, a Year of Plenty or a Monopoly moves its
  count, and the resource goes into `HiddenHand.flow`, which
  `PublicLedger.apply_hand_diff` reads as it reads a concrete hand's diff.
  The pile stays unreadable: `flow` is ledger bookkeeping, not a composition.
- What no seat can draw for itself comes from the host. The observed game's
  `chance` is a `chance.Hosted` source the caller feeds with
  `expect(kind, value)` immediately before applying the action each outcome
  belongs to: the dice total (`"roll"`), the card a steal moved or
  `chance.UNSEEN` when this seat was not party to it (`"steal"`), the card a
  purchase drew or `UNSEEN` for another seat's (`"draw"`), and the count each
  hidden seat surrendered to a Monopoly (`"surrender"`). An outcome it was not
  given, or one of the wrong kind, raises.
- The host is the referee. Of a hidden hand only the count can be checked,
  so applying another seat's action takes the table's word that it was legal;
  `legal_actions` answers for the observing seat only.
- A win is checked on the current player's countable points: its full count
  for the observing seat, public points for a hidden one. A win decided by
  hidden victory-point cards is the host's to announce.
- Trades are the host's: `trade_mode` is `"external"`, and an exchange the
  table made goes in through `trading.execute_agreed` with nobody asked.

Played in lockstep with the game it observes, an observed game stays equal to
`observed_by` of it: every public field, its own legal actions and the ledger
(`tests/test_observed.py`).

## The seat

`hexset.seat.Seat` is the observed game together with the seat's bot, and
the bookkeeping the engine's own drivers do around every decision. A client
mirroring a foreign table translates the host's events into `play` and
`traded` calls and puts the seat's decisions on the host's wire; it makes no
decision of its own.

```python
from hexset.actions import Action, ActionType
from hexset.cards import DevCard
from hexset.chance import UNSEEN
from hexset.seat import Seat

# board, me, bot and the event values stand for the client's own.
seat = Seat.sit(board, 4, me, bot, game_type=game_type)  # at the deal

# the table's events, in the order the host logged them
seat.play(actor, Action(ActionType.ROLL), roll=8)
seat.play(actor, Action(ActionType.MOVE_ROBBER, hexagon, victim), stolen=UNSEEN)
seat.play(me, Action(ActionType.BUY_DEV_CARD), drawn=int(DevCard.KNIGHT))
seat.traded(actor, other, received)          # an exchange the host executed

# this seat's decisions
seat.offer()           # the next offer of this turn's trade event, or None
seat.pick(responses)   # which answer to execute, priced by who gave it; or None
seat.close_round(trade)
seat.observed_round(offer, responses, trade)  # another seat's round closed
seat.answer(offer)     # accept, counter or pass another seat's offer
seat.counter(other, received)  # accept, counter or pass a counter to a counter
seat.shown(actor, received)    # an offer, an answer or an ask the table showed
seat.choose()          # its move
seat.discard()         # a whole discard, as resource indices
```

`Seat.sit(board, num_players, seat, bot, gate=None, *, game_type, first,
locked, turn_cap, counter_steps, deals)` deals a fresh observed game with no
turn cap by default; `Seat(game, seat, bot, gate=None, *, counter_steps=1,
deals=None)` wraps an observed game the client built. `gate` prices trades
and defaults to the bot when it has `gains_many`; a seat with no gate offers
nothing, passes every offer and promises nothing. `deals` is how the seat
makes robber deals (`hexset.deals.DealParams`): by default it answers
asks (`hexset.deals.ANSWERING`), and `DealParams()` promises nothing.
`play` requires `drawn` for this seat's own purchase and takes
`surrendered`, the hidden seats' Monopoly losses in seat order.

What `Seat` does for the client, because the engine's drivers do it at their
own tables:

- **The trade event.** `offer` opens the turn's one event where the engine
  would: on entering the main phase, or at the first decision point a gate's
  `trade_now` says yes, never between a Road Building card's free roads. It
  keeps offering through `trading.offer` (the gate's menu, its pick, its
  repeat rule, the bundles already put to the table) until the gate's own
  `trade_offer_budget` is spent, it passes, or this seat makes a move after
  the event opened. It returns `None` while a round of this seat's is open
  (`open_offer`).
- **The round.** `pick` is the gate's `pick`, or `default_pick`, which prices
  each acceptance by who gave it and declines one under the gate's floor
  ([trading.md](trading.md#an-agreement-binds)). At a table whose seats
  answer over time, `pick(responses, waiting=seats)` names those that have
  not answered yet: the answer picked is executed only if none of them would
  be picked over it by accepting the offer as put, and otherwise `pick`
  returns `None` and `held` says what is waited for. An open offer never
  waits. `close_round` tells the gate how its broadcast went
  (`trade_round_finished`), which is how a plan spread over several offers
  learns whether its fragment cleared.
- **Counters to counters.** `counter(other, received, any=0)` answers
  `other`'s counter to this seat's open offer, or, with `other` on turn, its
  counter to the counter this seat answered its offer with. Both sides share
  one `trading.CounterThread` for the turn, bounded by `counter_steps`; a
  counter past it, or one repeating a bundle already put, is a pass
  ([trading.md](trading.md#counters-to-counters)). A counter back is the
  client's to put on the wire; an acceptance of a counter to this seat's
  offer is this seat's to execute.
- **What the table traded.** Once a turn the gate's `observe_trade` hears the
  turn's exchanges and the hand sizes its trading opened with: on this
  seat's own turn when its event closes, on another seat's when that turn
  ends. Every round's close reaches the gate's `observe_round`
  ([trading.md](trading.md#which-hooks-a-parameter-set-installs)): this
  seat's own through `close_round`, another seat's through
  `observed_round(offer, responses, trade)`.
- **What each seat has shown.** The client reports every offer and answer the
  table shows, this seat's own included, and an ask with nothing named in
  return ("need wood" is `received` with one Wood and nothing given), through
  `shown`, in the table's order. It goes on the public ledger as wants and
  wastes ([trading.md](trading.md#what-a-seat-has-shown)). `offer`, `answer`
  and `pick` are decisions and record nothing, so a client may put an offer
  nobody made to `answer`.
- **Discards in one piece.** `discard` asks the bot for one card at a time,
  as the engine does, on a copy where each chosen card is already gone and
  only this seat owes, and returns the whole selection; `None` if the bot has
  no answer yet. `choose` during a discard asks for this seat's next card the
  same way.
- **The host's table rules.** `restrict_robber(hexes)` limits the robber
  move at hand to `hexes`; `None` restores the rulebook's.
  `allow_open_offers(False)` is a host whose offers name every card on both
  sides: this seat's open ask goes up as the one-for-one it stands for
  ([trading.md](trading.md#open-offers)). `set_embargoes(pairs)` states the
  table's trade embargoes, `(seat, against)` pairs: this seat's offers are
  built for the seats it can still trade with, and a seat under an embargo
  with it is out of its robber deals both ways. `limit_before_roll(cards)`
  names the development cards the host takes before a roll, `None` for the
  rulebook's any. Each holds for the rest of the game.
- **Gains on demand.** `gains(received, counterparty)` is the gate's own gain
  on each exchange, for a host protocol that asks a question the engine has
  no verb for.
- **Out of step.** An outcome the host revealed that the engine did not take,
  or an event it cannot apply, raises `OutOfStep`; the client rebuilds the
  game from the host's statement of the position (its `chance` a `Hosted`)
  and hands it to `resync`.

## Robber deals

A seat at a table where robber deals are made in chat answers and asks
through these verbs; the client translates the host's words into them and
their answers back ([trading.md](trading.md#robber-deals)).

- **Answering.** `deal_answer(mover, kind, resource, promised=(), to=None)`
  says whether this seat promises `mover` a card of `resource` for a deal of
  `kind` (`hexset.deals.NO_BLOCK` or `NO_STEAL`); `promised` is the seats
  that already have, in order, and `to` the seat the mover named, if it
  named one. A promise is the seat's consent. `deal_withdraw()` takes back
  the last promise before the mover heard it. `deal_targeted(...)`, with the
  same arguments, says whether this seat is the one the mover is likeliest
  to hit, holding the card or not, so a client can answer an ask aimed at
  it truthfully. Once the robber has moved this turn, the seat on turn is
  promised nothing.
- **Paying.** After the robber moves (`play`), a promise is owed where the
  move spared this seat, and void where it did not. `answer` then takes the
  mover's offer that collects the card, an open offer included.
- **Asking.** At this seat's own robber move, before it is chosen,
  `deal_ask()` says whether to ask and for what (`Ask`, or `None`); asked
  again for the same move, it answers the same. `deal_promised(promised)`
  takes the seats that promised and returns those the seat spares, which the
  robber move `choose` then makes spares. A deal from an earlier robber move
  this turn carries into a later one: a promiser that has paid is spared
  again. Once the robber has moved, `offer` puts each spared promiser's
  exchange to the table, and `pick` takes a spared promiser's acceptance as
  the payment at once.
- **The log.** `deal_log` is every deal this seat asked this game, with what
  came of it (`hexset.deals.Deal`).

A hosted seat's view of its observed game is the same information set as its
view of the true game, so every verb returns what the engine's own stage
returns there: the same offer, the same answer, the same move
(`tests/test_seat.py`).
