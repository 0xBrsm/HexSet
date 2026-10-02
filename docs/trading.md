# Trading: the referee, the parameters, the protocol

`hexset.trading` is split three ways, plus one helper module:

| Module | What it owns |
| --- | --- |
| `hexset.trading._engine` | The **referee**: candidate enumeration, validation, the trade round, the automatic clearing house, moving the cards |
| `hexset.trading._params` | What one gate **declares about itself**: `TradeParams` |
| `hexset.trading._policy` | The **protocol** those parameters drive: `TradeProtocol` |
| `hexset.trading._fragments` | The planning arithmetic behind a fragmented proposal |

The public surface is the names `hexset.trading` exports (its `__all__`):
`from hexset.trading import trade_round, Trade, bundle`. The submodules stay
importable, but what they hold beyond those names (underscore helpers, the
ledger-showing steps the round drivers run) is internal and may change.

A bot answers two questions. *What is this exchange worth to me?* is the
valuation, and it is the whole of what a bot brings. *What will I put to the
table, what will I answer, and what will I sign?* is the protocol, which is
the same for every valuation, so a Python bot and a neural checkpoint play
the same one.

## What the table sets

The table sets `Game.trade_mode` (`"round"`, the default propose-and-respond
round; `"auto"`, the clearing house; `"external"`, a caller-driven round),
`Game.counter_steps` (below), and for `"auto"` `Game.trade_rule`. It sets no
per-turn budget and no card rule. Everything else about bargaining is one
seat's own `TradeParams`, binding nobody else: two seats at one table may
carry different ones.

How many cards an exchange moves is refused, if at all, at each side's own
consent: `max_give_cards` bounds what that seat parts with, and a seat
declaring none is bounded only by the cards it holds. A manual seat declares
nothing and its submission is its own consent, so a table of people has no
card limit.

The one number the engine keeps for itself is `ENUMERATION_CARDS` (3), a
**search bound, not a rule**: candidate generation walks every multiset of a
hand on both sides, so a gate that declares no cap is enumerated at three
cards a side. Nothing refuses an exchange for exceeding it, and a gate whose
`max_give_cards` is wider is enumerated that wide.

```python
from dataclasses import replace
from hexset.trading import UNLIMITED

replace(UNLIMITED, max_give_cards=2)   # this seat parts with two, at most
```

No server flag narrows the bots a server seats, and the wire does not cap
what a seat submits: a limit lives in the bot's own config or nowhere.

## When the turn's trade opens

Each turn has one trade event, belonging to the seat on the move. A gate
without `trade_now` opens it on entering the main phase, before its first
main-phase action. A gate with `trade_now(game) -> bool` is asked at every
main-phase decision point until the event is used: on entering the main
phase, and after each of its own main-phase actions, except between the
free roads of a Road Building card. The event runs at the first point it
answers yes; a turn it never answers yes in has no trade. `trade_now` gets
the `Game`, as `choose` does, under the same information contract.

Within the event the actor's gate offers round after round until its
`trade_offer_budget` (`max_offers`, `-1` for none declared) is spent or it
makes no offer; a bundle already offered this turn is not offered again. A
gate with `max_offers=0` opens no event. A round clears at most one
exchange.

A record files the event's exchanges under the step whose `apply` they
cleared inside, so a trade made after a build replays after that build.

## `TradeParams`

| Field | Default | What it is |
| --- | --- | --- |
| `max_offers` | `None` | Offers this gate makes on each of its own turns (`trade_offer_budget`). `None`: it offers while it has an offer not yet made this turn. `0`: it opens nothing and signs nothing; every offer put to it is a pass, and the clearing house never deals with it |
| `max_give_cards` | `None` | Cards it parts with in one exchange; `None` for no limit of its own |
| `trade_floor` | `0.0` | The floor its own gains must exceed, on its own value scale |
| `responder_card_risk` | `0.0` | Charge per outgoing card on a response and on final consent; never charged to the proposer |
| `fragment_trades` | `False` | Plan one gated target a turn and offer it as fragments |
| `fragment_threshold` | `0.0` | The proposer cutoff a full target is gated on |
| `max_fragments` | `1` | Fragments one target may be offered as (1 or 2) |
| `fragment_cards` | `None` | Cards either side of one fragment may move; `None` for `max_give_cards`, else `ENUMERATION_CARDS`. One side of a fragment is always a single card |
| `gate_plies` | `0` | Plies rolled forward over the exchanged hand before it is valued |
| `fit_offers` | `False` | Put first the offers and counters that fit what the counterparty has shown it wants and will give up (`fit`, off the public ledger); elsewhere the same choice as without it. Offers are planned only under `fragment_trades`, so there it orders both; under card caps or a responder price alone it orders counters |

`TradeParams` raises `ValueError` for a negative `max_offers`, floor, risk or
threshold, a cap below 1, `max_fragments` outside 1..2, a negative
`gate_plies`, and a `fragment_cards` wider than a declared `max_give_cards`
under `fragment_trades`.

One set ships: **`UNLIMITED`**, `TradeParams()`, with no card caps, a floor
at zero, no response charge, no offer budget and no planning. Every gate that
declares nothing is read at it. A bot's own config lives with the bot: in its
module for a Python bot, in its ONNX metadata for a checkpoint.

A declared card cap is a **refusal, not a preference**. It applies before
the valuation and binds the actor as well as the responder: `menu` drops
every offer giving more, and an answer giving more is masked out of `pick`
rather than chosen and then refused at execution. An undeclared cap is no
cap.

## Which hooks a parameter set installs

`hexset.trading._engine` reads a gate's `candidates`, `offer`, `respond`,
`respond_any`, `pick`, `consent_gain`, `allow_repeated_offer` and
`trade_round_finished` by name (`HOOKS`, every hook a `TradeProtocol` can
install). A missing hook falls back to the engine's menus (below),
`default_offer`, `default_respond`, `default_respond_any`, `default_pick`
and `valued`. The engine also reads names the protocol never supplies, which
stay the bot's own: the valuation (`gains_many`, else `accepts_many`, else
`accepts`; and `estimate_many`), `trade_floor`, `trade_params`, `trade_now`
and `observe_trade`.

`hooks_for(params)` says which hooks a parameter set needs.
`install(gate, protocol)` binds those as instance attributes, deletes the
other `HOOKS` from the instance, and keeps the protocol on the gate as
`_protocol` for `retune`; a hook the bot's class defines shows through where
the protocol does not need it.

| Parameters | Installed |
| --- | --- |
| Declares nothing (`UNLIMITED`) | nothing: the engine's defaults answer |
| Card caps or a responder price | `respond`, `respond_any`, `pick`, `consent_gain` |
| `fragment_trades` | all eight |
| `max_offers=0` | nothing: the engine passes for this gate without asking it |

A gate that constrains nothing therefore behaves as a gate with no
parameters. A cap alone does not install `candidates` or `offer`: the
protocol's replacement for the bot's own menu is one planned fragment, a
different protocol rather than a different limit.

`retune(gate, **changes)` replaces fields on the gate's `TradeParams` and
reinstalls its protocol; on a gate with no `TradeParams` it honours only
`max_offers`, through `trade_offer_budget`.

## What a seat may ask for

An offer is broadcast to every other seat, and whoever holds what it asks
for may take it. The engine's offer menu, `open_candidates`, is every give the
actor's hand covers against any ask the other seats can hold between them:
the public count of each resource outside the bank and the actor's hand, up
to the gate's enumeration width a side, the two sides on disjoint resources.
What any one seat is certified, believed or sampled to hold does not narrow
it. The fragmented policy's parent pool is the same enumeration.

A counter goes to the actor alone, so its menu, `known_candidates`, asks
only for what the ledger certifies the actor holds. A gate's own
`candidates` hook replaces either menu; it is called with `turn=None` for a
counter. An open offer (one with any cards, `is_open`) cannot be accepted as
it stands, only countered, and is answered by `respond_any`, which under the
protocol is the counter `respond` would send, under the same caps, response
charge and fit.

What a gate does with an ask is its own. The network gate scores `-1.0` for
a counterparty the record proves cannot give its side (`has_room`: its
uncertified asks exceed its untyped cards), so it never offers what nobody
can take; the menu still holds the ask.

The referee admits an answer (`respond`) only in this form; anything else is
a pass:

- it is the asked seat's;
- an acceptance is of the offer as made, of a concrete offer, and covered by
  the seat's own hand (read off its own view, so nothing hidden is
  consulted);
- a counter is a five-count exchange with cards both ways.

An exchange the referee executes (`execute_agreed`) is between two seats at
the table, neither locked, in `MAIN`, one of them the current player, with
cards both ways, and both hands covering their sides; it raises
`ValueError` otherwise. An offer nobody can cover goes unanswered.

## What a seat has shown

Every offer and answer is public, so each goes on the public ledger as it is
made (`trading.show`): what the seat asks for becomes a **want**, what it
gives a **waste**, each kept at the most it has shown. Cards clear them: a
want falls by each card of that resource the seat receives by a public route
(a trade, the bank, a roll, a card), a waste by each it parts with. A steal
moves neither, since it names no resource. The round drivers show each offer
before anyone answers it and each answer as it is given; at a hosted seat
the client reports what its table showed through `Seat.shown`, an ask with
nothing in return included.

An offer or answer whose giving side the referee checks the seat covers also
certifies those cards: the ledger's `known` rises to them, out of the seat's
untyped cards, so a counter may then ask for them.

A `Record` keeps each shown offer or answer among the step's exchanges
(`Record.shown`), so a replay reaches the same ledger.

`fit(view, seat, received)` reads it for a gate: one point for giving `seat`
a resource it wants, one for asking only for cards it has offered away,
within the counts it offered.

## An agreement binds

A seat judges an exchange once, when it agrees to it: a responder when it
answers an offer, the actor when it picks an answer. Nobody is asked again
when the cards move. An acceptance is a different trade depending on who
gives it, so `default_pick` prices each acceptance by its taker, as it
prices a counter, and picks the highest gain above the actor's floor, or
none.

Execution checks the rules and nothing else: both seats still cover their
sides. The round's drivers (the engine's `trade_round` and the served table)
price each bot side's gain on the executed `Trade`, for the record only. A
bundle composed by hand and put to a bot has no agreement behind it, so its
gate is asked then (`execute_agreed` with `ask_counterparty=True`).

## Counters to counters

By default a round carries one counter: a seat answers the offer with one,
and the actor takes it or leaves it. `Game.counter_steps` (and
`trade_round(..., counter_steps=)`) extends the thread. Where the actor picks
nothing, each counter in turn, in seat order, is answered by the actor with
its own `respond` (accept, counter or pass) as if it were an offer to it
alone, and a counter back is answered the same way by that seat, mover and
seat in turn (`CounterThread`, run to its end by `counter_thread`). A thread
holds at most `counter_steps` counters, the seat's first; a counter past
that, or one repeating a bundle already put in the thread, is a pass. The
first acceptance executes and ends the round.

Every counter is between the mover and one seat: the mover executes every
trade, so a seat countering another's counter is making its own offer to
the mover. The served table runs one counter. A hosted seat plays a thread
one answer at a time (`Seat.counter`, [hosted.md](hosted.md)).

## Wearing it

A Python bot and a checkpoint carry the same object.

```python
import random
from dataclasses import dataclass, replace

from hexset.actions import options_for
from hexset.trading import UNLIMITED, DeclaredTrade, TradeParams, TradeProtocol, install

MY_TRADE = replace(UNLIMITED, max_give_cards=2)


@dataclass
class MyBot(DeclaredTrade):
    rng: random.Random
    trade: TradeParams = MY_TRADE

    def __post_init__(self):
        install(self, TradeProtocol(self, self.trade, seed=0))

    def choose(self, game):
        return self.rng.choice(options_for(game))

    def gains_many(self, view, received, counterparties):
        return [float(sum(bundle)) for bundle in received]  # cards in less cards out


rng = random.Random(0)
bot = MyBot(rng)                                     # MY_TRADE by default
strict = MyBot(rng, trade=replace(MY_TRADE, trade_floor=0.5))
```

`trade` is the gate's whole parameter set. `DeclaredTrade` reads every limit
the engine and `retune` ask a gate for (`trade_params`, `max_offers`,
`trade_floor`, `gate_plies`, ...) through it, and its setters write through
`retune`.

A checkpoint declares its own in ONNX metadata; see
[onnx.md](onnx.md#model-metadata) for the keys, and `TradeParams.as_meta()`
for writing them out. `bot_for` and `searcher_for` read them, and
`GatedSearch` carries them too, so a searching entrant and a one-forward
entrant from the same file bargain identically.

In the arena an entrant names a whole parameter set, or none:

```python
from hexset.arena import Entrant, register_entrant_kind

register_entrant_kind(
    "mybot", lambda entrant, board, rng: MyBot(rng, trade=entrant.trade_params(MY_TRADE))
)

Entrant("a", kind="mybot")  # the bot's own default
Entrant("c", kind="network", weights="models/policy.onnx", trade=MY_TRADE)
```

`Entrant.trade=None` (the default) means the bot's own: the default a
registered factory passes to `entrant.trade_params`, and the file's
declaration for a checkpoint.

## Where the protocol does not apply

`trade_mode="auto"`, the clearing house, ranks candidates by the raw gains
under `Game.trade_rule` (`"egalitarian"` by default) and never asks a gate
to price its own consent, so neither a card cap nor a response charge is in
force there. Its trade event raises `ValueError` when any seated gate
defines `consent_gain`. A gate at `max_offers=0` is never its counterparty.
`"auto"` is for research only; see [the evaluation rules](evaluation.md).
