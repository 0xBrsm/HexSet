# Using HexSet

Running games, implementing and registering a bot, inspecting recorded games,
the bench commands and the public interface. Installation and a first game
are in the [README](../README.md#installation); designing a comparison is
[evaluation.md](evaluation.md); batched collection, model runtimes and the
Gymnasium adapters are [training.md](training.md); the server is
[server.md](server.md).

## Run a game

`deal_game` builds a game from a seed and a game index; `play_game` seats one
bot per seat and steps the game until it is over or `action_cap` actions have
been taken:

```python
import random

from hexset.arena import deal_board, deal_game, play_game
from hexset.bots import RandomBot

board = deal_board(seed=0, index=0)
game = deal_game(seed=0, index=0, players=4, board=board)
rng = random.Random(0)
bots = [RandomBot(rng) for _ in range(4)]
finished = play_game(game, bots)
```

A game is over when a seat wins or when `game.turns` reaches the game's
`turn_cap` (default `hexset.game.MAX_TURNS`, 300 turns summed over all
seats), which ends it with `won_by` `None`. `play_game`'s `action_cap`
defaults to `hexset.arena.MAX_ACTIONS` (20000). `play_game` raises nothing at
the turn cap; `compete` and `compete_batched` raise `Exhausted`.

`deal_board(seed, index)` is the board `deal_game` deals for that
`(seed, index)`; passing it explicitly lets a bot that needs the board be
built against it. Every random stream of game `index` is a function of
`(seed, index)`, so a tournament and a lockstep environment that deal the
same index play the same game.

`compete` deals a run of games, rotates a lineup through the seats and returns
a `Tournament`. `lineup_from_names(["mybot", "random"])` resolves entrant
names; `python -m hexset.bench.duel mybot random --runtime mybots --games 8`
does the same from the command line after importing the module that
registers `mybot`.

## Bundled opponents

HexSet ships no playing bot. These are the names it resolves itself; any
other name is registered by a runtime ([Implement a bot](#implement-a-bot)).

| Name | Seats |
| --- | --- |
| `random` | `RandomBot`: uniform over the legal actions. The control for throughput and harness checks |
| `retired` | No player: the seat is retired at the deal, so a four-entrant lineup with two `retired` is a two-seat game |
| `catanatron`, `catanatron:<key>=<value>[:...]` | Catanatron's alpha-beta player (`CatanatronBot`), depth 2; keys `depth`, `worlds`, `temperature`, `select`. Needs the `catanatron` extra |
| `network:<path>` | A checkpoint, one forward per decision; `network:<path>@0` is the same checkpoint with trading off |
| `mcts:<path>[@[<simulations>][w<wave>][:k=<worlds>]]` | A checkpoint under PUCT search; defaults 128 simulations, wave 16, `k=1` |

`catanatron` is hosted natively: HexSet owns the rules, the legal actions and
the ledger, and the adapter supplies the moves. At its default `worlds=0` the
mirror handed to Catanatron holds every seat's true hand and development
cards (the omniscient read); `catanatron:worlds=<n>` votes over `n` worlds
sampled from the mover's own `View` (the information-set read,
[worlds.md](worlds.md)). `python -m hexset.bench.duel random catanatron`
seats it directly.

`network:` and `mcts:` build through the checkpoint loader the process
registered with `hexset.clients.netbot.register_entrants`; without one they
raise `ValueError` ([training.md](training.md#model-runtimes)).

## Trade rounds

`Game.trade_mode` is the only bargaining setting a table has:

| Mode | Trading |
| --- | --- |
| `"round"` (default) | The engine runs offer-and-answer rounds. `play_game`, `compete`, `LaneEnv` and every bench command use it |
| `"auto"` | The engine's exhaustive clearing house deals until nothing clears, ranking candidates by `Game.trade_rule`: `"egalitarian"` (default) maximises the smaller of the two gains, `"nash"` their product, `"actor"` the actor's own. Research only |
| `"external"` | The engine runs nothing; a served session runs the same rounds across requests |

Each turn has one trade event, the acting seat's. It opens on entering the
main phase after the roll or the robber, or, for a gate with a
`trade_now(game)` hook, at the first main-phase decision point the hook
answers yes. A round has three steps: the actor broadcasts an offer; every
other active seat accepts, counters or passes; the actor executes one answer
or none. The event repeats rounds until the actor's offer budget is spent or
it has no offer it has not already made this turn.

The budget is the actor's gate's `trade_offer_budget`
(`TradeParams.max_offers`): `-1` declares no limit, and `0` is a gate that
opens nothing. The table has no per-turn budget and
no card limit: each gate bounds what it gives (`max_give_cards`) and the gain
it signs for (`trade_floor`) in its own `TradeParams`. Each side agrees once,
and execution checks only that both hands still cover the exchange.
`Entrant.trade` replaces an entrant's whole `TradeParams`;
`hexset.trading.retune` changes a built gate in place.

Menus, hooks, counters and the public ledger of offers are in
[trading.md](trading.md). Results from `"round"` and `"auto"` are not
comparable ([evaluation.md](evaluation.md)).

## Implement a bot

A Python bot implements `choose(game) -> Action`; no base class is required.
`hexset.actions.options_for(game)` returns the legal actions and
`game.state(hexset.game.to_move(game))` the acting seat's information set.
The acting seat differs from `game.current_player` during discards. `choose`
must not mutate the live game; a search works on copies (`hexset.game.imagine`).

A factory registered with the arena builds the bot from an `Entrant`, the
board and the runner's own generator:

```python
import random
from dataclasses import dataclass

from hexset.actions import Action, options_for
from hexset.arena import Entrant, compete, register_entrant_kind
from hexset.game import Game


@dataclass
class ExampleBot:
    rng: random.Random

    def choose(self, game: Game) -> Action:
        return self.rng.choice(options_for(game))


def make_bot(entrant, board, rng):
    return ExampleBot(rng)


def register_example() -> None:
    register_entrant_kind("example", make_bot)


tournament = compete(
    [Entrant("example", kind="example"), Entrant("random", kind="random")],
    games=2,
    seed=0,
    worker_initializer=register_example,
    records=True,
)
```

`tests/test_bot_integration.py` runs this code and replays the records it
returns.

`register_entrant_kind(kind, factory)` installs `factory(entrant, board, rng)
-> bot` for `Entrant.kind`. An `Entrant` holds serialisable settings, not a
built bot, so it crosses into worker processes. `compete(worker_initializer=f,
worker_initargs=args)` calls `f(*args)` once in every worker, or in the
calling process when `workers=1`. Register through it even when registering
in the parent appears to work: a worker inherits the parent's registrations
only under the `fork` start method, and some platforms and Python versions
default to `spawn` or `forkserver`. `start_method="spawn"` forces spawned
workers, which need importable factories and initialisers and a driver
guarded by `if __name__ == "__main__":`. A kind no process registered raises
`ValueError`.

A module that registers its bots at import is a runtime:

- `register_preset(name, entrant)` names an `Entrant` for `lineup_from_names`
  and every command line.
- `register_spec(prefix, parse)` resolves names starting with `prefix`
  (`mybot:depth=3`) through `parse(name) -> Entrant`. A preset of the exact
  name wins, then the longest matching prefix.
- `registered_presets()` lists every preset except `random` and `retired`.
- `load_runtime(*modules)` imports the modules;
  `compete(worker_initializer=load_runtime, worker_initargs=modules)` does so
  in every worker, and `--runtime <module>` does so on every command line.
- `Entrant.options` holds settings only that kind's factory reads, as
  `(key, value)` pairs or a mapping, stored sorted by key so equal settings
  make equal entrants; `entrant.option(key, default)` reads one.
- An unknown name raises `ValueError` saying to import the runtime that
  registers it.

A name means one thing in a process. Registering a different factory, entrant
or parser under a taken name raises `ValueError`; registering the same one
again does nothing, so a runtime imported twice is harmless.
`unregister_entrant_kind`, `unregister_preset` and `unregister_spec` free a
name. The names HexSet resolves itself (`random`, `retired`, `catanatron`)
and the prefixes it parses (`network:`, `mcts:`, `catanatron:`, and any
prefix overlapping one) are refused. The kinds `network` and `mcts` are the
exception: `register_entrants` replaces their factories, and the last call
wins.

### Information sets

`game.state(seat)` returns that seat's `View`: its own cards, the public
state, and each opponent's hand as the public ledger bounds it.
`game.state(seat, hidden=False)` returns the true `GameState`, the live
object itself. The live `Game` exposes full state to Python callers: the
information set is a policy contract, not a security boundary. An
information-set policy reads its `View` and samples worlds for search
(`View.sample`, [worlds.md](worlds.md)). The engine reads no hidden card for
it: `View.sample`, `hexset.bots.determinized.distinct_worlds` and
`hexset.mcts.Search` deal every unseen development card alike unless the
caller passes a `hexset.view.HoldReading`. An experiment that reads
opponents' true cards is reported as such. The `# true state:` comments in
the source mark every true-state read for review; behavioural invariance
tests, which redeal hidden cards and compare answers, are what show a policy
respects its information set.

### Trading

A bot values exchanges through `gains_many(view, received, counterparties)`,
one gain per candidate bundle on its own scale, and declares `trade_floor`:
a gate that returns a positive gain without one raises `TypeError`. It may
also implement `offer`, `respond`, `pick` and `estimate_many`; the defaults in
`hexset.trading` decide from the bot's own gain and its estimate of the other
seat's. A bot that wants limits on top of its valuation declares a
`TradeParams` and installs `hexset.trading.TradeProtocol` instead of writing
those methods ([trading.md](trading.md)). Without `gains_many`, a boolean
`accepts_many` or `accepts` maps to gains of +1 or -1; a bot with none of
them declines every player trade. Bank and port trades are ordinary actions.

The shared pieces:

- `hexset.bots.base`: `Bot`, `TradeGate`, `TradesBy`, `RandomBot`,
  `seat_at`.
- `hexset.trading`: the referee, `TradeParams` and `TradeProtocol`.
- `hexset.bots.stances`: per-seat search objectives (`own`, `relative`,
  `paranoid`).
- `hexset.bots.determinized`: information-set worlds for a search
  ([worlds.md](worlds.md)).
- `hexset.actions.options_for`: the legal actions, raising `Stuck` if there
  are none.

## Inspect behaviour

Take records from the run that produces the score.
`compete(..., records=True)` returns them in `Tournament.records`, in game
order; `compete_batched(..., episodes=True, records=True)` attaches one to
each episode. `hexset.record.write(path, records)` appends them as JSON lines
and `read(path)` yields them back. `replay(record)` replays every action,
checking legality, the chance stream and the recorded result, and returns the
final position; `replay_to(record, ply)` returns the position after `ply`
actions, unchecked. A
record carries the chance stream, the exchanges and the shown offers, so a
replay reproduces the game without running any bot.

Records hold game behaviour, not model activations, search trees or training
examples; store those separately, keyed by game index and action step. A
gate's private valuation of a trade cannot be recovered from the exchanged
cards.

## Benchmark commands

Commands are modules under `hexset.bench`; each takes `--help`.

| Module | Purpose | Games default |
| --- | --- | --- |
| `duel` | Two sides with configurable seating: win rate, Wilson and board-level intervals, paired victory-point margin | `--games 400` |
| `baselines` | A lineup of named entrants (`--lineup`, required), with per-entrant and pooled win rates | `--games 40` |
| `throughput` | Engine speed: `random` in every seat | `--games 200` |
| `generate` | Replayable records of one bot in every seat (`--bot`, `--out`, both required) | `--games 100` |
| `trade_census` | Executed trades by entrant label (`BotSummary` documents the fields); `--records` saves the games | `--games 96` |

- Every command except `throughput` takes `--runtime <module>`
  (repeatable), imported first in every worker.
- The same four take `--game-type` (default `standard`), `--turn-cap`
  (default `MAX_TURNS`) and `--trade-mode` (`round` or `auto`, default
  `round`).
- `--workers` defaults to every core (`os.cpu_count()`), except
  `trade_census` (1). Set it explicitly on a shared machine.
- `--games` must divide evenly over the seats so the rotation completes.
- The four keep each game in a journal as it finishes (`--journal`), and
  `--resume` plays only the games a journal lacks. Default journals: `duel`
  beside its verdict (`--verdicts`, default `runs/eval`), `baselines` under
  `runs/baselines`, `generate` beside `--out`, `trade_census` beside `--out`
  or under `runs/census`. Paths are relative to the working directory.

Batched model policies are evaluated with
`hexset.bench.versus.compete_batched`, which uses the arena's boards and
seating and batches inference across live games
([training.md](training.md#batched-evaluation)).

### Duel seating

`--geometry` is one letter per seat: the default `aabb` seats two copies of
each side at a four-seat table, and `ab` is a two-player game. A
comma-separated geometry may name other entrants (`a,b,random,random`).
These are individual players, not teams. The duel requires an even
`--games` that divides over the seats.

Each board is played twice. Where the geometry has as many `a` slots as `b`
slots (`aabb`, `abab`, `a,b,random,random`), the second game swaps the two
sides' seats and leaves every other seat in place. With unequal sides
(`aab`) no swap exists, and the second game uses the arena's half-turn
rotation. In a lineup with `retired` seats the opening seat rotates among the
playing seats, so each side opens equally often.

## Public interface

From 1.0.0 on, HexSet follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
a minor release adds to the public interface, and only a major release
removes or changes any of it.

The public interface is every module with no leading underscore in its path,
as far as its `__all__` lists, together with the HTTP API and its wire format
at `API_VERSION` 4 ([api.md](api.md)), the MCP tools ([mcp.md](mcp.md)), the
ONNX metadata keys ([onnx.md](onnx.md)) and every command's flags. Anything
else is internal and may change in any release: an underscore module (such
as `hexset.trading._engine` or `hexset.server._webplay`), and any name a
public module does not list. `tests/test_public_surface.py` holds every
public module to an `__all__` whose names exist and are documented.
