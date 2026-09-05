# The bot API

This is the complete interface a `.onnx` file must satisfy to plug in as an
opponent. It is the only thing `src/hexset/clients/onnxbot.py` reads — the file
does not need access to this repo's source, only to what is written here plus
the public [ONNX](https://onnx.ai/) format itself. `hexset.heximax` (the
default handcrafted opponent) and `hexset.bots`' `search2` implement the same
interface a different way and are the reference for what "correct" means when
in doubt.

Two independent parts make up the contract:

1. **Self-description** — `metadata_props` on the ONNX model, read once at
   load time.
2. **The graph itself** — named inputs in, named outputs out. Which shape
   this takes depends on the `contract` key below.

## 1. Self-description (`metadata_props`)

| key | meaning | default |
| --- | --- | --- |
| `players` | table size the graph was traced for | required |
| `num_hexes` / `num_vertices` / `num_edges` | board-shape fingerprint; a mismatched board fails the load rather than running on meaningless input | required |
| `contract` | which graph shape below applies: `6`, the record shape | refused if absent — see below |
| `max_trades` | `0` to switch trading off for this checkpoint | trading on |
| `search` | `mcts` to search over the model's own priors; anything else plays one forward pass | none |
| `simulations` | descents per decision, when `search=mcts` (clamped to 4096) | 128 |
| `wave` | leaves batched per expansion, when `search=mcts` (clamped to 256) | 16 |
| `iteration` | informational only; not read for behaviour | 0 |

Unreadable or missing optional keys fall back to their default rather than
failing the load — a typo'd hint costs the hint, not the whole opponent. See
`src/hexset/server/modelmeta.py` for the exact clamping.

Inference device (`cpu`/GPU) is deliberately **not** a metadata key — it is a
fact about the machine serving the game, not the checkpoint.

**The `contract` number is assigned by the exporter, not by this repo.**
`hexset.export_onnx._CONTRACT_VERSION` is the one definition; `hexset.server`
reads it and never writes it. A graph declares the fields it wants and is fed
exactly those, so a graph that predates a field this record has gained still
loads and plays. An unknown number is refused at load with the number named,
rather than failing later on its first move with a missing-input error.

**Only contract 6 is served.** 2, 3 and 4 are the offer protocol's
contracts — 3 added four live-offer fields, 4 the two public-knowledge ledger
fields, and all three declare a `pair_mask` input and a `pair_index` output
for the one-for-one give/want heads. Trading is now one engine event with no
actions at all (see §4), so those graphs describe a game this engine does not
play: there is no honest way to feed them, and they are refused by name.
Contract 5 is refused too now, for two independent reasons that happened to
land together: it declared a `valuations` field for the one-event mechanic's
public valuation vector, and that public layer is gone outright
(`agents/reference/trading-final.md`, item 1) rather than replaced; and the
knight two-step fix (a knight is played, then the robber moves through the
same phase a seven enters, rather than one action carrying both) dropped
`PLAY_KNIGHT`'s operands, shrinking the flat `ActionSpace` a contract-5
graph's `action_mask`/`prior` were traced against. Either change alone would
have forced the bump. Contract 1 — the original shape, where the engine
encoded the position into feature tensors and the graph was a bare
policy/value head masked in Python — went the same way on 2026-09-02
(`docs/engine-divergence-2026-09-02.md`, B5).

## 2. The graph — the record contract (`6`)

The engine builds a **record**: the position stated in the rules' own terms,
already filtered to what the perspective seat may legally know. The graph
owns everything downstream of that — encoding, masking, normalising,
argmax, un-rotating back to board-seat order. Built by
`hexset.onnx_record.record_from_game`; the full field-by-field derivation
lives in [`onnx-contract-v2.md`](onnx-contract-v2.md).

Leading batch axis `B` on every tensor.

**Inputs:**

| Name | Shape | dtype |
| --- | --- | --- |
| `terrain` | `(B, num_hexes)` | int64 |
| `token` | `(B, num_hexes)` | int64 |
| `port_code` | `(B, num_vertices)` | int64 |
| `robber` | `(B,)` | int64 |
| `vertex_owner` | `(B, num_vertices)` | int64 |
| `vertex_building` | `(B, num_vertices)` | int64 |
| `edge_owner` | `(B, num_edges)` | int64 |
| `bank` | `(B, NUM_RESOURCES)` | int64 |
| `knights_played` | `(B, players)` | int64 |
| `award_points` | `(B, players)` | int64 |
| `longest_road_holder` | `(B,)` | int64 |
| `largest_army_holder` | `(B,)` | int64 |
| `phase` | `(B,)` | int64 |
| `free_roads` | `(B,)` | int64 |
| `deck_size` | `(B,)` | int64 |
| `turns` | `(B,)` | int64 |
| `perspective` | `(B,)` | int64 |
| `own_hand` | `(B, NUM_RESOURCES)` | int64 |
| `hand_totals` | `(B, players)` | int64 |
| `own_dev` | `(B, NUM_DEV_CARDS)` | int64 |
| `dev_totals` | `(B, players)` | int64 |
| `ledger_known` | `(B, players, NUM_RESOURCES)` | int64 |
| `ledger_unknown` | `(B, players)` | int64 |
| `action_mask` | `(B, space.size)` | bool |

**Outputs:**

| Name | Shape | Note |
| --- | --- | --- |
| `action_index` | `(B,)` int64 | argmax over the masked distribution |
| `prior` | `(B, space.size)` | normalised over legal actions, zero elsewhere |
| `value` | `(B, players)` | board-seat order, already un-rotated |

`NetworkBot` reads `action_index`; searches read `prior`/`value`. One graph
serves both.

**The engine drift this section used to list is gone.** This server no longer
carries its own copy of the engine: it depends on the `hexset` package (now
one distribution together with the gym, see the CHANGELOG's "one
distribution" entry), so what it plays is exactly what dev-HexNet plays. See
[`engine-divergence-2026-09-02.md`](engine-divergence-2026-09-02.md) for the
full account of what the copy held and how each difference was resolved.

**The one mask difference that used to remain is gone too.** The
`action_mask` served here was built over an *honest* trade sample, because
the engine's own `legal_actions` filtered the offer sample by opponents'
true hands and telling a human that would give away a specific opponent's
hand. There is no offer sample: trading is not an action, no remaining
action's legality depends on another seat's hand, and there is now one list,
`hexset.actions.legal_actions`, for every seat.

## 3. Trading

**Status, 2026-09-05: rewritten for the trading-final mechanic
(`agents/reference/trading-final.md`); the human/LLM negotiation surface
this section used to describe (`PUT .../valuation`, confirm mode's
per-route default, `propose_trade`) is server-side work still in flight —
see that document's item 5 for what changes there.**

A checkpoint does not act to trade, and there is no public layer any more:
nothing is advertised, and no vector rides in this record. Instead, every
seat answers a private **gate** — `gains_many(view, received,
counterparties) -> list[float]`, that seat's own gain from each candidate
exchange, in whatever unit its value is, read through its own view. The
engine enumerates every coverable candidate bundle between the current
player and each other seat after the roll and the robber, and again after
every MAIN action the current player takes (build, buy, a bank/port trade, a
development card): any signed bundle on disjoint resources, each side
bounded only by what that hand holds (not one card for one card — a
candidate can give several resources and receive several back in the same
exchange). It asks the current player's gate once over every candidate,
keeps the strictly positive subset, asks each counterparty's gate once over
its own accepted subset, keeps the strictly positive subset of *that*, and
clears the one candidate `Game.trade_rule` ranks highest — the default,
`"egalitarian"`, maximises the smaller of the two private gains; ties fall
to the current player's own gain, then a canonical bundle order, then the
lower counterparty seat, for determinism. Then it loops until nothing
clears. A gate is a pure function of the current position, asked fresh
every time — there is no publish step and no timing to get right.

A checkpoint served embedded (`hexset.clients.onnxbot.NetworkBot`) trades
off the same `value` head this contract already declares: `accepts` is the
head's strict preference for the concrete post-trade hand over the current
one — the derivation `hexnet.policy.DerivedTrader` trains under,
reimplemented here against the wire record instead of a live forward — and
`accepts_many` batches it over up to `NETWORK_GATE_ROWS` candidates in one
graph call. There is no magnitude-valued `gains_many` here: `hexset.bots.
search2.Bot`'s structural default derives one from `accepts_many`
(`+1.0`/`-1.0`), which is all a boolean value-head gate can support; a
magnitude-valued network gate is HexNet's own concern. `max_trades=0` in the
metadata is still the explicit off switch — a seat with it set accepts
nothing, exactly like a bot with no trading methods at all.

A checkpoint served externally (`hexset.clients.botclient.RecordBrain`, the
`python -m hexset.clients.botclient` peer) does not share this brain and
does not trade at all: it reads `GET /api/record` for `action_index` alone
and is never seated as a gate.

**The negotiation interface (human and LLM seats).** Everything above is the
automatic event; a human or LLM seat additionally gets `POST
/api/games/<code>/trade` (`hexset.game.Game.execute_trade`) to compose and
submit a bundle directly, bypassing the automatic candidate search — any
bundle both sides can cover, not only what the event would have found. It is
legal on the proposer's own turn against any seat, or during another seat's
turn against that seat only, and requires the counterparty's own gate to
price the exchange strictly above zero; the proposer's own gate is never
consulted, since submitting is its own consent. A checkpoint served through
this contract is never itself a *proposer* here — nothing calls this route
on a bot's behalf — but it is a valid **counterparty**: a person or an LLM
may propose a bundle against a served checkpoint at any time, and the
checkpoint's gate answers it exactly as it would an automatically-found
candidate, because the call is the same.

## What is never part of this contract

`onnxbot.py`'s job stops at reading these names and shapes. It never imports
or inspects anything else about how a checkpoint was produced, and a
checkpoint's author never needs this repo's source to write one — only this
document, the ONNX spec, and the topology fingerprint of the board they are
targeting.
