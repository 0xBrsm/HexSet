# Native game lanes

`hexset.native` supplies an optional C++ state backend for no-trade games with
two or four seats. Installation builds it when a C++17 compiler is available.
`available()` reports whether it was built; `supported(game, contract)` checks
the imported state and observation contract before a caller chooses its
backend. Other configurations continue through the Python engine. A native
lane never switches engines after play begins.

For a source checkout, build the extension once with the same interpreter that
starts the workers, before launching them:

```sh
python -m pip install --no-deps -e .
python -c 'from hexset.gym.native import available; assert available()'
```

Run these commands at the package root with a C++17 compiler installed.
Editable installation places the extension beside the mounted Python source.
The build is optional, so verify availability explicitly when native lanes are
required; an installation without a working compiler still supports Python play.

Installation builds two optional extensions. `hexset._road_native` answers
the road-graph queries of `hexset.roads` (longest road, roads to take,
reach) from packed integer adjacency, with no game state crossing the
boundary; without it `hexset.roads` answers them in Python, with the same
results. `hexset._native_core` is the game-state backend this page
describes.

`NativeState(game)` owns its state, common and seat-specific ledgers, and a copy
of the game's existing Chance streams. It preserves the original Chance
callbacks and their Recording events. `encode9(state, perspective, contract)`
returns a fresh writable float32 observation for that perspective. Supported
contracts are `9`, `9.1`, and the two-seat `9d`, `9d.1` and `11d`, each byte for byte
as `hexset.contract9` encodes it; referee snapshots and true victory points
are separate from visible observations. `encode8(state, perspective,
contract)` returns the contract-8 (or 8.1) observation, byte for byte as
`hexset.encoding.encode` writes it: its hexes, vertices, edges and globals
flattened and packed in that order. An opponent's expected hand is
`hexset.view.View.expected_counts` exactly: where the seat's upper bounds
bind, every split of its hidden cards is weighed by its hypergeometric
chance in 128-bit integers and each mean is the correctly rounded quotient,
as Python's integer division gives it.

## The driver

Most callers do not use the classes below directly.
`hexset.gym.driver.Driver(env, seats, keep=..., lanes="python" | "native" |
"auto")` plays a `LaneEnv`'s games on either lane kind with one `Request`
type, and imports each dealt game into a native lane when `lanes="native"`
([batched games](training.md#one-loop-over-python-and-native-lanes)). The rest
of this page is the bridge it drives.

## Playout copies

Paired rollouts (`hexset.rollout`) hold each world's copy natively where they
can (`hexset.native.playout.NativeCopy`): the extension is built, the
searching policy reads one of the five contracts above (`native_contract`)
over every dealt seat, or reads a two-seat contract (9d, 9d.1 or 11d) at a
table with two live seats, and the position is a supported no-trade one. A table dealt four with
two seats retired is such a table: the copy counts its live seats alone, the
perspective first (`hexset.encoding.seat_frame`), as the Python engine does.
A copy answers what the Python engine answers for the same position: the same
legal moves, the same decision tensors and the same observation bytes,
drawing chance from a copy of the same streams. A position the native engine
refuses keeps its Python copy. `HEXSET_NATIVE_PLAYOUTS=0` in the environment
keeps every copy a Python one, to compare the two engines on the same games.

`NativeCopy.viewing(state, topology, dealt, locked)` is a read-only view of
a native lane's own state rather than a copy: what an atomic seat on a
native lane reads (`observation`, `observation8`, `tensors`, `frame`,
`legal`). `hexset.encoding.encode_batch` encodes such a position through its
`observation8`.

Batched requests (below) count every dealt seat, so they take the two-seat
contracts at a two-seat table only.

## Batched requests

`hexset.gym.native.NativeLane.import_game(game, *, index, seed, tape,
action_cap, cast=())` imports a recorded fresh stream with an empty `Tape`.
The caller owns dealing and whole-game resume: already retained games are
excluded before import. The lane retains no Python `Game` after import.
Trading gates, unresolved deals, hidden referee state and unsupported Chance
implementations are refused before the native lane is created.

`NativeBatch(lanes, *, space, pairs, contract, planned_indices, keep,
on_decision=None, discard_order="seat", finished_indices=(), atomic=())`
exposes one decision per active lane. A seat whose policy is in `atomic` is
asked one atomic move at a time: its request holds the legal `actions`
(`options`) in `hexset.actions.legal_actions`' order and a `view` of the lane
(`NativeCopy.viewing`), and no observation or decision tensors. `pairs` is the original topology's
`contract9.incidences`; setup coordinates remain Python integers. The caller
owns policy batching, sampling, draw keys, and when retired lanes are refilled.

`requests()` returns an immutable tuple of `NativeRequest` objects. Each has
an observation, a `decisions.Decision`, an epoch and seven identifiers:
game index, lane, seat, atomic step, policy, decision id and part. Its numeric
arrays are read-only views of owned immutable bytes and remain valid after
later requests. Requests contain neither a `Game` nor private record targets.
`targets()` separately returns referee-only points and turn counts for the
recording caller; these must never be supplied to a deciding seat.

`advance(answers)` accepts one object with `action` and
`decision=(DecisionKind, candidate_index)` for each request, in request order;
an atomic request's answer is one of its actions, with `decision` None.
Every answer is validated before any lane changes. Compounds apply one atomic
action at a time, with the action cap checked after each part. The optional
`on_decision(request, answer)` callback runs once, after the decision's first
successful atomic action and Tape append; it receives the original answer.

## Completion and failure

The mandatory `keep(completion)` callback receives a `NativeCompletion` with
the game index, seed, players, cast, `Outcome`, and complete `Record`. It must
durably retain the whole record before returning. Only then is the lane marked
finished. A non-`None` callback result is returned by `advance`; otherwise it
returns the completion. `replace_lane(slot, lane)` accepts a distinct missing
index from the complete plan, after the finished lane's request epoch has
closed.

An apply or callback error poisons the batch. It must be abandoned: there is
no retry or fallback that could repeat an applied move, a Chance draw, or a
record write. Invalid answers rejected before mutation leave the outstanding
requests available for correction. Imported state must remain exclusively
owned by the batch while it is running; a changed native generation is
rejected before applying an answer.
