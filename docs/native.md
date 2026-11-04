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

`NativeState(game)` owns its state, common and seat-specific ledgers, and a copy
of the game's existing Chance streams. It preserves the original Chance
callbacks and their Recording events. `encode9(state, perspective, contract)`
returns a fresh writable float32 observation for that perspective. Supported
contracts are `9`, `9.1`, and the two-seat `9d`; referee snapshots and true
victory points are separate from visible observations.

## Batched requests

`hexset.gym.native.NativeLane.import_game(game, *, index, seed, tape,
action_cap, cast=())` imports a recorded fresh stream with an empty `Tape`.
The caller owns dealing and whole-game resume: already retained games are
excluded before import. The lane retains no Python `Game` after import.
Trading gates, unresolved deals, hidden referee state and unsupported Chance
implementations are refused before the native lane is created.

`NativeBatch(lanes, *, space, pairs, contract, planned_indices, keep,
on_decision=None, discard_order="seat", finished_indices=())` exposes one
decision per active lane. `pairs` is the original topology's
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
`decision=(DecisionKind, candidate_index)` for each request, in request order.
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
