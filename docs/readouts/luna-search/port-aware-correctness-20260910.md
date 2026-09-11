# Port-aware correctness audit and isolated fix

Date: 2026-09-10

This is a local correctness repair only. No native or remote games were run,
and no private source was exported.

## Proven mismatch

The scalar hand-term path uses conversion-aware purchase progress when
`HonestEvaluator.port_aware` is true:

```python
actual_walk = self._walk(state, seat) if walk is None else walk
if self.port_aware:
    progress = port_aware_progress_fast(
        hand, actual_walk, deck_left=len(state.deck), bank=state.bank
    )
    _, spare, risk = hand_terms(
        hand, actual_walk, num_players=state.num_players,
        deck_left=len(state.deck)
    )
    return progress, spare, risk
return hand_terms(
    hand, actual_walk, num_players=state.num_players,
    deck_left=len(state.deck)
)
```

Before this repair, `score_many` always computed raw purchase progress:

```python
chosen = scored.argmax(axis=2)
progress = np.maximum(np.take_along_axis(
    scored, chosen[:, :, None], axis=2
)[:, :, 0], 0.0)
```

The repair applies `port_aware_progress_fast` per row and seat only when the
flag is enabled. It leaves the ordinary `port_aware=False` vectorized path
unchanged.

The evaluate cache key previously ended at:

```python
len(state.deck),
tuple(tuple(hand) for hand in state.hands),
None if belief is None else belief.signature(),
```

The repair appends `tuple(state.bank)` only for port-aware evaluators, because
scalar port-aware scoring reads the mutable bank. The ordinary cache key is
unchanged.

## Reachable tests

The scalar/batch test creates a real random board, places a settlement on a
specific 2:1 wheat port, gives seat 0 `[0, 1, 1, 4, 0]`, and compares every
seat and row from `score_many` with scalar `score` at absolute tolerance
`1e-12`.

The cache test uses the same reachable port state, retains one public belief,
evaluates once, removes ore from the public bank, and evaluates again. The
second value changes, proving bank-sensitive invalidation.

The complete focused suite passed:

```text
18 passed, 2 skipped in 0.26s
```

## Historical provenance limit

`docs/readouts/luna-search/results/08-port-aware.json` contains only
`candidate`, `games`, `seed`, and result rows. It has no source fingerprint.
The frozen Git `HEAD` source also contains no `port_aware` or
`port_progress_fast` implementation. Therefore the exact source used for the
historical port-aware screen (`41/120` versus AB2 and `16/120` versus shipped)
cannot be proven from the archived result. Those results must not be treated as
a test of this corrected implementation.

## Actual hashes

```text
0a9a4b5df187153484c3de15da14995a1938c891328ac3e9a705f2fd55220dce  src/hexset/bots/heximax/search.py
5e1a8415e27c66d2235d81aee3c2297c055707ed49560a565ed77558ea191dad  src/hexset/bots/heximax/evaluate.py
e3f2ec78fee372feeb2ff1dba72fe518c01f72764018d7cd36564e9e01cd3faa  src/hexset/bots/heximax/port_progress_fast.py
dc84fad6d6978514a970d7a18b927caedfed4d276fa7582000f83a1c85931e92  src/hexset/bots/heximax/port_variant.py
6dfea20caabcef722e2fc623281ee5734c33e4e6c2e24ea91c01721b0a3954d3  src/hexset/catanatron/searchcfg.py
bc4a8e67df4a4d5c300a73a3992873abbc56ad7d3634412290a84912dbf12135  src/hexset/bots/heximax/structural_features.py
86b153a5e44cd663789ca33795d818c5dc35d35b233e84ae6557cfdc5b572028  src/hexset/catanatron/structuralcfg.py
c5a2f30a275002adffd983bf83628bf2641212ae446bfeba44adeef35732dadd  src/hexset/catanatron/structural_validation.py
ed805bc8f52c594c526a1c7fdcf92bae6a5bbb221d4440127f3214ca65e83e7e  tests/test_port_progress.py
762219b7c21c5cf83e21621aef286915c9e83fba3e628b719a8d63d275507b7e  tests/test_structural_features.py
```
