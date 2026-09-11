# Cached votes over sampled worlds

`hexset.bots.determinized.determinized` wraps a `Game -> Action | None`
model or search callback. For each decision it samples the mover's `View`,
asks the callback once per distinct sampled-world key, and counts every draw
in the action vote. Three draws of one world contribute three votes even
though they require only one callback call. The cache is discarded when the
decision returns.

This is opt-in. Existing Heximax, network, and Catanatron arena presets retain
their current behavior. Heximax already has its own PIMC search; this helper
does not replace it. Catanatron remains an external reference; the games and
sampled information sets here belong to the native HexSet engine.

## Model callbacks

A runtime-neutral `Policy` can be used without importing Catanatron or a
Colonist adapter:

```python
import random
from hexset.actions import options_for
from hexset.bots.determinized import determinized, holdings_signature
from hexset.game import to_move

# `policy` is your loaded hexset.clients.policy.Policy implementation.
def choose_world(world):
    row = (world, to_move(world), tuple(options_for(world)))
    return policy.act_rows([row])[0]

choose = determinized(
    choose_world,
    worlds=40,
    rng=random.Random(7),
    # Use this only if the policy cannot observe the development deck.
    world_key=holdings_signature,
)
# action = choose(live_game)
```

The callback receives an isolated hypothetical game. Its encoder determines
what it observes: an encoder that reads only the original information set
may give the same answer for every sample. This helper does not turn a
belief-based encoder into a perfect-information encoder.

Wrap the policy callback, not a stateful bot's `choose` method that retains
its last game for trading. For example, `NetworkBot.choose` seats its trade
gate at the supplied game. A driver using this callback alongside that gate
must call `network_bot.seat_at(live_game)` so trading still evaluates the
live position.

## Cache contract and vote settings

- `worlds=0` returns the original callback without drawing randomness.
  Positive `worlds` is the number of samples, not a target count of distinct
  worlds. Sampling still costs time on cache hits.
- The default `world_signature` includes opponents' resources, mature and
  fresh development cards, and the sampled development deck. Public state
  and own holdings stay fixed during this decision.
- `holdings_signature` explicitly ignores the development deck, including
  its composition and order. Use it for a chooser that cannot observe the
  deck, or when deliberately choosing one representative deck per set of
  holdings. A search that consumes development cards may distinguish these
  worlds, so this coarser key is not the default.
- Custom `world_key(state, perspective)` callbacks must return a hashable
  key covering what the chooser observes. These keys omit the fixed root
  context and must not be used for caches shared across decisions.
- Stochastic choosers produce one answer per key. Duplicates reuse that
  answer rather than requesting fresh exploration. This changes the
  semantics of an uncached stochastic ensemble; it is not a free,
  behavior-preserving speedup for every search.
- `temperature=0` puts equal probability on tied winners; positive
  temperature raises vote fractions to `1 / temperature` and normalizes.
  `select="argmax"` chooses the most probable action with ties broken by
  action order. `select="sample"` samples that distribution. Temperature
  only affects selection when sampling.
- `None` answers are cached as abstentions. All abstentions return `None`.
- Sampling and hypothetical chance use separate RNG streams. Callback
  chance consumption does not change the sampled-world sequence. The
  live game's RNG, state and ledger are not consumed or mutated.

## External reference bot

```python
from hexset.catanatron.bot import CatanatronBot

reference = CatanatronBot(worlds=12, rng=random.Random(7))
# Uses the conservative key including the development deck.
# action = reference.choose(live_game)
```

`CatanatronBot` uses the currently pinned Catanatron `Params` API and keeps
its search RNG separate from the sampling RNG. Its default `worlds=0`
continues to search the supplied true state once, preserving the existing
reference entrant's RNG sequence. Determinization is currently a Python
constructor option; the stock arena `catanatron` entrant remains unchanged.

## Recovery provenance and verification

Recovered from Wintermute's clean
`/home/bsm/code/dev-hexset/tmp/HexSet-worlds`, branch
`feat/catanatron-determinized`, commit `34d33f5`. That was an existing HexSet
port based on `0e42c22`, before the Catanatron migration in PR #145.

The originating Clio work is in
`/home/bsm/code/dev-hexset/tmp/HexSet-Clio`, branch `feat/live-interface`:
`f76cba7` (world vote), `96f55ff` (duplicate-world reuse), and `784cff2`
(temperature and sampling). The remaining inspected stream concerns
Colonist capture, reconstruction, trading protocol translation and live
validation; those adapter changes are not part of this port. Both source
checkouts were left untouched.

The recovered implementation is now a generic helper. Compared with the
orphaned port it includes deck content/order in the default key, separates
sampling from callback chance, caches abstentions, validates settings, and
normalizes counts before applying temperature to avoid overflow.

Native, capture-free tests check duplicate vote mass, cache lifetime,
deck-sensitive keys, seeded sampling, public resource/card counts, live
game isolation, and a runtime-neutral policy callback. In the known-holdings
fixture, 40 draws with `holdings_signature` require one model call. That is
a call-count check, not a measured 40x end-to-end speedup or a strength result.
Reference integration tests also exercise actual pinned AB2 and the
existing mirror/RNG regressions. No broad evaluation sweep was run.
