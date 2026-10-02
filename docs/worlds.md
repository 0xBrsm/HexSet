# Cached votes over sampled worlds

`hexset.bots.determinized.determinized(choose, worlds, rng)` wraps a
`Game -> Action | None` model or search callback. For each decision it draws
`worlds` samples from the mover's `View`, calls the callback once per
distinct sampled-world key, and counts every draw in the action vote: three
draws of one world give three votes for one callback call. The cache is
discarded when the decision returns.

No entrant uses it by default. `catanatron:worlds=<n>` wraps the Catanatron
reference in it; a search with its own sampling, such as `hexset.mcts.Search`,
does not need it.

`determinized` takes no `HoldReading`, so every unseen development card is
dealt alike. The fold beneath it, `distinct_worlds(belief, rng, draws,
world_key, hold)`, returns `(share, state)` pairs and takes a
`hexset.view.HoldReading` as `hold` for a caller with its own reading of held
cards.

## Model callbacks

A runtime-neutral `Policy` works without Catanatron or any other adapter:

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
    # Only for a policy that cannot observe the development deck.
    world_key=holdings_signature,
)
# action = choose(live_game)
```

The callback receives an isolated hypothetical game and may mutate it. Its
encoder decides what it observes: an encoder that reads only the original
information set gives the same answer for every sample, and wrapping it does
not make it a perfect-information encoder.

Wrap the policy callback, not the `choose` method of a stateful bot that
keeps its last game for trading. `NetworkBot.choose` seats its trade gate at
the game it is given, so it would be seated at a sampled world; a driver
using the callback beside such a gate calls `network_bot.seat_at(live_game)`
so trading evaluates the live position.

## Cache contract and vote settings

- `worlds=0` returns the callback unchanged and draws no randomness.
  Positive `worlds` is the number of samples, not a count of distinct
  worlds. Sampling costs time on cache hits too.
- The default key, `world_signature`, covers opponents' resources, mature
  and fresh development cards, and the sampled development deck in order.
  Public state and the mover's own holdings are fixed within a decision.
- `holdings_signature` ignores the development deck, its composition and
  order. Use it for a chooser that cannot observe the deck, or to keep one
  representative deck per set of holdings. A search that draws development
  cards can tell those worlds apart, so it is not the default.
- A custom `world_key(state, perspective)` returns a hashable key covering
  everything the chooser observes. Keys omit the fixed root context, so they
  are not valid for a cache shared across decisions.
- A stochastic chooser answers once per key, and duplicates reuse that
  answer. This changes the semantics of an uncached stochastic ensemble; it
  is not a behaviour-preserving speedup for every search.
- `temperature=0` (default) splits probability equally among the tied
  winners; positive `temperature` raises vote shares to `1 / temperature`
  and normalises. `select="argmax"` (default) takes the most probable
  action, ties broken by action order; `select="sample"` samples the
  distribution with `rng`. Temperature changes the choice only under
  `"sample"`.
- `None` answers are abstentions; if every world abstains the result is
  `None`.
- Sampling and the hypothetical games' chance use separate streams: chance
  the callback consumes does not change the sequence of sampled worlds. The
  live game's generator, state and ledger are not consumed or mutated.

## External reference bot

```python
import random

from hexset.catanatron.bot import CatanatronBot

reference = CatanatronBot(worlds=12, rng=random.Random(7))
# action = reference.choose(live_game)
```

`CatanatronBot(player=None, *, rng=None, worlds=0, temperature=0.0,
select="argmax", world_key=world_signature)` seats a Catanatron player,
depth-2 alpha-beta by default (`alpha_beta(depth)`), on the pinned
Catanatron `Params` API. Its search draws from `rng`, and world sampling
from a copy of it, so sampling never consumes the search stream. At
`worlds=0` it searches the supplied true state once: the mirror holds every
seat's true hand and development cards (the omniscient read). `worlds>0` is
the information-set read, one mirror per world sampled from the mover's
`View`, keyed by `world_key`. Zero is the default because that is how
Catanatron's own players play. The arena entrant reads the same settings
from its spec: `catanatron:worlds=12:temperature=1:select=sample`
([guide.md](guide.md#bundled-opponents)).
