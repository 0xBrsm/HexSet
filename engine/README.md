# hexset

> **Provenance.** This directory's history was imported into the HexSet
> repo (`feat/engine-in-repo`) from `0xBrsm/dev-HexNet` (becoming HexNet),
> branch `refactor/hexnet-package-split` at commit
> `665ecb9da0a51f6c956e90282651bd0265f0a3e6`. The import used `git
> filter-repo` to keep only `src/hexset/`, `src/benchmarks/` and the
> torch-free slice of `src/tests/`, stripped of the `src/` prefix (tip
> `5210f5b862152eb5e1db41b9689c9a98531fb443`), merged in with `git subtree
> add --prefix=engine` so every original commit and its authorship survive
> under `engine/` rather than arriving as one squashed import. `heximax`
> split out of `hexset` into its own top-level package afterwards, in this
> repo, via `git mv` (see [`CHANGELOG.md`](../CHANGELOG.md)'s "Unreleased" entry).

A full-rules four-player engine and a self-play agent that reads the board as a
graph, implementing the classic ruleset published as *Settlers of Catan* — see
[Trademarks](#trademarks) below. Hexes, vertices (settlement sites) and edges (road
sites) are nodes of a heterogeneous graph; a message-passing network runs over that
graph and emits its policy as a per-node readout, so action legality is a node
property and masking falls out of the board rather than being bolted on. Published
agents for this game instead flatten the hex board onto a rectangular grid so an
ordinary CNN can approximate hex adjacency — this works on the real adjacency.

The `hexset` package ships the rules engine, handcrafted evaluations and search
bots, a seat-balanced arena for measuring one against another, the ledger of
public knowledge a policy may honestly read, and the observation encoder. It
depends on nothing but numpy. The learning layer — model, PPO self-play, batched
MCTS, expert iteration and the resumable training loop — is the sibling `hexnet`
package: split out so the engine, bots and ledger stay installable (and
testable) on a machine without PyTorch, with `hexset` never importing `hexnet`
in the other direction. `hexset.arena` exposes a small registry
(`register_entrant_kind`) that `hexnet.netbot` populates at import, which is
how a duel can seat a trained checkpoint without the arena itself needing torch.

## Quick Start

Everything runs as a module from this directory, so there is nothing to install
first beyond the dependencies.

```bash
cd src
python -m pytest tests -q

# how fast the engine plays random games
python -m benchmarks.throughput --games 200 --workers 4

# a seat-balanced duel with Wilson intervals
python -m benchmarks.baselines --lineup search2 search2 greedy greedy --games 400

# PPO self-play, resumable (hexnet, needs PyTorch)
python -m hexnet.train --lanes 128 --iterations 100 --checkpoint-dir runs/ppo
```

Requires Python 3.11+. `hexset` — the rules engine, the arena and the self-play
collector — needs only numpy, which is the one declared dependency; `hexnet`
needs PyTorch, provisioned separately (the training/GPU image) rather than a
hard dependency of this distribution, so `hexset` can be tested on a machine
where torch cannot be installed at all. `docker/Dockerfile` builds the ROCm
training image.

## Any layout, no new code

`board.topology` derives vertices, edges and adjacency from a bare set of hex
coordinates. A vertex is keyed by the three hex positions touching it, so the key is
canonical regardless of which hex reaches it, and nothing in the rules layer knows
the shape of the board: placement legality, longest road, coastlines and ports are
all graph queries. The base board, small boards, and Seafarers-style multi-island
layouts (`board.maps.islands`) go through one code path, with sea and gold terrain
already in the type system. Disconnected and touching islands are tested, as is
Euler's characteristic across radii 0–4. Because a graph model shares weights across
nodes, a bigger board costs no extra parameters either.

## Bots, and how they are measured

`bots.Bot` is the protocol a network also satisfies. `greedy` is one ply over a
handcrafted per-seat evaluation; `SearchBot` is max^n over decisions with dice
expanded across all eleven outcomes and weighted rather than sampled. This game has
four players and one winner, so every evaluation and the value head return one
number per seat, and a *stance* says how a seat collapses that vector: `own` is plain max^n,
`relative` subtracts the mean of the other seats, `paranoid` the largest. Player-to-
player trading is implemented and on, which published agents generally disable.

`arena` rotates a lineup so every entrant sits every seat the same number of times,
refuses a game count the rotation cannot balance, reports Wilson intervals, and fans
out over a process pool with results identical at any worker count. Entrants are
frozen picklable descriptions, so a lineup goes into a run manifest verbatim. A
checkpoint enters as `network:<path>`, or as the leaf evaluation of the ordinary
search with `netsearch:<path>` / `netgreedy:<path>`, or as a tree with `mcts:<path>`.

## Catanatron adapter (optional)

`hexset/catanatron/` plugs this engine into [Catanatron](https://github.com/bcollazo/catanatron)'s
arena as an external benchmark, in both directions: any `hexset` bot can be seated as
a Catanatron `Player` (`DC:<entrant>`, e.g. `DC:search2-offers3` or
`DC:network:<checkpoint>.pt`), and any Catanatron bot can be seated in a `hexset`
lineup the same way. Catanatron's `Player` interface is the adapter surface —
`hexset.catanatron.player.DevCatanPlayer` re-translates a live `catanatron.Game` into
a `hexset.Game`/`GameState` fresh on every decision (`state.py`), rather than
mirroring the two engines' turn machines incrementally. Our engine, ledger, and
trading are unchanged on this path; the translation is one-way and stateless, so an
honest bot seated through the bridge sees a *memoryless* public ledger — nothing
certified by type, the whole opponent hand counted as unknown — a deliberate lower
bound on what an information-set-honest bot could know there, not a limitation of
the ledger itself (see `state.py`'s `translate` docstring). `python -m
hexset.catanatron.duel` shards a duel across worker processes and stamps
catanatron's resolved commit, seed, worker count, and `PYTHONHASHSEED` on every
report — see `_ensure_pythonhashseed_zero`'s docstring for why the last one matters
(catanatron's own robber-move tie-break resolves via hash-order-sensitive set
iteration, so a reproducibility check across two launches needs it pinned).
Install with `pip install -e "src[catanatron]"`; catanatron is a real dependency of
this extra, never vendored, and no catanatron import reaches the base package. See
`agents/reference/bridge-container.md` for the containerized recipe.

## Source Layout

`hexset` (engine, bots, ledger — numpy only) and `hexnet` (PPO/training
research — needs PyTorch) are two packages in this one `src/` tree, split so
the former can be extracted and depended on as a plain package without the
latter. `hexset` never imports `hexnet`.

| Path | Purpose |
|------|---------|
| `hexset/board/` | `coords` (cube hex), `topology` (vertices, edges, adjacency from any layout), `terrain`, `board` (tokens, setup bags), `ports` (coastlines), `maps` (base, mini, multi-island) |
| `hexset/state.py`, `economy.py` | Occupancy, hands, bank stock, placement legality as graph queries; costs, payment, production, port rates |
| `hexset/game.py`, `actions.py` | The turn and phase machine, and a flat action space sized from the board with legality masking |
| `hexset/roads.py`, `robber.py`, `devcards.py`, `cards.py`, `victory.py`, `trading.py`, `ledger.py` | Longest road as a longest trail, the robber and discards, the development deck, victory conditions, player-to-player offers, and the public-knowledge ledger an honest policy reads instead of the true hidden state |
| `hexset/evaluate.py`, `evaluate_tiered.py` | Two handcrafted per-seat evaluations — nine fitted blended terms, and a tiered priority-order reimplementation kept as a comparison baseline |
| `hexset/bots.py`, `arena.py` | The `Bot` protocol, random / greedy / max^n search with stances; seat-rotated head-to-head play with confidence intervals and the `register_entrant_kind`/`register_preset` registry `hexnet.netbot` and `heximax` populate |
| `heximax/` | A sibling top-level package, not part of `hexset`: the PIMC handcrafted bot that reads its own belief through the ledger, registering the "heximax"/"heximax-omni"/"heximax-notrade" presets and evaluators with `hexset.arena`/`hexset.tuning` on import rather than being imported by either |
| `hexset/tuning.py`, `fitting.py`, `behaviour.py` | Fitting evaluation weights by hill climbing and by logistic regression, and reporting what the bot actually does in the shape published aggregates are quoted in |
| `hexset/encoding.py`, `onnx_record.py` | The seat-relative, information-set-correct graph observation, and the torch-free information-set record contract a served checkpoint (or the gym) reads instead of reimplementing the encoder — see `onnx_record.py`'s own docstring |
| `hexset/mcts.py`, `record.py`, `dataset.py`, `play.py` | PUCT with leaves gathered into waves, backing up a per-seat vector (evaluator supplied by the caller — `hexnet.netbot.LeafEvaluator` in training); replayable game records, labelled positions from them, and a random player |
| `hexset/catanatron/` | Optional adapter to Catanatron's arena (`pip install -e "src[catanatron]"`) — see "Catanatron adapter" above |
| `hexnet/model.py`, `readout.py` | Message passing over the graph observation, and the index map from heads to flat action slots |
| `hexnet/selfplay.py`, `policy.py`, `rewards.py` | Vectorised lockstep rollout collection behind a `BatchPolicy` protocol, the torch policy, and the per-seat terminal scalarisation |
| `hexnet/ppo.py`, `train.py`, `league.py`, `schedule.py` | GAE, clipped surrogate and value loss; the runnable, resumable training loop; the self-play ladder; learning-rate schedules |
| `hexnet/expert.py`, `distill.py`, `distill_train.py` | Expert iteration through the existing collector, and distilling a search's target back into the policy |
| `hexnet/netbot.py`, `export_onnx.py`, `migrate.py`, `widen.py`, `ddp.py` | A checkpoint as an arena entrant (registers with `hexset.arena`); the traced encoder (`RecordEncoder`) and ONNX export reading `hexset.onnx_record`'s contract; checkpoint migration and function-preserving widening; multi-GPU update |
| `hexnet/run/` | `run.init`/`run.manifest`: freezing a run's full parameter set before launch, and reading it back |
| `benchmarks/` | hexset-side: throughput, baselines, ablations, weight fitting, encoder cost, duels, human agreement, AIVAT |
| `hexnet/benchmarks/` | hexnet-side: forward-pass and rollout cost, value-head diagnostics, ranking probes, training-loop profiling |
| `tests/` | 607 tests across 31 files (`hexset`, torch-free); `tests/hexnet/` carries the `hexnet`-specific tests alongside their modules, `tests/catanatron/` is skipped unless the `catanatron` extra is installed |
| `docker/` | ROCm training image |

## Design Notes

- **Information-set correct by construction.** The encoder rotates seats so the
  player to move is always seat 0, and encodes only what that player may legally
  know: own hand and development cards exactly, opponents as counts. Nothing
  downstream can read a hidden card.
- **Records outlive encoders.** A record stores the board and the action sequence,
  not features, so a dataset survives every change to the encoding. Dice and steals
  come back from the seeded stream, which turns a change in how randomness is drawn
  into a replay mismatch rather than into quietly wrong training data.
- **Reproducible runs.** A game is a pure function of the seed and its index, and the
  game counter is checkpointed with the weights, so a resumed run continues its
  training set instead of replaying it. Checkpoints are written to a temporary file
  and renamed.
- **Vector value, not scalar.** The value head emits one number per seat and every
  output is trained from every position, which is what lets the search back it up
  with max^n instead of a minimax sign flip.

## Trademarks

CATAN and SETTLERS OF CATAN are trademarks of Catan GmbH and Catan Studio. This
project is not affiliated with, endorsed by, or sponsored by either, and it ships
no Catan artwork, text, or other content. Those names appear here only to identify
which game's rules this implements — nominative use, not a claim on the marks.
hexset is the name of this software.
