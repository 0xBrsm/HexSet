# AB:2 bridge re-read — heximax-notrade through the collapsed adapter

Re-reads `heximax-notrade` against three Catanatron `AB:2` players through
`src/hexset/catanatron/duel.py` (the collapsed `hexset.catanatron` adapter;
the old `catanatron_bridge` image is legacy) after the structural pass, the
omniscience fix and the latest speedups — all claimed census-identical, so
this number was expected to reproduce the prior record within noise.

`DC:heximax-notrade,AB:2,AB:2,AB:2`, seed 7, 500 games, `--workers=8`
(duel.py's own documented default and example). catanatron pinned to
`d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816` per `pyproject.toml`'s
`catanatron` extra, confirmed via the installed dist's `direct_url.json`.
`PYTHONHASHSEED=0` pinned before the interpreter starts.

## Result — seed 7, 500 games

| | |
|---|---|
| wins | **146 / 500 = 29.2%** |
| Wilson 95% | [25.4%, 33.3%] |
| fair share | 25% (4-seat table) |
| avg VP, heximax-notrade | 7.02 |
| avg VP, AB:2 (mean of 3 seats) | 6.66 (6.56 / 6.81 / 6.60) |
| wall time | 2776.9s (46.3 min), 0.18 games/sec |
| provenance | `catanatron 3.3.0 @ d3f4ad05bb78 \| PYTHONHASHSEED=0 \| seed 7 \| workers 8 \| 8 shards of 63, seeds 7-14` |

Raw runner output: `seed7-500games.txt`.

Seed 8 was not run: at the measured per-game rate (44.1s/game/worker,
read directly off this run's wall time and shard size), even 12 workers
projects to ~31 minutes for 500 games — over the 20-minute budget set for
that leg, so it was skipped rather than started and left unfinished.

## Does it reproduce 160/500?

**No — 146/500, 14 games short of the 160/500 recorded as reproducing
bit-for-bit twice before.** 29.2% sits inside this run's own Wilson interval
alongside 32.0%, but that interval describes sampling uncertainty over the
space of possible games at a *given* configuration — it is not the right
tool for judging a determinism check between two runs that are each supposed
to be a single fixed, reproducible outcome. A 14-game gap between two
notionally-deterministic replays of the same seed is a real discrepancy to
explain, not noise to wave at.

Two candidate explanations, weighed rather than guessed:

**1. A worker-count / shard-seeding mismatch (more likely).** `run_duel`
derives one RNG seed per *shard* as `seed + i`, and the shard count is
`ceil(num_games / workers)` — so `--seed` alone does not pin the games
played; `--seed` **and** `--workers` together do. `duel.py`'s own
`provenance()` docstring flags exactly this risk ("a 500-game pair measured
this way differed by 4 points on one checkpoint" from a workers mismatch
alone). This run used `--workers=8`, matching `duel.py`'s module docstring
example and the README's own documented invocation — but nothing committed
in this public repo (docs/, git history of `duel.py`, the test suite) pins
the workers count the *original* `heximax-notrade` vs `AB:2` record used.
That record isn't reproduced anywhere in this repository, so the comparison
here is against a number this read could not independently verify the
invocation of. If the original run used a different `--workers`, a
double-digit swing on 500 games is exactly the documented failure mode.

**2. A real behaviour change slipping through the "census-identical" claim
(less likely, not ruled out).** The structural pass, omniscience fix and
last night's speedups all live in `hexset.bots` / `hexset.trading`, not in
`hexset/catanatron/`, which is untouched by this read (read-only checkout,
no source files changed to produce this number). If any of those changes
altered a choice despite being verified byte-identical against the internal
choice census, it would show up here as exactly this kind of swing. Nothing
in this read isolates that possibility from (1) — doing so needs either the
original run's `--workers` value or a byte-level choice diff against a
pre-speedup checkout at matching `--workers`, neither of which was available
in the time box.

**On priors, (1) is the likelier cause**: it is a known, previously-observed
failure mode with a specific mechanism and a citation in this repo's own
code, whereas (2) requires the census-identity claims for three separate
changes to all be wrong at once. But this read cannot fully close the
question without the original invocation's worker count.

## A gap found along the way (not a source change)

`hexset.catanatron.duel.main()` never imports `hexset.bots` (or
`hexset.bots.heximax`), so `DC:heximax-notrade` (and any other heximax-family
entrant) fails inside the worker pool with `KeyError: 'heximax-notrade'` when
run via the bare documented command
(`python -m hexset.catanatron.duel --players=DC:heximax-notrade,...`).
`search2`/`search2-notrade` don't hit this because they're module-level
entries in `hexset.arena.PRESETS`; heximax's presets are registered by
`hexset.bots.heximax` at import time, and nothing on the `DC:` player path
(`duel.py`, `player.py`, `register.py`) imports that module. The repo's own
test suite (`tests/catanatron/test_catanatron_duel.py`) only exercises
`search2`/`search2-notrade` DC entrants, so this gap has no test coverage
either.

Worked around here without touching `src/`: `run_ab2_duel.py` (this
directory) imports `hexset.bots` in the parent process before calling
`run_duel` — `multiprocessing.Pool`'s fork start method (the Linux default)
copies the now-populated `hexset.arena.PRESETS` into every worker — and
relies on `PYTHONHASHSEED=0` already being set in the shell environment
before Python starts, so `_ensure_pythonhashseed_zero()`'s re-exec (which
would discard the pre-import by replacing the process) is a no-op. See the
script's own docstring for the full reasoning. Command used:

```
PYTHONHASHSEED=0 python run_ab2_duel.py \
    --players=DC:heximax-notrade,AB:2,AB:2,AB:2 --num=500 --workers=8 --seed=7
```

This is a real fix candidate for `hexset.catanatron.duel.py` itself (one
`import hexset.bots` line), left unfixed here per the read-only scope of
this task.
