# AB2 state/evaluator and worker-throughput follow-up

Measured September 11, 2026 UTC on Wintermute. All games use
`DC:heximax-notrade,AB:2,AB:2,AB:2`, standard 10-VP/7-discard rules and the
unchanged native 20-second search deadline. The measured source is commit
`6d774bcdb3cf7a6b6c35c7240a457bc0b73ed261`, fingerprint
`8df53df693c6bc279618d775b388d132295f9224870d9a974b79efe28ee7fdd0`.
The pinned image is
`sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`,
with Catanatron `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816` and Python 3.12.14.

## Clean one-CPU full-game comparison

Eight predetermined seeds (`650100000..650100007`), three fresh-process arms
per seed with rotating order. Both optimized arms matched the native arm's
actual game seed, complete ordered action trace, winners and VP on all seeds.
These timings exclude leaf instrumentation and interpreter startup.

| Mode | Wall seconds | CPU seconds | Throughput vs native |
| --- | ---: | ---: | ---: |
| off | 111.000 | 110.954 | 1.000x |
| cached | 54.978 | 54.955 | 2.019x |
| fast | 35.365 | 35.344 | 3.139x |

Fast adds 55.46% aggregate throughput over cached, reducing its wall time by
35.67%. Relative to native, fast cuts wall time by 68.14%. It was faster than
cached on all eight seeds. These are full-game timings with three AB2 opponents,
not timings of an isolated evaluator or an estimate of parallel throughput.

| Seed | Off seconds | Cached seconds | Fast seconds |
| --- | ---: | ---: | ---: |
| 650100000 | 2.386 | 1.313 | 1.024 |
| 650100001 | 12.152 | 5.888 | 3.849 |
| 650100002 | 9.446 | 4.826 | 3.063 |
| 650100003 | 16.585 | 7.934 | 5.065 |
| 650100004 | 19.034 | 9.249 | 6.339 |
| 650100005 | 13.526 | 6.448 | 4.190 |
| 650100006 | 21.762 | 11.636 | 7.148 |
| 650100007 | 16.108 | 7.685 | 4.686 |

Fast recorded 463,363 board-feature hits and 19,032 misses (96.05% hits), versus
460,326 hits and 22,069 misses (95.43%) for cached. These are feature-cache
calls, not whole-search transpositions. Fast also removes the remaining state
pickle round trip and extracts only the evaluated player's resource hand.

## Independent score/search audit

Two disjoint seeds (`650000000..650000001`) compare cached and fast with leaf
instrumentation. All 103,317 leaf values match the original native scalar
evaluator exactly. Both arms match leaf counts, ordered score digests, full
action traces and outcomes. No nonterminal depth-positive deadline cutoff was
observed. Instrumented audit timings are excluded from performance claims.

## Measured 30-worker batch throughput

Each trial replays the same 240 predetermined seeds (`650200000..650200239`)
in a fresh parent interpreter and worker pool, bounded to 30 CPUs. Trials run
serially, with profile order reversed on repetition two. The 1,440 game plays
therefore represent 240 unique seeds, not 1,440 independent samples. All six
trials matched every actual game seed, full ordered action trace, winner and VP.

| Profile | First seconds | Second seconds | Mean seconds | Games/second | Worker occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| cached:static | 184.539 | 185.287 | 184.913 | 1.298 | 79.96% |
| fast:static | 120.011 | 123.430 | 121.720 | 1.972 | 80.46% |
| fast:dynamic | 108.944 | 109.363 | 109.153 | 2.199 | 92.67% |

On aggregate, fast/static adds 51.92% throughput over cached/static.
Dynamic scheduling adds another 11.51% over fast/static. Combined,
fast/dynamic delivers 1.694x throughput (69.41% more games/second)
and cuts batch wall time by 40.97% versus cached/static.

These timings include pool startup, per-game trace hashing/JSON recording and
pool shutdown. Worker occupancy is summed task wall time divided by 30 times
batch wall time; it measures occupied worker capacity, not OS CPU utilization.
The same recording runs in every profile. Per-game scheduling adds some setup
and recording overhead but shortens the idle tail. Native/off was measured only
in the separate one-CPU comparison; no native 30-worker speedup is inferred by
multiplying ratios from different benchmarks.

## Validation and limitations

The local full suite passed 850 tests with 14 optional skips. Integration CI
passed 879 tests with 8 skips; core CI passed on Python 3.11 and 3.13. A real
local two-worker forkserver run also matched all game traces across the three
throughput profiles.

Native search remains time-limited. Matching finite samples do not guarantee
policy equivalence for every deadline-limited position. Acceleration remains
explicitly selected with `--catanatron-speedups=fast`; the reference default is
`off`. Dynamic scheduling is the duel default, with static available explicitly.
The existing evolve/port discovery loops already assign games dynamically; the
scheduler improvement applies to the duel/confirmation path.

## Reproduce and audit

Run with the pinned image and source, `PYTHONHASHSEED=0` and all BLAS/OpenMP
thread limits set to one. Use one CPU for the first two commands and 30 for
the third; run the commands sequentially with fresh output paths.

```sh
python -m hexset.catanatron.speed_benchmark --modes cached fast --audit \
  --games 2 --seed 650000000 --out /out/audit.json
python -m hexset.catanatron.speed_benchmark --modes off cached fast \
  --games 8 --seed 650100000 --out /out/clean.json
python -m hexset.catanatron.throughput_benchmark --games 240 --workers 30 \
  --repeats 2 --seed 650200000 --out /out/throughput.json
```

The JSON summaries contain every game record. Compact runtime receipts include
image identity, mounts, selected environment, CPU limits, command and exit state.

- [audit-runtime.json](audit-runtime.json) — SHA-256 `ba8b408a7b8a5a453904ba4be4d1d898729bf8472716a8083259d90c54e26c36`
- [audit.json](audit.json) — SHA-256 `c94cc2280b4f026a1d7d22e1897d1e79ceb44cbe08036cf6362804c6f22c9608`
- [clean-runtime.json](clean-runtime.json) — SHA-256 `3935481e30edd7980675fed4efa49260225e41b5df20dc587c2a3c51c4c76bd0`
- [clean.json](clean.json) — SHA-256 `b660abff51a7b2127014a12a11bce06ec4fe6d291f84b6e9cb5452810180c408`
- [throughput-runtime.json](throughput-runtime.json) — SHA-256 `c2e7e5aa7d14c0ac4ba137daabf905e8b8ca54cc1aab228a20c0c92febf09cb0`
- [throughput.json](throughput.json) — SHA-256 `f9780c251ea4f185b4de6d788e781c28207d32e154ffaff790b1773061bb1fc9`
