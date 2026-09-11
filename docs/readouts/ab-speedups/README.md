# AB2 acceleration measurement

Measured September 11, 2026 UTC on Wintermute, one CPU per container and one
serial game worker. Every game used `DC:heximax-notrade,AB:2,AB:2,AB:2`, standard
10-VP/7-discard rules, no domestic trading, and the native 20-second deadline.
The source is commit `af44fbec2a7e212b84ed598457ce7472f8b5939d`, source fingerprint
`21b9b7f3c156b23b06ad6fb831fc70545c5de104a8726063cdfaf01334975f69`.
The pinned image is `sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`.

## Clean full-game timings

Eight predetermined seeds (`630100000..630100007`), three fresh-process arms
per seed with rotating order. Both optimized arms matched the baseline's actual
game seed, complete ordered action trace, winners and VP on all eight seeds.

| Mode | Total wall seconds | Speedup | Time reduction |
| --- | ---: | ---: | ---: |
| off | 103.039 | 1.000x | — |
| basic | 55.901 | 1.843x | 45.75% |
| cached | 52.186 | 1.974x | 49.35% |

CPU-time ratios independently agree: 1.843x basic and 1.974x cached. Basic was
faster than baseline on all eight seeds; cached was faster than basic on seven.
Cached adds 7.12% aggregate throughput over basic in this sample.

| Seed | Off seconds | Basic seconds | Cached seconds |
| --- | ---: | ---: | ---: |
| 630100000 | 13.537 | 7.246 | 6.917 |
| 630100001 | 12.353 | 6.887 | 5.756 |
| 630100002 | 10.495 | 5.930 | 5.188 |
| 630100003 | 5.517 | 3.017 | 2.509 |
| 630100004 | 12.135 | 6.567 | 5.787 |
| 630100005 | 9.611 | 5.282 | 4.616 |
| 630100006 | 12.360 | 6.869 | 6.820 |
| 630100007 | 27.032 | 14.103 | 14.593 |

The cached arm recorded 451,381 board-feature hits and 19,697 misses
(95.82% hits), with no more than 4096 retained entries. These
are board-feature calls, not whole-tree transposition hits.

## Separate score/search audit

Two disjoint seeds (`630000000..630000001`) ran all three modes with leaf
instrumentation. Each optimized arm matched all 88,126 leaf values, ordered
score digests, leaf counts, full action traces and outcomes. Every instrumented
leaf was also checked directly against the original scalar evaluator. No
nonterminal depth-positive deadline cutoff was observed in those audits.
Audit timings are excluded from the speed claim.

## Validation and limits

The full local suite passed 831 tests (14 optional skips); GitHub integration
CI passed 860 tests (8 skips), and core CI passed on Python 3.11 and 3.13.
A real two-worker CLI smoke passed with cached mode reported explicitly.

This is eight paired games on one CPU, not a measured 30-worker campaign
throughput gain. Clean runs have no leaf/deadline instrumentation. Native
search remains time-limited, so these matching traces do not guarantee policy
equivalence for every deadline-limited position. The reference default stays
`off`; accelerated runs are explicitly labeled `basic` or `cached`.

## Artifacts

The JSON summaries retain every child record; compact Docker receipts record
the exact image, read-only source mount, one-CPU limit, environment and exit.

- [audit-runtime.json](audit-runtime.json) — SHA-256 `15eb6001c5a4aeb6ff52c7ca85245eb533aec61721992da47a8d7458f3a80d29`
- [audit.json](audit.json) — SHA-256 `65542734160bb1ff187cc21c808b8761004de974e747a31c81da88355b38979e`
- [clean-runtime.json](clean-runtime.json) — SHA-256 `5feed02ea2cd64a8d516ae59def63362fbabf7fcffddedf40adef051c6429197`
- [clean.json](clean.json) — SHA-256 `33926e0f5842d735fd25e0dc2e81f00bc82aee72d3978a389358e4ea5110b5e2`
