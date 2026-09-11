# Timing observations

The matched native preflight used four workers in both off and fast modes.
Its four AB2 matchups took 37.215 summed worker-wall seconds without the
speedups and 12.763 with them (2.916x). All complete action traces matched.
This is a small sanity sample, not a new large performance benchmark. The
four shipped-only games varied from 5.167 to 4.437 seconds; AB2 patches do not
act on that matchup, so that difference is timing variability.

Completed full confirmation batches, each 512 AB2 plus 512 shipped games:

| Candidate | Workers | Wall seconds | Observed games/second |
|---|---:|---:|---:|
| p10-exp025 | 16 | 381.217 | 2.686 |
| road-zero-exp025 | 16 | 364.330 | 2.811 |

These are actual batch throughputs, not isolated speedup comparisons: seeds,
policies, game lengths and opponent mixes affect timing. Sixteen workers was
a conservative resource choice, not a measured optimum. Following the user's
correction, remaining work uses 30 workers on Wintermute's 32 available CPUs.
Docker reported about 30 CPUs in use and negligible competing container CPU.

The final holdout started at 16 workers and was handed over at 30, retaining
completed checkpoints. Its final timing must include that short initial
period when reporting end-to-end throughput; the resumed runner's elapsed
seconds alone exclude it. Start and runtime receipts are saved alongside the
results. No interim win counts are consulted for timing or stopping decisions.
