# Second batch: 400 additional games per format

This is the fixed extension requested after the first batch, using the same
current unpinned adaptive Heximax and Catanatron AB2 configuration in HexSet.
The [plan](PLAN.md) was committed before launch as `d1cc670`. The runner and
audit differ from the original batch only in their seed constants.

| Matchup | Heximax wins | Win rate | 95% Wilson interval |
| --- | --- | --- | --- |
| One Heximax vs one AB2 | 298 / 400 | **74.5%** | 70.0–78.5% |
| One Heximax vs three AB2 | 226 / 400 | **56.5%** | 51.6–61.3% |

All 800 games finished and passed independent replay of 199,085 legal
actions. There were no AB2 deadline hits or domestic exchanges; the unpinned
adaptive slider remained at zero. Source fingerprints match the first batch.
All six off/fast preflight pairs matched complete records. The strength
batch took 201.31 seconds, excluding the 18.15-second preflight and audit.

See the [pooled readout](../README.md) for headline results over both batches.
The original batch remains preserved separately. No games are replaced or
excluded, and the 12 extension preflight games are outside the strength sample.

Artifacts: [summary](summary.json), [audit](audit.json),
[manifest](manifest.json), [preflight verdict](preflight-summary.json),
[runtime receipt](runtime-receipt.json), [raw records](records.tar.gz),
[checksums](SHA256SUMS), [runner](run.py), and [audit script](audit.py).

Reproduce this batch from the recorded HexSet policy and Catanatron pin:

```sh
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python docs/readouts/current-heximax-ab2/extension/run.py --out /tmp/current-ab2-extension --workers 30
PYTHONPATH=src python docs/readouts/current-heximax-ab2/extension/audit.py /tmp/current-ab2-extension
```
