# Heximax versus AB2: host throughput and bridge cost

Measured September 11, 2026 UTC on Wintermute, with one `heximax-notrade`
against three depth-two Catanatron AB players, standard 10-VP/7-discard rules.
HexSet source is `824d3acbd74b73611943a203ec23169a47fc662e` (includes PR #138),
fingerprint `0cc6d8a2d6d65da7bacd801062655f22fc4c2c0ca903ced93ced11a9ad561b44`.
Catanatron is the supported pin `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`,
fingerprint `0968d060809ce8cb0037ec96a65df77c85e14ab802b145b4fe636a65e9c445af`.
Both hosts enable the same fast speedups. Heximax always searches in HexSet;
AB2 always searches in Catanatron. The selected host runs the real game and
bridges decisions for the guest bot(s).

## Results

Two reversed-order rounds, 120 predetermined games per host per round, 30
workers bounded to 30 CPUs. Each arm has a fresh interpreter and dynamic pool.
Rates below use 240 game plays divided by total batch seconds for each host.

| Host | First batch seconds | Second batch seconds | Games/s | CPU seconds/game | Effective CPU concurrency |
| --- | ---: | ---: | ---: | ---: | ---: |
| HexSet | 53.570 | 54.919 | 2.212 | 11.053 | 24.45 |
| Catanatron | 50.318 | 51.142 | 2.365 | 11.022 | 26.07 |

**Heximax's guest overhead did not reverse the observed host throughput
advantage.** Catanatron completed this workload at 6.93% higher throughput;
HexSet's rate was 6.48% lower. However, total CPU per game differs by only
0.28%. Most of the batch wall-time difference is accounted for by different
effective pool utilization/tails, not more total CPU work in HexSet. This is
an operational sample of these 120 seeds, repeated twice, not a claim about
intrinsic engine speed or statistical significance.

## CPU attribution

Per-decision process CPU timers surround AB2 `decide`, Heximax `choose`, and
the entire host-facing bot call. Guest bridge cost is outer call time minus
inner search/decision time; it includes translation, initial bot/map setup,
action mapping and timer/call overhead. It is not a cProfile estimate. Trace
recording and remaining host work are included in full-game timing. The tiny
outer-minus-inner time for a local bot is wrapper overhead, not a bridge.

| Mean CPU seconds per game | HexSet host | Catanatron host |
| --- | ---: | ---: |
| Three AB2 search/decision calls | 9.788 | 10.021 |
| Three AB2 host-facing calls, including bridge if needed | 9.935 | 10.022 |
| Heximax search/decision calls | 0.981 | 0.945 |
| Heximax host-facing calls, including bridge if needed | 0.982 | 0.964 |

Heximax's Catanatron bridge costs **0.0185 CPU seconds/game**, about **1.92% of
Heximax's own host-facing CPU time** and **0.168% of total game CPU time**.
That overhead is small because the search itself continues to use HexSet.
The three AB2 bridges in HexSet cost approximately 0.147 CPU seconds/game.
Search work differs between hosts, so the per-game search rows do not measure
an isolated implementation slowdown. In particular, there is no fixed AB2
speed multiplier from the earlier four-AB2 experiment that can be transferred
to this mixed lineup.

## Protocol and behavioral limits

- Predetermined seeds: 690200000..690200119; Heximax occupies `index % 4`, giving
  30 initial games per seat. Both hosts receive the same native-generated board,
  deck, seating and initial setup state. HexSet translates that initial state
  and subsequently plays through its own arena. Heximax's initial RNG seed
  matches the Catanatron bridge's color-derived seed on each paired game.
- Order: HexSet then Catanatron in round one; Catanatron then HexSet in round two.
  Timings include pool startup/shutdown and per-game setup. The workers use
  lightweight CPU timers and chosen-action tracing in both arms. The unchanged
  20-second native search deadline remains enabled.
- All **480 game plays** reached a winner. All **240 within-host repeat pairs**
  match action digests/counts, winner, all-seat VP, turns, per-bot decision counts
  and fallback counts. There were **zero Heximax adapter fallbacks**.
- Cross-host trajectories differ: mean actions are 304.35 in HexSet versus
  292.32 in Catanatron; mean turns are 78.66 versus 82.14. Identical initial seeds
  do not imply identical subsequent games or search workloads.
- There is also an information difference: `state.translate` currently rebuilds
  a memoryless public hand ledger for Heximax in Catanatron. HexSet-hosted games
  retain their accumulated public resource history. Heximax reads that ledger
  when forming beliefs, so this can change choices as well as search cost.
  The benchmark measures the current supported integrations; it is **not a
  controlled strength comparison or an isolated translation-cost experiment**.
  The direct CPU attribution above identifies the bridge's own cost separately.

These results support using the actual mixed lineup for throughput decisions
and retaining HexSet as the preferred evaluation host when public-history
behavior matters. They do not establish the result for a two-player matchup
or for another ratio of Heximax and AB2 seats.

## Reproduction and artifacts

Use the immutable Python 3.12.14 image
`sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`,
30 CPUs, network disabled, PYTHONHASHSEED=0 and BLAS/OpenMP thread limits of one.
Mount measured source as `/study/src`, these scripts together under `/study`,
and a fresh writable directory as `/out`. Run
`PYTHONPATH=/study/src python /study/run_compare.py`.
The runner rejects existing outputs, checks repeat traces and records both
engine source fingerprints. The [runtime receipt](runtime.json) retains image,
mounts, CPU limit, environment and successful exit.

- [run_compare.py](run_compare.py), [worker.py](worker.py): exact harness.
- [summary.json](summary.json): aggregate rates and CPU attribution.
- [verified.json](verified.json): repeat verification receipt.
- [0-hexset.json](0-hexset.json), [1-hexset.json](1-hexset.json): HexSet games.
- [0-catanatron.json](0-catanatron.json), [1-catanatron.json](1-catanatron.json): Catanatron games.
