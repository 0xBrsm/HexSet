# Native Heximax expansion holdout, September 11, 2026

The fixed holdout passed both registered endpoints. All real games ran in
HexSet with its native public-history ledger, standard rules, and no domestic
trading. Catanatron AB2 is the pinned external reference opponent; its internal
hypothetical search uses Catanatron. These results do not use the historical
memoryless DC adapter protocol.

| Four-player matchup | Wins / games | Win rate | 95% Wilson lower bound | 97.5% lower bound |
| --- | ---: | ---: | ---: | ---: |
| Candidate vs three AB2 | 1,086 / 2,048 | 53.03% | 50.86% | 50.55% |
| Candidate vs three frozen Heximax-notrade | 714 / 2,048 | 34.86% | 32.83% | 32.54% |

The thresholds were 50% and 25%, respectively. The stricter bounds use the
registered first-attempt alpha of .025. Each gate has 2,048 independent boards,
512 games in each candidate seat, seed 716000000, and all indices 0–2047.
No holdout success was checked before all 4,096 games completed. These results
establish the registered absolute gates, not optimality or a paired gain over
the old policy against AB2.

The candidate `road-zero-exp025` keeps the shipped depth 2, width 6, 600-leaf
budget, k=1, win stance, temperature 2.476644394795811, opening prior, and
weights except road=0. It adds at most .25 VP for the best legal settlement
site reachable now or after one additional road; a one-road-away site gets
half credit. Production pips determine quality. The score uses a maximum,
not a sum over mutually exclusive opportunities. The frozen opponent retains
road=.1237 and expansion=0, with all other settings identical. The positive
bonus replaces unconditional road credit. Trading mode was not evaluated.

Selection used small exploratory screens, a separate local screen at seed
714000000, and fresh 512-game/gate confirmation at 715000000 (273 AB2 wins,
171 frozen-Heximax wins). The complete campaign, including unsuccessful arms,
is recorded in PLAN.md, STATUS.md, immutable manifests, and stage archives.

## Adapter repair and checkpoint preservation

After 1,978 completed AB2 games, original index 1935 exposed a reference-adapter
deadlock: 15 roads, one unspendable Road Building credit, no legal road site.
HexSet allowed END_TURN, while the mirror offered no action. Commit 788f27e
repairs only this previously fatal empty-offer case; normal offers are
unchanged. The source differs from the original evaluated d3d4618 only in
catanatron/state.py. MAIN and pre-roll cases have regression tests.

All 1,978 original records retain their original source identity and bytes.
Four registered full-action replays matched exactly under the repair. The
original failing game and every remaining fixed index were completed with
the patched source. The compatibility argument, exact fingerprints, diagnostic,
replay proof and per-file hashes are preserved in holdout-repair-plan.md and
the raw archive. No failed game was dropped, replaced, or scored as a loss.
An independent local check verified record completeness, unique indices,
balanced candidate seats, all legacy hashes and both confidence calculations.

Archive: holdout-road-zero-exp025.tar.gz
SHA256: `8388e6b9dd7c49d50c8c7c1e026216d65a64bfb02e801cb8706c8860f3d868b8`.
It contains all 4,096 game records, repair replays, hash manifest, diagnostic,
and final verdict. Replay games are excluded from the holdout sample.

The run moved from 16 to 30 workers on the user's instruction. The resumed
pool used about 30 cores. Its elapsed time excludes the earlier work and
repair downtime, so it must not be presented as whole-holdout throughput.
