# Production-weighted port liquidity: local implementation

Date: 2026-09-10. This is an opt-in local strength family; no remote game was
launched and the shipped evaluator/presets were not edited.

The candidate replaces the flat `NO_TRADE_WEIGHTS.port` term with

```
sum(rate_r * (1 / ratio_r - 1 / BASE_TRADE_RATIO))
```

where `rate_r` follows `Evaluator.survey`: per-vertex pips, settlement/city
multiplicity, and zero contribution for a robber-blocked hex. Gold is included
in the native total rate but omitted from the fixed-resource liquidity sum.
The cache key contains the evaluator board, all public vertex/edge occupancy,
the robber, and seat.

Files:

- `src/hexset/bots/heximax/port_liquidity.py`
- `src/hexset/catanatron/port_liquiditycfg.py`
- `tests/test_port_liquidity.py`

The six planned arms are `control` (ordinary `heximax-notrade`),
`drop-flat-port` (`port=0`), `flat-port-4x` (`port=0.12252`), and liquidity
coefficients `1.3925`, `2.785`, and `5.57` with `port=0`. Every arm uses
notrade mode, depth 2, width 6, max_nodes 600, k=1, win stance, temperature
unset, and max_trades 0. Each arm has AB2 and frozen-original gates, 120 games
and 30 workers, with reserved paired per-game seeds from bases 600000000 and 600100000. Controls are
marked in the manifest and are never promotion candidates.

Validation: `PYTHONPATH=/data/data/com.termux/files/home/code/HexSet/src pytest -q tests/test_port_liquidity.py tests/test_ports.py tests/test_structural_features.py` -> **33 passed**. Both new modules and the test file compile with `python -m py_compile`.

SHA-256:

- `port_liquidity.py`: `e9ab7a80922a3230795fa8351e566baa93a215d6c65f0cd3dff66590233af358`
- `port_liquiditycfg.py`: `c7b73fed5a4aa46de0fdfef0ed698de7ce697ef2900530f3b0ad444e7f65c981`
- `test_port_liquidity.py`: `fe608efcd98299d4e953e81d1a4440db99aba0802730619a7c5dd5357ba79b32`
- current config manifest source fingerprint: `1fc07ae035e5a3708131d145eb7774098a9c5d9d2a6f843685e9fbd0b03b95f6`
