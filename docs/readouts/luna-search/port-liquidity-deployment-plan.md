# Port-liquidity deployment plan

This is a permission-ready payload. No upload or game launch is performed by this
plan.

The existing isolated Wintermute destination is:

```text
/home/bsm/tmp/hexset-port-liquidity-15e00c6b295792c1
```

It was created from the certified source tree
`/home/bsm/tmp/hexset-luna-followon-certified`. The payload contains exactly
four files; it contains no private ledgers, histories, or dirty workspace files.
The remote layout is `ROOT/src/...`.

| file | SHA-256 |
|---|---|
| `src/hexset/bots/heximax/port_liquidity.py` | `e9ab7a80922a3230795fa8351e566baa93a215d6c65f0cd3dff66590233af358` |
| `src/hexset/catanatron/port_liquiditycfg.py` | `686894bdfe601985ac6d9935f8fe963c58fb515ba8e2bcb4846b9064a685fcf5` |
| `src/hexset/catanatron/port_liquidity_runner.py` | `da8e757edc01582068eb2b6ac57ae5a4b2001918cc3b8ed6f5e28e2e2a729c98` |
| `scripts/port_liquidity_queue.sh` | `01dd6dac19ee20806a187edcae9ce33f36448a7a43cb53d5941b70238277c538` |

The reviewed transport helper is
`scripts/port_liquidity_stage.sh`. It streams only those four files through
`ssh wintermute` to the existing WSL destination and verifies the four hashes.
The queue is `scripts/port_liquidity_queue.sh`; it requires a confirmation state
whose first line is `COMPLETE`, uses one output lock, invokes the runner for every
job so existing records are revalidated, and runs six arms × two gates at 240
games and 30 workers serially.

After upload, the bounded native smoke is one `_play_one` invocation in the
pinned image with one CPU and `PYTHONHASHSEED=0`. The 30-worker queue remains
unstarted until the round6000 confirmation has exited and its validated terminal
state is available.
