# DCP 1,000-game follow-up plan

Prepared after the p06 discovery selection and while the p06 1,000-game
follow-up is active. This is a plan only; no DCP container has been launched.
The gate jobs are serial and must wait for the p06 container to exit normally.

## Certified identity

- Image: `58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`
- Source root: `/home/bsm/tmp/hexset-luna-followon-certified`
- Public-ledger source: `src/hexset/catanatron/public_ledger.py`
- Public-ledger SHA-256: `f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea`
- DCP CLI SHA-256: `bc8d23876693434009c0ec9847da25bde7dbd4257473bab05518e40bb8f4ac0d`
- Protocol: `public-ledger-v1`
- Output root: `/home/bsm/tmp/luna-20260910-discovery-3000/dcp-followup`

The certified oracle used the same image, source root, and public-ledger hash.
It checked typed lower bounds against private hands and the invariant
`sum(known) + unknown == actual public hand size`, with zero fallbacks. DCP
never reads hidden opponent card identities. `DevCatanPlayer` forces
`max_trades=0` for the bridge, while normal maritime trade actions remain
available and are tracked by the public ledger.

## Exact existing CLI

`dclcfg.main()` parses `--candidate {dcl,dcp}`, `--gate {ab2,shipped}`,
`--games`, `--workers`, `--seed`, and `--out`. It imports
`hexset.catanatron.public_ledger`, which registers `DCP` as
`PublicProductionLedgerPlayer`, then constructs these exact lineups:

```python
lineup = (
    ('DCL' if a.candidate == 'dcl' else 'DCP') + ',AB:2,AB:2,AB:2'
    if a.gate == 'ab2' else
    ('DCL' if a.candidate == 'dcl' else 'DCP') +
    ',DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade'
)
```

The planned DCP lineups are therefore:

```text
DCP,AB:2,AB:2,AB:2
DCP,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade
```

## Deferred commands

After the p06 container `luna-discovery-followup-p06` exits with status 0 and
no other 30-worker container is active:

```bash
mkdir -p /out/dcp-followup
python -m hexset.catanatron.dclcfg --candidate dcp --gate ab2 \
  --games 1000 --workers 30 --seed 410000000 \
  --out /out/dcp-followup/dcp-vs-ab2-1000-410000000.json
python -m hexset.catanatron.dclcfg --candidate dcp --gate shipped \
  --games 1000 --workers 30 --seed 410100000 \
  --out /out/dcp-followup/dcp-vs-shipped-1000-410100000.json
```

Both commands belong in one `set -eu` container command so the second gate
cannot overlap the first. The Docker mounts are:

```text
/home/bsm/tmp/hexset-luna-followon-certified:/followon:ro
/home/bsm/tmp/luna-20260910-discovery-3000/dcp-followup:/out:rw
```

with `PYTHONPATH=/followon/src`, UID 1000, `--cpus 30`, `PYTHONHASHSEED=0`,
and `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

Each output must retain `candidate=dcp`, the exact gate, 1,000 games, 30
workers, its seed, exact lineup, `source_sha256=f4b09d92...`, and
`protocol=public-ledger-v1`. A separate run manifest should bind both output
SHA-256 values, the full image digest, source-root path, public-ledger hash,
lineups, seeds, and serial container identity. Results remain independent
assessment evidence and are not pooled with the prior 120-game DCP screen.
