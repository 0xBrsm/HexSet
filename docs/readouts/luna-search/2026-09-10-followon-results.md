> Evaluation-scope correction — 2026-09-11
>
> Stock `DC:` screens use a memoryless public ledger and cannot directly select or reject native HexSet policies with public history. The explicitly labeled `DCP` row uses a separate partial-ledger protocol and must be audited separately; it is not proof of equivalence to HexSet's native ledger.
> Raw results and historical verdicts are preserved. See [scope audit](../ledger-scope-audit/README.md).

# Luna follow-on screens — 2026-09-10

These are exploratory screen results only. They are not fresh validation, and they do not establish optimality.

## Closed screen results

All rows below used 120 games per gate, 30 workers, and `max_trades=0`.

| Arm | AB2 | Frozen Heximax-notrade | Role / phenotype |
|---|---:|---:|---|
| bankext | 50/120 | 31/120 | bank extension; 600 nodes, width 6 |
| bankext-wide | 57/120 | 27/120 | bank extension; 2400 nodes, width 12 |
| control | 54/120 | 26/120 | matched control; 600 nodes, width 6 |
| control-wide | 53/120 | 24/120 | matched control; 2400 nodes, width 12 |
| DCP | 57/120 | 23/120 | public ledger candidate; audited ledger source `f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea` |
| fair k1 control | 59/120 | 25/120 | `heximax-notrade-wide`; depth 2, k 1, width 12, 2400 nodes |
| fair k4 candidate | 59/120 | 25/120 | `heximax-fair-k4`; depth 2, k 4, width 12, 9600 nodes |

The fair k1 raw files are `k1-control-ab2.json` and `k1-control-shipped.json`; the fair k4 raw files are `k4-candidate-ab2.json` and `k4-candidate-shipped.json`, under the remote root:

`/home/bsm/tmp/hexset-fair-18789ed84f64707d8734fc06637747d3aa80488c299469151d3274996c4c999d/artifacts/fair-screen/`

Fair k4 output hashes:

```text
k4-candidate-ab2.json     a6df9a5480984c9ceb85a41eb8bb6f48c76a64627457c103b58cb53796b7a79c
k4-candidate-shipped.json 99320325d2d879f547b3bd718d99161b30590450a22395598167e7156832dfcd
```

The fair source hash is `11a0cbfef3280c5a9040ad6bcfbd3dff3dbf10c636b0f4e159d1b0073a47ec19`. The DCP raw files were `dcp-ab2-120-3000000.json` and `dcp-shipped-120-3100000.json`; both used source hash `f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea` and protocol `public-ledger-v1`.

## Evolution screen audit

Generation 0 contained 2,880 valid discovery artifacts. The checkpoint recorded `promoted: []` and generated all 12 generation-1 candidates from `g00-p00`; its summary rows were empty. A further 582 generation-1 artifacts were produced before interruption at approximately 17:52:59 UTC and are ancillary/excluded from inference. The promotion/correction audit remains incomplete; this readout does not treat those artifacts as a valid promotion result.

