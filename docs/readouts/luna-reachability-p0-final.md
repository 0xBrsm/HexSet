# Reachability P0 final benchmark

Result SHA-256: `72487289e8f2a06483768c84626e31de892c84fca6cff8087c950b1d53c9d1d4`  
Status: `PASS`; equivalence qualified: `True`; rows: `8`.

Source SHA-256: `11a0cbfef3280c5a9040ad6bcfbd3dff3dbf10c636b0f4e159d1b0073a47ec19`  
Image: `sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`  
Players: `DC:heximax-notrade,AB:2,AB:2,AB:2`  
Max trades: `0` (player-to-player `OFFER_TRADE` count was checked; maritime actions remain enabled)  
Native timeout: `20` seconds.

All eight candidate runs matched baseline winner, wins, points, ordered action trace, spectrum, feature trace, and domestic-trade telemetry.

| Seed | Baseline wall s | Candidate wall s | Wall ratio | Baseline CPU s | Candidate CPU s | CPU ratio | Equivalence |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 295000000 | 23.335015630 | 23.712709933 | 1.016185732 | 23.329639792 | 23.673427458 | 1.014736090 | PASS |
| 295000001 | 35.071865354 | 42.070983100 | 1.199565027 | 34.903513710 | 41.841236509 | 1.198768607 | PASS |
| 295000002 | 12.944453103 | 22.843918979 | 1.764765093 | 12.867091867 | 22.708854583 | 1.764878561 | PASS |
| 295000003 | 46.875346494 | 40.037367753 | 0.854124198 | 46.649976309 | 39.742044230 | 0.851919923 | PASS |
| 295000004 | 44.140877367 | 37.676292002 | 0.853546514 | 43.304432036 | 36.959407764 | 0.853478640 | PASS |
| 295000005 | 23.213624452 | 18.615354246 | 0.801915026 | 22.645867478 | 18.329508700 | 0.809397508 | PASS |
| 295000006 | 20.933067317 | 16.232240481 | 0.775435355 | 20.395396324 | 15.978287958 | 0.783426206 | PASS |
| 295000007 | 54.702197957 | 46.902523676 | 0.857415706 | 45.433813874 | 39.034832181 | 0.859158166 | PASS |

Aggregate baseline → candidate wall: `261.216447674 → 248.091390170 s`, ratio `0.949754092`, delta `-13.125057504 s`.  
Aggregate baseline → candidate CPU: `249.529731390 → 238.267599383 s`, ratio `0.954866573`, delta `-11.262132007 s`.

The harness recorded no native timeout counter. Its deadline observation proxy was present only in instrumented equivalence runs and is not a native timeout telemetry source. Clean timings have no feature or spectrum instrumentation. This is a one-CPU paired benchmark and does not establish broad throughput.
