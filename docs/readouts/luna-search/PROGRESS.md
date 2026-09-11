# Luna search validation progress

Updated 2026-09-10 from the live `luna-heximax-validation` container.

- Image: `58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`
- Source mount: `/home/bsm/tmp/hexset-luna-validation-corrected` → `/study`
- Controller SHA-256: `4508676f53043e506a83451b62729c84b7d9de97bad762ff3d9bae0a17a09e46`
- Screen: all 9 archived under `results/`, each 120 games.
- Selected top three from the screen: `nodes2400-width12`, `nodes2400-k4`, `nodes600-k4`.
- First confirmation archived under `validation/confirm-00-nodes2400-width12.json`; SHA-256 `148a560089079dee6a78ea2d23f41531fc897878efda70e08961c4eb11af24d3`.
- First confirmation counts: `vs-ab2` 441/1024 (`43.1%`), `vs-shipped` 269/1024 (`26.3%`). The report records 1439.3 seconds for AB2 and 180.9 seconds for shipped.
- At the last audit, the second confirmation (`nodes2400-k4`, seed `1201000`) was active with 30 workers.

These are confirmation-stage observations only. No finalist or fresh 2048-game holdout verdict is available in this readout. Bank-quiescence, stance aggregation, and adaptive evolution stages have not been run.
- Second confirmation counts: `vs-ab2` 464/1024 (`45.3%`), `vs-shipped` 272/1024 (`26.6%`).
