# Fresh-process diagnostic final readout

- Status `PASS`, cases `4/4`, children `8/8`.
- Result archive `/data/data/com.termux/files/usr/tmp/luna-fresh-diagnostic-294000-final.json` SHA256 `1dfce96828c38d81d27fb0b04e59d37438dbf839e865c9b5687d8149ba15b74f`.
- Source `11a0cbfef3280c5a9040ad6bcfbd3dff3dbf10c636b0f4e159d1b0073a47ec19`, image `sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`, timeout `20s`.

| arm | seed | mode | baseline wall | candidate wall | baseline CPU | candidate CPU | leaf calls baseline/candidate | deadline observations |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| v2 | 294000005 | instrumented | 46.815756767 | 43.921427156 | 46.605085049 | 43.593639689 | 73548/73548 | 0/0 |
| v2 | 294000005 | clean | 45.073078619 | 44.667707888 | 44.858813051 | 44.330241442 | None/None | None/None |
| ownreach | 295000002 | instrumented | 25.477434618 | 20.168617506 | 25.394438395 | 20.089494914 | 43486/43486 | 0/0 |
| ownreach | 295000002 | clean | 25.930609749 | 20.106794332 | 25.785017700 | 20.032827192 | None/None | None/None |

- All four pairs were exact on winner, wins, points, action trace, and domestic-trade count.
- Instrumented score calls matched: V2 73,548/73,548; own-reachability 43,486/43,486. Approximate deadline observations were zero; native timeout telemetry is unavailable.
- Clean timing is diagnostic only and does not establish game-throughput speedup.
