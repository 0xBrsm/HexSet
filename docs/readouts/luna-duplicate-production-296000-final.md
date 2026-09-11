# Duplicate production benchmark final readout

- Remote terminal status: `PASS`; equivalence pairs `8/8`; clean pairs `8/8`; mismatch `None`.
- Frozen source SHA: `11a0cbfef3280c5a9040ad6bcfbd3dff3dbf10c636b0f4e159d1b0073a47ec19`; helper SHA: `2454b37e2f50ae2ccc66a1944dd7cf83506d0d9f5e5be89dd6057f9450ecf1f8`; image: `sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`; timeout: `20s`.
- Archived local JSON: `/data/data/com.termux/files/usr/tmp/luna-duplicate-production-296000-output.json`; SHA256 `bd5249dad1e0ceb1445d4d0a400fb4bbb82745ab9a8aa79f2938361f5a814943`.

## Clean timing pairs

| seed | baseline wall | candidate wall | baseline CPU | candidate CPU | wall ratio | CPU ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 296000000 | 31.372963663 | 36.049091643 | 31.365879307 | 36.017288776 | 1.149050 | 1.148295 |
| 296000001 | 28.455993715 | 82.392748765 | 28.428262579 | 67.415658827 | 2.895444 | 2.371431 |
| 296000002 | 33.089076628 | 29.439717156 | 28.738154748 | 24.307725930 | 0.889711 | 0.845835 |
| 296000003 | 23.842734643 | 24.508953505 | 23.820781469 | 24.439315747 | 1.027942 | 1.025966 |
| 296000004 | 51.474927922 | 48.469719979 | 51.419185784 | 48.426317262 | 0.941618 | 0.941795 |
| 296000005 | 10.291464168 | 10.117726964 | 10.273715585 | 10.110919446 | 0.983118 | 0.984154 |
| 296000006 | 20.407866206 | 18.275166730 | 20.375022195 | 18.252860154 | 0.895496 | 0.895845 |
| 296000007 | 30.624885305 | 29.453527171 | 30.594584999 | 29.436903199 | 0.961751 | 0.962161 |

- Aggregate wall: `229.559912250 -> 278.706651913s`, ratio `1.214091124`, delta `49.146739663s`.
- Aggregate CPU: `225.015586666 -> 258.406989341s`, ratio `1.148395954`, delta `33.391402675s`.
- All 8 action/outcome equivalence pairs passed. Maritime trades remain enabled; domestic player-to-player trade count was zero. Timing is clean native timing with the unchanged 20-second deadline; no native timeout counter was available.
