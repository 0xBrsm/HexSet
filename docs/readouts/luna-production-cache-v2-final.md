# Production cache V2 final
Status: `PASS`; updated `2026-09-10T21:37:15.572017+00:00`; all 8 rows passed with mismatch `None`.

| Seed | Clean wall ratio | Clean CPU ratio | Baseline wall s | Cached wall s |
|---:|---:|---:|---:|---:|
| 294000000 | 0.965560511 | 0.966233470 | 10.490291 | 10.129010 |
| 294000001 | 0.959008696 | 0.954981222 | 16.315773 | 15.646968 |
| 294000002 | 0.985777058 | 0.986604767 | 9.311582 | 9.179144 |
| 294000003 | 0.977653881 | 0.962269258 | 12.559973 | 12.279306 |
| 294000004 | 0.973349636 | 0.974947689 | 7.747147 | 7.540682 |
| 294000005 | 1.596252261 | 1.591877440 | 31.238633 | 49.864739 |
| 294000006 | 0.910061222 | 0.908190500 | 57.977921 | 52.763458 |
| 294000007 | 0.938731511 | 0.936103896 | 44.457826 | 41.733962 |

Aggregate baseline/cached wall: `190.099145020s` / `199.137269475s`, ratio `1.047544267`.
Aggregate baseline/cached CPU: `189.423154755s` / `197.843644619s`, ratio `1.044453329`.

This is a paired one-CPU benchmark, with no broad throughput claim. Native timeout remained 20 seconds; wrapper observations are not a native timeout counter.
