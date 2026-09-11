# Fixed scalar leaf benchmark final readout

- Status `PASS`; result SHA256 `505ef8233bf2ffccb52c3acaa85206e60e72e986ed2d612a95e1a0996a25c60c`; archive `/data/data/com.termux/files/usr/tmp/luna-fixed-leaf-296900-result.json`.
- Certified source `11a0cbfef3280c5a9040ad6bcfbd3dff3dbf10c636b0f4e159d1b0073a47ec19`; image `sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`; benchmark kind `fixed_state_scalar_leaf_kernel`.
- Four passes, 100 repeats, 6,400 timed calls per arm/pass. Deserialization and exact score verification were outside timing.

| arm | aggregate wall | aggregate CPU | wall / baseline | CPU / baseline | score digests | checksums |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 6.173619797s | 6.127073046s | 1.000000000 | 1.000000000 | 1 | 1 |
| v2 | 5.174793367s | 5.104942974s | 0.838210570 | 0.833178083 | 1 | 1 |
| ownreach | 4.464211245s | 4.433545180s | 0.723110815 | 0.723599204 | 1 | 1 |
| duplicate | 5.296609642s | 5.227155262s | 0.857942312 | 0.853124359 | 1 | 1 |

## Per-pass ratios

| pass | v2 wall/CPU | ownreach wall/CPU | duplicate wall/CPU |
|---:|---:|---:|---:|
| 0 | 0.843075 / 0.846564 | 0.710515 / 0.712970 | 0.864240 / 0.864869 |
| 1 | 0.874363 / 0.852878 | 0.723259 / 0.716849 | 0.882737 / 0.860589 |
| 2 | 0.772948 / 0.776487 | 0.719439 / 0.722012 | 0.802776 / 0.807225 |
| 3 | 0.863500 / 0.857396 | 0.739692 / 0.742863 | 0.882818 / 0.880239 |

## Fixture coverage

| target | buildings | cities | canonical roads | map port nodes | occupied port nodes |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 18 | 0 |
| 16 | 8 | 0 | 8 | 18 | 0 |
| 40 | 8 | 0 | 8 | 18 | 0 |
| 80 | 8 | 0 | 8 | 18 | 0 |

- All 16 arm runs have the same exact score digest and numeric checksum as baseline; all have 6,400 timed scalar calls.
- Coverage gaps are explicit: no city buildings and no occupied ports occur in the 0/16/40/80 targets. The map contains 18 static port nodes. This is a warm fixed-state kernel benchmark, not complete-game throughput evidence.
- Related archives: duplicate cProfile `fc8f8fbf74cf486c76710b9aa8287b7b04f14d041111fc1f4c62f69995f4eed4`, reachability cProfile `1f5cf34eb4778b8c130270292cfb1dbcde1da61069953fb311b3c79519df11e2`, fresh diagnostic result `1dfce96828c38d81d27fb0b04e59d37438dbf839e865c9b5687d8149ba15b74f`.
