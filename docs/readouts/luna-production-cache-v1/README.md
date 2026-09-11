# Public production cache V1 final readout

The pinned one-CPU run completed all eight paired seeds `293000000..293000007` with `PASS` status and no equivalence or clean-timing mismatches. The cache arm preserved winners, points, action traces, spectra, ordered feature digests, and zero player-to-player trades.

Clean totals were baseline `323.0517160508316s` wall / `321.03427563000014s` CPU and cached `314.1851188200526s` / `311.294663668s`. Aggregate cached ratios were `0.9725536290623391` wall and `0.9696617691588008` CPU: 2.7446% wall and 3.0338% CPU reductions over these eight paired games. This is a paired benchmark result and carries no broad throughput claim.

Native Catanatron exposed no timeout-hit counter. The harness retained the native `MAX_SEARCH_TIME_SECS=20` deadline and reported only its own wrapper observations, so zero observations do not prove that no native deadline was reached. Fallback telemetry was unavailable. Maritime/bank/port trades remained enabled; player-to-player `OFFER_TRADE` was absent from generated actions.

Result JSON archive: `/data/data/com.termux/files/usr/tmp/luna-shared-production-cache/v1-final-archive/native-cache-equivalence-final.json`

- Result SHA256: `5607e589f0ca1a92ffda606a4d90ce50015e9a22a0e412a01a6073793b1b6346`
- Metadata SHA256: `4b5b4c6741ba7aa9db8538adf7b74b2225e1fe7d3ee73f0181b17186352f9222`
- Final timestamp: `2026-09-10T21:18:40.388534+00:00`
- Certified source: `11a0cbfef3280c5a9040ad6bcfbd3dff3dbf10c636b0f4e159d1b0073a47ec19`
- Pinned image: `sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`
