# Heximax campaign recovery

Recovered on 2026-09-10 from the stalled local Codex session
`01a0898c-8dea-7d32-b59a-6f999e13fd94`.

The user requested continuation, retaining the earlier instructions: prioritize
Heximax strength, use Luna for experiment work, use Wintermute with 30 workers,
and keep domestic trading disabled. Training and experiment uploads were
explicitly authorized. The targets remain above 50% against three AB2 opponents
and above 25% against three frozen Heximax no-trade opponents. Further speed
experiments are paused. No successful target or exhausted-search claim has been
established.

The original session's last substantive progress was saved before it stopped
responding. Its later turns show starts and interruptions without additional
work. This establishes the recovery point, but does not establish the cause of
the Codex hang.

## Work recovered

- The 5,760-game generation-6000 weight search uses the certified source on
  Wintermute. Inspect live status before scheduling another main worker pool.
  `scripts/round6000_postvalidate.py` validates its complete raw archive;
  `scripts/round6000_continuation_plan.py` selects the best joint-margin
  phenotype for independent 1,000-game gates. Preserve completed game records.
- The v5 maritime fallback replay established a forced Road Building phase
  mismatch. Native Catanatron offered only free-road actions, while translated
  HexSet also offered a bank trade. Hand, bank, and port ratio were valid.
  See the final replay entry in `CAMPAIGN-STATUS.md`.
- The opt-in production-weighted port evaluator and local tests are saved.
  See `port-liquidity-local-20260910.md` and `port-scale-audit-20260910.md`.
  Its game experiment had not launched when this session recovered the work.

## Current ownership

- `resume_weight_search`: live run status, archive validation, independent
  weight confirmation, and queue ownership.
- `resume_fallback_fix`: forced-road adapter correction and regression proof.
- `resume_port_feature`: port evaluator checks and runnable strength experiment.

All three are Luna agents. Preserve the certified running source and original
opponent behavior. Stage new candidates separately; do not combine results
across source changes or select from incomplete groups. The port screen is
being prepared with 240 games per arm per gate; candidate selection must be
followed by independent validation.

## Verified recovery progress

Generation 6000 passed strict validation for all 5,760 records. Selection from
the raw records chose `g6000-p10`: 113/240 wins against AB2 and 68/240 against
frozen Heximax. Both campaign targets remain unproven. The full table is in
`round6000-recovery-20260910.md`; strict validation and selected-phenotype
sidecars are also copied into the adjacent `round6000/` directory.

Independent confirmation is running as `luna-round6000-confirm-vs-ab2`
(1,000 games, 30 workers, seed 620000000). A remote serial supervisor waits for
successful completion before starting `luna-round6000-confirm-vs-shipped`
(seed 620100000). Its recovered script is under
`/data/data/com.termux/files/usr/tmp/luna-round6000-recovery/serial-supervisor.sh`.
Final outcome validation remains separate from container completion.

The forced-road correction is opt-in through `Entrant.native_action_compat`;
existing registrations retain their defaults. Candidate search keeps the
translated post-roll state in MAIN and limits actions while free roads remain.
Local forced-road, port-feature/runner, and existing Heximax tests passed
together: **31 passed in 7.83 seconds**. Luna also ran forced-road and native
state-translation checks in the pinned Wintermute image.

The six-arm port runner uses 240 requested games per gate, per-game RNG seeds,
and atomic resumable records. Native smoke and game screens are pending upload.
Automatic approval review rejected the exact three-module plus queue-script
payload to Wintermute, requiring explicit approval for those files and that
destination. See `port-liquidity-deployment-plan.md` for the local prepared
action. This note does not authorize retrying the rejected upload. Existing
weight confirmations continue independently.
