# Luna adaptive campaign status

Checkpoint recorded: 2026-09-10T16:13:23Z (local UTC clock).

## Objective

Find an entrant exceeding 50% against four player AB2 and 25% against three frozen Heximax no-trade opponents. Research is ongoing; no exhaustion claim is made.

## Verified completed work

- Search confirmations: 441/269, 464/272, and 479/259 per 1024 games. The abandoned holdout is excluded.
- Stance screen: current 14/16; T1 eligible 63/34 per 120.
- DCP public ledger oracle: 30 pinned Wintermute games passed all typed-bound and total invariants, with zero bridge fallbacks. Exact audited source hash: `f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea` (also recorded in [dcp-oracle/source.sha256](dcp-oracle/source.sha256)). Results and harness are in [dcp-oracle](dcp-oracle/).
- DCP parser proof uses `DCP:heximax-notrade,AB:2,AB:2,AB:2` and `DCP:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade`; no additional games are implied by this checkpoint.

## Operations and resumption

The primary controller owns the 30-worker Wintermute queue. Recovery should preserve immutable source snapshots, explicit seeds, per-game atomic checkpoints, and source/protocol/config validation before resuming. Completed generations and completed game files must be reused; only missing or invalid checkpoints may be rerun. The current controller and validation paths are managed outside this checkpoint; inspect the active controller state before starting any work.

## Controls and limitations

DCL remains disabled pending specific oracle export approval and a fresh source audit. Do not use DCP as a workaround for that blocked export. DCP is an opt-in experimental entrant: it tracks public named costs and maritime trades, and credits only the latest non-seven roll when the unchanged board and aggregate post-roll bank demand make the payout conservative. Hidden events clear typed certainty without reading hidden payloads; older rolls and ambiguous production histories are skipped. This is not a full historical replay.

The code and readouts are not a completed campaign result. Preserve unrelated user documents and do not overwrite existing result directories. Before any future queue action, use a fresh immutable snapshot, verify the complete relevant source hash, use disjoint deterministic seeds, and record the exact lineup, gate, stage, worker count, and output path.

Related evidence: [DCP oracle](dcp-oracle/), [ledger oracle](../ledger-oracle/), and [search readouts](.).

## Latest recovery state

At 16:13:03 UTC, 16 stance files were complete. The per-120 table is: baseline 47/33; T1 63/34; T1.5 44/25; T4 55/25; T6 48/26; own 39/26; relative 46/27; paranoid 49/27. T1 is the sole eligible stance under the current screen.

The post-container input mount handoff is corrected (`future:/post`, `validation:/study`). A prior validation attempt failed before games because its aggregate reader expected top-level points while the artifact stores rows; post-stance is fixing this with real-artifact tests. A fresh 1024-game confirmation is pending relaunch after the aggregate-reader fix; no games had started in the failed attempt. No screen rerun is authorized by this checkpoint. Operational messages should use follow-up tasks when continuity is required; sending a message alone does not restart a completed agent.

## Operations proof at 16:22:06 UTC

Operations verified a coherent `post` source and `post/src` mount. The T1 fresh confirmation has been active since approximately 16:18:41 UTC (about 3:25 elapsed at verification), covering both 1024-game gates with seed base 2100000 and 30 workers. Earlier packaging, schema, and fingerprint failures started no games; the completed stance screen remains preserved. The exact new post output path and source hash were not included in this proof and must be obtained from operations before being recorded. T1 source is treated as immutable. Bank/DCP follow-on source verification remains separate.

## Latest handoff state

The latest T1 confirmation is closed and rejected at 445/1024 versus AB2 and 263/1024 versus shipped. The follow-on watcher attempted handoff but the container failed before games with `No module named hexset.catanatron`; the follow-on source mount was incomplete or nested incorrectly. Recovery is preparing a packaging retry with the complete certified source rooted at `/followon/src` and a shared import preflight. Until that preflight passes, follow-on games remain at zero and the handoff is pending.

The packaging failure is confirmed as a permission mismatch, not a missing package: the follow-on snapshot is owned by UID/GID 1000/1000; `src` is mode 705 and `src/hexset` is mode 700. Image `58c092875440` defaults to user `hexset` (UID/GID supplied by the image), so `/post/src/hexset` is not traversable. Recovery should align the final preflight UID with the snapshot ownership or stage a narrowly readable isolated copy; broad tree permission changes are unnecessary.

## Follow-on audit at 18:00 UTC

The certified DCP continuation ran with the pinned oracle and cached eight bank/control screens. DCP scored 57/120 (47.5%) against AB2 and 23/120 (19.2%) against shipped Heximax, so it returned `NO_CANDIDATE`; no confirmation or holdout was scheduled. The DCP screen artifact uses source `f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea` and disjoint seeds 3,000,000/3,100,000.

The adaptive evolution run completed generation-00 discovery (12 candidates × 120 games × 2 gates = 2,880 records), then was stopped at 17:53:10 UTC after an aggregation defect was confirmed. Its checkpoint had `complete: true`, `promoted: []`, and empty summary rows because the final 300-game summarizer discarded 120-game discovery rows when no candidate was promoted. `next_generation()` then fell back to `previous[:1]`, producing generation-01 candidates all descended from `g00-p00`. The 582 generation-01 partial records are ancillary and excluded. No validation was run.

The readiness fingerprint `9d1bf3f...` and controller fingerprint `11f2ae...` are distinct named SHA-256 schemes over the same 104 Python files: the former hashes absolute-path `sha256sum` lines; the latter hashes relative POSIX path, NUL, file bytes, NUL. A local migration archive preserves the original checkpoint/raw records and an explicit source-file hash list. A repaired controller passed local recovery tests but remains pending strict stale-checkpoint/provenance review and explicit upload authorization. The 30-worker pool is idle; no validated winner exists.

The separate fair-budget screen completed four literal width-12 jobs with source `11a0cbfef3280c5a9040ad6bcfbd3dff3dbf10c636b0f4e159d1b0073a47ec19`: k1 control (2,400 nodes, k=1) scored 59/120 AB2 and 25/120 shipped; k4 candidate (9,600 nodes, k=4) independently scored 59/120 AB2 and 25/120 shipped. Both miss the strict targets, so no fair fresh validation was run. The results are archived with their manifest and per-file hashes.

## Recovery repair audit at 18:42 UTC

The untouched remote baseline was rechecked read-only: checkpoint SHA-256 `e0c5a13a41e166c7919d60c9cac7b7ac00e2eee0ce3efa9359acf344f7a67716`, with 3,462 raw files (2,880 complete generation-00 discovery records and 582 incomplete generation-01 ancillary records). The local archive is `/data/data/com.termux/files/usr/tmp/luna-repair-archive-11f2ae`; its rich migration manifest has 2,880 and 582 per-file entries and local SHA-256 `06f62ae431fd9170415aea118e075e6d74508ce30ff9bf1c1b72a764b11d4bb2`; the canonical raw-manifest sidecar derived from those same files is SHA-256 `20c307eabdb4b0cd9eb2feebfb36fa8e6a3c21caf92c0bfc99a205eb643ce5d3` (the exact checkpoint identity). Generation-00 and generation-01 byte-stream digests are `df98603b6f140ce07f16f3977e9725003d169498befa03e79f6e6e66ca050ce8` and `425aef01721239729032954a14223cf471f67ae3efe349ccdc35928f58f1d0b2`. The raw JSON files have not been rewritten.

The final isolated controller is `/data/data/com.termux/files/usr/tmp/fair-evolve-repair/src/hexset/catanatron/evolve.py`, SHA-256 `3b39bf32966aeacf6960019c95257fbb19b16179954c59fdca5f4c5b6cba5c13`. Its strict stale-checkpoint suite passed 5/5, and `py_compile`, CLI help, worker registration, and the candidate-color identity test passed. A disposable copy containing the actual 2,880 discovery files requested exactly 1,440 confirmation jobs for `g00-p06`, `g00-p02`, `g00-p04`, and `g00-p03`, no discovery jobs, and retained all 582 generation-01 files as explicitly excluded archive data. The mock confirmation records are confined to that disposable copy and must never be used as campaign evidence. Production resume still requires the final controller plus an honest sidecar manifest over the pristine remote files; no repaired code has been uploaded and no production workers are active.

The opt-in future structural family is queued locally under `src/hexset/bots/heximax/structural_features.py` and `src/hexset/catanatron/structuralcfg.py`. It preregisters backup-progress (`+0.45` applied to the residual second-purchase value), award-fragility (`-0.25` against a public longest-road/largest-army holder margin), and their combined arm, with the ordinary `heximax-notrade` control unchanged. These are future screens only; the negative DCP/fair results and the invalid adaptive evolution records do not qualify a candidate.

The current runbook is [luna-evolve-repair/READY.md](../luna-evolve-repair/READY.md). Its canonical raw-manifest digest `20c307e...` and the rich migration-manifest digest `06f62ae...` are now reconciled: they hash different explicitly named serializations over the same verified records. The final user authorization for uploading the isolated controller payload remains pending.

The broader orchestration wrappers also require review before use: `followon_pipeline.main()` currently creates a new `root/evolve/checkpoint.json` in its callback and does not stage the supplied checkpoint/games tree, so invoking it as-is would discard the archived discovery records; `post_stance.py` contains the literal option typo `--promote-top4` where the evolve CLI requires `--promote-top 4`. The production handoff must use a reviewed wrapper or the exact repaired evolve command followed by `evolve_validation.run_fresh`, with the pristine checkpoint and raw tree explicitly mounted/staged.

A validation boundary also needs a reviewed fix: because repaired evolution keeps 120-game discovery rows for unpromoted candidates, `evolve_validation.select_candidate()` must require both gate rows to have the checkpoint's full `total_games` (300) before fresh validation. Its current `len(rows) == 2` check alone would admit an unpromoted discovery-only candidate.

## Final recovery handoff audit at 18:50 UTC

The authoritative local provenance record is `/data/data/com.termux/files/usr/tmp/luna-repair-archive-authority.json`. It binds `/data/data/com.termux/files/usr/tmp/luna-repair-archive-11f2ae/migration-manifest.json` at SHA-256 `06f62ae431fd9170415aea118e075e6d74508ce30ff9bf1c1b72a764b11d4bb2`; all 3,462 listed records verify with zero mismatches, and the pristine checkpoint remains `e0c5a13a41e166c7919d60c9cac7b7ac00e2eee0ce3efa9359acf344f7a67716`. The `20c307e...` digest is the canonical raw-manifest sidecar, while `06f62ae...` is the richer migration-manifest file; they hash different explicitly named objects and are both reproducible from this archive.

The fresh artifact contract defect has a saved reproducer at `/data/data/com.termux/files/usr/tmp/luna-fresh-malformed/holdout-vs-ab2-block-0.json`, SHA-256 `9fcdd81e7ef3089a5ef9479a9f39decf881c7718e6c49e877e45de864f6ed329`. It contains candidate/games/gate/wins only; `evolve_validation._read_fresh()` accepts it despite absent workers, seed, source, and phenotype identity. The reviewed recovery-to-validation wrapper must validate those fields before reusing any fresh artifact.

The new recovery-to-validation wrapper was independently reviewed before launch. Its selection logic requires generations 0 through 5 complete, candidate membership in each generation's `promoted` list, exactly two 300-game gate rows, and matching ordinary Heximax phenotype fields; this is the intended boundary. Its current regression tests still need correction (the fixture puts the later candidate at generation 5 but asserts generation 1, and the fake fresh evaluator omits fields the wrapper now validates). The wrapper also cannot verify source/controller identity inside reused fresh files because `evolve_candidate_eval` does not emit those fields; a sidecar binding or mandatory regeneration is required before accepting existing fresh artifacts.

The reviewed recovery-to-validation wrapper now passes 4/4 tests, including dynamic six-generation finalist selection and fresh worker/seed/phenotype reuse checks. One terminal-state requirement remains: evaluator/selection exceptions currently propagate without writing an atomic error verdict; the wrapper should persist `pipeline-verdict.json` with an explicit `ERROR` status and bound hashes on every failure path, alongside its existing rejection/unresolved persistence.

Final wrapper review note: candidate weights and search bounds are cross-checked against each generation entry, but summary stance/temperature should also be cross-checked against the registered candidate before fresh evaluation. This remains a small fail-closed guard for the wrapper owner; no production launch is authorized.


## Recovery wrapper correction at 19:00 UTC

The final wrapper review closed the remaining summary identity gap: a promoted 300-game summary must now match its registered candidate's `stance` and `temperature` as well as weights and search bounds. The wrapper is at SHA-256 `6e67177389f444850a233ae627d509c9ac95f5d4aa3c887aa21c62650b291e0f`; its ten strict recovery-wrapper tests pass under the absolute source `PYTHONPATH`. It already writes an atomic `pipeline-verdict.json` for PASS, rejection/unresolved outcomes, and exceptions. The earlier entries stating that terminal error persistence or summary stance/temperature checks remained are superseded by this entry.

The saved malformed fresh artifact remains `/data/data/com.termux/files/usr/tmp/luna-fresh-malformed/holdout-vs-ab2-block-0.json` (SHA-256 `9fcdd81e7ef3089a5ef9479a9f39decf881c7718e6c49e877e45de864f6ed329`); the existing evaluator reader accepts its missing worker, seed, source, and phenotype fields, while the wrapper rejects it before reuse. The pristine migration authority remains `/data/data/com.termux/files/usr/tmp/luna-repair-archive-11f2ae/migration-manifest.json` at SHA-256 `06f62ae431fd9170415aea118e075e6d74508ce30ff9bf1c1b72a764b11d4bb2`; the canonical raw sidecar is the separately named `20c307e...` serialization over the same 3,462 verified records.


## Large-sample throughput and queue at 19:10 UTC

A one-game direct worker profile of `DC:heximax-notrade,AB:2,AB:2,AB:2` ran in the pinned image `58c092875440` from the approved `/home/bsm/tmp/hexset-luna-followon-certified` source, seed `190000000`, with one CPU and no pool-wait measurement. It took 33.125 seconds and recorded 75,563,370 calls. Cumulative attribution was dominated by Heximax minimax: `minimax.decide` 32.65 s and `alphabeta` 32.59 s; value evaluation 17.20 s; spectrum expansion/execution 14.14/14.10 s; and game/state/board copies 9.82/9.73/7.49 s. These cumulative figures overlap by call nesting and are not additive. Profile SHA-256 is `27d236f5059220fd5fa3194289614216fb6393d387d979639108f14298e73b22`; summary SHA-256 is `3dd935a78edc29f527fa98f5482267f3634f47f525b0b61f114e219f4e2fdfe5`.

The authorized large queue is running from remote `/home/bsm/tmp/luna-20260910-large/luna_queue.sh` (SHA-256 `1d4203dbc9da62ccac3e472079833fd107f76c361993f4d0d4b2da1ec76f3b0e`) under SSH session `89259`, with output-only fair runner SHA-256 `8bacb89294b1bb8d377ee5a782e4ea28fac82a0f5b06a975acef4f2c56f3e7dc`. Bankext-wide AB2 is active as `luna-bankwide-ab2`, 1,000 games, 30 workers, seed `170000000`. The queue waits for it, then runs bankext-wide shipped at `170100000`, fair-k4 AB2 at `180000000`, and fair-k4 shipped at `180100000`, each 1,000 games and 30 workers, sequentially. The live AB2 container showed 30 child workers at one thread each; its environment had no explicit OMP/OPENBLAS/MKL settings, while all queued subsequent jobs pin those libraries to one thread per worker. All four measurements are fresh assessment data and are excluded from the old 120-game screen and holdout claims.


The queued bank jobs mount `/home/bsm/tmp/hexset-luna-followon-certified` read-only (readiness tree SHA `9d1bf3f060ace5cdfc31ed65ac58fc98a8e9611e37146b5bd7068c4dcd816b4e`); the queued fair jobs mount `/home/bsm/tmp/hexset-fair-18789ed84f64707d8734fc06637747d3aa80488c299469151d3274996c4c999d` read-only (fair runtime source SHA `11a0cbfef3280c5a9040ad6bcfbd3dff3dbf10c636b0f4e159d1b0073a47ec19`). Both use image SHA `58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`, UID 1000, `PYTHONHASHSEED=0`, and output-only writable mounts.


Bankext-wide 1,000-game results now include AB2 `458/1000` (45.8%, 1,689.4 s, artifact SHA `46c8704d68695b9f2b4ee63e4302c5d8f9199608f330426dd64fd3af141c3e98`) and frozen self `259/1000` (25.9%, 219.6 s, artifact SHA `e96c73c7f741572c88a977830c895b2b883716e32260f03268d7b8d4da219825`). The candidate is below the AB2 point target in this fresh assessment; these two gates remain separate and are not combined with earlier screens or promoted to a holdout claim. The sequential queue has advanced to fair-k4 AB2 seed `180000000`; fair-k4 shipped seed `180100000` remains queued.


## Queue state correction at 19:45 UTC

The large queue has completed both bankext-wide gates: AB2 `458/1000` and frozen self `259/1000`, with the artifact hashes recorded above. Fair-k4 AB2 is the sole active 1,000-game main pool, started at approximately 19:41:47 UTC with seed `180000000`; fair-k4 shipped seed `180100000` is the next automatic queue stage. The queue supervisor remains PID `148279` / SSH session `89259`; no source or policy code was changed in these measurements.


## Post-queue discovery continuation staged at 19:59 UTC

The certified original controller CLI and source hash were checked read-only. A concrete isolated plan is [luna-discovery-3000-PLAN.md](luna-discovery-3000-PLAN.md), SHA-256 `15b68c2717d53a48cbd6c7dbad934b5932f15a418f37f294be51ae129bb4195e`. It records source tree readiness `9d1bf3f...`, original controller file `9ac7fd9...`, original `_source_hash()` `11f2ae...`, image `58c092875440...`, isolated root `/home/bsm/tmp/luna-20260910-discovery-3000`, and exact 5,760-game generation-3000 discovery command. Generation 3000 is a seed offset, not prior generations; expected AB2 seeds are `310000000..310000239` and shipped seeds `310050000..310050239` for each of twelve candidates. The run has no confirmation stage and its raw records are primary; promotion/offspring metadata is excluded from inference. Future structural families reserve 400m+ seed ranges.

A detached supervisor is staged at `/home/bsm/tmp/luna-20260910-discovery-3000/luna_postqueue_discovery.sh`, SHA-256 `6e11d4a6e327f2e7ec7435a7955079d98d7065d6a49f42c74414fc1fa6f74666`, PID `150100`. It currently logs `WAITING_FOR_CURRENT_QUEUE`. Before launch it requires `QUEUE_COMPLETE`, both fair-k4 1,000-game artifacts with source `11a0cbfe...` and exact seeds `180000000/180100000`, and no active current queue container; it then launches one 30-worker original-controller container with thread caps and writes terminal state. The current four-job queue and containers were left untouched.

## Discovery continuation guard correction at 20:12 UTC

The waiting continuation was corrected before its queue guard could pass. The fair prerequisite validator now imports `hexset.catanatron.fair_budget` from the approved fair source mounted at `/study`, reads the two fair-1,000 artifacts from `/queue`, and writes `fair-prereq-verdict.json` to the isolated discovery root mounted at `/out`. It validates protocol, source fingerprint, candidate, gate, seeds, worker count, players, phenotype, self opponent, and a complete 1,000-game win total. It computes the two-sided 95% Wilson interval and returns `PROCEED_DISCOVERY` only when at least one gate's interval upper bound is strictly below its target; otherwise it writes `NEEDS_FAIR_FOLLOWUP` and exits 10. The validator is `/home/bsm/tmp/luna-20260910-large/validate_fair_prereq.py`, SHA-256 `b0207aa717ea25d924c08486fcd7514d62f899f5e0e4b039c1f130449912a672`.

A disposable pinned-image preflight using fair-source mount `/home/bsm/tmp/hexset-fair-18789ed84f64707d8734fc06637747d3aa80488c299469151d3274996c4c999d` and a separate 500/1,000 fixture returned exit 10 and persisted `NEEDS_FAIR_FOLLOWUP`; both intervals were [0.4690696004, 0.5309303996]. Fixture input SHA-256 values were `3d1741690a49dc2fda68a4d3d8bad83f9e6bfa922dc092880612f241ca4230b0` (AB2) and `14e961b61511459fd4b9c3371070cdb3b3dae96b4f1f3ca2ae4d122c2728874e` (shipped). The fixture was kept outside the large-output root.

The corrected supervisor is `/home/bsm/tmp/luna-20260910-discovery-3000/luna_postqueue_discovery.sh`, SHA-256 `bbaefc2b474625c95a38a695178d04bf4f27dbc2855b77875c9ec99217c9e366`, running as PID `151431` and waiting on the unchanged four-job queue. Its Wilson exit-10 branch writes an atomic `NEEDS_FAIR_FOLLOWUP` state and stops the new discovery supervisor; permanent validator errors write `ERROR` and stop rather than retrying. Before discovery it independently recomputes the original certified controller source hash inside the pinned image and compares it with `11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb`. No discovery container has launched.

The approved fair-k4 AB2 artifact completed at 434/1,000 (43.4%, 95% Wilson interval 40.4%–46.5%), runtime 1,665.7 seconds, source `11a0cbfef3280c5a9040ad6bcfbd3dff3dbf10c636b0f4e159d1b0073a47ec19`, artifact SHA-256 `8d56420a5daefbfde33fc03d48242ed2fecbab96b31139bb74d8fd0f6670d5ce`. Its upper interval is below the 50% AB2 target, so a valid shipped artifact will permit the isolated discovery branch under the stated guard; it does not establish a shipped-gate result or a comparative control claim.

The certified generation-3000 proposal manifest was generated read-only inside the pinned image from the original source and saved at `/home/bsm/tmp/luna-20260910-discovery-3000/expected-phenotypes.json`, SHA-256 `41ec094177f72efc9dee3a52469e0bcb5353fdaa6b62f4603a9d308508d2a2f8`. It contains IDs `g3000-p00` through `g3000-p11`, source hash `11f2ae...664eb`, and the exact AB2/shipped seed vectors 310000000–310000239 and 310050000–310050239. The generator script is SHA-256 `2d3f86d5362b6682d0208846b4c439419ede2f6cfbc7e40b7f757be11a3cab8c`. The manifest is metadata only; no discovery game records or synthetic fixture files were copied into the experiment root.

The first post-upload SSH background process did not survive session teardown; it produced no launch or state output. The same corrected supervisor was then relaunched with `setsid nohup` and is live as PID `151850` at 20:14:29 UTC. This changed only the supervisor process, not the strength queue or any game container; the earlier PID `151431` is superseded.

## Fair prerequisite completion and discovery launch at 20:17 UTC

The fair-k4 shipped gate completed independently: 246/1,000 (24.6%), report Wilson interval 22.0%–27.4%, runtime 328.4 seconds, seed `180100000`, artifact SHA-256 `944bd4a77ced1445695454d1373c8aaae1649d505920ec072c794373d6858901`. Together with fair-k4 AB2 434/1,000 and upper Wilson bound 46.4913%, the validator recorded `PROCEED_DISCOVERY`; the shipped upper bound was 27.3632%. The two fair artifacts were identity-validated from the approved fair source before proceeding.

The isolated discovery container `luna-evolve-discovery-3000` is running from the certified original source, PID `152065`, container ID `c3973970fb053c0241531f8bedfd234c51f9986bfea67f574f11e257a98a7edf`, with 30 CPUs/workers and one thread per numeric library. Its exact command is recorded above. At 20:16:17 UTC it had written 55 raw game records; at 20:17 UTC the read-only count was 101. The source hash guard logged the expected `11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb`.

A second stale supervisor raced the first launch and received Docker name-conflict exit 125; it did not start a second container or write game records. The active first container was retained, and the state was atomically corrected to `RUNNING` with the active container ID. The duplicate attempt is recorded in the supervisor log and is an orchestration incident only; discovery records remain from the single active container.

The supervisor's queue guard was strengthened for future restarts: if `QUEUE_COMPLETE` is absent and all four expected queue containers are absent for three consecutive 60-second checks, it atomically records `ERROR` stage `current_queue`, code 5, message `queue_process_died_or_incomplete`, instead of waiting forever. The updated supervisor SHA-256 is `5cfb770bbec46897b465eead4bc9aed9296b4b7c6235babca5ba425debfaea4d`; it was installed while the already-running discovery container continued unchanged. The active run's source and game code remain the original certified snapshot.

At 20:20 UTC the active discovery container had PID `152065`, Docker PID `58887`, CPU cap 30, and 164 raw records. The only other named containers visible in the read-only Docker audit were exited historical jobs; `ecstatic_bose` was absent from this Docker context, so no sidecar status or log was fabricated. The active discovery container was not interrupted.

The authoritative future supervisor hash is now `0f77c5ff176f10f4a975d2b1d66d9c6772115aacdfe1ec694f14e6b80505dd24`. It adds an atomic `mkdir` lock at `.supervisor.lock` (duplicate invocations exit 6 without overwriting state), retains the three-check queue-death guard, and was installed while container `luna-evolve-discovery-3000` remained running. The prior `5cfb...` file is superseded for future launches; the active process had already passed its guard and was not restarted.

## Supervisor race and live discovery audit at 20:22 UTC

The live discovery Docker command is owned by host process `152065`, whose parent shell is PID `151344` (`bash -s`, session/process group `151344`, started 20:11:09 UTC). The container started at 20:15:14.355998754 UTC, has ID `c3973970fb053c0241531f8bedfd234c51f9986bfea67f574f11e257a98a7edf`, Docker PID `58887`, status `running`, and CPU cap `30000000000` nanocpus (30 CPUs). The first waiting supervisor reached the launch at 20:15:14; a second previously detached waiter reached it at 20:15:34 and received Docker name-conflict exit 125. No second container was created, and the duplicate process is no longer present. The race occurred because the earlier waiting process predated the lock; the final script now acquires an atomic lock before waiting.

At 20:22:26 UTC, the isolated discovery root contained 543 raw records of the expected 5,760. The state file truthfully records `RUNNING`, the active container ID, source hash `11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb`, fair prerequisite verdict path, and no exit code. The race/error remains in the supervisor log as an infrastructure event; raw records were preserved.

The final future supervisor was uploaded atomically through a temporary remote filename then `mv`, so an active shell could not observe a truncated script. Its lock cleanup now removes `pid` only when it matches the current shell PID, then removes the lock directory. Final installed SHA-256: `983ab5da5db2a6fb1272d78c71e0f9100c344a794b41ddef2ce624fd913d9f43`. The currently running supervisor process had already read the prior script and was not restarted.

## Read-only discovery progress audit at 20:30 UTC

At `2026-09-10T20:29:49Z`, the active container `luna-evolve-discovery-3000` remained running from the certified source with source hash `11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb`, Docker ID `c3973970fb053c0241531f8bedfd234c51f9986bfea67f574f11e257a98a7edf`, host PID `152065`, Docker PID `58887`, and `30,000,000,000` nanocpus. `docker top` showed one Python controller plus 30 Python worker processes, each with one thread and approximately 97–98% CPU; the process command and mount identity matched the planned generation-3000 run.

A read-only audit validated all 1,106 raw files against `expected-phenotypes.json`: source fingerprint, candidate IDs, exact filename, gate, discovery stage, generation 3000, game index, expected seed, uniqueness, and candidate-win field. There were zero validation errors or duplicate keys. Four candidate/gate groups are complete at 240 games and are the only groups with reported wins so far: `g3000-p00` AB2 103/240 (42.9167%), shipped 62/240 (25.8333%); `g3000-p01` AB2 104/240 (43.3333%), shipped 72/240 (30.0000%). `g3000-p02` AB2 has 146 records and is incomplete; all other groups are incomplete or empty. No partial group was used for a final rate, promotion, or optimality claim.

The audit script is `/home/bsm/tmp/luna-20260910-discovery-3000/luna_discovery_progress_audit.py`, SHA-256 `40b808e43d36ddc8f99f1e87e958a25bb5a1e31b09a1e98e6e0c556cba482801`. It mounted both source and discovery data read-only and wrote no campaign output.

## Generation-3000 discovery and p06 follow-up at 21:30 UTC

The isolated generation-3000 discovery completed cleanly with all 5,760 raw
records (12 candidates × 2 gates × 240 games), source hash
`11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb`, and
manifest SHA-256 `41ec094177f72efc9dee3a52469e0bcb5353fdaa6b62f4603a9d308508d2a2f8`.
Selection maximized `min(AB2 rate - .50, shipped rate - .25)` with ascending
candidate-ID tie breaking. `g3000-p06` was selected at 116/240 AB2 and 78/240
shipped, joint margin `-0.0166666667`, ahead of every other candidate.

The discovery controller exited normally at 21:22:20 UTC. Its raw archive is
`/data/data/com.termux/files/usr/tmp/luna-discovery-3000-pristine.tar`, SHA-256
`0882506fbd11a1b89b8c266ce1c9158fd872920b04e6eed53b618585b8994894`. The
immutable selection sidecar is
`/data/data/com.termux/files/usr/tmp/luna-discovery-3000-pristine/g3000-selection-sidecar.json`,
SHA-256 `e04e4f2345e1bd73a0326d243f45fd32c81dabd0cc07f1180ca068e3e8c3f57e`.
The sidecar records declared scarcity separately from the effective baseline
scarcity `0.91 * 2.785 / 36 = 0.07039861111111112`, because the existing
registration path excludes candidate-specific `scarce` values.

The approved p06 fresh assessment is active in remote container
`luna-discovery-followup-p06` (ID
`11ca60b189c66f2da8dca07ba71d7b61c23249908876244d8b78267ae54aacad`), started
21:28:37 UTC, with 30 workers, source mounted read-only from
`/home/bsm/tmp/hexset-luna-followon-certified`, and output mounted at
`/home/bsm/tmp/luna-20260910-discovery-3000/followup`. It runs AB2 seed
`500000000` followed serially by shipped seed `500100000`; no result is
interpreted as a pass from point rate alone.

The existing V1 production-cache benchmark remains separate evidence: its
paired eight-game wall reduction was 2.7446% and is not combined with strength
measurements. A DCP 1,000-game follow-up plan is recorded in
[dcp-1000-followup-plan.md](dcp-1000-followup-plan.md), using certified
public-ledger source `f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea`,
image `58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`,
and disjoint seeds `410000000`/`410100000`. It is deferred until p06 exits
normally and awaits parent review; it has not been launched.

## DCP 1,000-game deferred follow-up identity at 21:31 UTC

The certified DCP phenotype was preflighted read-only in image
`58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be` from
`/home/bsm/tmp/hexset-luna-followon-certified`. The imported class is
`PublicProductionLedgerPlayer` with MRO
`PublicProductionLedgerPlayer -> PublicLedgerPlayerMixin -> DevCatanPlayer -> Player`;
`_public_production=True`, and its mixin initializes `_public_book=None`,
`_public_token=None`, and `_public_history_cursor=0`. The bridge resolves
`DC:heximax-notrade` through the native Entrant
`kind=heximax, depth=2, width=6, max_nodes=None, k=1, mode=notrade,
max_trades=0`, with the native win stance and temperature defaults. The bridge
forces `max_trades=0`; maritime trade actions remain enabled and are observed by
DCP's public ledger.

The exact certified source hashes are `public_ledger.py`
`f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea` and
`dclcfg.py` `bc8d23876693434009c0ec9847da25bde7dbd4257473bab05518e40bb8f4ac0d`.
The 30-game oracle on this same public-ledger hash verified typed lower bounds
against private hands, public hand-total invariants, and zero fallbacks.

The full deferred identity sidecar is
[dcp-1000-followup-identity.json](dcp-1000-followup-identity.json), SHA-256
`5982145e66015c3ca6548a00463db22a291feb291169faccd21582ee5bd8a7c2`. The
fail-closed dependency queue is
[dcp-after-p06-1000.sh](../../scripts/dcp-after-p06-1000.sh), SHA-256
`b4a08b02969e2181b57eaf91defda7c6b63b48854af961058ca28f454c6924fe`; it is
prepared but inactive. It waits for p06 to exit with status 0, verifies both
p06 1,000-game outputs against the selected effective weights, temperature,
gates, seeds, workers, lineups, and row counts inside the pinned image, then
runs DCP AB2 seed `410000000` and shipped seed `410100000` serially with one
30-worker pool. Any failed precondition or nonzero DCP exit stops the queue.

The DCP identity sidecar was extended after a pinned-source preflight to bind
DCP's actual `entrant_spec=heximax-notrade` and spawned Heximax fields. Its
updated SHA-256 is `cde5daf6e6d5ec0b42bb5d4fe12f53a225572c9054d56ed8ff1fe102c9b6c027`.
The spawned bot is `depth=2,width=6,max_nodes=600,k=1,mode=notrade,
max_trades=0,stance=win,placement=true`, with effective weights
`production=2.785, diversity=0.358, scarce=0.07039861111111112,
buy_progress=0.45, road=0.1237, knight=0.1026, spare_card=0.1065,
robber_risk=-0.30, port=0.03063`; its `temperature` input is `None`, which
means the pinned `WIN_TEMPERATURE=2.476644394795811`.

The inactive DCP dependency queue was tightened to verify the pinned image ID
and certified `public_ledger.py` SHA before validating p06 outputs. Its updated
SHA-256 is `bd5af9dea84ccbe529eefefdbf983f5b95f71eb8bab3f804afb05cf709bbf03e`;
the earlier `b4a08...` script digest is superseded.

The deferred queue now retains DCP containers as `luna-dcp-ab2-410000` and
`luna-dcp-shipped-410100`, captures their Docker IDs/start/finish timestamps,
and writes an output SHA-256 execution manifest after both gates. The current
inactive script SHA-256 is `75ebf795a4d46754a177202d5a89f9e751d82fe4a409d44ded2c814f2919c2da`.
It also pins and verifies the p06 sidecar SHA `e04e4f2345e1bd73a0326d243f45fd32c81dabd0cc07f1180ca068e3e8c3f57e`,
full evolution source `11f2ae...`, dclcfg SHA, public-ledger SHA, and image
ID before starting DCP.

The inactive DCP queue's p06 artifact validator now follows native
`StatisticsAccumulator` semantics: winner counts must be nonnegative integers
with total at most the requested games; point arrays must have equal length to
that completed-winner total, and any residual games are recorded as documented
draws rather than mislabeled wins. A completed native 1,000-game artifact
verified this format directly: wins `Color.RED=458, BLUE=149, ORANGE=186,
WHITE=207` sum to 1,000, and every point list has length 1,000. The queue
script's updated SHA-256 is
`30504fe504be8dad45578959c5ee7e783a3f973891a18ecf272e4950daa602a1`.

## DCP queue activation after review at 21:47 UTC

The revised generic queue was staged with no private controller source. Its
local and remote script SHA-256 is
`210011e604a61361771e8e6e23023b5c453cb888e76d5e7607d1bef8f97b491d`.
The immutable DCP identity sidecar was copied alongside the future outputs at
`/home/bsm/tmp/luna-20260910-discovery-3000/dcp-followup/dcp-1000-followup-identity.json`
with SHA-256 `cde5daf6e6d5ec0b42bb5d4fe12f53a225572c9054d56ed8ff1fe102c9b6c027`.

One detached waiter is active as PID `157241`, command
`/home/bsm/tmp/luna-20260910-discovery-3000/dcp-after-p06-1000.sh`, with lock
owner PID `157241`. At the activation check, p06 container
`luna-discovery-followup-p06` was still `running` with exit code 0; both retained
DCP containers were absent and the queue log was empty. The waiter therefore
has not started DCP. It will require p06 terminal status 0, exact p06 artifact
identity, sidecar SHA, full source/dclcfg/public-ledger/image hashes, and native
wins/points validation before starting the retained containers
`luna-dcp-ab2-410000` and `luna-dcp-shipped-410100` serially.

## Placement semantics correction at 21:49 UTC

A certified-source introspection compared actual DCP and stock
`DC:heximax-notrade` instances independently. Both resolve `entrant_spec` to
`heximax-notrade` and spawn the same effective Heximax: `depth=2,width=6,
max_nodes=600,k=1,mode=notrade,max_trades=0,stance=win,temperature=None,
placement=True`, with the baseline no-trade weights. `placement=False` belongs
to the Entrant metadata; `_spawn()` omits that argument, so `heximax()` uses
its default `placement=True`. This is a metadata distinction, not a measured
policy difference.

The actual setup branch is:

```python
if game.phase is Phase.SETUP_SETTLEMENT and self.placement:
    options = options_for(game)
    chosen = best_opening(
        game.state(seat, hidden=False), seat, [a.a for a in options]
    )
    return Action(ActionType.SETUP_SETTLEMENT, chosen)
```

`best_opening` is `hexset.placement.best`, a separate public opening prior. It
ranks each candidate greedily by `pips + 1.19 * distinct_resources + 0.91 *
scarce_resources`, with vertex-index tie breaking. It does not use
`NO_TRADE_WEIGHTS`, candidate weights, or hidden hands. The same prior is active
for DCP, stock, and p06's spawned Heximax.

## p06 1,000-game follow-up and DCP handoff at 21:57 UTC

The approved p06 follow-up exited normally at `2026-09-10T21:52:43Z`.
Read-only copies of both fresh artifacts are archived at
`/data/data/com.termux/files/usr/tmp/luna-p06-1000/`; their bytes match the
remote files. The AB2 artifact is
`g3000-p06-vs-ab2-1000.json`, SHA-256
`3a704b10823c9ccf0b10938655e1485bb921d65fd34d2cabecafddc41583ba91`, with
`447/1,000` candidate wins and no draws (Wilson 95% `41.64%–47.80%`). The
shipped artifact is `g3000-p06-vs-shipped-1000.json`, SHA-256
`5c8548fea85c97eb9b8f46da2550a013e149dbcef1ae8351f302f9c2addda837`, with
`269/1,000` and no draws (Wilson 95% `24.24%–29.73%`). Both retain the
candidate, effective weights, temperature, gate, seed, and 30-worker identity.
The AB2 upper bound is below 50%, so p06 receives no holdout continuation.

The detached DCP waiter remains PID `157241`, with its lock owned by that PID.
It passed the p06 terminal guard and started the retained container
`luna-dcp-ab2-410000` at `2026-09-10T21:53:16Z`; it is running the approved
1,000-game AB2 gate with seed `410000000` and 30 workers. The shipped DCP gate
has not started. This is an operational phase transition only; no DCP result
is available yet.

The 40,803-game opening-fit claim has no retained dataset, generator command,
seed, engine/version, source hash, or trading-mode metadata. The repository
therefore cannot establish whether that cohort used trading, no-trade, or a
mixture. The separate 5,000-game trading and 5,000-game no-trade evaluator-fit
cohorts in `docs/readouts/heximax-fit/README.md` are not evidence about the
40,803 opening cohort.

No-trade evaluator port ablations are independently evidenced by historical
`sweep-notrade.json` in commit `acd0575` (1,024 games per port rung) and by
`results/07-drop-port.json` (SHA-256
`f54cf6ad9190d97932de7c2693d3a05e9889dde2238e9b5aa07285ee008b7251`). These
ablations set the later evaluator's `port` weight to zero while retaining the
same setup prior; they do not test a port-aware opening prior.

## Round6000 continuation planning snapshot

The existing certified round6000 discovery run remains active in the pinned
source/image with no source or game-policy changes. At the latest read-only
snapshot, container `luna-evolve-round6000` was running with 30 worker
processes plus its controller, a 30-CPU cap, and approximately 2,943%
aggregate reported CPU. It had written 1,126 valid JSON records in the
generation-6000 output, with zero JSON parse errors, 30 total fallbacks, and
98,685 decisions. Only the complete groups `g6000-p00` (AB2 `111/240`, shipped
`70/240`) and `g6000-p01` (AB2 `108/240`, shipped `70/240`) have been inspected
as full groups; the partial `g6000-p02` AB2 group had 166 records and is
excluded from selection. No partial scores have been promoted or used for a
follow-up.

The planned continuation is stored in
`scripts/round6000_continuation_plan.py` (SHA-256
`8c97300ca9798bb50d97c27a0c34d750302bc681aee7dd9545126a5472f242ae`). After
strict validation of all 5,760 records, it will select the maximum
`min(AB2 rate - .50, shipped rate - .25)` candidate and emit, without
executing, serial 1,000-game commands at seeds `620000000` and `620100000`.
The sidecar binds the full effective weights, scarce baseline, placement
metadata, evaluator SHA, input binding SHA, and a filename-plus-bytes digest
of the consumed raw archive. The planner remains `NOT_EXECUTED`.

## Round6000 bridge fallback audit (22:59 UTC)

A second read-only snapshot of the still-running round6000 container found
1,126 raw records, all parseable. The raw `fallbacks` fields summed to 48
across 20 records and 98,685 decisions; this replaces the earlier in-progress
30-fallback count from the preceding 1,098-record snapshot. By group, the
complete p00/p01 groups had respectively AB2 fallbacks `4/20,164` and
`3/20,313`, and shipped fallbacks `11/22,141` and `10/22,185`; p02 AB2 was
complete with `3/20,257`, while p02 shipped had `15/22,418` at the snapshot.
The remaining p03 AB2 group was partial (`37` records, `2` fallbacks).

In the certified bridge, `DevCatanPlayer.decide()` catches only `ValueError`
from `to_catanatron()` and then chooses uniformly from the native playable
actions (`src/hexset/catanatron/player.py:120-126`). The mapping helper raises
when the chosen dev-catan action has no native playable-action match; the
source documents piece-cap and stale-flag rule differences as the known
causes. Raw round6000 artifacts record only aggregate fallback counts, with
no exception text or action trace, so the audit cannot identify the exact
failed action or establish systematic loss of a winning or maritime action.
The fallback rate is low in this snapshot (48/98,685 decisions), but its
random replacement remains a possible source of noise and is retained in the
artifact metadata rather than corrected post hoc.

## Round6000 current progress correction (23:02 UTC)

The active container remains `eaa81c1d6305ac84ef19183d6d3b4fe291b99326cb4ceaa61131b7ceac31cc96` with the same certified source, image, 30-CPU cap, and 30 one-thread workers. The latest read-only count is 1,558 parseable raw records, zero parse errors, 49 aggregate bridge fallbacks, and 137,204 decisions. Complete groups are now p00 AB2 `111/240`, shipped `70/240`; p01 AB2 `108/240`, shipped `70/240`; and p02 AB2 `109/240`, shipped `72/240`. p03 AB2 is partial at `118` records and `51` wins. No partial group has been selected or promoted.

## Fallback reproducer prepared (review only)

The isolated diagnostic is `scripts/round6000_fallback_diagnostic.py`, SHA-256
`3bab374782bf429839e264ae71bca2e102c6c62524917bb634be4a2a13b1fbe8`. It
imports the certified `hexset.catanatron.evolve._play_one` with exact signature
`_play_one(job: tuple[str, str, str, int, int, int]) -> dict`, registers one
manifest phenotype, and monkeypatches only the `hexset.catanatron.player`
module's `to_catanatron` symbol. On `ValueError`, the wrapper records the
exception text, selected dev action, and native playable-action list, then
re-raises the same exception. It restores the symbol in `finally` and compares
all reference outcome/counter fields while ignoring only Catanatron's fresh
UUID.

The chosen real reference is remote
`/home/bsm/tmp/luna-round6000-execution/games/generation-6000/g6000-p00-vs-ab2-discovery-6000-0000.json`, SHA-256
`249abf35dbeceda31260f5bcb906fe92e47d2d2c76a4347c92e503dabc7c448e`:
seed `610000000`, candidate win `1`, decisions `62`, fallbacks `1`, and
points `BLUE=2, ORANGE=4, RED=10, WHITE=5`. The diagnostic has not been run;
the active 30-worker pool is untouched.

The fallback diagnostic was tightened before any execution. Current SHA-256 is
`ca48c2d2c446222709409b16afeb72bf3b8f1cb1c13a79d419fbd9543abbef38`. It now
pins the staged manifest SHA `dd99767f08ac115da9251345113b308ca7f4593bd1b6f6343d32252f1bebc2b3`,
checks its schema/top-level identity and controller-file SHA, checks the exact
`to_catanatron` parameter signature, tags each event with
`playable_actions[0].color`, compares fallback counts only for the candidate's
native color, and atomically persists observed events and failure metadata even
when replay comparison fails. A local manifest/candidate preflight passed for
`g6000-p00`; native replay remains pending review of the upload/run slot.

## Round6000 fallback diagnostic replay (2026-09-10 23:14 UTC)

A one-game replay of the retained real reference
`g6000-p00-vs-ab2-discovery-6000-0000.json` was run in a fresh retained
container using the certified source and pinned image, with the diagnostic
wrapper as the only uploaded file. The first retained attempt failed before
Python startup because the script mount was omitted; it produced no game and
was preserved as an infrastructure audit. The corrected run was
`luna-round6000-fallback-diagnostic-v2`, container
`6ccde0b7c227d170b69ece2b0eb12d05ffcfca7252401251e340aaef0db29f16`, cpuset
0, one CPU, image
`sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`,
started `2026-09-10T23:14:12.512350461Z`, exited 0 at
`2026-09-10T23:14:30.036301349Z`. Diagnostic script SHA-256 is
`ca48c2d2c446222709409b16afeb72bf3b8f1cb1c13a79d419fbd9543abbef38`;
result SHA-256 is
`e7cc57031ab5926d5efd78fcbaf53b4f2c3734f46a1b939864cd87bb5162894d`;
execution manifest SHA-256 is
`14f754243ec459ff207687d26810cbc2d9807db1041dbd8e798eedb47a9dccb6`.

The replay matched the reference exactly: candidate win 1, 62 decisions,
one fallback, and candidate actor color RED. The sole logged exception was
`ValueError: no catanatron playable_action matched: MARITIME_TRADE SHEEP->WOOD`.
The selected DevCatan action was `[13, 2, 0]` (bank trade, SHEEP to WOOD),
while the native playable-action list had no matching maritime action. The
pinned native source confirms `maritime_trade_possibilities` computes a rate
of 4 by default, 3 with a generic port, or 2 with a matching resource port;
it emits a trade only when the hand has that many cards and the bank has the
requested resource. Thus this replay proves a DevCatan/native action-space
mismatch that invokes the random fallback and can discard a candidate trade
intent. The action tuple does not encode the rate or port, so this single
replay does not prove that the failed trade was port-specific or quantify a
strength bias. No strength source, policy, or 30-worker pool was modified.

### Corrected replay context (v5)

The earlier wording that described a “one-card intent” was withdrawn: the
`[13, 2, 0]` tuple names the resource pair only and does not encode quantity.
A second retained replay with logging SHA
`83d99fd0660de07016fdff256dbd919a80711cc9274e60574cd8d2b31af54287` ran as
container `luna-round6000-fallback-diagnostic-v5`, Docker ID
`1bb763433faddc85fcd463b99d1fa7b0739551f98294152d891178d5f816fcd6`, image
`sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`,
cpuset 0/NanoCpus 1000000000, started
`2026-09-10T23:25:43.729654239Z`, exited 0 at
`2026-09-10T23:26:01.447538344Z`; result SHA is
`359e06e94bf6680be9066e939a88fd5ec9030446980e3aa9e569827b926eaa9e`.

The native and translated contexts both had hand `[0,2,3,2,0]`, bank
`[17,16,13,13,17]`, SHEEP port ratio 2, and `has_rolled=true`. Native state
also had `is_road_building=true` and `free_roads_available=1`; translated
HexSet had `free_roads=1` and phase MAIN. The pinned native
`generate_playable_actions` implementation returns immediately through
`road_building_possibilities` when `is_road_building` is true, so it does not
append maritime actions. HexSet `_trade_actions` still exposes bank trades in
MAIN while free roads remain. The actual fallback is therefore the free-road
branch/action-space mismatch, not a card-quantity or bank-shortage failure.
