# Served-protocol confirmation of the exchange robber coefficient

**+0.075 beat −0.30 in 272/392 fresh games: 69.39%, 95% Wilson
64.66–73.74%.** Restore +0.075 for exchange pricing on this new served-protocol
evidence. The earlier automatic-clearing studies remain research-only.
This confirms the previously selected candidate against the restored baseline;
it does not revalidate the entire old grid or locate a unique optimum.

Two +0.075 seats faced two −0.30 seats, with cyclic A,A,B,B seating and each
bot playing individually. Candidate entrant wins were 149 and 123; control
wins were 54 and 66. All 392 games finished. Only exchange robber pricing
differed: spare_card stayed 0.15, floor zero, stock depth 2 / width 6 / 600
leaves / k=1 and default stance, temperature and opening prior.

## Actual served protocol and its current limitation

The driver calls production `GameSession.submit` directly. The session uses
`begin_round`, `_broadcast`, `_resolve` and `_execute_round_trade` for one
initial broadcast per turn, recipient accepts/counters/passes and proposer
choice. Initial offers and counters allow three cards per side. Production
`default_offer` uses the engine referee's coverable candidate list. This is
the current embedded server policy, not the older Clio custom view-safe
proposer. No arena game loop or automatic exchange search is used.

Automatic clearing is disabled with game.max_trades=0 while all bot gates
remain enabled. Entering automatic clearing search raises. Every action event
was checked to contain zero automatic trades; served trades are separate
journal steps. Two stock/instrumented full-trace pairs matched, including
round notes. Every native action/chance tape matched independent server-journal
conversion, and every confirmation record passed independent legal replay.

**Current served games hold the unpinned move slider at zero.** The game-level
automatic-off switch forces this in Heximax's profile update; its observer
receives empty automatic-off notifications, not completed served exchanges.
Both sides reproduce that existing production behavior here. This result
neither fixes nor validates adaptive movement under served trading. That
wiring remains an explicit follow-up, rather than being changed silently
inside this coefficient test.

## Verification and budget

[Plan](PLAN.md) and initial driver frozen in `80764af`. Six initial preflight
attempts failed at serialization of the first RoundNote before any confirmation
launched. The [amendment](AMENDMENT.md), frozen in `4757b28`, changed only that
serialization, charged every failed attempt, moved to fresh seeds and reduced
confirmation from 400 to 392. No strength results existed when the amendment
was fixed. Failed journals and the nonzero runtime receipt are preserved.

Baseline source: `36c62009d206f5a04c1db7102a643fb125ef8c96`, SHA256
`2e7b8d58e7a020723e180374397f3235de213b11bd83aaf4ac799724ba802d85`.
Confirmation seed 734000000, indices 0–391 (98 appearances per physical seat).
Preflight seed 734900000, indices 0–1 for stock/instrumented pairs, and
734900100 for two candidate checks. No antithetic repeats or selection on the
confirmation sample. Wintermute Python 3.12.14, 30 workers, one BLAS/OpenMP
thread each, pinned image, network disabled and no Catanatron imports.

Six failed attempts + six successful checks + 392 confirmation games = 404
attempts, 135.31 cumulative evaluation seconds before adoption checks.
Both reserved adopted-default replays matched full native tapes and served
round notes. Final total: **406 attempts and 141.83 seconds (2 minutes 22
seconds)**, including failed preflight work, under the unchanged 408-attempt /
900-second cap. The first saved adoption game initially tripped a tuple/list
comparison in its verifier, although its serialized tape and notes matched
exactly. The verifier was corrected and completed only the unstarted second
game; the first was checked from its existing saved record, not rerun. Independent audit and ordinary unit tests are
separate verification work. No further sweep is launched.

The audit replayed 102,089 legal actions and reconstructed 25,429 public
notifications (all carrying empty participants, matching current served
behavior). It verified source, coefficient vectors, boards, seat balance,
final scores, journal hashes, offer/response counts and the confidence interval.
The coefficient passed 37 focused Heximax tests.

For this **mixed-coefficient matchup**, there were 15,690 offers, 14,026 accept
responses, 9,481 counters, 23,563 passes and 13,347 executed exchanges. Multiple
seats may accept one offer; response counts are not offer acceptance rates.
Mean game length was 65.00 turns including the winning turn. These are not a
four-current-Heximax behavior baseline or a matched human comparison.

## Artifacts and reproduction

[Summary](summary.json), [audit](audit.json), [manifest](manifest.json),
[preflight](preflight-summary.json), [runtime](runtime-receipt.json),
[records and journals](records.tar.gz), [failed preflight](failed-preflight.tar.gz),
[failed runtime](failed-preflight-receipt.json),
[adoption checks](adoption-verification.json), [adoption records](adoption-records.tar.gz),
[first-check receipt](first-check-receipt.json),
[adoption runtime](adoption-runtime-receipt.json), [checksums](SHA256SUMS).

```sh
git worktree add /tmp/served-robber-baseline 4757b28
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/served-robber-baseline/src timeout --signal=KILL 895s \
  python docs/readouts/served-robber-confirmation/run.py --out /tmp/served-robber --workers 30
PYTHONPATH=/tmp/served-robber-baseline/src \
  python docs/readouts/served-robber-confirmation/audit.py /tmp/served-robber
```

The baseline guard deliberately rejects promoted source. The default-adoption
verifier is separate from strength inference and spends only reserved checks.
