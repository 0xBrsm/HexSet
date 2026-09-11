# Cheap follow-on queue

After stance completes and its strict 16-file aggregation fails to produce a
qualifying candidate, the future snapshot runs
`python -m hexset.catanatron.cheap_followon --root /study/cheap-followon`.
It evaluates `bankext` and `bankext-wide` for 120 games per frozen gate using
seeds 3,000,000+, and adds `dcl` only when `/study/cheap-followon/dcl-oracle.json`
exists. Missing oracle evidence creates an explicit `dcl-skipped.json` record.

Only a candidate meeting both point targets in both 120-game gates gets a
1,024-game per-gate confirmation and then independent 4,096-game per-gate
holdouts, all from disjoint seed families. A screen or confirmation failure
does not promote the candidate; the parent controller may then continue to
the adaptive evolution branch. The incumbent remains the shipped
`heximax-notrade` policy and all candidate entrants set `max_trades=0`.

The queue modules are deployed in the isolated Wintermute snapshot at
`/home/bsm/tmp/hexset-luna-post-stance-future/src/hexset/catanatron/`.
