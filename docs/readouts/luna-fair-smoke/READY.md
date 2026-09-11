# Fair budget runtime preflight — smoke excluded

Source archive SHA256: `18789ed84f64707d8734fc06637747d3aa80488c299469151d3274996c4c999d`
Runtime image: `sha256:58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be`
Snapshot: `/home/bsm/tmp/hexset-fair-18789ed84f64707d8734fc06637747d3aa80488c299469151d3274996c4c999d`
Runtime source hash: `11a0cbfef3280c5a9040ad6bcfbd3dff3dbf10c636b0f4e159d1b0073a47ec19`

The pinned image ran exactly two AB2 smoke duels with UID 1000, read-only `/fair` snapshot, writable `/out`, `PYTHONPATH=/fair/src`, `PYTHONHASHSEED=0`, and one worker: k1 seed `120000000` completed in 30.5s with candidate `0/1`; k4 seed `120000001` completed in 51.7s with candidate `1/1`. Catanatron was `3.3.0 @ d3f4ad05bb78`.

Raw output does not expose `DevCatanPlayer.fallbacks`; normalized artifacts record `fallback_count: null` and preserve that limitation. No fallback telemetry was emitted by the raw report. These files are smoke excluded and no 30-worker screen was run.

Exact duel command template:

```sh
ssh wintermute \"wsl.exe -d Debian -- docker run --rm --network none --user 1000:1000 -e PYTHONHASHSEED=0 -e PYTHONPATH=/fair/src -v /home/bsm/tmp/hexset-fair-<archive-sha>:/fair:ro -v /home/bsm/tmp/hexset-fair-<archive-sha>/artifacts:/out:rw -w /fair --entrypoint python 58c092875440 -m hexset.catanatron.duel --players=<lineup> --num=1 --workers=1 --seed=<seed>\"
```


## Handoff commands (production queue remains stopped)

After operations confirms the DCP/evolution failure handoff and selects this frozen snapshot, run from the snapshot root with the local source first on `PYTHONPATH`:

```sh
export PYTHONPATH=/followon/src
python -m hexset.catanatron.fair_budget --root /study/fair-screen --run > /study/fair-screen.json
HASH=$(PYTHONPATH=/followon/src python -c 'from hexset.catanatron.fair_budget import source_fingerprint; print(source_fingerprint())')
python -m hexset.catanatron.fair_budget_validation --root /study/fair-validation --screen /study/fair-screen.json --source-hash "$HASH" --run
```

The screen command is the only command that starts the four 120-game jobs; validation starts only after the screen JSON is complete and source/phenotype checks pass. Keep the smoke JSON files marked `smoke_excluded`; they are not eligible for promotion.

The local explicit `multiprocessing` spawn harness also resolved both aliases as native Heximax Entrants with exit code 0. `duel.py:186` constructs a real `Pool`, so the one-game runtime checks used one child process; its start method is platform default (fork on Wintermute WSL, forkserver locally), while `player.py:113-115` imports fair registrations before alias resolution under spawn.
