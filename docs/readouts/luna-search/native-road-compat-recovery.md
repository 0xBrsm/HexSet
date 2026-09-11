# Native forced-road adapter recovery

Date: 2026-09-10

The retained round6000 replay identified the fallback cause as an action-space mismatch. Native Catanatron had `is_road_building=true` and `free_roads_available=1`; its `generate_playable_actions` path returned only remaining road placements. The translated HexSet state correctly remains `Phase.MAIN` for post-roll semantics, but ordinary Heximax legality also exposed bank trades. The selected `SHEEP -> WOOD` action therefore had no native match and entered the random fallback.

The recovery is opt-in. `Entrant.native_action_compat` defaults to `False`; shipped controls and existing evolution registrations do not enable it. A candidate that explicitly enables the field causes Heximax `_options_in` to restrict a post-roll state with `Phase.MAIN` and `free_roads > 0` to `BUILD_ROAD`. After applying the final free road, the state remains MAIN and normal actions return. If native forced-road state has no placeable road, native itself offers an empty action list; the candidate root raises `ValueError` instead of reopening unrestricted MAIN actions. This is an invalid/unactionable native position guard, not a recovery policy.

Local validation used `tests/test_heximax_native_compat.py`:

```text
PYTHONPATH=src pytest -q tests/test_heximax_native_compat.py --disable-warnings
4 passed
```

Pinned Wintermute validation used image `58c092875440`, `--network none`, and `--cpus 1`, with a staged source mounted at `/study` and `PYTHONPATH=/study/src`. The unit runner printed:

```text
HEXSET_NATIVE_COMPAT_PASS
NATIVE_STATE_TRANSLATION_PASS 11
```

The second check constructed a real Catanatron game state, set the replay hand/bank values `[0,2,3,2,0]` and `[17,16,13,13,17]`, applied the native forced-road flags, and verified translation remained MAIN while preserving ordinary HexSet bank-trade legality. The first runner covered intermediate roads, final-road transition, pre-roll/default behavior, and no-placement root behavior.

Relevant staged local file SHA-256 values:

```text
eff6c659137acddadb042736c1cfc20e3952d127d5250abe7a270a70dcbfd7c0  src/hexset/arena.py
be2e93eb00b217c5f30a2773e6b41fe9049d84fad1368cdf7e417613a6711d62  src/hexset/bots/heximax/search.py
4d3a2c3452b18d45ec8fad6a93d3136f064499c776bb24cd0e8fd268ede531a7  src/hexset/bots/heximax/presets.py
6d0316d57f8b6acafa0b3ada73214bf3c30609a1b94222b12de2a23829cd0464  src/hexset/catanatron/player.py
37ffa6a251a0323eb1594cbc5ab3685ebb6ae3befea59b2a211fc80dfd5d054b  src/hexset/catanatron/state.py
0bdd378c9a8144da930851c8385c136e5061fde549f5b72f99b07c827887e29c  tests/test_heximax_native_compat.py
```

The certified round6000 strength source and active 30-worker runtime were not modified or restarted.
