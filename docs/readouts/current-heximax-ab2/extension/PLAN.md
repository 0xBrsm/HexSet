# Fixed extension requested after the initial benchmark

The user requested another 400 games per matchup to tighten the headline
intervals. This plan fixes that addition before any new games are played.
It extends the initial 400-per-format study after its results were visible;
the combined Wilson intervals are descriptive, not a sequential stopping test.

- Same policy, dependency, runtime image, native HexSet host and all bot/rule
  settings as the original [plan](../PLAN.md); 30 workers, one BLAS/OpenMP thread.
- Exactly 400 additional 1v1 games: seed 727300000, indices 0–399.
- Exactly 400 additional four-player games: seed 727400000, indices 0–399.
- Fresh independent boards; 200/100 games per focal seat, respectively.
- No parameter changes, replacements, exclusions or further automatic extension.
- Repeat the original off/fast trace preflight on fresh seeds 727920000 (1v1)
  and 727930000 (four players); exclude these 12 preflight games from results.
- Independently replay and audit all 800 new game records, using the original
  audit with only the registered seed constants changed.
- Preserve the original batch and this batch separately; verify source hashes
  match and game seeds do not overlap, then report pooled wins out of 800 per
  format with descriptive 95% Wilson intervals and both batch counts.
