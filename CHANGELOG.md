# Changelog

All notable changes to HexSet-UI are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The engine moved into this repo, under `engine/`.** `engine/hexset`
  (the rules engine, bots, ledger, arena and tuning) and `engine/heximax`
  (the honest handcrafted baseline, its own top-level package) were
  imported with full commit history from `0xBrsm/dev-hexset` (becoming
  HexNet), branch `refactor/hexnet-package-split` — see `engine/README.md`
  for the exact SHAs and how the import was done (`git filter-repo` +
  `git subtree add`). `heximax` was then split out of `hexset` in this
  repo, the same shape as the prior `hexset`/`hexnet` split: it registers
  its entrant kind, presets and tuning evaluators with `hexset.arena`/
  `hexset.tuning` on import rather than being imported by either.
- **A single `pip install -e .` from the repo root now provides `hexset`,
  `heximax` and `hexset_ui`.** Two (now three) package roots in one
  `pyproject.toml` (`where = ["src", "engine"]`), rather than a path
  dependency or a second `pip install -e` invocation — the simplest thing
  that works with plain setuptools. `hexset` is no longer installed from a
  separate checkout of `dev-hexset`; see the README's "Running it" section.
- **`hexset.build_info()`**: version and git commit for a consumer (e.g.
  HexNet's run manifest) to stamp into its own provenance records.

### Changed

- The engine's own test suite (`engine/tests`, previously run from a
  separate `dev-hexset` checkout) now runs in place: `pytest engine/tests`
  is 609 passed/6 skipped (534 hexset, 73 heximax, 2 for
  `hexset.build_info`); `pytest tests` (this package's own suite) is
  unchanged at 173 passed/4 skipped.
