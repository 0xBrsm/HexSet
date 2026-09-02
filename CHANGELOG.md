# Changelog

All notable changes to HexSet-UI are recorded here.

The engine's own history lives at [`engine/CHANGELOG.md`](engine/CHANGELOG.md) until
the two version lineages are unified.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **The engine moved into this repo, under `engine/`.** `engine/hexset`
  (the rules engine, bots, ledger, arena and tuning) and `engine/heximax`
  (the honest handcrafted baseline, its own top-level package) were
  imported with full commit history from `0xBrsm/dev-hexset`. `heximax`
  is now split out of `hexset` in this repo: it registers its entrant
  kind, presets and tuning evaluators with `hexset.arena`/`hexset.tuning`
  on import rather than being imported by either.
- **A single `pip install -e .` from the repo root now provides `hexset`,
  `heximax` and `hexset_ui`.** `hexset` is no longer installed from a
  separate checkout of `dev-hexset`; see the README's "Running it"
  section.
- **`hexset.build_info()`**: version and git commit for a consumer (e.g.
  HexNet's run manifest) to stamp into its own provenance records.

### Changed

- The engine's own test suite now runs in place at `engine/tests` instead
  of a separate `dev-hexset` checkout.
