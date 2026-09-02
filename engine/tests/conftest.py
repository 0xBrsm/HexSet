# SPDX-License-Identifier: GPL-3.0-only
"""Project-wide pytest options.

`--write-census` is the escape hatch for
`test_heximax.test_choices_are_byte_identical_to_the_recorded_census`: that
test is a behaviour-preservation gate, not a spec, so when a change to
`heximax` deliberately changes what it chooses, the fixture has to be
regenerated on purpose rather than hand-edited. Passing the flag makes the
test recompute the census and overwrite
`tests/fixtures/heximax_census_ecb5252.json` instead of asserting against it.
"""

from __future__ import annotations


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--write-census",
        action="store_true",
        default=False,
        help=(
            "regenerate tests/fixtures/heximax_census_ecb5252.json from the "
            "current heximax instead of asserting against it"
        ),
    )
