"""What a real `pip install` actually ships.

Every other way this repo gets imported bypasses packaging entirely: the test
suite puts `src` on the path (`[tool.pytest.ini_options]`), the Docker image
sets `PYTHONPATH=/app/src` and never installs the package at all, and
`pip install -e .` reads the source tree in place. All three work fine with a
`pyproject.toml` that would produce a broken wheel, which is how
`static/index.html` came to be missing from one once — caught only by building
and running the image back when it did install the package, as a
FileNotFoundError on GET /.

So this builds an actual wheel and looks inside it. It is the only test here
that exercises the packaging config rather than the code.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "src" / "hexset_ui"

# Everything a fresh clone would not have. `.egg-info` is the one that
# matters and the reason this test builds from a copy at all: setuptools
# defaults `include-package-data` to true under pyproject.toml and will happily
# take package data from a stale SOURCES.txt left by an earlier
# `pip install -e .`, so a wheel built in a working tree can contain files
# that `[tool.setuptools.package-data]` never asked for. That passes here and
# fails for anyone installing from a clean checkout, which is precisely the
# class of bug this file exists to catch.
NOT_IN_A_CLEAN_CHECKOUT = shutil.ignore_patterns(
    ".git", "*.egg-info", "build", "dist", "__pycache__", ".venv", "venv",
    ".pytest_cache", "*.onnx",
)


@lru_cache(maxsize=1)
def packaged_names() -> tuple[str, ...]:
    """Everything inside a freshly built wheel, minus its metadata.

    Built from a pristine copy of the tree (see NOT_IN_A_CLEAN_CHECKOUT), so
    what comes out is what someone cloning and installing would get rather
    than whatever local build artifacts happen to be lying around.

    `--no-build-isolation` keeps this offline and quick by using the
    environment's own setuptools instead of fetching the pinned one; that is
    what the importorskip guards. `--no-deps` because numpy and onnxruntime
    have nothing to do with whether our own files are included. `--no-cache-dir`
    because pip caches wheels it builds from a local directory, and a cached
    one would make a change to the packaging config invisible here.
    """
    pytest.importorskip("setuptools", reason="building a wheel needs setuptools")
    workspace = Path(tempfile.mkdtemp())
    clean = workspace / "checkout"
    shutil.copytree(REPO_ROOT, clean, ignore=NOT_IN_A_CLEAN_CHECKOUT)
    out = workspace / "wheel"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--no-cache-dir", "--wheel-dir", str(out), str(clean)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"could not build a wheel:\n{result.stdout}\n{result.stderr}")
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0]) as archive:
        return tuple(n for n in sorted(archive.namelist()) if ".dist-info/" not in n)


def test_the_frontend_ships_with_the_package():
    """`static/index.html` is the entire frontend and the one file setuptools
    will not include on its own — it is a static asset inside a package
    directory, so it lives or dies by `[tool.setuptools.package-data]`."""
    assert "hexset_ui/static/index.html" in packaged_names()


def test_every_module_in_the_source_tree_ships():
    """A module dropped from the wheel is an ImportError for anyone who
    installed it and nothing at all for anyone running from source, so
    compare the two directly rather than trusting `packages.find`."""
    expected = {
        f"hexset_ui/{path.relative_to(PACKAGE_SRC).as_posix()}"
        for path in PACKAGE_SRC.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    missing = sorted(expected - set(packaged_names()))
    assert not missing, f"in src/hexset_ui but not in the wheel: {missing}"


def test_the_wheel_carries_nothing_from_outside_the_package():
    """`packages.find` is scoped to `hexset_ui*`; tests/, models/ and docker/ have
    no business in an installed copy."""
    strays = sorted(n for n in packaged_names() if not n.startswith("hexset_ui/"))
    assert not strays, f"unexpected files in the wheel: {strays}"
