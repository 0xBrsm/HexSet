"""Build a clean wheel and verify modules and browser assets are included."""

from __future__ import annotations

import email.message
import email.parser
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOTS = {"hexset": REPO_ROOT / "hexset"}

# Everything a fresh clone would not have. `.egg-info` is why this builds from
# a copy at all: setuptools takes package data from a stale SOURCES.txt, so a
# wheel built in a working tree can carry files the packaging config never
# asked for -- passing here and failing from a clean checkout.
NOT_IN_A_CLEAN_CHECKOUT = shutil.ignore_patterns(
    ".git", "*.egg-info", "build", "dist", "__pycache__", ".venv", "venv",
    ".pytest_cache", "*.onnx",
)


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory) -> Path:
    """A freshly built wheel. `--no-build-isolation` stays offline on the
    environment's own setuptools, which the importorskip guards;
    `--no-cache-dir` because a cached wheel would make a packaging-config
    change invisible here.
    """
    pytest.importorskip("setuptools", reason="building a wheel needs setuptools")
    workspace = tmp_path_factory.mktemp("packaging")
    clean = workspace / "checkout"
    clean.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(REPO_ROOT / name, clean / name)
    for name, root in PACKAGE_ROOTS.items():
        shutil.copytree(root, clean / name, ignore=NOT_IN_A_CLEAN_CHECKOUT)
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
    return wheels[0]


@pytest.fixture(scope="session")
def packaged_names(built_wheel) -> tuple[str, ...]:
    """Everything inside the wheel, minus its metadata."""
    with zipfile.ZipFile(built_wheel) as archive:
        return tuple(n for n in sorted(archive.namelist()) if ".dist-info/" not in n)


def _metadata(wheel: Path) -> email.message.Message:
    with zipfile.ZipFile(wheel) as archive:
        (name,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        return email.parser.Parser().parsestr(archive.read(name).decode("utf-8"))


def test_the_wheel_describes_itself_to_an_index(built_wheel):
    """An SPDX license expression with the license file beside it (PEP 639),
    the README as the long description, and the classifiers and links an
    index shows."""
    meta = _metadata(built_wheel)
    assert meta["Name"] == "hexset"
    assert meta["License-Expression"] == "GPL-3.0-only"
    assert "LICENSE" in meta.get_all("License-File", [])
    assert meta["Description-Content-Type"] == "text/markdown"
    assert meta.get_payload().lstrip().startswith("#"), "the README is the long description"
    assert meta["Author"]
    classifiers = meta.get_all("Classifier", [])
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert not [c for c in classifiers if c.startswith("License ::")], (
        "a license classifier beside a license expression is refused by an index"
    )
    urls = dict(u.split(", ", 1) for u in meta.get_all("Project-URL", []))
    assert {"Homepage", "Source", "Changelog"} <= urls.keys()


def test_the_frontend_ships_with_the_package(packaged_names):
    """A static asset: it lives or dies by `[tool.setuptools.package-data]`."""
    assert "hexset/server/static/index.html" in packaged_names


def test_every_module_in_the_source_tree_ships(packaged_names):
    expected = {
        f"{name}/{path.relative_to(root).as_posix()}"
        for name, root in PACKAGE_ROOTS.items()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    missing = sorted(expected - set(packaged_names))
    assert not missing, f"in the source tree but not in the wheel: {missing}"


def test_the_wheel_carries_nothing_from_outside_the_package(packaged_names):
    prefixes = tuple(f"{name}/" for name in PACKAGE_ROOTS)
    strays = sorted(n for n in packaged_names if not n.startswith(prefixes))
    assert not strays, f"unexpected files in the wheel: {strays}"
