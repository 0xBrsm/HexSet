# SPDX-License-Identifier: GPL-3.0-only
"""The game page is an installable app: it links a manifest, the manifest
names its icons, and the server answers each of them with its own type. Both
pages show docs/logo.svg as their icon."""

from __future__ import annotations

import json
import re
import struct
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from _page_server import serving
from hexset.server.web import APP_FILES, SOUND_FILES, STATIC_DIR

SRC = Path(__file__).resolve().parents[2]
PORT = 8814


def links(page: str) -> dict[str, str]:
    html = (STATIC_DIR / page).read_text(encoding="utf-8")
    return dict(re.findall(r'<link rel="([a-z-]+)"[^>]* href="([^"]*)">', html))


def png_size(body: bytes) -> str:
    assert body[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", body[16:24])
    return f"{width}x{height}"


@pytest.fixture(scope="module")
def site():
    with serving(PORT) as base:
        def get(path: str) -> tuple[str, bytes]:
            with urllib.request.urlopen(base + path) as response:
                return response.headers["Content-Type"], response.read()
        yield get


def test_the_icon_is_the_logo() -> None:
    assert (STATIC_DIR / "icon.svg").read_bytes() == (SRC / "docs" / "logo.svg").read_bytes()
    assert links("index.html")["icon"] == links("rankings.html")["icon"] == "/icon.svg"


def test_every_app_file_is_there() -> None:
    assert all((STATIC_DIR / name).is_file() for name, _ in APP_FILES.values())


def test_the_game_page_is_installable(site) -> None:
    page = links("index.html")
    content_type, body = site(page["manifest"])
    assert content_type == "application/manifest+json"
    manifest = json.loads(body)
    assert manifest["display"] == "standalone" and manifest["start_url"] == "/"
    purposes = {icon["purpose"] for icon in manifest["icons"]}
    assert purposes == {"any", "maskable"}
    for icon in manifest["icons"] + [{"src": page["apple-touch-icon"], "type": "image/png"}]:
        content_type, body = site(icon["src"])
        assert content_type == icon["type"], icon["src"]
        if icon.get("sizes", "any") != "any":
            assert png_size(body) == icon["sizes"], icon["src"]
    sizes = {icon["sizes"] for icon in manifest["icons"] if icon["purpose"] == "any"}
    assert {"192x192", "512x512"} <= sizes


def test_every_sound_the_page_plays_is_served_and_nothing_else(site) -> None:
    """The page's clips are the `.ogg` files under `static/sounds`, each
    answered as `audio/ogg`, beside the note of their licence; a name that is
    not one of them, or that reaches out of the directory, is a 404."""
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    block = page[page.index("const SOUNDS = {"):page.index("};", page.index("const SOUNDS = {"))]
    named = set(re.findall(r'"([A-Za-z0-9_-]+)"', block))
    # Every clip played is served, and none is shipped that is not played.
    assert named and {f"/sounds/{name}.ogg" for name in named} == set(SOUND_FILES)
    assert "CC0" in (STATIC_DIR / "sounds" / "README.md").read_text(encoding="utf-8")
    content_type, body = site("/sounds/dice-throw-1.ogg")
    assert content_type == "audio/ogg" and body[:4] == b"OggS"
    for path in ("/sounds/nope.ogg", "/sounds/README.md", "/sounds/../index.html"):
        with pytest.raises(urllib.error.HTTPError) as refused:
            site(path)
        assert refused.value.code == 404
