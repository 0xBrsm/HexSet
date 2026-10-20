"""An ONNX file's `metadata_props`, read and rewritten without a model runtime.

A served model names itself: the server's picker offers it as
`<name>-<version>`, at the game types its `modes` declare
(`hexset.rules.GAME_TYPES` names, comma separated), and `commit` is the code
it was built from. The filename plays no part. Files may share a name as long
as their versions differ. A file may also declare an `owner`, a `url` and a
`description` (`PROFILE_KEYS`), shown with it as a bot's profile is.

Reading parses only the top level of the `ModelProto` protobuf, so listing a
models directory never builds an inference session, and `with_metadata`
copies every other field byte for byte: a stamped file's graph is the file's
graph. `python -m hexset.clients.onnxmeta` shows or stamps a file::

    python -m hexset.clients.onnxmeta show model.onnx
    python -m hexset.clients.onnxmeta stamp in.onnx out.onnx \\
        --name mynet --modes standard --version 1.0 --commit 0123abcd \\
        --owner octocat --description "A small policy net with 128-visit search."
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

from hexset.rules import GAME_TYPES, STANDARD_GAME

__all__ = [
    "SERVE_KEYS",
    "PROFILE_KEYS",
    "read_metadata",
    "with_metadata",
    "modes_of",
    "stamp",
    "main",
]

# `ModelProto` field numbers: `graph` (7) is what makes a file a model,
# `metadata_props` (14) a repeated `StringStringEntryProto` of `key` (1) and
# `value` (2).
_GRAPH = 7
_METADATA = 14
_KEY = 1
_VALUE = 2
_LENGTH_DELIMITED = 2

#: The keys a served file names itself with (`stamp` writes all four).
SERVE_KEYS = ("name", "modes", "version", "commit")

#: What a served file may say about itself besides, shown with it as a
#: server bot's profile is (`stamp` writes those it is given).
PROFILE_KEYS = ("owner", "url", "description")


def _varint(data: bytes, at: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if at >= len(data):
            raise ValueError("not an ONNX model: truncated varint")
        byte = data[at]
        at += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, at
        shift += 7
        if shift > 63:
            raise ValueError("not an ONNX model: varint too long")


def _fields(data: bytes) -> Iterator[tuple[int, int, int, int, int]]:
    """`(number, wire type, start, payload start, end)` for each top-level
    field of a protobuf message, in file order."""
    at = 0
    while at < len(data):
        start = at
        key, at = _varint(data, at)
        number, wire = key >> 3, key & 7
        body = at
        if wire == 0:
            _, at = _varint(data, at)
        elif wire == 1:
            at += 8
        elif wire == _LENGTH_DELIMITED:
            length, body = _varint(data, at)
            at = body + length
        elif wire == 5:
            at += 4
        else:
            raise ValueError(f"not an ONNX model: wire type {wire}")
        if at > len(data):
            raise ValueError("not an ONNX model: truncated field")
        yield number, wire, start, body, at


def _entry(data: bytes) -> tuple[str, str]:
    key = value = ""
    for number, wire, _, body, end in _fields(data):
        if wire == _LENGTH_DELIMITED and number in (_KEY, _VALUE):
            text = data[body:end].decode("utf-8")
            if number == _KEY:
                key = text
            else:
                value = text
    return key, value


def read_metadata(data: bytes) -> dict[str, str]:
    """The `metadata_props` of the ONNX model serialised in `data`, as
    onnxruntime's `custom_metadata_map` reads them. A `ValueError` for bytes
    that are not a model."""
    meta: dict[str, str] = {}
    graph = False
    for number, wire, _, body, end in _fields(data):
        if number == _GRAPH and wire == _LENGTH_DELIMITED:
            graph = True
        elif number == _METADATA and wire == _LENGTH_DELIMITED:
            key, value = _entry(data[body:end])
            meta[key] = value
    if not graph:
        raise ValueError("not an ONNX model: no graph")
    return meta


def _delimited(number: int, payload: bytes) -> bytes:
    def varint(value: int) -> bytes:
        out = bytearray()
        while value >= 0x80:
            out.append(value & 0x7F | 0x80)
            value >>= 7
        out.append(value)
        return bytes(out)

    return varint(number << 3 | _LENGTH_DELIMITED) + varint(len(payload)) + payload


def with_metadata(data: bytes, updates: dict[str, str]) -> bytes:
    """`data` with each of `updates` set in its `metadata_props`, replacing
    any entry of that key; every other field is copied unchanged."""
    read_metadata(data)  # refuses what is not a model before rewriting it
    out = bytearray()
    for number, wire, start, body, end in _fields(data):
        if number == _METADATA and wire == _LENGTH_DELIMITED and _entry(data[body:end])[0] in updates:
            continue
        out += data[start:end]
    for key, value in updates.items():
        entry = _delimited(_KEY, key.encode("utf-8")) + _delimited(_VALUE, value.encode("utf-8"))
        out += _delimited(_METADATA, entry)
    return bytes(out)


def modes_of(meta: dict[str, str]) -> tuple[str, ...]:
    """The game types `meta` declares it plays (`modes`). A file that
    declares none plays the standard game only: no model plays every game
    type unless it says so."""
    modes = tuple(m.strip() for m in meta.get("modes", "").split(",") if m.strip())
    return modes or (STANDARD_GAME.name,)


def stamp(source: Path, target: Path, *, name: str, modes: list[str], version: str, commit: str,
          profile: dict[str, str] | None = None) -> str:
    """Write `source` to `target` named by `name`, `modes`, `version` and
    `commit`, with the `profile` fields given (`PROFILE_KEYS`, held to a
    client profile's rules), through a temporary file and a rename. Returns
    the new file's sha256."""
    from hexset.server.constants import profile_value

    declared = {}
    for key, value in (profile or {}).items():
        if key not in PROFILE_KEYS:
            raise ValueError(f"a served model's profile has no {key!r}; one of {PROFILE_KEYS}")
        value = profile_value(key, value)
        if value:
            declared[key] = value
    name, version, commit = name.strip(), version.strip(), commit.strip()
    modes = [m.strip() for m in modes if m.strip()]
    if not name or not version or not commit:
        raise ValueError("a served model needs a name, a version and a commit")
    unknown = [m for m in modes if m not in GAME_TYPES]
    if not modes or unknown:
        raise ValueError(f"modes are game types, one or more of {sorted(GAME_TYPES)}; got {modes}")
    data = with_metadata(
        source.read_bytes(),
        {"name": name, "modes": ",".join(modes), "version": version, "commit": commit, **declared},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)
    except BaseException:
        os.unlink(temporary)
        raise
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """`python -m hexset.clients.onnxmeta`: `show` a file's metadata or `stamp`
    a copy named for serving. Returns the exit status."""
    parser = argparse.ArgumentParser(prog="python -m hexset.clients.onnxmeta", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="print a file's metadata")
    show.add_argument("file", type=Path)
    stamped = commands.add_parser("stamp", help="write a copy named for serving")
    stamped.add_argument("source", type=Path)
    stamped.add_argument("target", type=Path)
    stamped.add_argument("--name", required=True)
    stamped.add_argument("--modes", required=True, help=f"comma separated, of {', '.join(GAME_TYPES)}")
    stamped.add_argument("--version", required=True)
    stamped.add_argument("--commit", required=True)
    for key in PROFILE_KEYS:
        stamped.add_argument(f"--{key}", default=None, help=f"the {key} shown with it (optional)")
    args = parser.parse_args(argv)
    if args.command == "show":
        for key, value in sorted(read_metadata(args.file.read_bytes()).items()):
            print(f"{key} = {value}")
        return 0
    digest = stamp(
        args.source, args.target, name=args.name, modes=args.modes.split(","),
        version=args.version, commit=args.commit,
        profile={key: getattr(args, key) for key in PROFILE_KEYS if getattr(args, key) is not None},
    )
    print(f"{args.target}  sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
