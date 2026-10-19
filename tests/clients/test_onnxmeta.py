"""`hexset.clients.onnxmeta`: a file's metadata read and rewritten without a
model runtime, the graph left byte for byte as it was."""

from __future__ import annotations

from pathlib import Path

import pytest

from hexset.clients.onnxmeta import _fields, main, modes_of, read_metadata, stamp, with_metadata

FIXTURE = Path(__file__).parent / "fixtures" / "stub-contract8.onnx"


def graph_bytes(data: bytes) -> list[bytes]:
    return [data[start:end] for number, _, start, _, end in _fields(data) if number != 14]


def test_metadata_reads_as_onnxruntime_reads_it():
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(FIXTURE), providers=["CPUExecutionProvider"])
    assert read_metadata(FIXTURE.read_bytes()) == dict(session.get_modelmeta().custom_metadata_map)


def test_a_rewrite_sets_and_replaces_keys_and_leaves_everything_else():
    data = FIXTURE.read_bytes()
    before = read_metadata(data)
    once = with_metadata(data, {"name": "a", "contract": before["contract"]})
    twice = with_metadata(once, {"name": "b"})
    after = read_metadata(twice)
    assert after == {**before, "name": "b"}
    assert graph_bytes(twice) == graph_bytes(data)


def test_bytes_that_are_not_a_model_are_refused():
    for data in (b"", b"not a model", b"\x08\x07"):
        with pytest.raises(ValueError, match="not an ONNX model"):
            read_metadata(data)
    with pytest.raises(ValueError):
        with_metadata(b"not a model", {"name": "x"})


def test_no_modes_is_the_standard_game_alone():
    assert modes_of({}) == ("standard",)
    assert modes_of({"modes": " "}) == ("standard",)
    assert modes_of({"modes": "standard, duel-variant"}) == ("standard", "duel-variant")


def test_stamp_names_a_copy_and_refuses_an_incomplete_name(tmp_path):
    target = tmp_path / "served" / "x.onnx"
    digest = stamp(FIXTURE, target, name="net", modes=["duel-variant"], version="2.0", commit="abc")
    meta = read_metadata(target.read_bytes())
    assert {k: meta[k] for k in ("name", "modes", "version", "commit")} == {
        "name": "net", "modes": "duel-variant", "version": "2.0", "commit": "abc",
    }
    assert len(digest) == 64 and list(target.parent.iterdir()) == [target]
    with pytest.raises(ValueError, match="game types"):
        stamp(FIXTURE, target, name="net", modes=["duel"], version="2.0", commit="abc")
    with pytest.raises(ValueError, match="game types"):
        stamp(FIXTURE, target, name="net", modes=[], version="2.0", commit="abc")
    with pytest.raises(ValueError, match="name, a version and a commit"):
        stamp(FIXTURE, target, name=" ", modes=["standard"], version="2.0", commit="abc")


def test_stamp_writes_the_profile_it_is_given_held_to_a_clients_rules(tmp_path):
    target = tmp_path / "x.onnx"
    stamp(FIXTURE, target, name="net", modes=["standard"], version="1", commit="abc",
          profile={"owner": " octocat ", "description": "A small net."})
    meta = read_metadata(target.read_bytes())
    assert (meta["owner"], meta["description"]) == ("octocat", "A small net.") and "url" not in meta
    for profile in ({"name": "x"}, {"url": "ftp://x"}, {"description": "x" * 101}):
        with pytest.raises(ValueError):
            stamp(FIXTURE, target, name="net", modes=["standard"], version="1", commit="abc", profile=profile)


def test_the_command_line_stamps_and_shows(tmp_path, capsys):
    target = tmp_path / "x.onnx"
    assert main(["stamp", str(FIXTURE), str(target), "--name", "net", "--modes", "standard,duel-variant",
                 "--version", "1.0", "--commit", "abc", "--owner", "octocat"]) == 0
    assert main(["show", str(target)]) == 0
    shown = capsys.readouterr().out
    assert "name = net" in shown and "modes = standard,duel-variant" in shown
    assert "owner = octocat" in shown
