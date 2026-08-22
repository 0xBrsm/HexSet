"""The duel's side split, which decides every verdict it reports."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from benchmarks.duel import _default_workers, sides  # noqa: E402
from catan.arena import base_name, lineup_from_names, pooled  # noqa: E402
from catan.arena import Standing  # noqa: E402


def test_two_checkpoints_arrive_sharing_one_name():
    """The reason `sides` exists, pinned so the fix cannot look gratuitous.

    `entrant_from_name` labels every `network:` spec "network" whatever
    checkpoint it carries, so `lineup_from_names` sees four repeats of one name
    rather than two of each side.
    """
    lineup = lineup_from_names(
        ["network:/runs/a.pt", "network:/runs/a.pt", "network:/runs/b.pt", "network:/runs/b.pt"]
    )

    assert {base_name(entrant.name) for entrant in lineup} == {"network"}


def test_naming_the_sides_separates_two_checkpoints():
    lineup = sides(
        lineup_from_names(
            ["network:/runs/a.pt", "network:/runs/a.pt", "network:/runs/b.pt", "network:/runs/b.pt"]
        ),
        "ppo6-655",
        "ppo4-585",
    )

    assert [base_name(entrant.name) for entrant in lineup] == [
        "ppo6-655",
        "ppo6-655",
        "ppo4-585",
        "ppo4-585",
    ]
    # What the rename is for: two sides to pool, not one.
    grouped = pooled([Standing(entrant.name, 1, 1.0) for entrant in lineup], 4)
    assert len(grouped) == 2


def test_the_weights_survive_the_rename():
    """`spawn` reads `kind` and `weights`; renaming must not touch either."""
    lineup = sides(
        lineup_from_names(
            ["network:/runs/a.pt", "network:/runs/a.pt", "network:/runs/b.pt", "network:/runs/b.pt"]
        ),
        "a",
        "b",
    )

    assert [entrant.weights for entrant in lineup] == [
        "/runs/a.pt",
        "/runs/a.pt",
        "/runs/b.pt",
        "/runs/b.pt",
    ]
    assert all(entrant.kind == "network" for entrant in lineup)


def test_a_checkpoint_duelled_against_itself_still_has_two_sides():
    """The harness check that must read 50%, not a single pooled side."""
    lineup = sides(
        lineup_from_names(["network:/runs/a.pt"] * 4), "ppo4-585", "ppo4-585"
    )

    assert [base_name(entrant.name) for entrant in lineup] == [
        "ppo4-585-a",
        "ppo4-585-a",
        "ppo4-585-b",
        "ppo4-585-b",
    ]


def test_the_arena_path_can_be_called_at_all():
    """A regression guard for the call site, not the helper.

    `sides` was extracted while `_via_arena` still bound a local of the same
    name further down, which makes the name local for the whole function and
    raises `UnboundLocalError` on the first line that uses it. The unit tests
    above all passed while the duel could not run.
    """
    import ast
    from pathlib import Path

    import benchmarks.duel as module

    tree = ast.parse(Path(module.__file__).read_text())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        assigned = {
            target.id
            for inner in ast.walk(node)
            if isinstance(inner, ast.Assign)
            for target in inner.targets
            if isinstance(target, ast.Name)
        }
        assert not assigned & functions, (
            f"{node.name} assigns a local shadowing {sorted(assigned & functions)}"
        )


def test_default_workers_is_one_for_two_bare_checkpoint_paths(tmp_path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.touch()
    b.touch()
    assert _default_workers(str(a), str(b)) == 1


def test_default_workers_is_one_for_two_network_prefixed_checkpoints():
    assert _default_workers("network:/runs/a.pt", "network:/runs/b.pt") == 1


def test_default_workers_is_26_against_a_preset_bot(tmp_path):
    """A scripted opponent cannot batch, and workers=1 left it unfinished."""
    a = tmp_path / "a.pt"
    a.touch()
    assert _default_workers(str(a), "search2-offers3") == 26


def test_default_workers_is_26_for_a_search_wrapped_checkpoint(tmp_path):
    """`netsearch:`/`netgreedy:`/`mcts:` still run a per-lane search."""
    a = tmp_path / "a.pt"
    a.touch()
    assert _default_workers(str(a), "netsearch:/runs/a.pt") == 26


def test_default_workers_is_26_when_neither_side_is_a_bare_network():
    assert _default_workers("search2-offers3", "random") == 26
