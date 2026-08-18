"""The duel's side split, which decides every verdict in `status.md`."""

from __future__ import annotations

from benchmarks.duel import sides
from catan.arena import base_name, lineup_from_names, pooled
from catan.arena import Standing


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
