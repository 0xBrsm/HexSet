# SPDX-License-Identifier: GPL-3.0-only
"""Create a run directory: `python -m hexset.run.init --mode M --name N -- <flags>`

Everything after `--` is the mode's own flags, parsed by that mode's parser and
frozen with every parameter explicit. The trainers then take the directory and
nothing else, so what a run meant is recoverable from the repository forever
rather than from whichever shell script in `tmp/` happened to survive.

    python -m hexset.run.init --mode league --name fac-r2a \
        --category heats --parent runs/lam095/latest.pt \
        --description "factorial replicate 2, both blocks" \
        --plan plans/heat-protocol.md \
        -- --base /w/runs/lam095/latest.pt --learner "" --learner lr=1.5e-4 \
           --iterations 60 --seed 102

`--category` groups the run under `runs/<category>/<name>`, which a sibling
project had already needed: a flat `runs/` reached 52 directories here before
anyone could tell which were live.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .manifest import freeze


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1 :]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("train", "league", "distill"))
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--category",
        default="",
        help="group the run under runs/<category>/<name>; omit for runs/<name>",
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--repo", default=".", help="where to read the git SHA from")
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--plan", default=None, help="the document registering this run's design"
    )
    parser.add_argument(
        "--parent",
        default=None,
        help="the checkpoint this run continues from, recorded as lineage so it "
        "never has to be recovered by matching iteration numbers again",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing run")
    parser.add_argument(
        "--git-commit",
        default=None,
        help="record this SHA instead of asking git. Needed only where git "
        "cannot read the repo -- a linked worktree's .git names an absolute "
        "gitdir, so inside a container that mounts the repo elsewhere git "
        "fails and the SHA would otherwise be silently absent.",
    )
    parser.add_argument(
        "--git-dirty",
        default=None,
        choices=("true", "false"),
        help="record the tree's cleanliness alongside --git-commit, since a "
        "container that cannot read the repo cannot determine it either",
    )
    parser.add_argument(
        "--allow-missing-sha",
        action="store_true",
        help="write a manifest with no commit recorded. Only for throwaway "
        "smoke runs: a result without a SHA cannot be cited.",
    )
    args = parser.parse_args(argv)

    root = Path(args.runs_root)
    directory = root / args.category / args.name if args.category else root / args.name
    if (directory / "run.json").exists() and not args.force:
        raise SystemExit(f"{directory} is already a run; pass --force to overwrite")

    manifest = freeze(
        args.mode,
        args.name,
        directory,
        passthrough,
        repo=Path(args.repo),
        description=args.description,
        plan=args.plan,
        parent=args.parent,
    )
    if args.git_commit:
        manifest.meta["git_commit"] = args.git_commit
        if args.git_dirty is not None:
            manifest.meta["git_dirty"] = args.git_dirty == "true"
        (directory / "run.json").write_text(
            __import__("json").dumps(manifest.meta, indent=1) + "\n"
        )
    if manifest.meta.get("git_commit") is None and not args.allow_missing_sha:
        raise SystemExit(
            f"refusing to create {directory} with no commit recorded.\n"
            f"  git could not read {args.repo!r}. Run init repo-side (where the\n"
            f"  worktree's gitdir resolves), pass --git-commit <sha>, or pass\n"
            f"  --allow-missing-sha for a throwaway smoke run.\n"
            f"  A run whose SHA is absent is the exact failure this replaces."
        )
    print(f"created {directory}")
    print(f"  run_id     {manifest.meta['run_id']}")
    print(f"  git        {manifest.meta['git_commit']} dirty={manifest.meta['git_dirty']}")
    print(f"  parent     {manifest.parent}")
    print(f"  config     {len(manifest.config)} parameters frozen")
    print(f"\nlaunch with:  python -m hexset.{args.mode} {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
