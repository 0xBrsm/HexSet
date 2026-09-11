#!/usr/bin/env python3
"""Strictly audit the 3000-generation discovery and prepare follow-up identity.

This is an offline artifact tool.  It never starts a duel or imports the game
runtime.  The raw discovery tree and expected-phenotypes manifest are treated
as immutable inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import tempfile
from pathlib import Path

SOURCE_HASH = "11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb"
IMAGE = "58c092875440"
SOURCE_ROOT = "/home/bsm/tmp/hexset-luna-followon-certified"
BASE_SCARCE = 0.91 * 2.785 / 36
FILE_RE = re.compile(r"^g3000-p(?P<p>\d\d)-(?P<gate>vs-ab2|vs-shipped)-discovery-3000-(?P<i>\d{4})\.json$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - artifact errors must be explicit
        raise SystemExit(f"ERROR malformed JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"ERROR artifact is not an object: {path}")
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--games-root", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--source-hash", default=SOURCE_HASH)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--source-root", default=SOURCE_ROOT)
    a = ap.parse_args()

    manifest_raw = a.manifest.read_bytes()
    manifest = read_json(a.manifest)
    if manifest.get("schema") != 1 or manifest.get("experiment") != "luna-discovery-3000":
        raise SystemExit("ERROR unexpected manifest schema/experiment")
    if (manifest.get("generation"), manifest.get("count"), manifest.get("seed")) != (3000, 12, 300000000):
        raise SystemExit("ERROR unexpected generation/count/base seed")
    if manifest.get("source_hash") != a.source_hash:
        raise SystemExit("ERROR manifest source hash mismatch")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 12:
        raise SystemExit("ERROR manifest must contain exactly 12 candidates")
    by_id = {}
    for c in candidates:
        cid = c.get("id")
        if not isinstance(cid, str) or cid in by_id:
            raise SystemExit("ERROR duplicate/missing candidate id")
        required = {"weights", "stance", "temperature", "depth", "width", "max_nodes", "k", "mode", "max_trades"}
        if not required.issubset(c):
            raise SystemExit(f"ERROR incomplete phenotype for {cid}")
        expected_weights = {"victory_point", "production", "diversity", "scarce", "buy_progress", "road", "knight", "spare_card", "robber_risk", "port"}
        if set(c["weights"]) != expected_weights or any(not math.isfinite(float(v)) for v in c["weights"].values()):
            raise SystemExit(f"ERROR incomplete/nonfinite weights for {cid}")
        phenotype = manifest.get("candidate_phenotype", {})
        for key in ("depth", "width", "max_nodes", "k", "mode", "max_trades"):
            if c.get(key) != phenotype.get(key):
                raise SystemExit(f"ERROR phenotype mismatch for {cid}: {key}")
        by_id[cid] = c

    expected_seeds = {
        "vs-ab2": set(range(310000000, 310000240)),
        "vs-shipped": set(range(310050000, 310050240)),
    }
    for gate, seeds in expected_seeds.items():
        spec = manifest.get("gates", {}).get(gate)
        if not isinstance(spec, dict) or spec.get("games") != 240 or set(spec.get("seeds", [])) != seeds:
            raise SystemExit(f"ERROR manifest seed specification mismatch: {gate}")

    records = {}
    files = sorted(a.games_root.rglob("g3000-p*-discovery-3000-*.json"))
    for path in files:
        m = FILE_RE.match(path.name)
        if not m:
            raise SystemExit(f"ERROR unexpected discovery filename: {path}")
        d = read_json(path)
        cid, gate, index = d.get("candidate"), d.get("gate"), d.get("game_index")
        filename_cid = f"g3000-p{m.group('p')}"
        filename_index = int(m.group("i"))
        if cid != filename_cid or gate != m.group("gate") or index != filename_index:
            raise SystemExit(f"ERROR filename/JSON identity mismatch: {path}")
        if cid not in by_id or gate not in expected_seeds:
            raise SystemExit(f"ERROR artifact identity not in manifest: {path}")
        if d.get("generation") != 3000 or d.get("stage") != "discovery":
            raise SystemExit(f"ERROR wrong generation/stage: {path}")
        if not isinstance(index, int) or not 0 <= index < 240:
            raise SystemExit(f"ERROR bad game index: {path}")
        if d.get("seed") != (310000000 + index + (50000 if gate == "vs-shipped" else 0)):
            raise SystemExit(f"ERROR seed mismatch: {path}")
        if d.get("candidate_wins") not in (0, 1):
            raise SystemExit(f"ERROR candidate_wins must be 0/1: {path}")
        key = (cid, gate, index)
        if key in records:
            raise SystemExit(f"ERROR duplicate raw record: {key}")
        records[key] = d

    expected_total = 12 * 2 * 240
    if len(records) != expected_total:
        raise SystemExit(f"ERROR incomplete discovery: {len(records)}/{expected_total} records")
    summaries = []
    for cid in sorted(by_id):
        rows = {}
        for gate in expected_seeds:
            group = [records[(cid, gate, i)] for i in range(240)]
            seeds = {d["seed"] for d in group}
            if seeds != expected_seeds[gate]:
                raise SystemExit(f"ERROR seed set mismatch: {cid} {gate}")
            wins = sum(d["candidate_wins"] for d in group)
            rate = wins / 240
            target = 0.50 if gate == "vs-ab2" else 0.25
            rows[gate] = {"games": 240, "wins": wins, "rate": rate, "target": target,
                          "margin": rate - target}
        score = min(rows["vs-ab2"]["margin"], rows["vs-shipped"]["margin"])
        summaries.append({"candidate": cid, "rows": rows, "joint_margin": score})
    # Higher joint margin wins; candidate ID is the deterministic ascending tie-break.
    selected = sorted(summaries, key=lambda x: (-x["joint_margin"], x["candidate"]))[0]
    chosen = by_id[selected["candidate"]]
    declared = {k: float(v) for k, v in chosen["weights"].items()}
    effective = dict(declared)
    effective["scarce"] = BASE_SCARCE
    weights_json = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    command_base = (
        "PYTHONPATH=/study/src python -m hexset.catanatron.evolve_candidate_eval "
        f"--candidate-id {shlex.quote(selected['candidate'])} "
        f"--weights-json {shlex.quote(weights_json)} "
        f"--stance {shlex.quote(str(chosen['stance']))} "
        f"--temperature {shlex.quote(str(chosen['temperature']))} "
        "--games 1000 --workers 30"
    )
    sidecar = {
        "schema": 1,
        "status": "READY_FOR_PARENT_REVIEW",
        "selection_rule": "maximize min(rate_vs_ab2-0.50, rate_vs_shipped-0.25); tie by ascending candidate id",
        "selected": selected,
        "all_scores": summaries,
        "weights_declared": declared,
        "weights_effective": effective,
        "effective_scarce_formula": "0.91 * 2.785 / 36",
        "search": {k: chosen[k] for k in ("depth", "width", "max_nodes", "k", "mode", "max_trades")},
        "stance": chosen["stance"],
        "temperature": float(chosen["temperature"]),
        "source_hash": a.source_hash,
        "source_root": a.source_root,
        "image": a.image,
        "manifest": {"path": str(a.manifest), "sha256": sha256_bytes(manifest_raw)},
        "raw_record_count": len(records),
        "followup": {
            "ab2": {"games": 1000, "workers": 30, "seed": 500000000},
            "shipped": {"games": 1000, "workers": 30, "seed": 500100000},
            "launch": "PARENT_REVIEW_REQUIRED",
            "commands": {
                "vs-ab2": command_base + " --gate vs-ab2 --seed 500000000 --out /out/" + selected["candidate"] + "-vs-ab2-1000.json",
                "vs-shipped": command_base + " --gate vs-shipped --seed 500100000 --out /out/" + selected["candidate"] + "-vs-shipped-1000.json",
            },
        },
    }
    a.sidecar.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=a.sidecar.parent, delete=False) as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
        f.write("\n")
        tmp = Path(f.name)
    tmp.replace(a.sidecar)
    a.sidecar.chmod(0o444)
    print(json.dumps({"raw_records": len(records), "all_scores": summaries,
                      "selected": selected["candidate"], "sidecar": str(a.sidecar)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
