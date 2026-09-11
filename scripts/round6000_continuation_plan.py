#!/usr/bin/env python3
"""Plan, but never execute, round6000's independent 1000-game confirmation.

The discovery root is validated strictly before a full phenotype is joined to
its selected candidate. This planner emits Docker commands as data; there is
no execution mode, so a malformed or partial archive cannot start games.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shlex
from pathlib import Path

VALIDATOR_SHA = "8c288a82c21f39a53bd70b712c2a0e989f6098788c1a40e6915bf2db8a168e1f"
VALIDATOR_PATH = Path(__file__).with_name("round6000_postvalidate.py")
EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "src/hexset/catanatron/evolve_candidate_eval.py"
SOURCE_ROOT = "/home/bsm/tmp/hexset-luna-followon-certified"
# Docker addresses the already-loaded Wintermute image by its local tag.
# IMAGE_SHA is the full local image ID returned by `docker image inspect`; it
# is deliberately checked before use. A repository digest would be spelled
# `name@sha256:...` and is a different reference, unavailable for this tag.
IMAGE = "58c092875440"
IMAGE_SHA = "58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be"
SOURCE_HASH = "11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb"
CONTROLLER_SHA = "9ac7fd9b875703e34f0f25a3de0f47b3265443ac15efd4463760b8ee8b07e84c"
EVALUATOR_SHA = "37244faaa029e2d75f2145e0942eb615f74271ae3252cf9aa610bba8bced212b"
MANIFEST_SHA = "dd99767f08ac115da9251345113b308ca7f4593bd1b6f6343d32252f1bebc2b3"
WORKERS = 30
GAMES = 1000
GATES = (("vs-ab2", 620_000_000, "round6000-confirmation-vs-ab2-620000000.json"),
         ("vs-shipped", 620_100_000, "round6000-confirmation-vs-shipped-620100000.json"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validator():
    if sha256(VALIDATOR_PATH) != VALIDATOR_SHA:
        raise RuntimeError("strict postvalidator SHA mismatch")
    if sha256(EVALUATOR_PATH) != EVALUATOR_SHA:
        raise RuntimeError("candidate evaluator SHA mismatch")
    spec = importlib.util.spec_from_file_location("round6000_postvalidate", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load strict postvalidator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_digest(raw_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((p for p in (raw_root / "games" / "generation-6000").glob("*.json") if p.is_file()), key=lambda p: p.name)
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def command(candidate: dict, gate: str, seed: int, filename: str, output_root: str) -> list[str]:
    weights = json.dumps(candidate["weights"], sort_keys=True, separators=(",", ":"))
    return [
        "docker", "run", "--name", f"luna-round6000-confirm-{gate}",
        "--network", "none", "--cpus", "30", "--user", "1000:1000",
        "-e", "PYTHONHASHSEED=0", "-e", "OMP_NUM_THREADS=1",
        "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
        "-e", "PYTHONPATH=/study/src", "-v", f"{SOURCE_ROOT}:/study:ro",
        "-v", f"{output_root}:/out:rw", "-w", "/study",
        "--entrypoint", "python", IMAGE, "-m",
        "hexset.catanatron.evolve_candidate_eval", "--candidate-id", candidate["id"],
        "--weights-json", weights, "--stance", candidate["stance"],
        "--temperature", str(candidate["temperature"]), "--gate", gate,
        "--games", str(GAMES), "--workers", str(WORKERS), "--seed", str(seed),
        "--out", f"/out/{filename}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-root", default="/home/bsm/tmp/luna-round6000-confirmation")
    parser.add_argument("--sidecar", type=Path, required=True)
    args = parser.parse_args()

    validator = load_validator()
    if sha256(args.manifest) != MANIFEST_SHA:
        raise RuntimeError("round6000 manifest SHA mismatch")
    manifest = validator.load_manifest(args.manifest)
    result = validator.validate(args.discovery_root, args.manifest, args.binding)
    if result.get("status") != "PASS" or result.get("raw_records") != 5760:
        raise RuntimeError("strict validation did not produce PASS/5760")
    binding_sha = sha256(args.binding)
    consumed_raw_sha = raw_digest(args.discovery_root)
    if len(consumed_raw_sha) != 64:
        raise RuntimeError("raw archive digest failed")
    selected_id = result.get("selected_by_joint_margin")
    population = manifest.get("population")
    candidates = {c.get("id"): c for c in population if isinstance(c, dict)}
    if selected_id not in candidates:
        raise RuntimeError("selected candidate is absent from full phenotype manifest")
    selected = candidates[selected_id]
    required = {"id", "weights", "stance", "temperature", "depth", "width", "max_nodes", "k", "mode", "max_trades"}
    if set(selected) != required:
        raise RuntimeError("selected phenotype fields are incomplete or changed")
    if selected["depth"] != 2 or selected["width"] != 6 or selected["max_nodes"] != 600:
        raise RuntimeError("selected search phenotype mismatch")
    if selected["k"] != 1 or selected["mode"] != "notrade" or selected["max_trades"] != 0:
        raise RuntimeError("selected no-trade phenotype mismatch")
    if selected["stance"] != "win" or selected["temperature"] != 2.476644394795811:
        raise RuntimeError("selected stance/temperature mismatch")
    fields = {"buy_progress", "diversity", "knight", "port", "production", "road", "robber_risk", "scarce", "spare_card", "victory_point"}
    if set(selected["weights"]) != fields:
        raise RuntimeError("selected full effective weight vector is incomplete")
    if any(not math.isfinite(float(v)) for v in selected["weights"].values()):
        raise RuntimeError("selected weight vector contains a non-finite value")
    commands = [command(selected, gate, seed, filename, args.output_root)
                for gate, seed, filename in GATES]
    for argv, (gate, seed, _filename) in zip(commands, GATES):
        assert argv[argv.index("--candidate-id") + 1] == selected_id
        assert json.loads(argv[argv.index("--weights-json") + 1]) == selected["weights"]
        assert argv[argv.index("--stance") + 1] == selected["stance"]
        assert float(argv[argv.index("--temperature") + 1]) == selected["temperature"]
        assert argv[argv.index("--gate") + 1] == gate
        assert int(argv[argv.index("--games") + 1]) == GAMES
        assert int(argv[argv.index("--workers") + 1]) == WORKERS
        assert int(argv[argv.index("--seed") + 1]) == seed
    sidecar = {
        "schema": 1,
        "status": "READY_FOR_ROOT_REVIEW",
        "execution": "NOT_EXECUTED_BY_THIS_PLANNER",
        "selected_candidate": selected_id,
        "selection": result,
        "phenotype": {**selected, "internal_placement_default": True,
                      "outer_entrant_placement": False, "port_aware": False,
                      "effective_temperature": selected["temperature"],
                      "scarce_binding": "baseline_no_trade"},
        "source": {"root": SOURCE_ROOT, "source_hash": SOURCE_HASH,
                   "controller_file_sha256": CONTROLLER_SHA,
                   "evaluator_file_sha256": EVALUATOR_SHA,
                   "image": IMAGE, "image_sha256": IMAGE_SHA,
                   "image_guard": f"docker image inspect {IMAGE} must return sha256:{IMAGE_SHA}"},
        "input_binding_sha256": binding_sha,
        "consumed_raw_filename_bytes_sha256": consumed_raw_sha,
        "discovery": {"records": 5760, "manifest_sha256": MANIFEST_SHA,
                      "validator_sha256": VALIDATOR_SHA,
                      "raw_filename_bytes_sha256": consumed_raw_sha},
        "confirmation": {"games_per_gate": GAMES, "workers": WORKERS,
                          "serial": True, "gates": [g for g, _, _ in GATES],
                          "commands": [{"argv": c, "shell": shlex.join(c)} for c in commands]},
    }
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.sidecar.with_suffix(args.sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.sidecar)
    args.sidecar.chmod(0o444)
    print(json.dumps(sidecar, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
