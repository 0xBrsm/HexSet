#!/usr/bin/env python3
"""Fail-closed post-validator for the isolated round-6000 discovery round.

The validator consumes only raw game JSON plus a run-binding sidecar.  It does
not invoke the controller, infer source identity from filenames, or perform
native promotion.  The run-binding sidecar must be written by the execution
wrapper before games start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

EXPECTED_SOURCE_HASH = "11f2ae946884b9062a07eb3fcc40e4350bf0d5f1aeda57dc25b46146c4f664eb"
EXPECTED_CONTROLLER_FILE_SHA = "9ac7fd9b875703e34f0f25a3de0f47b3265443ac15efd4463760b8ee8b07e84c"
EXPECTED_IMAGE_SHA = "58c092875440c770999bada97e0ec7c5178b68fbe640bf3f5a131bc8a624f7be"
EXPECTED_BINDING_SOURCE = "/home/bsm/tmp/hexset-luna-followon-certified"
EXPECTED_INPUT_MANIFEST_SHA = "dd99767f08ac115da9251345113b308ca7f4593bd1b6f6343d32252f1bebc2b3"
EXPECTED_CHECKPOINT_SHA = "9e3fe29f39f079272b1fbfc772e881a7b6dc2e3504fd6b5fd2df924315d60596"
EXPECTED_WORKERS = 30
EXPECTED_GAMES = 240
EXPECTED_CANDIDATES = [f"g6000-p{i:02d}" for i in range(12)]
EXPECTED_COLORS = {"RED", "WHITE", "BLUE", "ORANGE"}
EXPECTED_CANDIDATE_COLOR = "RED"
NAME_RE = re.compile(r"^(g6000-p\d{2})-(vs-ab2|vs-shipped)-discovery-6000-(\d{4})\.json$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_color(value: object, *, allow_enum: bool = True) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid color value: {value!r}")
    if value in EXPECTED_COLORS:
        return value
    if allow_enum and value.startswith("Color.") and value[6:] in EXPECTED_COLORS:
        return value[6:]
    raise ValueError(f"unknown color value: {value!r}")


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_nonnegative_int(value: object, label: str) -> None:
    if not is_int(value) or value < 0:
        raise ValueError(f"{label} is not a nonnegative integer")


def normalized_int_map(value: object, label: str, *, allow_empty: bool, enum_keys: bool = False) -> dict[str, int]:
    if not isinstance(value, dict) or (not allow_empty and not value):
        raise ValueError(f"{label} is not a nonempty object")
    result: dict[str, int] = {}
    for key, count in value.items():
        if enum_keys and (not isinstance(key, str) or not key.startswith("Color.")):
            raise ValueError(f"{label} key is not a Color enum string")
        color = norm_color(key, allow_enum=enum_keys)
        if color in result:
            raise ValueError(f"duplicate normalized color in {label}")
        require_nonnegative_int(count, f"{label}[{key!r}]")
        result[color] = count
    return result


def load_manifest(path: Path) -> dict:
    if sha256(path) != EXPECTED_INPUT_MANIFEST_SHA:
        raise ValueError("input manifest SHA-256 is not the staged rev2 manifest")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("input manifest is not an object")
    if manifest.get("status") != "LOCAL_DRY_CONTROL_FLOW_PASS":
        raise ValueError("input manifest is not an approved local protocol artifact")
    population = manifest.get("population")
    if not isinstance(population, list) or [c.get("id") for c in population] != EXPECTED_CANDIDATES:
        raise ValueError("manifest population is incomplete or reordered")
    if manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("manifest does not bind the staged rev2 checkpoint")
    for candidate in population:
        if not isinstance(candidate, dict):
            raise ValueError("manifest candidate is not an object")
        for value in candidate.get("weights", {}).values():
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite weight in {candidate['id']}")
    return manifest


def validate_binding(binding_path: Path, manifest_path: Path, manifest: dict) -> dict:
    binding = json.loads(binding_path.read_text())
    if not isinstance(binding, dict):
        raise ValueError("run-binding is not an object")
    expected = {
        "source_hash": EXPECTED_SOURCE_HASH,
        "controller_file_sha256": EXPECTED_CONTROLLER_FILE_SHA,
        "image_sha256": EXPECTED_IMAGE_SHA,
        "source_root": EXPECTED_BINDING_SOURCE,
        "workers": EXPECTED_WORKERS,
        "generation": 6000,
        "stage": "discovery",
        "games_per_candidate_per_gate": EXPECTED_GAMES,
        "gates": ["vs-ab2", "vs-shipped"],
        "input_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"run-binding mismatch: {key}")
    if binding.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("run-binding does not bind the input manifest")
    if binding.get("manifest_sha256") != EXPECTED_INPUT_MANIFEST_SHA:
        raise ValueError("run-binding manifest SHA-256 is not the staged rev2 manifest")
    return binding


def validate_game_manifest(game: object, path: Path) -> tuple[str, dict[str, int]]:
    if not isinstance(game, dict):
        raise ValueError(f"game payload is not an object: {path.name}")
    required = {"id", "seed", "seating", "candidate_color", "winner", "points"}
    if set(game) != required:
        raise ValueError(f"game payload fields mismatch: {path.name}")
    if not isinstance(game["id"], str) or not game["id"]:
        raise ValueError(f"game id is invalid: {path.name}")
    require_nonnegative_int(game["seed"], f"game seed in {path.name}")
    seating = game["seating"]
    if not isinstance(seating, list) or len(seating) != len(EXPECTED_COLORS):
        raise ValueError(f"game seating is invalid: {path.name}")
    seating_colors = [norm_color(color, allow_enum=False) for color in seating]
    if len(set(seating_colors)) != len(EXPECTED_COLORS) or set(seating_colors) != EXPECTED_COLORS:
        raise ValueError(f"game seating is not a color permutation: {path.name}")
    if norm_color(game["candidate_color"], allow_enum=False) != EXPECTED_CANDIDATE_COLOR:
        raise ValueError(f"candidate color is not fixed RED: {path.name}")
    winner = norm_color(game["winner"], allow_enum=False)
    points = normalized_int_map(game["points"], f"game points in {path.name}", allow_empty=False, enum_keys=False)
    if set(points) != EXPECTED_COLORS:
        raise ValueError(f"game points do not cover all colors: {path.name}")
    return winner, points


def validate_record(path: Path, candidate: dict, gate: str, index: int, binding: dict) -> int:
    match = NAME_RE.fullmatch(path.name)
    if not match or match.group(1) != candidate["id"] or match.group(2) != gate or int(match.group(3)) != index:
        raise ValueError(f"filename identity mismatch: {path.name}")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"record is not an object: {path.name}")
    required = {"candidate", "candidate_wins", "decisions", "fallbacks", "game_index",
                "games", "gate", "generation", "players", "points", "seed", "stage", "wins"}
    if not required.issubset(record):
        raise ValueError(f"incomplete record: {path.name}")
    for key in ("candidate", "gate", "stage", "players"):
        if not isinstance(record[key], str):
            raise ValueError(f"{key} is not a string: {path.name}")
    for key in ("generation", "game_index", "seed", "decisions", "fallbacks"):
        require_nonnegative_int(record[key], f"{key} in {path.name}")
    if record["candidate"] != candidate["id"] or record["gate"] != gate or record["stage"] != "discovery":
        raise ValueError(f"record identity mismatch: {path.name}")
    if record["generation"] != 6000 or record["game_index"] != index:
        raise ValueError(f"record generation/index mismatch: {path.name}")
    expected_seed = (610_000_000 if gate == "vs-ab2" else 610_050_000) + index
    if record["seed"] != expected_seed:
        raise ValueError(f"seed mismatch: {path.name}")
    expected_players = (f"DC:heximax-evolve-{candidate['id']},AB:2,AB:2,AB:2"
                        if gate == "vs-ab2" else
                        f"DC:heximax-evolve-{candidate['id']},DC:heximax-notrade,DC:heximax-notrade,DC:heximax-notrade")
    if record["players"] != expected_players:
        raise ValueError(f"lineup mismatch: {path.name}")
    if not is_int(record["candidate_wins"]) or record["candidate_wins"] not in (0, 1):
        raise ValueError(f"candidate_wins is not binary: {path.name}")

    games = record["games"]
    if not isinstance(games, list) or len(games) > 1:
        raise ValueError(f"expected zero or one game payload: {path.name}")
    wins = normalized_int_map(record["wins"], f"wins in {path.name}", allow_empty=True, enum_keys=True)
    if set(wins) - EXPECTED_COLORS:
        raise ValueError(f"wins contains an unknown color: {path.name}")
    points = record["points"]
    if not isinstance(points, dict):
        raise ValueError(f"points is not an object: {path.name}")
    point_lists: dict[str, list[int]] = {}
    for key, values in points.items():
        if not isinstance(key, str) or not key.startswith("Color."):
            raise ValueError(f"points key is not a Color enum string: {path.name}")
        color = norm_color(key, allow_enum=True)
        if color in point_lists:
            raise ValueError(f"duplicate normalized color in points: {path.name}")
        if not isinstance(values, list) or len(values) != len(games):
            raise ValueError(f"points list length mismatch: {path.name}")
        for value in values:
            require_nonnegative_int(value, f"points[{key!r}] in {path.name}")
        point_lists[color] = values
    if set(point_lists) != EXPECTED_COLORS:
        raise ValueError(f"points do not cover all colors: {path.name}")

    if not games:
        if record["candidate_wins"] != 0 or sum(wins.values()) != 0:
            raise ValueError(f"draw record has a winner: {path.name}")
        return 0

    winner, game_points = validate_game_manifest(games[0], path)
    if any(point_lists[color] != [game_points[color]] for color in EXPECTED_COLORS):
        raise ValueError(f"record points do not match game points: {path.name}")
    if sum(wins.values()) != 1:
        raise ValueError(f"wins is not one-hot: {path.name}")
    winner_keys = [key for key, value in wins.items() if key == winner and value == 1]
    if len(winner_keys) != 1:
        raise ValueError(f"wins winner mismatch: {path.name}")
    expected_win = int(winner == EXPECTED_CANDIDATE_COLOR)
    if record["candidate_wins"] != expected_win:
        raise ValueError(f"candidate_wins/winner mismatch: {path.name}")
    return record["candidate_wins"]


def validate(raw_root: Path, manifest_path: Path, binding_path: Path) -> dict:
    manifest = load_manifest(manifest_path)
    binding = validate_binding(binding_path, manifest_path, manifest)
    games_root = raw_root / "games" / "generation-6000"
    if games_root.is_symlink() or not games_root.is_dir():
        raise ValueError("generation-6000 raw directory is absent or symlinked")
    paths = sorted(games_root.glob("*.json"))
    if len(paths) != len(EXPECTED_CANDIDATES) * 2 * EXPECTED_GAMES:
        raise ValueError(f"expected 5760 raw records, found {len(paths)}")
    expected_names = {
        f"{candidate}-{gate}-discovery-6000-{index:04d}.json"
        for candidate in EXPECTED_CANDIDATES
        for gate in ("vs-ab2", "vs-shipped")
        for index in range(EXPECTED_GAMES)
    }
    actual_names = {path.name for path in paths}
    if actual_names != expected_names:
        raise ValueError("raw filename set has missing, duplicate, or unexpected records")
    population = {candidate["id"]: candidate for candidate in manifest["population"]}
    totals = {candidate: {"vs-ab2": 0, "vs-shipped": 0} for candidate in EXPECTED_CANDIDATES}
    for path in paths:
        match = NAME_RE.fullmatch(path.name)
        candidate, gate, index = match.group(1), match.group(2), int(match.group(3))
        totals[candidate][gate] += validate_record(path, population[candidate], gate, index, binding)
    scores = {}
    for candidate in EXPECTED_CANDIDATES:
        ab_rate = totals[candidate]["vs-ab2"] / EXPECTED_GAMES
        self_rate = totals[candidate]["vs-shipped"] / EXPECTED_GAMES
        scores[candidate] = {
            "vs-ab2-wins": totals[candidate]["vs-ab2"],
            "vs-shipped-wins": totals[candidate]["vs-shipped"],
            "vs-ab2-rate": ab_rate,
            "vs-shipped-rate": self_rate,
            "joint_margin": min(ab_rate - 0.50, self_rate - 0.25),
        }
    winner = min(EXPECTED_CANDIDATES, key=lambda candidate: (-scores[candidate]["joint_margin"], candidate))
    return {
        "status": "PASS",
        "raw_records": len(paths),
        "source_binding": binding,
        "scores": scores,
        "selected_by_joint_margin": winner,
        "native_promotions_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = validate(args.raw_root, args.manifest, args.binding)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
