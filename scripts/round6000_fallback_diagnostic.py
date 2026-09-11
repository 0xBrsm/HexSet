#!/usr/bin/env python3
"""Read-only one-game bridge-fallback diagnostic.

This is deliberately separate from the certified runtime source. It registers
one already-approved phenotype, wraps only ``player.to_catanatron`` to record a
failed mapping, re-raises the original ValueError, and calls the certified
``evolve._play_one`` once. It does not alter action choice, RNG, or policy.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

EXPECTED_MANIFEST_SHA = "dd99767f08ac115da9251345113b308ca7f4593bd1b6f6343d32252f1bebc2b3"
EXPECTED_CONTROLLER_SHA = "9ac7fd9b875703e34f0f25a3de0f47b3265443ac15efd4463760b8ee8b07e84c"
EXPECTED_FIELDS = (
    "candidate", "gate", "stage", "generation", "game_index", "seed",
    "players", "candidate_wins", "wins", "points", "decisions", "fallbacks",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_color(value: Any) -> str | None:
    if value is None:
        return None
    value = getattr(value, "value", value)
    value = str(value)
    return value.removeprefix("Color.").upper()


def describe(value: Any) -> Any:
    """Make action objects inspectable without calling policy or RNG methods."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    enum_value = getattr(value, "value", None)
    if enum_value is not None and isinstance(enum_value, (bool, int, float, str)):
        return {"type": type(value).__name__, "name": getattr(value, "name", None),
                "value": enum_value}
    fields = {}
    for name in ("type", "a", "b", "value", "action_type", "color"):
        if hasattr(value, name):
            fields[name] = describe(getattr(value, name))
    return {"type": type(value).__name__, "fields": fields,
            "repr": repr(value)}


def candidate_from_manifest(path: Path, candidate_id: str) -> tuple[dict, dict]:
    if sha256(path) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("staged round6000 manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    required_top = {"schema", "controller_file_sha256", "controller_source_hash", "population"}
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise RuntimeError("unexpected phenotype manifest schema")
    if not required_top.issubset(manifest):
        raise RuntimeError("staged manifest is missing required identity keys")
    if manifest["controller_file_sha256"] != EXPECTED_CONTROLLER_SHA:
        raise RuntimeError("manifest controller file SHA mismatch")
    candidates = manifest.get("population")
    if not isinstance(candidates, list):
        raise RuntimeError("manifest has no population")
    matches = [c for c in candidates if isinstance(c, dict) and c.get("id") == candidate_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest candidate {candidate_id!r}")
    return matches[0], manifest


def compare_reference(observed: dict, expected: dict) -> None:
    for key in EXPECTED_FIELDS:
        if observed.get(key) != expected.get(key):
            raise RuntimeError(f"reference mismatch in {key}: {observed.get(key)!r} != {expected.get(key)!r}")
    if len(observed.get("games", [])) != len(expected.get("games", [])):
        raise RuntimeError("reference mismatch in game count")
    for got, want in zip(observed.get("games", []), expected.get("games", [])):
        got = dict(got)
        want = dict(want)
        got.pop("id", None)
        want.pop("id", None)
        if got != want:
            raise RuntimeError(f"reference mismatch in game payload: {got!r} != {want!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-record", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--gate", choices=("vs-ab2", "vs-shipped"), required=True)
    parser.add_argument("--generation", type=int, default=6000)
    parser.add_argument("--game-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=610000000)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = json.loads(args.expected_record.read_text())
    candidate, manifest = candidate_from_manifest(args.manifest, args.candidate_id)
    if sha256(args.expected_record) != args.expected_sha:
        raise RuntimeError("expected reference record SHA mismatch")
    if manifest.get("controller_source_hash") != args.source_hash:
        raise RuntimeError("manifest/controller source hash mismatch")

    import hexset.bots  # noqa: F401 -- register native presets
    from hexset.arena import Entrant, register_preset
    from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
    from hexset.catanatron import evolve
    from hexset.catanatron import player as player_module
    from hexset.catanatron.actions import to_catanatron

    if evolve._source_hash() != args.source_hash:
        raise RuntimeError("certified source fingerprint mismatch")
    if tuple(inspect.signature(to_catanatron).parameters) != (
        "our_action", "our_game", "mapping", "seats", "playable_actions"
    ):
        raise RuntimeError("unexpected to_catanatron signature")
    required = {"id", "weights", "stance", "temperature", "depth", "width",
                "max_nodes", "k", "mode", "max_trades"}
    if set(candidate) != required or candidate["id"] != args.candidate_id:
        raise RuntimeError("candidate phenotype fields do not match manifest")
    name = "heximax-evolve-" + args.candidate_id
    entrant = Entrant(
        name, kind="heximax",
        weights=replace(NO_TRADE_WEIGHTS, **{
            key: value for key, value in candidate["weights"].items() if key != "scarce"
        }),
        depth=candidate["depth"], width=candidate["width"],
        max_nodes=candidate["max_nodes"], k=candidate["k"],
        mode=candidate["mode"], max_trades=candidate["max_trades"],
        stance=candidate["stance"], temperature=candidate["temperature"],
    )
    register_preset(name, entrant)

    events: list[dict] = []
    original = player_module.to_catanatron
    original_decide = player_module.DevCatanPlayer.decide
    native_context_stack: list[dict] = []

    def enum_name(value: Any) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "name", getattr(value, "value", value)))

    def native_context(game: Any, color: Any) -> dict:
        state = game.state
        index = state.color_to_index[color]
        prefix = f"P{index}"
        resources = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
        hand = [state.player_state[f"{prefix}_{resource}_IN_HAND"] for resource in resources]
        port_resources = sorted(
            (enum_name(resource) for resource in state.board.get_player_port_resources(color)),
            key=lambda value: "" if value is None else value,
        )
        ratios = {resource: 4 for resource in resources}
        if None in state.board.get_player_port_resources(color):
            ratios = {resource: 3 for resource in resources}
        for resource in state.board.get_player_port_resources(color):
            if resource is not None:
                ratios[enum_name(resource)] = 2
        return {
            "actor_color": enum_name(color),
            "state_current_color": enum_name(state.current_color()),
            "current_player_index": state.current_player_index,
            "current_turn_index": state.current_turn_index,
            "prompt": enum_name(state.current_prompt),
            "has_rolled": bool(state.player_state[f"{prefix}_HAS_ROLLED"]),
            "is_road_building": bool(state.is_road_building),
            "free_roads_available": state.free_roads_available,
            "hand": hand,
            "bank": list(state.resource_freqdeck),
            "port_resources": port_resources,
            "ratios": ratios,
        }

    def translated_context(our_game: Any, seats: Any) -> dict:
        state = our_game.state(0, hidden=False)
        seat = our_game.current_player
        resources = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
        ports = []
        for port in state.board.ports:
            if any(state.vertex_owner[v] == seat for v in port.vertices):
                ports.append({"resource": enum_name(port.resource), "ratio": port.ratio,
                              "vertices": list(port.vertices)})
        ratios = {resource: 4 for resource in resources}
        if any(port["resource"] is None for port in ports):
            ratios = {resource: 3 for resource in resources}
        for port in ports:
            if port["resource"] is not None:
                ratios[port["resource"]] = min(ratios[port["resource"]], port["ratio"])
        return {
            "actor_seat": seat,
            "actor_color": enum_name(seats.color_of[seat]),
            "current_player": our_game.current_player,
            "phase": enum_name(our_game.phase),
            "free_roads": our_game.free_roads,
            "hand": list(state.hands[seat]),
            "bank": list(state.bank),
            "ports": ports,
            "ratios": ratios,
        }

    def logging_decide(self, game, playable_actions):
        native_context_stack.append(native_context(game, self.color))
        try:
            return original_decide(self, game, playable_actions)
        finally:
            native_context_stack.pop()

    player_module.DevCatanPlayer.decide = logging_decide

    def logging_translation(action, our_game, mapping, seats, playable_actions):
        try:
            return original(action, our_game, mapping, seats, playable_actions)
        except ValueError as exc:
            native_color = None
            if playable_actions:
                native_color = normalize_color(getattr(playable_actions[0], "color", None))
            events.append({
                "native_actor_color": native_color,
                "error_type": type(exc).__name__, "error": str(exc),
                "selected_dev_action": describe(action),
                "native_playable_actions": describe(playable_actions),
                "native_maritime_actions": [
                    describe(a) for a in playable_actions
                    if enum_name(getattr(getattr(a, "action_type", None), "name", None))
                    == "MARITIME_TRADE"
                ],
                "native_context": native_context_stack[-1] if native_context_stack else None,
                "translated_context": translated_context(our_game, seats),
            })
            raise

    observed: dict | None = None
    status = "PASS"
    failure: dict | None = None
    player_module.to_catanatron = logging_translation
    try:
        observed = evolve._play_one((args.candidate_id, args.gate, "discovery",
                                     args.generation, args.game_index, args.seed))
        compare_reference(observed, expected)
        candidate_color = normalize_color(observed["games"][0]["candidate_color"])
        candidate_events = [e for e in events if e["native_actor_color"] == candidate_color]
        if len(candidate_events) != expected["fallbacks"]:
            raise RuntimeError(
                f"candidate logged ValueErrors {len(candidate_events)} != "
                f"reference fallbacks {expected['fallbacks']}"
            )
    except BaseException as exc:  # persist diagnostic evidence for every failure
        status = "ERROR"
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        player_module.to_catanatron = original
        player_module.DevCatanPlayer.decide = original_decide

    result = {
        "schema": 1, "status": status, "diagnostic": "fallback_mapping_only",
        "source_hash": args.source_hash, "manifest_sha256": sha256(args.manifest),
        "expected_record": str(args.expected_record), "expected_record_sha256": args.expected_sha,
        "job": {"candidate_id": args.candidate_id, "gate": args.gate,
                "generation": args.generation, "game_index": args.game_index,
                "seed": args.seed},
        "reference_counters": {k: expected[k] for k in ("decisions", "fallbacks", "candidate_wins")},
        "observed_counters": ({k: observed[k] for k in ("decisions", "fallbacks", "candidate_wins")}
                              if observed is not None else None),
        "events": events, "observed": observed, "failure": failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
