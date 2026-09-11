import json
from pathlib import Path

from hexset.catanatron import cheap_followon as cf


def test_run_executes_all_five_arms_and_promotes_dcp(tmp_path, monkeypatch):
    oracle = {"status": "PASS", "schema": 1,
              "source_sha256": "f4b09d92dda66feae5464939e24bd69f3ed5bec1b6c0b14c485c3b5f827a73ea"}
    (tmp_path / "dcp-oracle.json").write_text(json.dumps(oracle))
    calls = []
    def fake_run(cmd, check=True):
        calls.append(cmd)
        arm = cmd[cmd.index("--candidate") + 1]
        gate = cmd[cmd.index("--gate") + 1]
        games = int(cmd[cmd.index("--games") + 1])
        out = Path(cmd[cmd.index("--out") + 1])
        # Controls and bank extensions screen well, but only DCP clears holdout.
        if games == 120:
            wins = 80 if gate == "ab2" else 60
        elif arm == "dcp":
            wins = 600 if games == 1024 and gate == "ab2" else 320 if games == 1024 else 2200 if gate == "ab2" else 1200
        else:
            wins = 400 if gate == "ab2" else 100
        wins = {"Color.RED": wins, "Color.BLUE": games - wins}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"candidate": arm, "gate": "vs-" + gate,
            "games": games, "workers": 30, "seed": int(cmd[cmd.index("--seed") + 1]),
            "players": "DCP,AB:2,AB:2,AB:2" if arm == "dcp" and gate == "ab2" else "x",
            "wins": wins, "points": {"Color.RED": [1] * games}}))
    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    result = cf.run(tmp_path)
    assert result["status"] == "PASSED_VALIDATION"
    assert result["candidate"] == "dcp"
    assert len([c for c in calls if c[c.index("--games") + 1] == "120"]) == 10
    assert any("--candidate" in c and c[c.index("--candidate") + 1] == "dcp" for c in calls)
    assert not any(c[c.index("--candidate") + 1] in {"control", "control-wide"} and int(c[c.index("--games") + 1]) == 4096 for c in calls)


def test_cheap_seed_families_are_disjoint_except_screen_matching():
    screen = {3_000_000 + g * 100_000 + i for g in range(2) for i in range(30)}
    for rank in range(5):
        confirm = {4_000_000 + rank * 10_000 + g * 100_000 + i for g in range(2) for i in range(30)}
        holdout = {5_000_000 + rank * 1_000_000 + g * 100_000 + i for g in range(2) for i in range(30)}
        assert screen.isdisjoint(confirm | holdout)
        for other in range(rank):
            assert confirm.isdisjoint({4_000_000 + other * 10_000 + g * 100_000 + i for g in range(2) for i in range(30)})
            assert holdout.isdisjoint({5_000_000 + other * 1_000_000 + g * 100_000 + i for g in range(2) for i in range(30)})
