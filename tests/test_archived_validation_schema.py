import copy
import json
from pathlib import Path

from hexset.catanatron.validation import rank_screen


def test_archived_width12_result_uses_color_red_and_dual_margin(tmp_path):
    archived = Path("docs/readouts/luna-ablation/results/results/14-risk-141.json")
    document = json.loads(archived.read_text())
    # Keep the archived result schema/manifest, representing the width-12
    # candidate record whose observed gate counts are 69/120 and 30/120.
    document["candidate"] = "width12"
    document["manifest"]["width"] = 12
    document["rows"][0]["wins"] = {"Color.RED": 69, "Color.BLUE": 17, "Color.ORANGE": 17, "Color.WHITE": 17}
    document["rows"][1]["wins"] = {"Color.RED": 30, "Color.BLUE": 30, "Color.ORANGE": 30, "Color.WHITE": 30}
    paths = [tmp_path / "width12.json"]
    paths[0].write_text(json.dumps(document))
    for i in range(8):
        paths.append(tmp_path / f"other-{i}.json")
        paths[-1].write_text(json.dumps({
            "candidate": f"other-{i}", "games": 120,
            "rows": [
                {"gate": "vs-ab2", "games": 120, "wins": {"Color.RED": 60, "Color.BLUE": 60}},
                {"gate": "vs-shipped", "games": 120, "wins": {"Color.RED": 20, "Color.BLUE": 100}},
            ],
        }))
    assert rank_screen(paths)[0] == "width12"
    assert min(69 / 120 - 0.50, 30 / 120 - 0.25) == 0
