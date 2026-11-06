from pathlib import Path

from bench.campaign import cells
from campaign import load_campaign

ROOT = Path(__file__).resolve().parents[1]


def test_the_rerun_campaign_matches_the_spec_count():
    campaign = load_campaign(ROOT / "configs/campaign/rerun.yaml")
    per_block = {b["block"]: len(cells(b)) for b in campaign["blocks"]}
    assert per_block == {"deep1m": 160, "gist1m": 160,
                         "clustered200k_d16": 200, "clustered200k_d32": 200}
    assert sum(per_block.values()) == 720
    assert all(b["widths"] == [None, 2, 4, 8] for b in campaign["blocks"])
