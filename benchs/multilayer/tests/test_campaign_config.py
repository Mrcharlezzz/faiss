from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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


def test_the_glove25_campaign_matches_the_spec_count():
    campaign = load_campaign(ROOT / "configs/campaign/glove25.yaml")
    assert campaign["name"] == "glove25"
    [block] = campaign["blocks"]
    assert block["seeds"] == list(range(1, 11))
    assert block["widths"] == [None, 2, 4, 8]
    assert block["ef_search"] == [10, 23, 52, 79, 120, 182, 275, 512]
    assert len(cells(block)) == 320


def test_glove25_runs_cosine_on_the_first_million_into_its_own_outputs():
    with initialize_config_dir(str(ROOT / "configs"), version_base="1.3"):
        cfg = compose("glove25", overrides=["dataset=glove25"], return_hydra_config=True)
    run_dir = OmegaConf.to_container(cfg.hydra.run, resolve=False)["dir"]
    assert run_dir.startswith("benchs/multilayer/outputs/glove25/")
    assert cfg.experiment_name == "glove25"
    assert (cfg.dataset.metric, cfg.dataset.normalize, cfg.dataset.database_size) == ("ip", True, 1_000_000)
