from types import SimpleNamespace

import pytest

from bench.campaign import Cell
from campaign import Ledger, guard_fingerprint, preflight, run_cell

_GRID = dict(config_name="rerun", dataset="gist1m", dimension=None, seeds=[1],
             widths=[None], ef_search=[10])
_CELL = Cell("c", "rerun", "deep1m", None, 1, "standard", None, 10)


def test_preflight_names_a_missing_dataset_file(tmp_path):
    blocks = [dict(block="gist1m", requires=["data/gist/gist_base.fvecs"], **_GRID)]
    with pytest.raises(FileNotFoundError, match="data/gist/gist_base.fvecs"):
        preflight(blocks, tmp_path)


def test_guard_fingerprint_refuses_a_changed_source_tree(tmp_path):
    guard_fingerprint(tmp_path, "aaa")
    guard_fingerprint(tmp_path, "aaa")
    with pytest.raises(RuntimeError, match="source fingerprint"):
        guard_fingerprint(tmp_path, "bbb")


def test_ledger_skips_recorded_cells_and_counts_duplicates_apart(tmp_path):
    ledger = Ledger(tmp_path / "campaign.jsonl")
    cell = Cell("c", "rerun", "deep1m", None, 1, "standard", None, 10)
    other = Cell("c", "rerun", "deep1m", None, 1, "explicit", 2, 10)
    assert not ledger.done(cell)
    ledger.record(cell, "complete", 1.0)
    ledger.record(other, "duplicate", 0.1)
    assert ledger.done(cell) and ledger.done(other)
    assert Ledger(tmp_path / "campaign.jsonl").counts() == {"complete": 1, "duplicate": 1}


def test_ledger_retries_failed_cells(tmp_path):
    ledger = Ledger(tmp_path / "campaign.jsonl")
    ledger.record(_CELL, "failed", 1.0)
    assert not ledger.done(_CELL)


def test_run_cell_reports_duplicate_from_the_child_status(tmp_path):
    def runner(argv):
        return SimpleNamespace(returncode=0, stdout=f"Artifacts: {tmp_path}\n")
    (tmp_path / "complete.json").write_text('{"schema_version": 1, "status": "duplicate"}')
    assert run_cell(_CELL, core=0, runner=runner) == "duplicate"


def test_run_cell_reports_failure_without_raising(tmp_path):
    assert run_cell(_CELL, core=0, runner=lambda argv: SimpleNamespace(returncode=1, stdout="boom")) == "failed"


def test_runner_pins_the_core_and_passes_the_overrides():
    seen = []
    run_cell(_CELL, core=3, runner=lambda argv: (seen.append(argv), SimpleNamespace(returncode=1, stdout=""))[1])
    assert seen[0][:3] == ["taskset", "-c", "3"]
    assert "strategy.ef_search=10" in seen[0]
