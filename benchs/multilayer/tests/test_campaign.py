import itertools

from bench.campaign import Cell, cells, order, overrides

BLOCK = dict(block="deep1m", config_name="rerun", dataset="deep1m", dimension=None,
             seeds=[1, 2, 3, 4, 5], widths=[None, 2, 4, 8],
             ef_search=[10, 23, 52, 79, 120, 182, 275, 512])


def test_cells_counts_the_spec_block():
    assert len(cells(BLOCK)) == 160
    assert len({(c.seed, c.schedule, c.width, c.ef_search) for c in cells(BLOCK)}) == 160


def test_order_is_a_permutation_and_is_reproducible():
    original = cells(BLOCK)
    assert sorted(order(original, 11), key=repr) == sorted(original, key=repr)
    assert order(original, 11) == order(original, 11)
    assert order(original, 11) != order(original, 12)


def test_order_keeps_the_schedules_of_one_seed_and_ef0_together():
    sequence = order(cells(BLOCK), 11)
    groups = [list(g) for _, g in itertools.groupby(sequence, key=lambda c: (c.seed, c.ef_search))]
    assert len(groups) == 40
    assert all(len(g) == 4 for g in groups)
    assert [g[0].seed for g in groups] == sorted(g[0].seed for g in groups)


def test_overrides_name_the_schedule_and_the_construction_seed():
    standard = overrides(Cell("c", "rerun", "deep1m", None, 3, "standard", None, 52))
    assert "strategy=standard" in standard
    assert "experiment_seed=3" in standard
    assert "strategy.ef_search=52" in standard
    assert not any(o.startswith("strategy.widths") for o in standard)
    beam = overrides(Cell("c", "rerun", "synthetic_clustered_200k", 16, 3, "explicit", 4, 52))
    assert "strategy=explicit" in beam
    assert "strategy.widths=[0,4]" in beam
    assert "dataset.dimension=16" in beam
