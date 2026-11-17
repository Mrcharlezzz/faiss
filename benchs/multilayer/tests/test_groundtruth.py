import numpy as np
import pytest

import faiss
from bench.groundtruth import exact_neighbors, load_s1, node_levels, upper_nodes


def _small_hnsw(n, d, M, seed):
    rng = np.random.default_rng(seed)
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.rng = faiss.RandomGenerator(seed)
    index.add(rng.normal(size=(n, d)).astype("float32"))
    return index


def test_node_levels_and_upper_nodes_agree_with_the_index():
    index = _small_hnsw(n=3000, d=8, M=8, seed=3)
    levels = node_levels(index)
    assert levels.shape == (3000,)
    assert levels.min() >= 1
    assert levels.max() == index.hnsw.max_level + 1
    upper = upper_nodes(levels, level=1)
    assert np.array_equal(upper, np.flatnonzero(levels >= 2))


def test_exact_neighbors_on_a_subset_returns_original_ids():
    rng = np.random.default_rng(0)
    database = rng.normal(size=(400, 5)).astype("float32")
    queries = rng.normal(size=(7, 5)).astype("float32")
    subset = np.arange(0, 400, 3, dtype="int64")
    distances, ids = exact_neighbors(database, queries, 4, "l2", 100, subset=subset)
    assert ids.shape == (7, 4)
    assert np.isin(ids, subset).all()
    expected = ((queries[:, None, :] - database[subset][None]) ** 2).sum(-1)
    assert np.array_equal(ids[:, 0], subset[expected.argmin(axis=1)])
    assert np.allclose(distances[:, 0], expected.min(axis=1), rtol=1e-5)


def test_s1_refuses_an_index_whose_sha_does_not_match(tmp_path):
    with pytest.raises(ValueError, match="index sha"):
        load_s1(tmp_path, index_sha="a" * 64, k=100, expected_sha="b" * 64)


def test_s1_stores_and_checks_the_sha_it_was_computed_from(tmp_path):
    from bench.groundtruth import s1_path, write_s1
    sha = "a" * 64
    write_s1(tmp_path, sha, 5, sha, ids=np.zeros((1, 5), dtype="int64"))
    s1_path(tmp_path, sha, 5).rename(s1_path(tmp_path, "b" * 64, 5))
    with pytest.raises(ValueError, match="index sha"):
        load_s1(tmp_path, index_sha="b" * 64, k=5, expected_sha="b" * 64)
