import h5py
import numpy as np
import pytest

from bench.datasets import FvecsDataset, write_fvecs
from convert_glove import convert


def test_write_fvecs_round_trips_through_the_reader(tmp_path):
    x = np.random.default_rng(0).normal(size=(7, 3)).astype(np.float32)
    write_fvecs(tmp_path / "x.fvecs", x)
    ds = FvecsDataset(directory=tmp_path, database_file="x.fvecs", query_file="x.fvecs",
                      database_size=None, query_size=None, metric="l2", normalize=False)
    assert np.array_equal(ds.database, x)


def test_convert_writes_train_as_base_and_test_as_queries(tmp_path):
    rng = np.random.default_rng(1)
    train, test = rng.normal(size=(20, 5)).astype(np.float32), rng.normal(size=(4, 5)).astype(np.float32)
    with h5py.File(tmp_path / "glove-5-angular.hdf5", "w") as f:
        f["train"], f["test"] = train, test
    base, query = convert(tmp_path / "glove-5-angular.hdf5", tmp_path / "out")
    assert (base.name, query.name) == ("glove-5-angular.base.fvecs", "glove-5-angular.query.fvecs")
    ds = FvecsDataset(directory=tmp_path / "out", database_file=base.name, query_file=query.name,
                      database_size=None, query_size=None, metric="l2", normalize=False)
    assert np.array_equal(ds.database, train) and np.array_equal(ds.queries, test)


def test_convert_rejects_a_file_without_test_vectors(tmp_path):
    with h5py.File(tmp_path / "bad.hdf5", "w") as f:
        f["train"] = np.zeros((3, 2), np.float32)
    with pytest.raises(ValueError):
        convert(tmp_path / "bad.hdf5", tmp_path / "out")
