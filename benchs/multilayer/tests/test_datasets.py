import numpy as np

from bench.datasets import SyntheticDataset


def test_clustered_dataset_labels_name_the_nearest_centre():
    ds = SyntheticDataset(database_size=500, query_size=50, dimension=4, seed=7,
                          distribution="clustered", clusters=5, cluster_scale=10.0,
                          noise=0.01, metric="l2", normalize=False)
    assert ds.database_clusters.shape == (500,)
    assert ds.query_clusters.shape == (50,)
    assert ds.centres.shape == (5, 4)
    for points, labels in ((ds.database, ds.database_clusters), (ds.queries, ds.query_clusters)):
        nearest = np.argmin(((points[:, None, :] - ds.centres[None]) ** 2).sum(-1), axis=1)
        assert np.array_equal(nearest, labels)


def test_gaussian_dataset_has_no_cluster_labels():
    ds = SyntheticDataset(database_size=100, query_size=10, dimension=4, seed=7,
                          distribution="gaussian", clusters=5, cluster_scale=4.0,
                          noise=1.0, metric="l2", normalize=False)
    assert (ds.centres, ds.database_clusters, ds.query_clusters) == (None, None, None)
