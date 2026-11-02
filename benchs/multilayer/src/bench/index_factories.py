from .datasets import Dataset

import faiss


class HNSWIndexFactory:
    def __init__(
        self,
        M: int,
        ef_construction: int,
        seed: int,
        build_threads: int,
    ) -> None:
        if min(M, ef_construction, build_threads) < 1:
            raise ValueError("index widths and thread counts must be positive")
        self.M = M
        self.ef_construction = ef_construction
        self.seed = seed
        self.build_threads = build_threads

    def build(self, dataset: Dataset) -> faiss.IndexHNSWFlat:
        faiss.omp_set_num_threads(self.build_threads)
        metric = faiss.METRIC_L2 if dataset.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        index = faiss.IndexHNSWFlat(dataset.database.shape[1], self.M, metric)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.rng = faiss.RandomGenerator(self.seed)
        index.add(dataset.database)
        return index
