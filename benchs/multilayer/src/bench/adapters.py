import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

import faiss


class HNSWAdapter:
    def __init__(
        self, 
        name: str, 
        ef_search: int, 
        schedule: str, 
        width: int,
        r_mult: float, 
        widths: Sequence[int] | None,
        bounded_queue: bool, 
        check_relative_distance: bool
    ) -> None:
        self.name = name
        self.ef_search = ef_search
        self.schedule = schedule
        self.width = width
        self.r_mult = r_mult
        self.widths = widths
        self.bounded_queue = bounded_queue
        self.check_relative_distance = check_relative_distance

    def prepare(self, index: faiss.IndexHNSW) -> None:
        faiss.cvar.hnsw_level_stats.enabled = False
        faiss.cvar.hnsw_level_stats.reset()
        self.r = self.r_mult / index.hnsw.nb_neighbors(1)
        if self.ef_search < 1 or self.width < 1 or not 0 <= self.r <= 1:
            raise ValueError("invalid search width or r_mult")
        L = index.hnsw.max_level + 1
        widths = [1] * L
        if self.schedule == "layer1":
            if L > 1:
                widths[1] = max(1, math.floor(self.ef_search * self.r))
        elif self.schedule == "constant":
            widths = [1] + [self.width] * (L - 1)
        elif self.schedule == "exponential":
            widths = [1] + [max(1, math.floor(self.ef_search * self.r**l)) for l in range(1, L)]
        elif self.schedule == "explicit":
            if self.widths is None:
                raise ValueError("explicit schedule requires widths")
            if any(int(w) != w for w in self.widths):
                raise ValueError("widths must be integers")
            widths = [max(1, int(w)) for w in self.widths[:L]]
            widths += [1] * (L - len(widths))
        elif self.schedule != "standard":
            raise ValueError(f"unknown schedule: {self.schedule}")
        widths[0] = 0
        self.effective_widths = widths
        self.params = faiss.SearchParametersHNSW()
        self.params.efSearch = self.ef_search
        self.params.bounded_queue = self.bounded_queue
        self.params.check_relative_distance = self.check_relative_distance
        faiss.copy_array_to_vector(np.array(widths, dtype="int32"), self.params.efSearch_by_level)

    def search(
        self, 
        index: faiss.IndexHNSW, 
        queries: NDArray[np.float32], 
        k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        return index.search(queries, k, params=self.params)

    def collect_diagnostics(
        self, 
        index: faiss.IndexHNSW, 
        queries: NDArray[np.float32],
        k: int, 
        candidates: bool,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, int]]:
        stats = faiss.cvar.hnsw_level_stats
        stats.enabled = False
        stats.reset()
        faiss.cvar.hnsw_stats.reset()
        stats.enabled = True
        try:
            self.search(index, queries, k)
            arrays = {name: faiss.vector_to_array(getattr(stats, name)).reshape(stats.nq, stats.L) for name in ("ndis", "nhops")}
            if candidates:
                arrays.update(
                    {
                        name: faiss.vector_to_array(getattr(stats, field)).reshape(stats.nq, stats.S) 
                        for name, field in (("candidate_ids", "level_eps"), ("candidate_distances", "level_ep_dis"))
                    }
                )
                arrays["level_offsets"] = faiss.vector_to_array(stats.level_offsets)
            aggregate = {name: int(getattr(faiss.cvar.hnsw_stats, name)) for name in ("ndis", "nhops", "n1", "n2")}
            return arrays, aggregate
        finally:
            stats.enabled = False
            stats.reset()
