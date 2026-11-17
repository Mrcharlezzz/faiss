import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Score:
    recall_at_k: float
    recall_at_1: float
    results: list[tuple[NDArray[Any], NDArray[Any]]]


def score(strategy: Any, index: Any, batches: list[tuple[int, NDArray[np.float32]]], k: int,
          order: NDArray[np.int64], exact_ids: NDArray[np.int64], passes: int) -> Score:
    for _ in range(max(1, passes)):
        results = [strategy.search(index, batch, k) for _, batch in batches]
    correct = correct_at_1 = 0
    for (start, batch), (_, ids) in zip(batches, results):
        for query_index, neighbors in zip(order[start:start + len(batch)], ids):
            correct += np.intersect1d(neighbors[neighbors >= 0], exact_ids[query_index]).size
            correct_at_1 += int(neighbors[0] == exact_ids[query_index][0])
    total = sum(len(batch) for _, batch in batches)
    return Score(float(correct / (total * k)), float(correct_at_1 / total), results)


def timed_passes(strategy: Any, index: Any, batches: list[tuple[int, NDArray[np.float32]]], k: int,
                 repetitions: int, search_threads: int) -> list[dict[str, int]]:
    rows = []
    for trial in range(repetitions):
        for start, batch in batches:
            t0 = time.perf_counter_ns()
            strategy.search(index, batch, k)
            rows.append({
                "trial": trial,
                "query_start": start,
                "query_count": len(batch),
                "elapsed_ns": time.perf_counter_ns() - t0,
                "search_threads": search_threads,
            })
    return rows
