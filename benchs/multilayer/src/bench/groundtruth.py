import uuid
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import faiss
from bench.export import arrays


def node_levels(index: faiss.IndexHNSW) -> NDArray[np.int32]:
    return faiss.vector_to_array(index.hnsw.levels).astype(np.int32)


def upper_nodes(levels: NDArray[np.int32], level: int = 1) -> NDArray[np.int64]:
    return np.flatnonzero(levels >= level + 1).astype(np.int64)


def exact_neighbors(
    database: NDArray[np.float32],
    queries: NDArray[np.float32],
    k: int,
    metric: str,
    batch_size: int,
    subset: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    points = database if subset is None else database[subset]
    flat = (faiss.IndexFlatL2 if metric == "l2" else faiss.IndexFlatIP)(points.shape[1])
    flat.add(np.ascontiguousarray(points))
    distances, ids = [], []
    for start in range(0, len(queries), batch_size):
        d, i = flat.search(np.ascontiguousarray(queries[start:start + batch_size]), k)
        distances.append(d)
        ids.append(i)
    ids = np.concatenate(ids).astype(np.int64)
    if subset is not None:
        ids = np.where(ids >= 0, subset[np.maximum(ids, 0)], -1)
    return np.concatenate(distances), ids


def s0_path(cache: Path, database_id: str, query_id: str, k: int) -> Path:
    return cache / f"gt-s0-{database_id}-{query_id}-k{k}.npz"


def levels_path(cache: Path, index_sha: str) -> Path:
    return cache / f"levels-{index_sha}.npz"


def s1_path(cache: Path, index_sha: str, k: int) -> Path:
    return cache / f"gt-s1-{index_sha}-k{k}.npz"


def clusters_path(cache: Path, database_id: str, query_id: str) -> Path:
    return cache / f"clusters-{database_id}-{query_id}.npz"


def write_artifact(path: Path, **values: NDArray) -> None:
    temp = path.with_name(f".{uuid.uuid4().hex}.npz")
    arrays(temp, **values)
    temp.replace(path)


def write_s1(cache: Path, index_sha: str, k: int, expected_sha: str, **values: NDArray) -> Path:
    _check_sha(index_sha, expected_sha)
    path = s1_path(cache, index_sha, k)
    write_artifact(path, **values)
    return path


def load_s1(cache: Path, index_sha: str, k: int, expected_sha: str) -> dict[str, NDArray]:
    _check_sha(index_sha, expected_sha)
    with np.load(s1_path(cache, index_sha, k)) as stored:
        return {name: stored[name] for name in stored.files}


def _check_sha(index_sha: str, expected_sha: str) -> None:
    if index_sha != expected_sha:
        raise ValueError(f"index sha {index_sha} does not match the searched index {expected_sha}")
