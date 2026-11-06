from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(kw_only=True)
class Dataset(ABC):
    metric: Literal["l2", "ip"]
    normalize: bool
    database: NDArray[np.float32] = field(init=False, repr=False)
    queries: NDArray[np.float32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.metric not in {"l2", "ip"}:
            raise ValueError("dataset.metric must be l2 or ip")
        database, queries = self.load()
        arrays = []
        for x in (database, queries):
            x = np.ascontiguousarray(x, dtype=np.float32)
            if x.ndim != 2 or not len(x) or x.shape[1] == 0 or not np.isfinite(x).all():
                raise ValueError("vectors must be nonempty finite matrices")
            if self.normalize:
                x = x.copy()
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                x /= np.where(norms > 0, norms, 1)
            arrays.append(x)
        if arrays[0].shape[1] != arrays[1].shape[1]:
            raise ValueError("database and query dimensions differ")
        self.database, self.queries = arrays

    @abstractmethod
    def load(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        raise NotImplementedError


@dataclass(kw_only=True)
class SyntheticDataset(Dataset):
    database_size: int
    query_size: int
    dimension: int
    seed: int
    distribution: str
    clusters: int
    cluster_scale: float
    noise: float
    centres: NDArray[np.float32] | None = field(init=False, repr=False, default=None)
    database_clusters: NDArray[np.int32] | None = field(init=False, repr=False, default=None)
    query_clusters: NDArray[np.int32] | None = field(init=False, repr=False, default=None)

    def load(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if self.database_size < 1 or self.query_size < 1:
            raise ValueError("database and query sizes must be positive")
        rng = np.random.default_rng(self.seed)
        shape = (self.database_size + self.query_size, self.dimension)
        if self.distribution == "gaussian":
            x = rng.normal(size=shape)
        elif self.distribution == "clustered":
            centers = rng.normal(size=(self.clusters, self.dimension)) * self.cluster_scale
            assignment = rng.integers(self.clusters, size=shape[0])
            x = centers[assignment]
            x += rng.normal(scale=self.noise, size=shape)
            self.centres = centers.astype(np.float32)
            self.database_clusters = assignment[:self.database_size].astype(np.int32)
            self.query_clusters = assignment[self.database_size:].astype(np.int32)
        else:
            raise ValueError("distribution not supported")
        x = x.astype(np.float32)
        return x[:self.database_size], x[self.database_size:]


@dataclass(kw_only=True)
class FvecsDataset(Dataset):
    directory: str | Path
    database_file: str
    query_file: str
    database_size: int | None
    query_size: int | None

    def load(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        directory = Path(self.directory).expanduser().resolve()
        return self._read_fvecs(directory / self.database_file, self.database_size), self._read_fvecs(directory / self.query_file, self.query_size),
        
    @staticmethod
    def _read_fvecs(path: Path, limit: int | None) -> NDArray[np.float32]:
        raw = np.memmap(path, dtype="<i4", mode="r")
        if raw[0] <= 0:
            raise ValueError(f"invalid fvecs file: {path}")
        d = int(raw[0])
        if len(raw) % (d + 1):
            raise ValueError(f"truncated fvecs file: {path}")
        rows = raw.reshape(-1, d + 1)
        if limit is not None and limit < 1:
            raise ValueError("dataset limits must be positive")
        rows = rows[:limit]
        if not np.all(rows[:, 0] == d):
            raise ValueError(f"inconsistent fvecs dimensions: {path}")
        return rows[:, 1:].view("<f4")


@dataclass(kw_only=True)
class FbinDataset(Dataset):
    directory: str | Path
    database_file: str
    query_file: str
    database_size: int | None
    query_size: int | None

    def load(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        directory = Path(self.directory).expanduser().resolve()
        return (
            self._read_fbin(directory / self.database_file, self.database_size),
            self._read_fbin(directory / self.query_file, self.query_size),
        )

    @staticmethod
    def _read_fbin(path: Path, limit: int | None) -> NDArray[np.float32]:
        with path.open("rb") as stream:
            header = np.fromfile(stream, dtype="<u4", count=2)
        if len(header) != 2 or np.any(header == 0):
            raise ValueError(f"invalid fbin header: {path}")
        count, dimension = map(int, header)
        if path.stat().st_size != 8 + count * dimension * 4:
            raise ValueError(f"file size does not match fbin header: {path}")
        if limit is not None and not 1 <= limit <= count:
            raise ValueError(f"requested vector count must be between 1 and {count}: {path}")
        return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(count if limit is None else limit, dimension))
