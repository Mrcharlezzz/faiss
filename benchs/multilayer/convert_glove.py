import sys
from pathlib import Path

import h5py
import numpy as np

from bench.datasets import write_fvecs


def read_matrix(f: h5py.File, name: str) -> np.ndarray:
    if name not in f:
        raise ValueError(f"missing dataset: {name}")
    x = np.asarray(f[name], dtype=np.float32)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return x


def convert(source: Path, directory: Path) -> tuple[Path, Path]:
    with h5py.File(source, "r") as f:
        train, test = read_matrix(f, "train"), read_matrix(f, "test")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train and test dimensions differ")
    directory.mkdir(parents=True, exist_ok=True)
    base, query = directory / f"{source.stem}.base.fvecs", directory / f"{source.stem}.query.fvecs"
    write_fvecs(base, train)
    write_fvecs(query, test)
    return base, query


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: convert_glove.py <hdf5> <out_dir>")
    base_path, query_path = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    for path in (base_path, query_path):
        raw = np.memmap(path, dtype="<i4", mode="r")
        print(f"wrote: {path} shape=({len(raw) // (raw[0] + 1)}, {raw[0]})")
