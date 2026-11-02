import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def vector_digest(x: NDArray[Any]) -> str:
    h = hashlib.sha256(str((x.shape, x.dtype.str)).encode())
    for start in range(0, len(x), 10000):
        h.update(np.ascontiguousarray(x[start:start + 10000]).tobytes())
    return h.hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
