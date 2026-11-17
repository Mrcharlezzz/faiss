import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray


def json_file(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def arrays(path: str | Path, **values: NDArray[Any]) -> None:
    np.savez_compressed(path, **values)

def table(path: str | Path,  rows: list[dict[str, Any]], compression: str) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression=compression)
