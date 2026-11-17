from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Cell:
    block: str
    config_name: str
    dataset: str
    dimension: int | None
    seed: int
    schedule: str
    width: int | None
    ef_search: int


def cells(block: dict[str, Any]) -> list[Cell]:
    return [
        Cell(
            block["block"], block["config_name"], block["dataset"], block["dimension"], seed,
            "standard" if width is None else "explicit", width, ef_search,
        )
        for seed in block["seeds"]
        for width in block["widths"]
        for ef_search in block["ef_search"]
    ]


def order(items: list[Cell], seed: int) -> list[Cell]:
    rng = np.random.default_rng(seed)
    graphs: dict[tuple[str, int], dict[int, list[Cell]]] = {}
    for cell in items:
        graphs.setdefault((cell.block, cell.seed), {}).setdefault(cell.ef_search, []).append(cell)
    blocks = list(dict.fromkeys(block for block, _ in graphs))
    sequence = []
    for block in blocks:
        for graph in sorted(key for key in graphs if key[0] == block):
            groups = list(graphs[graph].values())
            for position in rng.permutation(len(groups)):
                group = groups[position]
                sequence.extend(group[i] for i in rng.permutation(len(group)))
    return sequence


def overrides(cell: Cell) -> list[str]:
    result = [
        "hydra.mode=RUN",
        f"dataset={cell.dataset}",
        f"experiment_seed={cell.seed}",
        f"strategy={cell.schedule}",
        f"strategy.ef_search={cell.ef_search}",
    ]
    if cell.dimension is not None:
        result.append(f"dataset.dimension={cell.dimension}")
    if cell.schedule == "explicit":
        result.append(f"strategy.widths=[0,{cell.width}]")
    return result
