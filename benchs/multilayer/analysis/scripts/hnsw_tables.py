"""Shared run_2 table schema and per-run conversion of benchmark artifacts.

A benchmark run, whatever exported it, leaves the same files in its artifact
directory: metadata.json, timings.parquet, diagnostics/batch-N.npz and
results/batch-N.npz. This module turns one such directory into rows of the
per-query tables and streams them to Parquet. Readers for a particular layout
(W&B export, local Hydra outputs) only have to find the run directories and
build the run-level row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCHEMAS = {
    "entry_points": pa.schema([
        ("run_id", pa.int32()), ("query_id", pa.int64()), ("level", pa.int8()),
        ("rank", pa.int16()), ("candidate_id", pa.int32()),
        ("candidate_distance", pa.float32()),
    ]),
    "level_stats": pa.schema([
        ("run_id", pa.int32()), ("query_id", pa.int64()), ("level", pa.int8()),
        ("ndis", pa.int64()), ("nhops", pa.int64()),
    ]),
    "neighbors": pa.schema([
        ("run_id", pa.int32()), ("query_id", pa.int64()), ("rank", pa.int16()),
        ("neighbor_id", pa.int64()), ("distance", pa.float32()),
    ]),
    "timings": pa.schema([
        ("run_id", pa.int32()), ("trial", pa.int64()), ("query_start", pa.int64()),
        ("query_count", pa.int64()), ("elapsed_ns", pa.int64()),
        ("search_threads", pa.int64()),
    ]),
    "search_batches": pa.schema([
        ("run_id", pa.int32()), ("batch", pa.int32()), ("ndis", pa.int64()),
        ("nhops", pa.int64()), ("n1", pa.int64()), ("n2", pa.int64()),
    ]),
}

# Run-level columns taken from metadata.json, in run_2's order after the 77
# inherited columns. Identity columns (job_id, run_dir, run_id, status,
# sweep_dir) are filled by the reader.
METADATA_COLUMNS = [
    "database_dim", "database_id", "database_rows", "distance_semantics",
    "effective_search_threads", "effective_widths", "faiss_version", "index_id",
    "index_reused", "job_id", "machine", "platform", "python", "query_id",
    "query_rows", "r", "run_dir", "run_id", "source_fingerprint", "status",
    "sweep_dir",
]


class IncompleteRun(RuntimeError):
    pass


@dataclass
class RunData:
    """Arrays of one run, checked for completeness."""
    metadata: dict
    query_ids: np.ndarray            # [Q], diagnostics order
    ndis: np.ndarray                 # [Q, L]
    nhops: np.ndarray                # [Q, L]
    candidate_ids: np.ndarray        # [Q, S]
    candidate_distances: np.ndarray  # [Q, S]
    level_offsets: np.ndarray        # [L+1]
    batch_sizes: list[int]           # diagnostics queries per batch
    batch_counters: list[dict | None]  # diagnostics/batch-N.json, if present
    result_query_ids: np.ndarray     # [Q], results order
    ids: np.ndarray                  # [Q, k]
    distances: np.ndarray            # [Q, k]
    timings: pa.Table


def load_run(artifact_dir: Path) -> RunData:
    """Load one run's artifacts, failing loudly on anything missing or inconsistent."""
    d = Path(artifact_dir)
    for name in ("metadata.json", "timings.parquet"):
        if not (d / name).is_file():
            raise IncompleteRun(f"{d}: missing {name}")
    metadata = json.loads((d / "metadata.json").read_text())
    diag_files = sorted((d / "diagnostics").glob("batch-*.npz"))
    res_files = sorted((d / "results").glob("batch-*.npz"))
    if not diag_files or not res_files:
        raise IncompleteRun(f"{d}: {len(diag_files)} diagnostics / {len(res_files)} results batches")

    diag = [np.load(f) for f in diag_files]
    offsets = {tuple(z["level_offsets"].tolist()) for z in diag}
    if len(offsets) != 1:
        raise IncompleteRun(f"{d}: level_offsets differ between batches: {offsets}")
    level_offsets = np.asarray(offsets.pop())
    widths = metadata["effective_widths"]
    expected = np.cumsum([0, 0] + list(widths[1:]))
    if not np.array_equal(level_offsets, expected):
        raise IncompleteRun(f"{d}: level_offsets {level_offsets.tolist()} != widths {widths}")

    counters = []
    for f in diag_files:
        j = f.with_suffix(".json")
        counters.append(json.loads(j.read_text()) if j.is_file() else None)

    res = [np.load(f) for f in res_files]
    run = RunData(
        metadata=metadata,
        query_ids=np.concatenate([z["query_ids"] for z in diag]),
        ndis=np.concatenate([z["ndis"] for z in diag]),
        nhops=np.concatenate([z["nhops"] for z in diag]),
        candidate_ids=np.concatenate([z["candidate_ids"] for z in diag]),
        candidate_distances=np.concatenate([z["candidate_distances"] for z in diag]),
        level_offsets=level_offsets,
        batch_sizes=[len(z["query_ids"]) for z in diag],
        batch_counters=counters,
        result_query_ids=np.concatenate([z["query_ids"] for z in res]),
        ids=np.concatenate([z["ids"] for z in res]),
        distances=np.concatenate([z["distances"] for z in res]),
        timings=pq.read_table(d / "timings.parquet"),
    )

    nq = metadata["query_shape"][0]
    for name, q in (("diagnostics", run.query_ids), ("results", run.result_query_ids)):
        if not np.array_equal(np.sort(q), np.arange(nq)):
            raise IncompleteRun(f"{d}: {name} query_ids are not a permutation of 0..{nq - 1}")
    if run.ndis.shape[1] != len(widths):
        raise IncompleteRun(f"{d}: {run.ndis.shape[1]} levels in diagnostics, {len(widths)} widths")
    if (run.ids < 0).any():
        raise IncompleteRun(f"{d}: results contain missing neighbors (id -1)")
    if int(pc.sum(run.timings["query_count"]).as_py()) % nq:
        raise IncompleteRun(f"{d}: timings do not cover whole passes over {nq} queries")
    return run


def run_metrics(run: RunData) -> dict:
    """Run-level means recomputed from raw arrays, named as in runs.csv."""
    t = run.timings
    out = {
        "metric.ndis_per_query": run.ndis.sum(1).mean(),
        "metric.nhops_per_query": run.nhops.sum(1).mean(),
        "metric.mean_query_ms": pc.sum(t["elapsed_ns"]).as_py()
        / pc.sum(t["query_count"]).as_py() / 1e6,
    }
    for lvl in range(run.ndis.shape[1]):
        out[f"metric.level_{lvl}/ndis_per_query"] = run.ndis[:, lvl].mean()
        out[f"metric.level_{lvl}/nhops_per_query"] = run.nhops[:, lvl].mean()
    return out


def metadata_columns(metadata: dict) -> dict:
    """The run_2 columns that come straight from metadata.json."""
    return {
        "database_dim": metadata["database_shape"][1],
        "database_id": metadata["database_id"],
        "database_rows": metadata["database_shape"][0],
        "distance_semantics": metadata["distance_semantics"],
        "effective_search_threads": metadata["effective_search_threads"],
        "effective_widths": list(metadata["effective_widths"]),
        "faiss_version": metadata["faiss_version"],
        "index_id": metadata["index_id"],
        "index_reused": metadata["index_reused"],
        "machine": metadata["machine"],
        "platform": metadata["platform"],
        "python": metadata["python"],
        "query_id": metadata["query_id"],
        "query_rows": metadata["query_shape"][0],
        "r": metadata["r"],
        "source_fingerprint": metadata["source_fingerprint"],
    }


def run_tables(run: RunData, run_id: int) -> dict[str, pa.Table]:
    """The per-query and per-batch rows of one run, in run_2's schemas.

    Hand-down slots marked unused (id -1, distance +inf) are dropped: they occur
    where a level's width exceeds the number of nodes on that level.
    """
    nq, n_levels = run.ndis.shape
    k = run.ids.shape[1]
    tables = {}

    # entry_points: one row per used hand-down slot, levels ascending per query
    off = run.level_offsets
    slot_level = np.repeat(np.arange(len(off) - 1), np.diff(off))
    slot_rank = np.arange(off[-1]) - off[slot_level]
    used = run.candidate_ids >= 0
    if not np.array_equal(used, np.isfinite(run.candidate_distances)):
        raise IncompleteRun("candidate ids and distances disagree on unused slots")
    qi, si = np.nonzero(used)
    tables["entry_points"] = pa.table({
        "run_id": np.full(len(qi), run_id, np.int32),
        "query_id": run.query_ids[qi].astype(np.int64),
        "level": slot_level[si].astype(np.int8),
        "rank": slot_rank[si].astype(np.int16),
        "candidate_id": run.candidate_ids[qi, si].astype(np.int32),
        "candidate_distance": run.candidate_distances[qi, si].astype(np.float32),
    }, schema=SCHEMAS["entry_points"])

    tables["level_stats"] = pa.table({
        "run_id": np.full(nq * n_levels, run_id, np.int32),
        "query_id": np.repeat(run.query_ids, n_levels).astype(np.int64),
        "level": np.tile(np.arange(n_levels), nq).astype(np.int8),
        "ndis": run.ndis.reshape(-1).astype(np.int64),
        "nhops": run.nhops.reshape(-1).astype(np.int64),
    }, schema=SCHEMAS["level_stats"])

    tables["neighbors"] = pa.table({
        "run_id": np.full(nq * k, run_id, np.int32),
        "query_id": np.repeat(run.result_query_ids, k).astype(np.int64),
        "rank": np.tile(np.arange(k), nq).astype(np.int16),
        "neighbor_id": run.ids.reshape(-1).astype(np.int64),
        "distance": run.distances.reshape(-1).astype(np.float32),
    }, schema=SCHEMAS["neighbors"])

    t = run.timings
    tables["timings"] = pa.table(
        [pa.array(np.full(t.num_rows, run_id, np.int32))]
        + [t[c].cast(pa.int64()) for c in SCHEMAS["timings"].names[1:]],
        schema=SCHEMAS["timings"])

    # search_batches: ndis/nhops sum the batch's per-query values (identical to
    # Faiss's counters, checked on run_2); n1 is the batch's query count. n2
    # only exists in diagnostics/batch-N.json and is null without it.
    bounds = np.cumsum([0] + run.batch_sizes)
    rows = {"ndis": [], "nhops": [], "n1": [], "n2": []}
    for b, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        c = run.batch_counters[b] or {}
        rows["ndis"].append(int(run.ndis[lo:hi].sum()))
        rows["nhops"].append(int(run.nhops[lo:hi].sum()))
        rows["n1"].append(c.get("n1", int(hi - lo)))
        rows["n2"].append(c.get("n2"))
    tables["search_batches"] = pa.table({
        "run_id": np.full(len(run.batch_sizes), run_id, np.int32),
        "batch": np.arange(len(run.batch_sizes), dtype=np.int32),
        **rows,
    }, schema=SCHEMAS["search_batches"])
    return tables


@dataclass
class TableWriter:
    """Streams per-run tables to one Parquet file per table, in large row groups."""
    out_dir: Path
    row_group_rows: int = 1_000_000
    _writers: dict = field(default_factory=dict)
    _buffers: dict = field(default_factory=dict)
    rows: dict = field(default_factory=dict)

    def write(self, tables: dict[str, pa.Table]) -> None:
        for name, t in tables.items():
            buf = self._buffers.setdefault(name, [])
            buf.append(t)
            self.rows[name] = self.rows.get(name, 0) + t.num_rows
            if sum(x.num_rows for x in buf) >= self.row_group_rows:
                self._flush(name)

    def _flush(self, name: str) -> None:
        buf = self._buffers.get(name)
        if not buf:
            return
        if name not in self._writers:
            self._writers[name] = pq.ParquetWriter(
                Path(self.out_dir) / f"{name}.parquet", SCHEMAS[name], compression="zstd")
        self._writers[name].write_table(pa.concat_tables(buf), row_group_size=self.row_group_rows)
        self._buffers[name] = []

    def close(self) -> None:
        for name in list(self._buffers):
            self._flush(name)
        for w in self._writers.values():
            w.close()
