"""Convert a campaign (run_3 rerun, run_4 GloVe-25) into run_2's Parquet tables.

Usage (from analysis/):
    python scripts/convert_run3.py ../outputs/rerun data/run_3 --cache ../cache
    python scripts/convert_run3.py ../outputs/glove25 data/run_4 --cache ../cache \
        --source run_4 --run-id-offset 4000
    python scripts/convert_run3.py <same arguments> --check-only

The raw directory holds campaign.jsonl (one line per cell) and one directory per
run: per-graph setup runs first, then the cells in campaign order. Cells with
status "duplicate" hold no arrays and point at the setup run that measured them.
The per-run conversion lives in hnsw_tables.py; this file maps cells to run
directories, builds the run-level row and adds per-query recall.

Inner-product runs (cosine on unit vectors) are stored as squared L2 between the
unit vectors, 2 - 2 cos, so that every table reads like an L2 run; runs keeps
the raw distance_semantics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnsw_tables import (SCHEMAS, TableWriter, load_run, metadata_columns,  # noqa: E402
                         run_metrics, run_tables)

PER_RUN_TABLES = ["entry_points", "level_stats", "neighbors", "timings", "search_batches"]
# A seventh table, so the five per-query/per-batch tables above keep run_2's
# schema exactly and still concatenate with it.
SCHEMAS["query_recall"] = pa.schema([
    ("run_id", pa.int32()), ("query_id", pa.int64()), ("recall_at_10", pa.float64()),
])
EXTRA_RUN_COLUMNS = [
    ("source", pa.large_string()),
    ("metric.level_4/ndis_per_query", pa.float64()),
    ("metric.level_4/nhops_per_query", pa.float64()),
    ("block", pa.large_string()),
    ("dimension", pa.int64()),
    ("seed", pa.int64()),
    ("schedule", pa.large_string()),
    ("w", pa.int64()),
    ("ef0", pa.int64()),
    ("n_levels", pa.int64()),
    ("cell_dir", pa.large_string()),
    ("duplicate_of", pa.large_string()),
    ("campaign_seconds", pa.float64()),
    ("index_sha256", pa.large_string()),
    ("ground_truth_file", pa.large_string()),
    ("ground_truth_available", pa.bool_()),
    ("recall_at_10_from_ground_truth", pa.float64()),
]
DATASET_SIGNATURE = {
    "deep1m": ("bench.datasets.FbinDataset", 1_000_000, 96),
    "gist1m": ("bench.datasets.FvecsDataset", 1_000_000, 960),
    "synthetic_clustered_200k": ("bench.datasets.SyntheticDataset", 200_000, None),
    "glove25": ("bench.datasets.FvecsDataset", 1_000_000, 25),
}
SANITY_TARGETS = {"deep1m": {1: (0.7511, 493, 53.6), 2: (0.7498, 504, 77.2)}}  # w: (R@10, ndis, L1 ndis)


def flatten(d: dict, prefix: str = "config.") -> dict:
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def check_cell_config(cell: dict, cfg: dict, where: str) -> None:
    """Fail unless a run directory's config is the campaign cell it should be."""
    target, size, native_dim = DATASET_SIGNATURE[cell["dataset"]]
    ds, st = cfg["dataset"], cfg["strategy"]
    width = cell["width"]
    problems = []
    if ds["_target_"] != target or ds["database_size"] != size:
        problems.append(f"dataset {ds['_target_']} size {ds['database_size']}")
    if cell["dimension"] is not None and ds.get("dimension") != cell["dimension"]:
        problems.append(f"dimension {ds.get('dimension')}")
    if cfg["index"]["seed"] != cell["seed"] or cfg["experiment_seed"] != cell["seed"]:
        problems.append(f"seed {cfg['index']['seed']}/{cfg['experiment_seed']}")
    if st["schedule"] != cell["schedule"] or st["ef_search"] != cell["ef_search"]:
        problems.append(f"schedule {st['schedule']} ef {st['ef_search']}")
    expected_widths = None if width is None else [0, width]
    if st["widths"] != expected_widths:
        problems.append(f"widths {st['widths']}")
    if problems:
        raise RuntimeError(f"{where} does not match campaign cell {cell}: {problems}")


def map_cells(raw: Path) -> tuple[list[dict], list[str]]:
    """Pair every campaign line with its directory, verifying config and status."""
    lines = [json.loads(x) for x in (raw / "campaign.jsonl").read_text().splitlines() if x.strip()]
    dirs = sorted(p.name for p in raw.iterdir() if p.is_dir())
    n_setup = len(dirs) - len(lines)
    setup, cell_dirs = dirs[:n_setup], dirs[n_setup:]
    keys = [tuple(sorted(x["cell"].items(), key=lambda kv: kv[0])) for x in lines]
    if len(set(keys)) != len(keys):
        raise RuntimeError("campaign.jsonl contains repeated cells")

    used_setup: dict[str, int] = {}
    cells = []
    for i, (line, name) in enumerate(zip(lines, cell_dirs)):
        cell = line["cell"]
        cfg = yaml.safe_load((raw / name / "config.yaml").read_text())
        complete = json.loads((raw / name / "complete.json").read_text())
        check_cell_config(cell, cfg, name)
        if complete["status"] != line["status"]:
            raise RuntimeError(f"{name}: complete.json {complete['status']} vs campaign {line['status']}")
        data_dir, duplicate_of = name, None
        if line["status"] == "duplicate":
            duplicate_of = data_dir = Path(complete["of"]).name
            if data_dir not in setup:
                raise RuntimeError(f"{name}: 'of' {complete['of']} is not a setup run")
            used_setup[data_dir] = used_setup.get(data_dir, 0) + 1
            setup_cfg = yaml.safe_load((raw / data_dir / "config.yaml").read_text())
            if setup_cfg != cfg:
                raise RuntimeError(f"{name}: config differs from its setup run {data_dir}")
            if cell["schedule"] != "standard" or cell["ef_search"] != 10:
                raise RuntimeError(f"{name}: duplicate cell is not standard ef=10")
        elif line["status"] != "complete":
            raise RuntimeError(f"{name}: unexpected status {line['status']}")
        metadata = json.loads((raw / data_dir / "metadata.json").read_text())
        expected_w = cell["width"] or 1
        if metadata["effective_widths"][1] != expected_w or metadata["effective_widths"][0] != 0 \
                or any(x != 1 for x in metadata["effective_widths"][2:]):
            raise RuntimeError(f"{name}: effective widths {metadata['effective_widths']} for w={expected_w}")
        native_dim = DATASET_SIGNATURE[cell["dataset"]][2] or cell["dimension"]
        if metadata["database_shape"] != [DATASET_SIGNATURE[cell["dataset"]][1], native_dim]:
            raise RuntimeError(f"{name}: database_shape {metadata['database_shape']}")
        cells.append(dict(job_id=i, cell=cell, status=line["status"], seconds=line["seconds"],
                          cell_dir=name, data_dir=data_dir, duplicate_of=duplicate_of, config=cfg))

    if sorted(used_setup) != sorted(setup) or set(used_setup.values()) != {1}:
        raise RuntimeError("setup runs and duplicate cells are not one-to-one")
    for name in setup:
        cfg = yaml.safe_load((raw / name / "config.yaml").read_text())
        if cfg["strategy"]["schedule"] != "standard" or cfg["strategy"]["ef_search"] != 10:
            raise RuntimeError(f"{name}: setup run is not standard ef=10")
    print(f"mapping verified: {len(cells)} cells in campaign order after {len(setup)} setup runs; "
          f"{len(used_setup)} duplicates each resolve to a distinct setup run with an identical config",
          flush=True)
    return cells, setup


def runs_schema(template: Path) -> pa.Schema:
    base = pq.read_schema(template).remove_metadata()
    return pa.schema(list(base) + [pa.field(n, t) for n, t in EXTRA_RUN_COLUMNS])


def per_query_recall(ids: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """Set overlap of returned and exact ids over k, bench.measure.score's rule per query."""
    return (ids[:, None, :] == exact[:, :, None]).any(2).sum(1) / exact.shape[1]


def squared_l2_from_cosine(d: np.ndarray) -> np.ndarray:
    """2 - 2 cos, clipped at 0; unused (non-finite) slots stay +inf."""
    finite = np.isfinite(d)
    return np.where(finite, np.maximum(2 - 2 * d.astype(np.float64), 0), np.inf).astype(d.dtype)


def as_squared_l2(run, cfg: dict) -> None:
    semantics = run.metadata["distance_semantics"]
    if semantics == "squared_l2":
        return
    if semantics != "inner_product" or not cfg["dataset"].get("normalize"):
        raise RuntimeError(f"no squared-L2 form for {semantics} (normalize={cfg['dataset'].get('normalize')})")
    run.candidate_distances = squared_l2_from_cosine(run.candidate_distances)
    run.distances = squared_l2_from_cosine(run.distances)


def experiment_column(cell: dict) -> str:
    return "standard" if cell["width"] is None else f"explicit_0-{cell['width']}"


def build_row(c: dict, run, run_id: int, metrics: dict, schema: pa.Schema, gt_name: str,
              gt_available: bool, gt_recall: float | None, label: str, source: str) -> dict:
    cell, cfg = c["cell"], c["config"]
    row = {f.name: None for f in schema}
    flat = flatten(cfg)
    unknown = [k for k in flat if k not in row]
    if unknown:
        raise RuntimeError(f"config keys with no runs column: {unknown}")
    row.update(flat)
    row.update(metadata_columns(run.metadata))
    row.update({f"metric.{k}": v for k, v in metrics.items()})
    row.update(
        dataset=cell["dataset"], experiment=cfg["experiment_name"],
        experiment_column=experiment_column(cell), repeat=1, run_id=run_id, job_id=c["job_id"],
        run_dir=f"{label}/{c['data_dir']}", sweep_dir=label, status=c["status"], source=source,
        block=cell["block"], dimension=run.metadata["database_shape"][1], seed=cell["seed"],
        schedule=cell["schedule"], w=cell["width"] or 1, ef0=cell["ef_search"],
        n_levels=len(run.metadata["effective_widths"]), cell_dir=f"{label}/{c['cell_dir']}",
        duplicate_of=c["duplicate_of"], campaign_seconds=c["seconds"],
        index_sha256=run.metadata["index_sha256"], ground_truth_file=gt_name,
        ground_truth_available=gt_available, recall_at_10_from_ground_truth=gt_recall,
    )
    return row


def convert(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob("*.parquet")):
        sys.exit(f"{args.out} already contains Parquet files; remove them first")
    cells, _ = map_cells(args.raw)
    if args.limit:
        cells = cells[:args.limit]
    schema = runs_schema(args.template)
    writer = TableWriter(args.out)
    ground_truth: dict[str, np.ndarray | None] = {}
    run_rows = []
    worst = 0.0
    t0 = time.time()

    for c in cells:
        run_id = args.run_id_offset + c["job_id"]
        d = args.raw / c["data_dir"]
        run = load_run(d)
        as_squared_l2(run, c["config"])
        reported = json.loads((d / "metrics.json").read_text())

        # The raw arrays must reproduce the run's own metrics.json.
        for name, value in run_metrics(run).items():
            ref = reported[name.removeprefix("metric.")]
            rel = abs(value - ref) / max(abs(ref), 1e-12)
            worst = max(worst, rel)
            if rel > args.metric_tolerance:
                raise RuntimeError(f"{d.name}: {name} recomputed {value} vs metrics.json {ref}")
        bounds = np.cumsum([0] + run.batch_sizes)
        for b, counter in enumerate(run.batch_counters):
            lo, hi = bounds[b], bounds[b + 1]
            if counter is None or counter["ndis"] != int(run.ndis[lo:hi].sum()) \
                    or counter["nhops"] != int(run.nhops[lo:hi].sum()):
                raise RuntimeError(f"{d.name}: batch {b} counters {counter} disagree with per-query arrays")

        gt_name = Path(run.metadata["ground_truth_path"]).name
        if gt_name not in ground_truth:
            path = args.cache / gt_name
            ground_truth[gt_name] = np.load(path)["ids"] if path.is_file() else None
        exact = ground_truth[gt_name]
        tables = run_tables(run, run_id)
        gt_recall = None
        if exact is not None:
            if exact.shape != (run.metadata["query_shape"][0], run.ids.shape[1]):
                raise RuntimeError(f"{gt_name}: shape {exact.shape} does not fit run {d.name}")
            recall = per_query_recall(run.ids, exact[run.result_query_ids])
            gt_recall = float(recall.mean())
            tables["query_recall"] = pa.table({
                "run_id": np.full(len(recall), run_id, np.int32),
                "query_id": run.result_query_ids.astype(np.int64),
                "recall_at_10": recall,
            }, schema=SCHEMAS["query_recall"])
        writer.write(tables)
        run_rows.append(build_row(c, run, run_id, reported, schema, gt_name, exact is not None, gt_recall,
                                  args.raw.name, args.source))
        if (c["job_id"] + 1) % 50 == 0:
            print(f"{c['job_id'] + 1}/{len(cells)} cells, {time.time() - t0:.0f}s", flush=True)

    writer.close()
    columns = {}
    for f in schema:
        values = [r[f.name] for r in run_rows]
        values = [None if isinstance(v, float) and math.isnan(v) else v for v in values]
        columns[f.name] = pa.array(values, type=f.type)
    pq.write_table(pa.table(columns, schema=schema), args.out / "runs.parquet", compression="zstd")
    print(f"done: {len(run_rows)} runs in {time.time() - t0:.0f}s; "
          f"worst relative metric mismatch {worst:.2e}")
    for name, n in sorted(writer.rows.items()):
        print(f"  {name}: {n} rows")


def print_cell(args: argparse.Namespace, runs: pd.DataFrame, block: str, ef0: int, w: int,
               target: tuple | None) -> None:
    row = runs[(runs.dataset == block) & (runs.seed == 1) & (runs.ef0 == ef0) & (runs.w == w)]
    assert len(row) == 1
    row = row.iloc[0]
    expected = f"expected R@10 {target[0]} ndis {target[1]} L1 {target[2]}" if target else f"{block} ef0={ef0}"
    stats = pq.read_table(args.out / "level_stats.parquet", filters=[("run_id", "=", int(row.run_id))]).to_pandas()
    by_level = stats.groupby("level").ndis.mean()
    recall = pq.read_table(args.out / "query_recall.parquet", filters=[("run_id", "=", int(row.run_id))]).to_pandas()
    eps = pq.read_table(args.out / "entry_points.parquet", filters=[("run_id", "=", int(row.run_id))]).to_pandas()
    print(f"  w={w} run_id={row.run_id} status={row.status} {expected}; "
          f"actual R@10 {row['metric.recall_at_k']:.5f} (per-query mean {recall.recall_at_10.mean():.5f}), "
          f"total ndis {row['metric.ndis_per_query']:.2f} (level_stats {stats.groupby('query_id').ndis.sum().mean():.2f}), "
          f"L1 ndis {row['metric.level_1/ndis_per_query']:.2f} (level_stats {by_level[1]:.2f})")
    print(f"    mean ndis by level index: {by_level.round(2).to_dict()}")
    print(f"    entry rows per query by level: {(eps.groupby('level').size() / row.query_rows).to_dict()}")
    nearest = eps[eps['rank'] == 0].pivot(index="query_id", columns="level", values="candidate_distance")
    levels = sorted(nearest.columns)
    steps = {f"{hi}->{lo}": bool((nearest[lo] <= nearest[hi]).all()) for lo, hi in zip(levels, levels[1:])}
    print(f"    mean rank-0 entry distance by level: {nearest.mean().round(4).to_dict()}; "
          f"never farther when descending: {steps}")


def run_check(args: argparse.Namespace) -> None:
    """Sanity checks over the written tables; prints values, never adjusts them."""
    pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 100)
    runs = pd.read_parquet(args.out / "runs.parquet")
    print(f"runs: {len(runs)} rows; run_id {runs.run_id.min()}..{runs.run_id.max()}; "
          f"status {runs.status.value_counts().to_dict()}")
    print(runs.groupby(["dataset", "dimension"]).agg(
        cells=("run_id", "size"), seeds=("seed", "nunique"), widths=("w", lambda s: sorted(set(s))),
        efs=("ef0", lambda s: sorted(set(s))), gt=("ground_truth_available", "first"),
        queries=("query_rows", "first"), n_levels=("n_levels", lambda s: sorted(set(s)))))

    for name in PER_RUN_TABLES:
        schema = pq.read_schema(args.out / f"{name}.parquet").remove_metadata()
        same = schema.equals(SCHEMAS[name])
        print(f"{name}: {pq.read_metadata(args.out / f'{name}.parquet').num_rows} rows; "
              f"schema identical to run_2: {same}")
    base = pq.read_schema(args.template).remove_metadata()
    print(f"runs: first {len(base)} columns identical to run_2: "
          f"{pa.schema(list(pq.read_schema(args.out / 'runs.parquet').remove_metadata())[:len(base)]).equals(base)}")

    if (runs.distance_semantics != "squared_l2").any():
        print(f"\ndistance_semantics {runs.distance_semantics.value_counts().to_dict()}; "
              "inner-product distances stored as 2 - 2 cos")
    present = [b for b in SANITY_TARGETS if b in set(runs.dataset)]
    if present:
        print("\nspecified sanity targets (Deep1M seed 1, ef=10):")
        for block in present:
            for w, target in SANITY_TARGETS[block].items():
                print_cell(args, runs, block, 10, w, target)
    else:
        print("\nno specified sanity targets; seed 1 at the smallest ef0:")
        for block in sorted(set(runs.dataset)):
            ef = int(runs[runs.dataset == block].ef0.min())
            for w in (1, 2):
                print_cell(args, runs, block, ef, w, None)

    print("\nper-query recall mean vs metrics.json recall_at_k:")
    with_gt = runs[runs.ground_truth_available]
    diff = (with_gt.recall_at_10_from_ground_truth - with_gt["metric.recall_at_k"]).abs()
    print(f"  {len(with_gt)} runs with ground truth; max |difference| {diff.max():.3e}; "
          f"without ground truth: {(~runs.ground_truth_available).sum()} runs "
          f"{sorted(set(runs[~runs.ground_truth_available].block))}")
    recall = pq.read_table(args.out / "query_recall.parquet").group_by("run_id").aggregate(
        [("recall_at_10", "mean"), ("recall_at_10", "count")]).to_pandas().set_index("run_id")
    joined = with_gt.set_index("run_id").join(recall)
    print(f"  from query_recall.parquet: max |difference| "
          f"{(joined.recall_at_10_mean - joined['metric.recall_at_k']).abs().max():.3e}; "
          f"rows per run equal query_rows: {bool((joined.recall_at_10_count == joined.query_rows).all())}")

    print("\nschedule-level and graph-level invariants:")
    upper = runs.groupby(["block", "seed", "w"])["metric.level_1/ndis_per_query"].nunique()
    print(f"  distinct L1 ndis across ef within block/seed/w: {upper.value_counts().to_dict()}")
    per_graph = runs.groupby(["block", "seed"]).agg(index=("index_id", "nunique"), db=("database_id", "nunique"))
    print(f"  index_id per block/seed unique: {bool((per_graph['index'] == 1).all())}; "
          f"database_id across seeds per block: {runs.groupby('block').database_id.nunique().to_dict()}")
    print(f"  index_reused: {runs.index_reused.value_counts().to_dict()}")
    print(f"  metric.level_4 rows non-null: {runs['metric.level_4/ndis_per_query'].notna().sum()} "
          f"(n_levels=5 rows: {(runs.n_levels == 5).sum()})")
    print(runs.groupby("block").seed.apply(lambda s: sorted(set(s[runs.loc[s.index, 'n_levels'] == 5]))).to_string())
    drops = runs.sort_values("ef0").groupby(["block", "seed", "w"])["metric.recall_at_k"].apply(
        lambda s: float(s.diff().min()))
    print(f"  largest recall drop with rising ef: {drops.min():.5f}")
    print("\nmean R@10 and QPS at the largest ef per block and w:")
    top = runs[runs.ef0 == runs.groupby("block").ef0.transform("max")]
    print(top.groupby(["block", "ef0", "w"])[["metric.recall_at_k", "metric.queries_per_second"]].mean().round(4))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path, help="directory holding campaign.jsonl and the run directories")
    ap.add_argument("out", type=Path, help="output directory (must not contain tables yet)")
    ap.add_argument("--template", type=Path, default=Path(__file__).resolve().parent / "runs_template.parquet",
                    help="empty run_2 runs table, for column order and dtypes")
    ap.add_argument("--cache", type=Path, required=True, help="directory holding ground-truth-*.npz")
    ap.add_argument("--run-id-offset", type=int, default=3000,
                    help="run_id of the first cell; keeps ids disjoint from run_1 and run_2")
    ap.add_argument("--source", default="run_3", help="value of the runs.source column")
    ap.add_argument("--metric-tolerance", type=float, default=1e-9)
    ap.add_argument("--limit", type=int, help="convert only the first N cells (for testing)")
    ap.add_argument("--check-only", action="store_true", help="run the sanity checks on existing tables")
    args = ap.parse_args()
    if not args.check_only:
        convert(args)
    if not args.limit:
        run_check(args)


if __name__ == "__main__":
    main()
