"""Run 3 data: loading, sanity checks and every aggregation behind the figures.

No plotting here. Every reported number goes through the same three steps:

  1. query -> run          run_means(), and the `runs` table (mean over the run's queries)
  2. run -> seed-paired    paired() at equal ef0, matched() at matched recall; always
                           w minus standard on the same graph seed
  3. seeds -> value        across_seeds(): mean, 95% t interval, seed range, sign test

Tidy tables (readable names, one function each):

  runs      one row per run                         load_runs()
  queries   one row per (w > 1 run, query), joined  load_queries()
            to its standard twin (same seed, ef0, query)
  paired    one row per (dataset, seed, width, ef0) paired()
  matched   one row per (dataset, seed, width, ef0) matched()

`dist_*` means distance computations (Faiss ndis) per query.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- constants

BLOCKS = ["deep1m", "gist1m", "clustered200k_d16", "clustered200k_d32"]
DATASETS = ["Deep1M", "GIST1M", "Clustered d=16", "Clustered d=32"]  # run_3; tables use the datasets loaded
DSNAME = {**dict(zip(BLOCKS, DATASETS)), "glove25": "GloVe-25"}
CLUSTERED = DATASETS[2:]
WIDTHS = [1, 2, 4, 8]  # 1 = standard
WS = [2, 4, 8]

MATCH_EF_MAX = 52  # clustered sets: matched recall uses w points with ef0 <= 52 only
MATCH_METHOD = "log1m"  # standard curve interpolated in log(1 - recall); "linear" = in recall
NEAR = -np.log(0.7)  # Figure 6 "near" fit: entry point at most about 30% closer or farther

LABELS = {
    "dataset": "dataset",
    "seed": "graph seed",
    "width": "level-1 beam width w",
    "ef0": "ef0",
    "recall_at_10": "recall@10",
    "one_minus_recall": "1 − recall@10",
    "latency_ms": "MQL (ms)",
    "dist_total": "total distance computations per query",
    "dist_layer0": "level-0 distance computations per query",
    "dist_layer1": "level-1 distance computations per query",
    "dist_upper": "level ≥ 2 distance computations per query",
    "layer0_saved": "level-0 saved per query",
    "layer1_extra": "extra level-1 per query",
    "net_saved": "net saved per query",
    "layer0_saved_pct": "level-0 saved (% of std total)",
    "layer1_extra_pct": "extra level-1 (% of std total)",
    "net_saved_pct": "net saved (% of std total)",
    "recall_gain": "recall@10 gain vs standard",
    "dist_total_delta_pct": "Δ total distance computations at matched recall (%)",
    "latency_delta_pct": "Δ MQL at matched recall (%)",
    "delta_pct": "Δ vs standard at matched recall (%)",
    "entry_dist": "L2 distance, query to ep1",
    "entry_closer_logratio": "entry point closer by (log-ratio scale)",
    "layer0_saved_rel_pct": "level-0 distance computations saved (%)",
    "ns_per_dist": "ns per distance computation (MQL / total)",
}


# --------------------------------------------------------------------------- loading


def _names(df: pd.DataFrame) -> list:
    """Dataset order of df: its categories, else DSNAME order of the names present."""
    s = df["dataset"]
    if isinstance(s.dtype, pd.CategoricalDtype):
        return list(s.cat.categories)
    present = set(s)
    return [d for d in DSNAME.values() if d in present]


@dataclass
class Run3Data:
    runs: pd.DataFrame
    queries: pd.DataFrame
    load_checks: dict = field(default_factory=dict)  # checks that need raw columns not kept in the tables


def load(data_dir="data/run_3") -> Run3Data:
    """Load `runs` and `queries` from the Parquet tables in `data_dir`."""
    runs = load_runs(data_dir)
    queries, load_checks = load_queries(data_dir, runs)
    return Run3Data(runs, queries, load_checks)


def load_runs(data_dir) -> pd.DataFrame:
    """One row per run (campaign cell).

    recall_at_10  = metric.recall_at_k (metrics.json, mean over queries)
    latency_ms    = MQL = sum over query batches of median(timed passes of the batch) / n_queries
    dist_layerL   = mean over queries of level-L distance computations
    dist_upper    = dist_layer2 + dist_layer3 + dist_layer4 (missing levels count 0)
    dist_total    = metric.ndis_per_query (= sum over levels)
    """
    data_dir = Path(data_dir)
    rename = {"metric.recall_at_k": "recall_at_10", "metric.ndis_per_query": "dist_total",
              "metric.mean_query_ms": "metric_latency_ms",
              **{f"metric.level_{l}/ndis_per_query": f"dist_layer{l}" for l in range(5)}}
    cols = ["run_id", "block", "seed", "w", "ef0", "n_levels", "status", "ground_truth_available", *rename]
    r = pd.read_parquet(data_dir / "runs.parquet", columns=cols).rename(columns=rename)
    names = r.pop("block").astype(str).map(DSNAME)
    assert names.notna().all(), "block missing from DSNAME"
    r.insert(1, "dataset", pd.Categorical(names, [d for d in DSNAME.values() if d in set(names)], ordered=True))
    r = r.rename(columns={"w": "width"})
    r["width"] = r.width.astype("int8")
    r["seed"] = r.seed.astype("int8")
    r["ef0"] = r.ef0.astype("int16")
    r["dist_upper"] = r[[f"dist_layer{l}" for l in (2, 3, 4)]].fillna(0).sum(axis=1)
    r = r.drop(columns=[f"dist_layer{l}" for l in (2, 3, 4)])

    t = pd.read_parquet(data_dir / "timings.parquet", columns=["run_id", "query_start", "query_count", "elapsed_ns"])
    batch = t.groupby(["run_id", "query_start"]).agg(ns=("elapsed_ns", "median"), n=("query_count", "first"))
    per_run = batch.groupby("run_id").agg(ns=("ns", "sum"), nq=("n", "sum"))
    r["latency_ms"] = r.run_id.map(per_run.ns / per_run.nq / 1e6)
    r["n_queries"] = r.run_id.map(per_run.nq).astype("int32")
    r["one_minus_recall"] = 1 - r.recall_at_10

    order = ["run_id", "dataset", "seed", "width", "ef0", "recall_at_10", "one_minus_recall", "latency_ms",
             "dist_total", "dist_layer0", "dist_layer1", "dist_upper", "n_queries",
             "n_levels", "status", "ground_truth_available", "metric_latency_ms"]
    return r[order].sort_values(["dataset", "seed", "width", "ef0"]).reset_index(drop=True)


def _row_groups(path, cols, fn):
    """Call fn on each row group of a Parquet file (keeps peak memory at one row group)."""
    pf = pq.ParquetFile(path)
    for g in range(pf.num_row_groups):
        fn(pf.read_row_group(g, columns=cols))


def load_queries(data_dir, runs: pd.DataFrame):
    """One row per (w > 1 run, query), with the standard twin's values beside it.

    Twin = standard run on the same dataset, seed and ef0, same query_id.
    dist_layer0 / dist_layer1      per-query level-0 / level-1 distance computations
    entry_id / entry_dist          ep1 = level-1 rank-0 entry point handed to level 0;
                                   entry_dist = sqrt(candidate_distance), true L2
    entry_unchanged                entry_id == entry_id_std
    entry_closer_logratio          x = ln(entry_dist_std / entry_dist); exactly 0 when the
                                   node is unchanged; NaN when either distance is 0
    layer0_saved_rel_pct           y = 100 (dist_layer0_std - dist_layer0) / dist_layer0_std
    recall_at_10(_std)             per-query recall, NaN where ground truth is missing
    Returns (queries, load_checks).
    """
    data_dir = Path(data_dir)
    by_id = runs.sort_values("run_id")
    offset = np.zeros(int(by_id.run_id.max()) + 1, dtype=np.int64)
    offset[by_id.run_id.to_numpy()] = np.r_[0, by_id.n_queries.cumsum().to_numpy()[:-1]]
    n = int(by_id.n_queries.sum())

    c0 = np.full(n, -1, np.int32)
    c1 = np.full(n, -1, np.int32)
    ep1_d2 = np.full(n, np.nan, np.float32)
    ep1_id = np.full(n, -1, np.int32)
    ep2_id = np.full(n, -1, np.int32)  # level-2 rank-0: start of level 1, for a check only
    rec = np.full(n, np.nan, np.float32)

    def take_levels(tb):
        lvl = tb["level"].to_numpy()
        pos = offset[tb["run_id"].to_numpy()] + tb["query_id"].to_numpy()
        ndis = tb["ndis"].to_numpy()
        for l, arr in ((0, c0), (1, c1)):
            m = lvl == l
            arr[pos[m]] = ndis[m]

    def take_eps(tb):
        m = tb["rank"].to_numpy() == 0
        lvl = tb["level"].to_numpy()
        pos = offset[tb["run_id"].to_numpy()] + tb["query_id"].to_numpy()
        cid, dst = tb["candidate_id"].to_numpy(), tb["candidate_distance"].to_numpy()
        m1, m2 = m & (lvl == 1), m & (lvl == 2)
        ep1_id[pos[m1]], ep1_d2[pos[m1]] = cid[m1], dst[m1]
        ep2_id[pos[m2]] = cid[m2]

    recall_sum = {}

    def take_recall(tb):
        rid = tb["run_id"].to_numpy()
        pos = offset[rid] + tb["query_id"].to_numpy()
        v = tb["recall_at_10"].to_numpy()
        rec[pos] = v
        s = pd.Series(v).groupby(rid).agg(["sum", "count"])
        for k, row in s.iterrows():
            a = recall_sum.setdefault(k, [0.0, 0])
            a[0] += row["sum"]
            a[1] += row["count"]

    _row_groups(data_dir / "level_stats.parquet", ["run_id", "query_id", "level", "ndis"], take_levels)
    _row_groups(data_dir / "entry_points.parquet",
                ["run_id", "query_id", "level", "rank", "candidate_id", "candidate_distance"], take_eps)
    _row_groups(data_dir / "query_recall.parquet", ["run_id", "query_id", "recall_at_10"], take_recall)
    ep1 = np.sqrt(ep1_d2)  # true L2
    del ep1_d2
    assert (c0 >= 0).all() and (c1 >= 0).all() and (ep1_id >= 0).all() and (ep2_id >= 0).all()
    assert np.isfinite(ep1).all()

    # w > 1 runs in (dataset, seed, width, ef0) order, each with its standard twin
    std_id = runs[runs.width == 1].set_index(["dataset", "seed", "ef0"]).run_id
    wr = runs[runs.width > 1].sort_values(["dataset", "seed", "width", "ef0"])
    twin = std_id.reindex(pd.MultiIndex.from_frame(wr[["dataset", "seed", "ef0"]])).to_numpy()
    nq = wr.n_queries.to_numpy()
    idx_w = np.concatenate([offset[r] + np.arange(k) for r, k in zip(wr.run_id, nq)])
    idx_s = np.concatenate([offset[r] + np.arange(k) for r, k in zip(twin, nq)])

    rep = lambda col, dt: np.repeat(wr[col].to_numpy(), nq).astype(dt)  # noqa: E731
    q = pd.DataFrame({
        "dataset": pd.Categorical.from_codes(np.repeat(wr.dataset.cat.codes.to_numpy(), nq),
                                             wr.dataset.cat.categories, ordered=True),
        "seed": rep("seed", "int8"),
        "width": rep("width", "int8"),
        "ef0": rep("ef0", "int16"),
        "query_id": np.concatenate([np.arange(k, dtype=np.int32) for k in nq]),
        "dist_layer0": c0[idx_w], "dist_layer0_std": c0[idx_s],
        "dist_layer1": c1[idx_w], "dist_layer1_std": c1[idx_s],
        "entry_id": ep1_id[idx_w], "entry_id_std": ep1_id[idx_s],
        "entry_dist": ep1[idx_w], "entry_dist_std": ep1[idx_s],
        "recall_at_10": rec[idx_w], "recall_at_10_std": rec[idx_s],
    })
    q["entry_unchanged"] = q.entry_id.to_numpy() == q.entry_id_std.to_numpy()
    d_w, d_s = q.entry_dist.to_numpy(), q.entry_dist_std.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = d_w / d_s
    ratio[q.entry_unchanged.to_numpy()] = 1.0
    ratio[(d_s == 0) | (d_w == 0)] = np.nan
    q["entry_closer_logratio"] = -np.log(ratio)
    q["layer0_saved_rel_pct"] = (100 * (q.dist_layer0_std.to_numpy(np.float64) - q.dist_layer0.to_numpy())
                             / q.dist_layer0_std.to_numpy()).astype(np.float32)

    # Checks that need raw columns the tables do not keep.
    base = runs.groupby("dataset", observed=True).ef0.min()
    wb = wr[wr.ef0.to_numpy() == wr.dataset.map(base).to_numpy().astype(int)]
    ep2_mism = sum(int((ep2_id[offset[r]:offset[r] + k] != ep2_id[offset[s]:offset[s] + k]).sum())
                   for r, s, k in zip(wb.run_id, std_id.reindex(pd.MultiIndex.from_frame(wb[["dataset", "seed", "ef0"]])), wb.n_queries))
    rs = pd.DataFrame(recall_sum, index=["sum", "count"]).T
    qr_mean = rs["sum"] / rs["count"]
    metric = runs.set_index("run_id").recall_at_10.reindex(qr_mean.index)
    load_checks = dict(n_rows_held=n, ep2_mismatches=ep2_mism,
                       recall_runs=len(qr_mean), recall_worst_diff=float((qr_mean - metric).abs().max()))
    return q, load_checks


def entry_distances(queries: pd.DataFrame, ef0=None) -> pd.DataFrame:
    """Per-query ep1 distances at one ef0 (default: each dataset's smallest), standard included.

    Standard rows (width 1) come from the w=2 rows' twin columns, so each standard query
    appears once. ep1 does not depend on ef0 (checked), so one ef0 stands for all.
    """
    base = queries.groupby("dataset", observed=True).ef0.min() if ef0 is None else None
    keep = (queries.ef0.to_numpy() == (queries.dataset.map(base).to_numpy().astype(int) if ef0 is None else ef0))
    q = queries.loc[keep, ["dataset", "seed", "width", "query_id", "entry_dist", "entry_dist_std",
                           "entry_unchanged", "entry_closer_logratio"]]
    s = q[q.width == 2].assign(width=np.int8(1), entry_dist=lambda d: d.entry_dist_std,
                               entry_unchanged=True, entry_closer_logratio=np.float32(0))
    out = pd.concat([s, q], ignore_index=True).drop(columns="entry_dist_std")
    return out.sort_values(["dataset", "width", "seed", "query_id"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- step 1: query -> run


def run_means(queries: pd.DataFrame, cols) -> pd.DataFrame:
    """Step 1. Mean over a run's queries: value(run) = (1 / n_queries) * sum_q value(q)."""
    return queries.groupby(["dataset", "seed", "width", "ef0"], observed=True)[cols].mean().reset_index()


# --------------------------------------------------------------------------- step 2: run -> seed-paired


def paired(runs: pd.DataFrame) -> pd.DataFrame:
    """Step 2 at equal ef0: each w run beside its standard twin (same dataset, seed, ef0).

    layer0_saved = dist_layer0_std - dist_layer0
    layer1_extra = dist_layer1 - dist_layer1_std
    net_saved    = layer0_saved - layer1_extra  (= dist_total_std - dist_total, levels >= 2 are equal)
    *_pct        = 100 * value / dist_total_std
    recall_gain  = recall_at_10 - recall_at_10_std
    """
    keys = ["dataset", "seed", "ef0"]
    vals = ["recall_at_10", "latency_ms", "dist_total", "dist_layer0", "dist_layer1"]
    std = runs.loc[runs.width == 1, keys + vals]
    p = runs.loc[runs.width > 1, keys + ["width"] + vals].merge(std, on=keys, suffixes=("", "_std"), validate="many_to_one")
    p["layer0_saved"] = p.dist_layer0_std - p.dist_layer0
    p["layer1_extra"] = p.dist_layer1 - p.dist_layer1_std
    p["net_saved"] = p.layer0_saved - p.layer1_extra
    for c in ("layer0_saved", "layer1_extra", "net_saved"):
        p[c + "_pct"] = 100 * p[c] / p.dist_total_std
    p["recall_gain"] = p.recall_at_10 - p.recall_at_10_std
    return p.sort_values(["dataset", "seed", "width", "ef0"]).reset_index(drop=True)


def interp_std(xp, fp, x, method=MATCH_METHOD):
    """Standard value at recall x from one seed's standard curve (xp = recall, fp = value).

    "linear": piecewise linear in recall.  "log1m": piecewise linear in -log(1 - recall),
    dropping standard points with recall 1.  NaN outside the curve's range: no extrapolation.
    """
    if not (xp.min() <= x <= xp.max()):
        return np.nan
    if method == "log1m":
        if x >= 1 or xp.max() >= 1:
            keep = xp < 1
            xp, fp = xp[keep], fp[keep]
            if x >= 1 or not (xp.min() <= x <= xp.max()):
                return np.nan
        return float(np.interp(-np.log1p(-x), -np.log1p(-xp), fp))
    if method != "linear":
        raise ValueError(method)
    return float(np.interp(x, xp, fp))


def matched(runs: pd.DataFrame, method=MATCH_METHOD, cut_std_curve=False,
            cols=("dist_total", "latency_ms")) -> pd.DataFrame:
    """Step 2 at matched recall: each w point against its own seed's standard curve.

    value_std = interp_std(standard recall curve, standard value, recall of the w point)
    delta_pct = 100 * (value_w / value_std - 1),  for value = dist_total and latency_ms
    Extra columns in `cols` (e.g. from query_tails) get their value_std too; their deltas are the caller's.
    w points: all ef0 on Deep1M/GIST1M, ef0 <= MATCH_EF_MAX on the clustered sets.
    cut_std_curve=True also cuts the clustered standard curve at ef0 <= MATCH_EF_MAX (sensitivity).
    Rows whose recall is outside the standard range stay in, with NaN.
    """
    cols = list(dict.fromkeys(["dist_total", "latency_ms", *cols]))
    rows = []
    for ds in _names(runs):
        rd = runs[runs.dataset == ds]
        efmax = MATCH_EF_MAX if ds in CLUSTERED else rd.ef0.max()
        for s in sorted(rd.seed.unique()):
            std = rd[(rd.seed == s) & (rd.width == 1)].sort_values("ef0")
            assert (std.recall_at_10.diff().dropna() >= 0).all(), "standard recall must rise with ef0"
            if cut_std_curve and ds in CLUSTERED:
                std = std[std.ef0 <= MATCH_EF_MAX]
            xp = std.recall_at_10.to_numpy()
            for r in rd[(rd.seed == s) & (rd.width > 1) & (rd.ef0 <= efmax)].sort_values(["width", "ef0"]).itertuples():
                row = dict(dataset=ds, seed=r.seed, width=r.width, ef0=r.ef0, recall_at_10=r.recall_at_10,
                           one_minus_recall=r.one_minus_recall, **{c: getattr(r, c) for c in cols})
                for c in cols:
                    row[c + "_std"] = interp_std(xp, std[c].to_numpy(), r.recall_at_10, method)
                rows.append(row)
    m = pd.DataFrame(rows)
    m["dataset"] = pd.Categorical(m.dataset, _names(runs), ordered=True)
    m["dist_total_delta_pct"] = 100 * (m.dist_total / m.dist_total_std - 1)
    m["latency_delta_pct"] = 100 * (m.latency_ms / m.latency_ms_std - 1)
    m.attrs["method"] = method
    return m.sort_values(["dataset", "seed", "width", "ef0"]).reset_index(drop=True)


def seed_means_over_ef(m: pd.DataFrame, col) -> pd.DataFrame:
    """Per (dataset, width, seed): mean of `col` over the seed's matched ef0 points (NaN points skipped)."""
    return m.dropna(subset=[col]).groupby(["dataset", "width", "seed"], observed=True)[col].mean().reset_index()


def matched_summary(m: pd.DataFrame, col) -> pd.DataFrame:
    """Matched-recall headline: seed_means_over_ef() then across_seeds() by (dataset, width)."""
    return across_seeds(seed_means_over_ef(m, col), col, ["dataset", "width"])


# --------------------------------------------------------------------------- step 3: seeds -> value

_T975 = {1: 12.7062, 2: 4.30265, 3: 3.18245, 4: 2.77645, 5: 2.57058, 6: 2.44691, 7: 2.36462, 8: 2.306, 9: 2.26216,
         10: 2.22814, 11: 2.20099, 12: 2.17881, 13: 2.16037, 14: 2.14479, 15: 2.13145, 19: 2.09302, 29: 2.04523}


def t975(df):
    """0.975 quantile of Student's t with `df` degrees of freedom (two-sided 95%)."""
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        return _T975[df]


def sign_p(values):
    """Exact two-sided sign test of median 0; zeros are dropped."""
    v = np.asarray(values)
    v = v[v != 0]
    k, n = int((v < 0).sum()), len(v)
    if n == 0:
        return np.nan
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n)


def _seed_stats(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    mean = v.mean() if n else np.nan
    half = t975(n - 1) * v.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan
    return dict(mean=mean, ci_lo=mean - half, ci_hi=mean + half,
                seed_min=v.min() if n else np.nan, seed_max=v.max() if n else np.nan,
                n_seeds=n, n_below_0=int((v < 0).sum()), sign_p=sign_p(v))


def across_seeds(df: pd.DataFrame, value_cols, by) -> pd.DataFrame:
    """Step 3. Per group in `by`, over the seeds (one row per seed, NaN skipped):

    mean, 95% t interval mean ± t(0.975, n-1) * sd / sqrt(n), seed_min, seed_max,
    n_seeds, n_below_0 (seeds with value < 0) and sign_p (exact two-sided sign test).
    `value_cols` a str gives flat columns; a list gives (column, statistic) columns.
    """
    single = isinstance(value_cols, str)
    cols = [value_cols] if single else list(value_cols)
    by = [by] if isinstance(by, str) else list(by)
    assert not df.duplicated(by + ["seed"]).any(), "across_seeds needs one row per seed and group"
    parts = {}
    for c in cols:
        recs = {k: _seed_stats(g[c].to_numpy()) for k, g in df.groupby(by, observed=True, sort=True)}
        t = pd.DataFrame.from_dict(recs, orient="index")
        t.index = pd.MultiIndex.from_tuples([k if isinstance(k, tuple) else (k,) for k in t.index], names=by)
        t[["n_seeds", "n_below_0"]] = t[["n_seeds", "n_below_0"]].astype(int)
        parts[c] = t
    out = parts[cols[0]] if single else pd.concat(parts, axis=1)
    if len(by) == 1:
        out.index = out.index.get_level_values(0)
    if "dataset" in by:
        out = out.reset_index()
        out["dataset"] = pd.Categorical(out.dataset, _names(df), ordered=True)
        out = out.sort_values(by).set_index(by)
    return out


# --------------------------------------------------------------------------- figure tables


def fig1_table(runs):
    """Recall and MQL per (dataset, ef0, width): seed mean, and the MQL 95% t half-width."""
    a = across_seeds(runs, ["recall_at_10", "latency_ms"], ["dataset", "ef0", "width"])
    t = pd.concat({"recall_at_10": a["recall_at_10"]["mean"], "latency_ms": a["latency_ms"]["mean"],
                   "latency_ms ±": (a["latency_ms"]["ci_hi"] - a["latency_ms"]["ci_lo"]) / 2}, axis=1)
    return t.unstack("width")


def fig2_table(runs, efs=(10, 23, 52)):
    """Total distance computations (seed mean) and ns per distance computation = 1e6 * MQL / dist_total."""
    d = runs[runs.ef0.isin(efs)].assign(ns_per_dist=lambda r: 1e6 * r.latency_ms / r.dist_total)
    a = across_seeds(d, ["dist_total", "ns_per_dist"], ["dataset", "ef0", "width"])
    return pd.concat({"dist_total": a["dist_total"]["mean"], "ns_per_dist": a["ns_per_dist"]["mean"]}, axis=1).unstack("width")


def matched_by_ef(m, col, stats=("mean",)):
    """Per (dataset, ef0, width): across_seeds of `col` at matched recall (step 3 at each ef0)."""
    a = across_seeds(m.dropna(subset=[col]), col, ["dataset", "ef0", "width"])
    return a[list(stats)]


def fig4_table(p):
    """Net balance at equal ef0: across_seeds of the paired columns per (dataset, ef0, width)."""
    a = across_seeds(p, ["layer0_saved", "layer1_extra", "net_saved", "net_saved_pct", "recall_gain"], ["dataset", "ef0", "width"])
    out = pd.DataFrame({
        "layer0_saved": a["layer0_saved"]["mean"], "layer1_extra": a["layer1_extra"]["mean"],
        "net_saved": a["net_saved"]["mean"], "net_ci_lo": a["net_saved"]["ci_lo"], "net_ci_hi": a["net_saved"]["ci_hi"],
        "net_min": a["net_saved"]["seed_min"], "net_max": a["net_saved"]["seed_max"],
        "net_pct": a["net_saved_pct"]["mean"], "net_pct_ci_lo": a["net_saved_pct"]["ci_lo"],
        "net_pct_ci_hi": a["net_saved_pct"]["ci_hi"], "net_pct_min": a["net_saved_pct"]["seed_min"],
        "net_pct_max": a["net_saved_pct"]["seed_max"],
        "seeds_net_pos": p.groupby(["dataset", "ef0", "width"], observed=True).net_saved.apply(lambda v: int((v > 0).sum())),
        "seeds": a["net_saved"]["n_seeds"], "net_sign_p": a["net_saved"]["sign_p"],
        "recall_gain": a["recall_gain"]["mean"]})
    return out


def entry_table(ent):
    """Figure 5. Seeds pooled per (dataset, width): ep1 distance quartiles; query shares with ep1
    unchanged, closer (x > 0) and farther (x < 0); unchanged share also as seed mean with 95% t CI."""
    rows = []
    for (ds, w), g in ent.groupby(["dataset", "width"], observed=True):
        v = g.entry_dist.to_numpy()
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        x = g.entry_closer_logratio.to_numpy()
        rows.append(dict(dataset=ds, width=w, median=med, q1=q1, q3=q3, unchanged_id=g.entry_unchanged.mean(),
                         closer=np.mean(x > 0), farther=np.mean(x < 0)))
    t = pd.DataFrame(rows).set_index(["dataset", "width"])
    per_seed = ent.groupby(["dataset", "width", "seed"], observed=True).entry_unchanged.mean().reset_index()
    a = across_seeds(per_seed, "entry_unchanged", ["dataset", "width"])
    t["unchanged_ci_lo"], t["unchanged_ci_hi"] = a.ci_lo, a.ci_hi
    return t


def fig6_pairs(queries, ds, ef0):
    """Per-query (x, y) for Figure 6 at one dataset and ef0, widths pooled: rows with finite x.

    x = entry_closer_logratio = ln(d_std / d_w),
    y = layer0_saved_rel_pct = 100 (L0_std - L0_w) / L0_std, recomputed in float64.
    """
    q = queries[(queries.dataset == ds) & (queries.ef0 == ef0)]
    q = q[np.isfinite(q.entry_closer_logratio.to_numpy())]
    y = 100 * (q.dist_layer0_std.to_numpy(np.float64) - q.dist_layer0.to_numpy()) / q.dist_layer0_std.to_numpy()
    return q.assign(layer0_saved_rel_pct=y)


def elasticity(queries, exclude_unchanged=False):
    """Figure 6. Per (dataset, ef0, seed), widths pooled, least-squares slope b of
    ln(L0_std / L0_w) = a + b * ln(d_std / d_w), restated as pct_per_10pct = 100 (1 - 0.9^b),
    the % level-0 saving for an entry point 10% closer. near_*: same fit on |x| <= NEAR.
    exclude_unchanged=True drops queries whose ep1 node did not change (x = 0).
    Then across_seeds per (dataset, ef0).
    """
    rows = []
    for ds in _names(queries):
        for ef in sorted(queries.loc[queries.dataset == ds, "ef0"].unique()):
            q = fig6_pairs(queries, ds, ef)
            x_all = q.entry_closer_logratio.to_numpy()
            share_near = np.mean(np.abs(x_all) <= NEAR)
            if exclude_unchanged:
                q = q[~q.entry_unchanged.to_numpy()]
            x, ln_c = q.entry_closer_logratio.to_numpy(), -np.log1p(-q.layer0_saved_rel_pct.to_numpy() / 100)
            near = np.abs(x) <= NEAR
            s = q.seed.to_numpy()
            for sd in np.unique(s):
                m = s == sd
                b = np.polyfit(x[m], ln_c[m], 1)[0]
                bn = np.polyfit(x[m & near], ln_c[m & near], 1)[0]
                rows.append(dict(dataset=ds, ef0=ef, seed=sd, elasticity=b, pct_per_10pct=100 * (1 - 0.9 ** b),
                                 near_pct_per_10pct=100 * (1 - 0.9 ** bn), share_near=share_near, n=int(m.sum())))
    per_seed = pd.DataFrame(rows)
    a = across_seeds(per_seed, ["elasticity", "pct_per_10pct", "near_pct_per_10pct"], ["dataset", "ef0"])
    return pd.DataFrame({
        "elasticity": a["elasticity"]["mean"], "pct_per_10pct": a["pct_per_10pct"]["mean"],
        "ci_lo": a["pct_per_10pct"]["ci_lo"], "ci_hi": a["pct_per_10pct"]["ci_hi"],
        "seed_min": a["pct_per_10pct"]["seed_min"], "seed_max": a["pct_per_10pct"]["seed_max"],
        "share_near": per_seed.groupby(["dataset", "ef0"], observed=True).share_near.first(),
        "near_pct_per_10pct": a["near_pct_per_10pct"]["mean"], "near_min": a["near_pct_per_10pct"]["seed_min"],
        "near_max": a["near_pct_per_10pct"]["seed_max"],
        "queries_per_seed": per_seed.groupby(["dataset", "ef0"], observed=True).n.mean()})


def zero_table(queries):
    """Figure 6. Seeds and widths pooled: level-0 saving (%) for queries with ep1 unchanged (x = 0)
    vs closer (x > 0), and the share of the mean saving that comes from unchanged-ep1 queries
    = share_x0 * mean_y(x = 0) / mean_y(all)."""
    rows = []
    for ds in _names(queries):
        for ef in sorted(queries.loc[queries.dataset == ds, "ef0"].unique()):
            q = fig6_pairs(queries, ds, ef)
            x, y = q.entry_closer_logratio.to_numpy(), q.layer0_saved_rel_pct.to_numpy()
            z = x == 0
            rows.append(dict(dataset=ds, ef0=ef, share_x0=z.mean(), y_mean_x0=y[z].mean(), y_median_x0=np.median(y[z]),
                             y_mean_xpos=y[x > 0].mean(), y_median_xpos=np.median(y[x > 0]),
                             y_mean_all=y.mean(), y_median_all=np.median(y)))
    t = pd.DataFrame(rows)
    t["dataset"] = pd.Categorical(t.dataset, _names(queries), ordered=True)
    t = t.set_index(["dataset", "ef0"])
    t["saving_share_unchanged"] = t.share_x0 * t.y_mean_x0 / t.y_mean_all
    return t


# --------------------------------------------------------------------------- sensitivity helpers


def heldout_interp_error(runs):
    """Drop one interior standard point, predict its dist_total from the rest of the seed's curve.
    Error = 100 (predicted / measured - 1), > 0 = standard overstated; seed mean per (dataset, ef0).
    Clustered sets: held-out points with ef0 <= MATCH_EF_MAX only."""
    rows = []
    for ds in _names(runs):
        rd = runs[(runs.dataset == ds) & (runs.width == 1)]
        for s in sorted(rd.seed.unique()):
            std = rd[rd.seed == s].sort_values("ef0")
            x, f = std.recall_at_10.to_numpy(), std.dist_total.to_numpy()
            for i in range(1, len(x) - 1):
                if ds in CLUSTERED and std.ef0.iloc[i] > MATCH_EF_MAX:
                    continue
                k = np.r_[0:i, i + 1:len(x)]
                rows.append(dict(dataset=ds, ef0=std.ef0.iloc[i],
                                 linear=100 * (interp_std(x[k], f[k], x[i], "linear") / f[i] - 1),
                                 log1m=100 * (interp_std(x[k], f[k], x[i], "log1m") / f[i] - 1)))
    t = pd.DataFrame(rows)
    t["dataset"] = pd.Categorical(t.dataset, _names(runs), ordered=True)
    return t.groupby(["dataset", "ef0"], observed=True)[["linear", "log1m"]].mean()


def dropped_points(m):
    """Seeds whose w recall falls outside their standard curve (no extrapolation), per (dataset, ef0, width)."""
    d = m[m.dist_total_std.isna()]
    return d.pivot_table(index=["dataset", "ef0"], columns="width", values="seed", aggfunc="size",
                         observed=True, fill_value=0)


# --------------------------------------------------------------------------- checks


def checks(data: Run3Data) -> pd.DataFrame:
    """Sanity checks behind the figures. Asserts each one that must hold; returns a summary table."""
    runs, q = data.runs, data.queries
    rows = []

    def add(name, result, ok):
        rows.append(dict(check=name, result=result, ok=bool(ok)))

    # step 1: per-query means equal the run metrics, for w runs and their standard twins
    keys = ["dataset", "seed", "width", "ef0"]
    rm = run_means(q, ["dist_layer0", "dist_layer1", "dist_layer0_std", "dist_layer1_std"])
    r = runs.set_index(keys)
    w = rm.set_index(keys)
    s = rm[rm.width == 2].assign(width=np.int8(1)).set_index(keys)
    e0 = max((w.dist_layer0 - r.dist_layer0.reindex(w.index)).abs().max(),
             (s.dist_layer0_std - r.dist_layer0.reindex(s.index)).abs().max())
    e1 = max((w.dist_layer1 - r.dist_layer1.reindex(w.index)).abs().max(),
             (s.dist_layer1_std - r.dist_layer1.reindex(s.index)).abs().max())
    add("per-query means vs run metrics, worst |diff| level 0 / level 1", f"{e0:.1e} / {e1:.1e}", e0 < 1e-9 and e1 < 1e-9)
    tot = (runs.dist_layer0 + runs.dist_layer1 + runs.dist_upper - runs.dist_total).abs().max()
    add("total = sum of levels, worst |diff|", f"{tot:.1e}", tot < 1e-9)

    # ep1 and level-1 work do not depend on ef0 (every dataset x seed x width, standard included)
    mism_id = mism_c1 = groups = 0
    for (ds, sd, wd), g in q.groupby(["dataset", "seed", "width"], observed=True):
        nef = g.ef0.nunique()
        groups += 1
        for col, cnt in (("entry_id", "id"), ("dist_layer1", "c1")):
            a = g[col].to_numpy().reshape(nef, -1)
            mm = int((a != a[0]).sum())
            if cnt == "id":
                mism_id += mm
            else:
                mism_c1 += mm
        if wd == 2:
            groups += 1
            for col in ("entry_id_std", "dist_layer1_std"):
                a = g[col].to_numpy().reshape(nef, -1)
                mm = int((a != a[0]).sum())
                mism_id += mm if col == "entry_id_std" else 0
                mism_c1 += mm if col == "dist_layer1_std" else 0
    add(f"ep1 and level-1 work across ef0, {groups} dataset×seed×width groups: mismatching queries",
        f"{mism_id} ep1, {mism_c1} level-1", mism_id == 0 and mism_c1 == 0)
    add("ep2 (start of level 1), w vs standard on the same graph: mismatching queries",
        str(data.load_checks["ep2_mismatches"]), data.load_checks["ep2_mismatches"] == 0)
    upper = runs.groupby(["dataset", "seed", "ef0"], observed=True).dist_upper.nunique().max()
    add("distinct level ≥ 2 work values within dataset×seed×ef0", str(upper), upper == 1)

    # same ep1 node with a slightly different stored distance (treated as x = 0)
    base = runs.groupby("dataset", observed=True).ef0.min()
    qb = q[q.ef0.to_numpy() == q.dataset.map(base).to_numpy().astype(int)]
    same = qb.entry_unchanged.to_numpy()
    a, c = qb.entry_dist.to_numpy()[same], qb.entry_dist_std.to_numpy()[same]
    neq = (a != c) & (c > 0)
    rel = float(np.abs(a[neq] / c[neq] - 1).max()) if neq.any() else 0.0
    add("same ep1 node, different stored distance (queries; largest relative diff)",
        f"{int(neq.sum())} of {int(same.sum())}; {rel:.1e}", rel < 1e-5)
    nz = int((~np.isfinite(qb.entry_closer_logratio.to_numpy())).sum())
    add("query×w rows with a zero ep1 distance (x undefined, dropped from Figure 6)", str(nz), True)

    # recall
    lc = data.load_checks
    add("query_recall: runs with per-query recall; worst |mean − metric.recall_at_k|",
        f"{lc['recall_runs']}; {lc['recall_worst_diff']:.1e}", lc["recall_worst_diff"] < 1e-9)
    add("metric.recall_at_k nulls", f"{int(runs.recall_at_10.isna().sum())} of {len(runs)}", runs.recall_at_10.notna().all())
    ratio = runs.latency_ms / runs.metric_latency_ms
    add("MQL vs metric.mean_query_ms: median ratio (p1–p99)",
        f"{ratio.median():.3f} ({ratio.quantile(.01):.3f}–{ratio.quantile(.99):.3f})", 0.95 < ratio.median() < 1.05)

    # step 2 identities
    p = paired(runs)
    dn = (p.net_saved - (p.dist_total_std - p.dist_total)).abs().max()
    add("net saved = total difference at equal ef0, worst |diff|", f"{dn:.1e}", dn < 1e-9)
    m = matched(runs, "linear")
    worst = 0.0
    present = _names(runs)
    hand = [d for d in CLUSTERED if d in present] or present
    efs = set()
    for ds in hand:  # matched recall by hand, seed 1, w=2, smallest ef0 inside the standard range, runs table only
        std = runs[(runs.dataset == ds) & (runs.seed == 1) & (runs.width == 1)].sort_values("ef0")
        w2 = runs[(runs.dataset == ds) & (runs.seed == 1) & (runs.width == 2)].sort_values("ef0")
        rr = w2[w2.recall_at_10.between(std.recall_at_10.min(), std.recall_at_10.max())].iloc[0]
        efs.add(int(rr.ef0))
        lo, hi = std[std.recall_at_10 <= rr.recall_at_10].iloc[-1], std[std.recall_at_10 >= rr.recall_at_10].iloc[0]
        v = lo.dist_total if hi.recall_at_10 == lo.recall_at_10 else \
            lo.dist_total + (rr.recall_at_10 - lo.recall_at_10) / (hi.recall_at_10 - lo.recall_at_10) * (hi.dist_total - lo.dist_total)
        tab = m[(m.dataset == ds) & (m.seed == 1) & (m.width == 2) & (m.ef0 == rr.ef0)].dist_total_delta_pct.iloc[0]
        worst = max(worst, abs(100 * (rr.dist_total / v - 1) - tab))
    which = "clustered" if hand[0] in CLUSTERED else ", ".join(hand)
    ef_text = "/".join(map(str, sorted(efs)))
    add(f"matched recall by hand ({which}, seed 1, w=2, ef0={ef_text}, linear) vs table, |diff| in %", f"{worst:.1e}", worst < 1e-9)
    h = heldout_interp_error(runs)
    add("held-out standard point, interpolation error range (%): linear / log1m",
        f"{h.linear.min():.1f}…{h.linear.max():.1f} / {h.log1m.min():.1f}…{h.log1m.max():.1f}", True)

    out = pd.DataFrame(rows).set_index("check")
    bad = out[~out.ok]
    assert bad.empty, f"failed checks:\n{bad}"
    return out


# --------------------------------------------------------------------------- Figures 7–8: entry-gain groups

GT_DATASETS = ["Deep1M", "Clustered d=16"]  # run_3 datasets with per-query recall (ground truth present)
N_GAIN_GROUPS = 3  # quantile groups of the entry gain among queries whose ep1 changed
RECALL_TARGETS = {"Deep1M": [0.80, 0.90, 0.95, 0.98], "Clustered d=16": [0.95, 0.98, 0.99, 0.995],
                  "GloVe-25": [0.85, 0.90, 0.95, 0.99]}


def entry_gain_groups(queries: pd.DataFrame, n_groups=N_GAIN_GROUPS) -> pd.DataFrame:
    """Per-query entry-gain group, on the datasets with per-query recall (ground truth present).

    entry_gain_pct = 100 * (1 - entry_dist / entry_dist_std): how much closer ep1 is under w.
    A query's ep1 does not depend on ef0, so its group is the same at every ef0.
    gain_group 0 = "unchanged" (same ep1 node); 1..n_groups = quantiles of entry_gain_pct among
    the changed queries, cut per (dataset, width) with all seeds pooled, so the edges are shared by seeds.
    """
    has_gt = queries.recall_at_10.notna().groupby(queries.dataset, observed=True).any()
    q = queries[queries.dataset.isin(has_gt.index[has_gt])].copy()
    q["dataset"] = q.dataset.cat.remove_unused_categories()
    q["entry_gain_pct"] = (100 * (1 - np.exp(-q.entry_closer_logratio.astype(np.float64)))).astype(np.float32)
    q["gain_group"] = np.int8(0)
    names = {}
    for (ds, w), g in q[~q.entry_unchanged].groupby(["dataset", "width"], observed=True):
        one = g[g.ef0 == g.ef0.min()]  # one row per (seed, query): the gain is the same at every ef0
        edges = np.quantile(one.entry_gain_pct.dropna(), np.linspace(0, 1, n_groups + 1))
        idx = np.clip(np.searchsorted(edges, g.entry_gain_pct.to_numpy(), side="right"), 1, n_groups)
        q.loc[g.index, "gain_group"] = idx.astype(np.int8)
        names[(ds, w)] = ["unchanged"] + [f"{edges[i]:.1f}–{edges[i + 1]:.1f}% closer" for i in range(n_groups)]
    q.attrs["group_names"] = names
    return q


def group_curves(qg: pd.DataFrame) -> pd.DataFrame:
    """Step 1 inside each entry-gain group: one row per (dataset, seed, width, gain_group, ef0), with the
    mean over the group's queries of recall and level-0/level-1 distance computations, for w and for
    the same queries under standard (the *_std columns). n_queries = queries in the group."""
    keys = ["dataset", "seed", "width", "gain_group", "ef0"]
    cols = ["recall_at_10", "recall_at_10_std", "dist_layer0", "dist_layer0_std", "dist_layer1", "dist_layer1_std",
            "entry_gain_pct"]
    c = qg.groupby(keys, observed=True)[cols].mean()
    c["n_queries"] = qg.groupby(keys, observed=True).size()
    c["one_minus_recall"] = 1 - c.recall_at_10
    c["one_minus_recall_std"] = 1 - c.recall_at_10_std
    return c.reset_index()


def _at_recall(ef, recall, value, target):
    """value at `target` recall on one curve over the ef0 grid, linear in -log(1 - recall), no extrapolation.
    Recall is made non-decreasing in ef0 first (a running max), since group means can dip by noise."""
    o = np.argsort(ef)
    r = np.maximum.accumulate(np.asarray(recall, float)[o])
    v = np.asarray(value, float)[o]
    keep = np.r_[True, np.diff(r) > 0]  # one point per distinct recall
    return interp_std(r[keep], v[keep], target, "log1m")


def group_matched(curves: pd.DataFrame, targets=None) -> pd.DataFrame:
    """Step 2 inside each entry-gain group, at matched recall R* (per seed, w vs standard on the same queries):

    layer0_saved = L0_std(R*) - L0_w(R*)    each curve interpolated at R* in log(1 - recall)
    layer1_extra = L1_w - L1_std            level 1 does not depend on ef0, so a constant per group
    net_saved    = layer0_saved - layer1_extra;   payback = layer0_saved / layer1_extra
    NaN when R* is outside either curve's recall range.
    """
    targets = targets or RECALL_TARGETS
    rows = []
    for (ds, s, w, gg), c in curves.groupby(["dataset", "seed", "width", "gain_group"], observed=True):
        l1_extra = (c.dist_layer1 - c.dist_layer1_std).mean()
        for t in targets[ds]:
            l0w = _at_recall(c.ef0, c.recall_at_10, c.dist_layer0, t)
            l0s = _at_recall(c.ef0, c.recall_at_10_std, c.dist_layer0_std, t)
            rows.append(dict(dataset=ds, seed=s, width=w, gain_group=gg, recall_target=t,
                             entry_gain_pct=c.entry_gain_pct.mean(), dist_layer0_std=l0s, dist_layer0=l0w,
                             layer0_saved=l0s - l0w, layer1_extra=l1_extra))
    m = pd.DataFrame(rows)
    m["dataset"] = pd.Categorical(m.dataset, _names(curves), ordered=True)
    m["net_saved"] = m.layer0_saved - m.layer1_extra
    m["payback"] = m.layer0_saved / m.layer1_extra
    return m


LABELS.update({
    "entry_gain_pct": "entry point closer by (%)",
    "gain_group": "entry-gain group",
    "recall_target": "recall@10 target",
    "payback": "payback (level-0 saved / extra level-1)",
})


def entry_gain_distribution(queries: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, width): how much closer ep1 gets, over queries (seeds pooled; one ef0, since ep1
    does not depend on it). entry_gain_pct = 100 * (1 - entry_dist / entry_dist_std); unchanged = same node.
    Columns: share unchanged, share at least 5/10/30% closer, and percentiles among the changed queries."""
    base = queries.groupby("dataset", observed=True).ef0.transform("min")
    q = queries[queries.ef0 == base]
    gain = 100 * (1 - np.exp(-q.entry_closer_logratio.astype(np.float64)))
    rows = []
    for (ds, w), g in gain.groupby([q.dataset, q.width], observed=True):
        ch = g[~q.loc[g.index, "entry_unchanged"]].dropna()
        rows.append(dict(dataset=ds, width=w, queries=len(g), unchanged=q.loc[g.index, "entry_unchanged"].mean(),
                         **{f"at_least_{p}pct": (g >= p).mean() for p in (5, 10, 30)},
                         **{f"changed_p{p}": np.percentile(ch, p) for p in (25, 50, 75, 90)}))
    return pd.DataFrame(rows).set_index(["dataset", "width"])


# --------------------------------------------------------------------------- Figure 9: tail of the per-query work

TAIL_Q = 95  # percentile of per-query total distance computations


def query_tails(data_dir, runs: pd.DataFrame, q=TAIL_Q) -> pd.DataFrame:
    """Step 1 with a percentile instead of the mean: `runs` plus, per run,

    dist_total_p95  = p-th percentile over the run's queries of total distance computations
                      (per query: sum of ndis over all levels, level_stats.parquet)
    tail_recall     = mean recall@10 of the run's costliest (100 - q)% of queries by that total
                      (ties broken by query_id); NaN where there is no per-query recall
    tail_n          = queries in that tail
    Reads only the Parquet columns it needs; the per-query rows are not kept.
    """
    data_dir = Path(data_dir)
    by_id = runs.sort_values("run_id")
    offset = np.zeros(int(by_id.run_id.max()) + 1, dtype=np.int64)
    offset[by_id.run_id.to_numpy()] = np.r_[0, by_id.n_queries.cumsum().to_numpy()[:-1]]
    n = int(by_id.n_queries.sum())
    tot = np.zeros(n, np.float64)
    rec = np.full(n, np.nan, np.float32)

    def take_ndis(tb):
        pos = offset[tb["run_id"].to_numpy()] + tb["query_id"].to_numpy()
        np.add.at(tot, pos, tb["ndis"].to_numpy())

    def take_recall(tb):
        rec[offset[tb["run_id"].to_numpy()] + tb["query_id"].to_numpy()] = tb["recall_at_10"].to_numpy()

    _row_groups(data_dir / "level_stats.parquet", ["run_id", "query_id", "ndis"], take_ndis)
    _row_groups(data_dir / "query_recall.parquet", ["run_id", "query_id", "recall_at_10"], take_recall)

    pcol = f"dist_total_p{q}"
    out = {}
    for rid, k in zip(by_id.run_id.to_numpy(), by_id.n_queries.to_numpy()):
        t, r = tot[offset[rid]:offset[rid] + k], rec[offset[rid]:offset[rid] + k]
        m = int(math.ceil(k * (100 - q) / 100))
        top = np.argsort(t, kind="stable")[-m:]
        out[rid] = (np.percentile(t, q), t.mean(), float(r[top].mean()) if np.isfinite(r).all() else np.nan, m)
    o = pd.DataFrame.from_dict(out, orient="index", columns=[pcol, "_mean_check", "tail_recall", "tail_n"])
    r = runs.join(o, on="run_id")
    worst = float((r._mean_check - r.dist_total).abs().max())
    assert worst < 1e-6, f"per-query totals do not average to metric.ndis_per_query (worst {worst})"
    r.attrs["mean_check_worst"] = worst
    return r.drop(columns="_mean_check")


def tail_matched(runs_t: pd.DataFrame, method=MATCH_METHOD, q=TAIL_Q) -> pd.DataFrame:
    """Step 2 at matched mean recall, for the tail columns of query_tails():

    dist_total_p95_delta_pct = 100 * (p95_w / p95_std(R) - 1)   R = the w point's mean recall
    tail_recall_delta        = tail_recall_w - tail_recall_std(R)  (recall points, > 0 = w better)
    Both standard values are read off the seed's standard curve at R, as in matched().
    """
    pcol = f"dist_total_p{q}"
    m = matched(runs_t, method, cols=("dist_total", "latency_ms", pcol, "tail_recall"))
    m[pcol + "_delta_pct"] = 100 * (m[pcol] / m[pcol + "_std"] - 1)
    m["tail_recall_delta"] = m.tail_recall - m.tail_recall_std
    return m


LABELS.update({
    f"dist_total_p{TAIL_Q}": f"p{TAIL_Q} total distance computations per query",
    f"dist_total_p{TAIL_Q}_delta_pct": f"Δ p{TAIL_Q} total distance computations at matched recall (%)",
    "tail_recall": f"recall@10 of the costliest {100 - TAIL_Q}% of queries",
    "tail_recall_delta": f"Δ recall@10, costliest {100 - TAIL_Q}% of queries (points)",
})
