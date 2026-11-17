# Rerun experiment: `[0, w, 1, 1]` on Deep1M, GIST1M and clustered 200k

**Purpose:** confirm the reportable insights with multiple graph seeds, per-query data and exact ground truth.

## 1. What runs

| Item | Setting |
|---|---|
| Schedules | `standard` (stock greedy on every upper level) and `explicit` with `widths = [0, w]`, w ∈ {2, 4, 8}: beam of width w on level 1, greedy above, all candidates handed down (effective widths `[0, w, 1, 1]`) |
| Datasets | Deep1M (96-d, 10k queries), GIST1M (960-d, native 1000 queries), clustered 200k at d = 16 and 32 (200 clusters, centres ~ N(0, 4²I), points = centre + N(0, I), 10k queries) |
| Index | HNSW, M = 32, efConstruction = 100 |
| Search | k = 10, single thread |
| ef0 | 1M sets: 10, 23, 52, 79, 120, 182, 275, 512. Clustered: 10, 23, 52, 79, 120 |
| Graph seeds | 5 on Deep1M and GIST1M, 10 on clustered. Seeds change only the graph construction; the clustered point cloud is fixed (`dataset.seed: 20261106`) |
| Run count | 1M sets: 2 × 5 seeds × 4 schedules × 8 ef0 = 320. Clustered 200k: 2 × 10 × 4 × 5 = 400. Total 720 |

Level 1 switches from greedy (standard) to a beam with a visited set (w ≥ 2), so a small counting difference remains in the w = 1 → 2 step. Negligible on the 1M sets; a caveat for clustered.

## 2. Protocol

- Each graph is built once per seed and cached as an index file (multithreaded builds are not reproducible; the file also holds node levels). The cache key includes the `faiss/` source fingerprint.
- Timing: 1 warm-up + 3 timed passes; every pass and batch is stored. Recall is scored from the warm-up pass.
- One `run.py` process per cell, pinned with `taskset`, one at a time, run by `campaign.py` in a seeded random order that keeps one graph's cells together. Use a quiet machine.
- ndis is the primary metric; latency confirms.

## 3. What each run records

| Output | Content |
|---|---|
| `metrics.json` | recall@1, recall@10, mean ndis and nhops per level, mean ms, QPS |
| `diagnostics/` | per query, per level: ndis, nhops, and the handed-down candidates |
| `results/` | per query: the 10 returned ids and distances |
| `timings.parquet` | per pass and per batch |
| `metadata.json`, `config.yaml` | index id and sha256, effective widths, source fingerprint, environment |

## 4. GloVe-25 campaign

| Item | Setting |
|---|---|
| Data | ann-benchmarks `glove-25-angular.hdf5` (25-d), converted by `convert_glove.py`; first 1,000,000 of 1,183,514 base rows, all 10,000 queries |
| Metric | cosine via `metric: ip` + `normalize: true`; zero-norm rows: 0 in base, 0 in queries |
| Grid | M = 32, efConstruction = 100 (as the other sets, for fairness); `standard` + `explicit` `[0, w]`, w ∈ {2, 4, 8}; ef0 10, 23, 52, 79, 120, 182, 275, 512 |
| Seeds | 1-10 |
| Run count | 4 schedules × 10 seeds × 8 ef0 = 320 cells (`configs/campaign/glove25.yaml`) |

- Distance signs (measured): `results/` distances are +cos and `diagnostics/` candidate distances are +cos. Both are similarities, so larger is nearer, opposite of the L2 datasets.
- Times: graph build ~80-86 s per seed (4 threads); top-10 ground truth ~83 s once; `ground_truth.py` 93.5 s for seed 1 (S0 top-1000, levels, S1) and ~4-5 s for later seeds.
- Cross-check against the ann-benchmarks neighbours: 1815 queries have all reference neighbours in the first 1M; agreement 1.0.
- Campaign: 2026-11-11, 12:49-14:36 (~1 h 47 min, ~19 s per cell, one pinned core); 310 complete, 10 duplicate (the preparation ef0 = 10 standard cells), 0 failed.
- Analysis copy: `data_analysis/data/run_4/glove25_data_20261111/glove25/`.

