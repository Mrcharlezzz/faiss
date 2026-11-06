# Per-level efSearch on the HNSW upper levels

Stock HNSW descends the upper levels greedily, so it can stop in a local
minimum and hand a poor entry point to level 0. This experiment tests a beam
on the upper levels, with the decay schedule `ef_l = max(1, floor(efSearch * r^l))`,
to see whether better entry points buy recall per distance computed. A beam
level hands down its whole result (its `ef_l` best nodes, sorted nearest
first), and all of them seed the next level and level 0; a greedy level hands
down its nearest node, as in stock HNSW.

`SearchParametersHNSW.efSearch_by_level` sets the width per upper level
(empty is stock HNSW), and `hnsw_level_stats` records, per query and level,
the handed-down set and the distances and hops spent there. Tests:
`tests/test_hnsw_per_layer_efsearch.py`.

## Results

## Rerun campaign

1. Build each index once (a normal `run.py` cell with `index.cache_policy=reuse`), then compute the exact ground truth for it: `python benchs/multilayer/ground_truth.py dataset=<name> experiment_seed=<seed>`. This writes S0 (top-1000), node levels, S1 (top-100 over level >= 1 nodes) and, for clustered data, the cluster labels into `cache/`.
2. Run the timed cells: `python benchs/multilayer/campaign.py --config benchs/multilayer/configs/campaign/rerun.yaml --core <n>`. Use `--dry-run` to list the 720 cells in order and `--limit N` to run a few.

The Hydra sweeper is not used for the timed campaign: the driver runs one pinned `run.py` process per cell. A restart skips cells recorded in `outputs/rerun/campaign.jsonl`, and the per-run dedupe markers in `cache/` make a repeated cell cheap.
