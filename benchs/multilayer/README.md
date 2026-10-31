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
