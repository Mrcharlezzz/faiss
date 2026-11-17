# Analysis

Turns the campaign outputs into Parquet tables and reproduces the figures and numbers of the results.

| File | Does |
|---|---|
| `scripts/convert_run3.py` | Converts a campaign's `outputs/<name>/` into tables under `data/` (uses `hnsw_tables.py` and the empty `runs_template.parquet` for the column layout) |
| `scripts/run3_data.py`, `scripts/run3_plots.py` | Loading, matched-recall comparison, seed intervals and plots |
| `run3_extended.ipynb` | Deep1M, GIST1M and clustered 200k (d = 16, 32) |
| `run4_glove25.ipynb` | GloVe-25 |
| `scripts/export_results.py` | Writes the result numbers to a `results.js` |

Run from this directory, after the campaigns and `ground_truth.py` have filled `../outputs/` and `../cache/`:

```bash
pip install -r requirements.txt
python scripts/convert_run3.py ../outputs/rerun data/run_3 --cache ../cache
python scripts/convert_run3.py ../outputs/glove25 data/run_4 --cache ../cache --source run_4 --run-id-offset 4000
jupyter lab run3_extended.ipynb run4_glove25.ipynb
```
