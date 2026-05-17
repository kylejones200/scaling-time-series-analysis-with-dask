# Scaling Time Series Analysis with Dask

Companion repository for [Scaling Time Series Analysis with Dask](https://medium.com/@kyle-t-jones/scaling-time-series-analysis-with-dask-0d786fe4ebf6).

The article export lives in `article.md`. Runnable examples are in `main.py`.

## What this demonstrates

- Converting a pandas time series to a partitioned Dask DataFrame
- Resampling, rolling windows, and calendar-day groupby at scale
- Forward-filling missing values on partitioned data
- Parquet write/read for out-of-core storage
- A synthetic “smart meter” case study: many Parquet files aggregated to daily consumption
- Optional local `dask.distributed` cluster for the same aggregation

All demos use **synthetic** minute-level data so nothing needs to be downloaded.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --quick
```

Full-size run (default ~500k rows from `config.yaml`):

```bash
python main.py
```

With a local distributed scheduler:

```bash
python main.py --distributed
```

Generated Parquet files are written under `data/` (gitignored).

## Project layout

| Path | Purpose |
|------|---------|
| `main.py` | Entry point; runs all article-aligned demos |
| `config.yaml` | Row count, partitions, resample/rolling settings |
| `requirements.txt` | Python dependencies |
| `article.md` | Medium article export |
| `data/` | Created at runtime (Parquet outputs) |

## Configuration

Edit `config.yaml` to change:

- `data.n_rows` — synthetic series length
- `data.npartitions` — Dask partition count
- `operations.resample_freq` / `rolling_window` — time ops
- `dask.n_workers` — workers when using `--distributed`

## Notes

- For very large production workloads, point `main.py` at your own CSV or Parquet paths using the same Dask patterns shown in `demo_energy_case_study` and `demo_parquet`.
- Optional ML (`dask-ml`, XGBoost) from the article is not included here to keep installs light; add those packages if you extend the repo for forecasting.
