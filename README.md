# Scaling Time Series Analysis with Dask

Companion repository for [Scaling Time Series Analysis with Dask](https://medium.com/@kyle-t-jones/scaling-time-series-analysis-with-dask-0d786fe4ebf6).

The article export lives in `article.md`. Runnable examples are in `main.py`.

## Business context

When dealing with time series data, the scale of data can quickly exceed the capabilities of a single machine. Traditional tools like pandas work well for small to moderately sized datasets, but they struggle when the data exceeds available memory. This is where Dask comes in.

Dask is a parallel computing framework that extends pandas, NumPy, and scikit-learn to handle larger-than-memory datasets efficiently. It enables distributed processing, making it a powerful tool for time series analysis at scale.

- Why Dask is essential for time series data. - How Dask compares to pandas. - Key Dask functions for time series. - Real-world examples of Dask for large-scale time series processing.

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
uv sync
uv run python main.py --quick
```

Full-size run (default ~500k rows from `config.yaml`):

```bash
uv run python main.py
```

With a local distributed scheduler:

```bash
uv run python main.py --distributed
```

Generated Parquet files are written under `data/` (gitignored).

## Project layout

| Path | Purpose |
|------|---------|
| `main.py` | Entry point; runs all article-aligned demos |
| `config.yaml` | Row count, partitions, resample/rolling settings |
| `pyproject.toml` / `uv.lock` | Dependencies (managed with [uv](https://docs.astral.sh/uv/)) |
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

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).