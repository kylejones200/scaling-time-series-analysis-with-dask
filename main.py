#!/usr/bin/env python3
"""
Scaling Time Series Analysis with Dask.

Companion code for the Medium article. Demonstrates out-of-core time series
operations with Dask: resampling, rolling windows, groupby, missing data,
Parquet I/O, and a synthetic energy-grid aggregation case study.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_synthetic_series(config: dict, n_rows: int | None = None) -> pd.DataFrame:
    """Build a minute-level series large enough to benefit from partitioning."""
    data_cfg = config["data"]
    n = n_rows if n_rows is not None else int(data_cfg["n_rows"])
    rng = np.random.default_rng(data_cfg.get("seed", 42))
    ts_col = data_cfg["timestamp_column"]
    val_col = data_cfg["value_column"]
    timestamps = pd.date_range(data_cfg["start"], periods=n, freq=data_cfg["freq"])
    # Smooth trend + noise; occasional gaps for fillna demo
    t = np.arange(n, dtype=float)
    values = 50 + 0.001 * t + 5 * np.sin(t / (24 * 60)) + rng.normal(0, 2, n)
    mask = rng.random(n) < 0.002
    values = values.astype(float)
    values[mask] = np.nan
    return pd.DataFrame({ts_col: timestamps, val_col: values})


def to_dask(pdf: pd.DataFrame, npartitions: int) -> dd.DataFrame:
    return dd.from_pandas(pdf, npartitions=npartitions)


def demo_pandas_vs_dask(
    pdf: pd.DataFrame, ddf: dd.DataFrame, ts_col: str, val_col: str
) -> None:
    """Compare wall time for a daily mean on the same synthetic data."""
    logger.info("--- Pandas vs Dask (daily mean) ---")
    t0 = time.perf_counter()
    pandas_daily = pdf.set_index(ts_col)[val_col].resample("D").mean()
    _ = pandas_daily  # materialized eagerly
    pandas_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    dask_daily = ddf.set_index(ts_col)[val_col].resample("D").mean().compute()
    dask_sec = time.perf_counter() - t0
    logger.info("Partitions: %s", ddf.npartitions)
    logger.info("Pandas: %.3fs | Dask: %.3fs", pandas_sec, dask_sec)
    logger.info("Last 3 daily means (Dask):\n%s", dask_daily.tail(3))


def demo_resample(ddf: dd.DataFrame, config: dict) -> pd.Series:
    data_cfg = config["data"]
    ts_col = data_cfg["timestamp_column"]
    val_col = data_cfg["value_column"]
    freq = config["operations"]["resample_freq"]
    logger.info("--- Resample to %s ---", freq)
    daily = ddf.set_index(ts_col)[val_col].resample(freq).mean().compute()
    logger.info("Resampled length: %s", len(daily))
    return daily


def demo_rolling(ddf: dd.DataFrame, config: dict) -> pd.DataFrame:
    val_col = config["data"]["value_column"]
    window = int(config["operations"]["rolling_window"])
    logger.info("--- Rolling mean (window=%s) ---", window)
    rolled = ddf.assign(rolling_mean=ddf[val_col].rolling(window=window).mean())
    sample = rolled[[val_col, "rolling_mean"]].head(5, compute=True)
    logger.info("Sample rows:\n%s", sample)
    return sample


def demo_groupby(ddf: dd.DataFrame, config: dict) -> pd.Series:
    data_cfg = config["data"]
    ts_col = data_cfg["timestamp_column"]
    val_col = data_cfg["value_column"]
    logger.info("--- GroupBy calendar day ---")
    daily_means = ddf.groupby(ddf[ts_col].dt.date)[val_col].mean().compute()
    logger.info("Days aggregated: %s", len(daily_means))
    return daily_means


def demo_missing_data(ddf: dd.DataFrame, config: dict) -> None:
    val_col = config["data"]["value_column"]
    logger.info("--- Forward-fill missing values ---")
    missing_before = ddf[val_col].isna().sum().compute()
    filled = ddf[val_col].ffill()
    missing_after = filled.isna().sum().compute()
    logger.info("Missing before: %s | after ffill: %s", missing_before, missing_after)


def demo_parquet(ddf: dd.DataFrame, parquet_path: Path) -> dd.DataFrame:
    logger.info("--- Parquet round-trip: %s ---", parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        if parquet_path.is_dir():
            import shutil

            shutil.rmtree(parquet_path)
        else:
            parquet_path.unlink()
    ddf.to_parquet(parquet_path, engine="pyarrow")
    reloaded = dd.read_parquet(parquet_path)
    logger.info("Reloaded partitions: %s", reloaded.npartitions)
    return reloaded


def write_energy_parquet(output_dir: Path, config: dict, n_meters: int = 4) -> Path:
    """Write partitioned Parquet files mimicking many smart-meter feeds."""
    data_cfg = config["data"]
    ts_col = data_cfg["timestamp_column"]
    rng = np.random.default_rng(data_cfg.get("seed", 42))
    energy_dir = output_dir / config["output"]["parquet_subdir"]
    energy_dir.mkdir(parents=True, exist_ok=True)
    rows_per_meter = max(10_000, int(data_cfg["n_rows"]) // n_meters)
    for meter_id in range(n_meters):
        timestamps = pd.date_range(
            data_cfg["start"], periods=rows_per_meter, freq=data_cfg["freq"]
        )
        base = 2.0 + 0.3 * meter_id
        consumption = base + rng.normal(0, 0.2, rows_per_meter)
        pdf = pd.DataFrame(
            {
                ts_col: timestamps,
                "meter_id": f"meter_{meter_id}",
                "consumption": consumption,
            }
        )
        path = energy_dir / f"meter_{meter_id}.parquet"
        pdf.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Wrote %s meter Parquet files under %s", n_meters, energy_dir)
    return energy_dir


def demo_energy_case_study(energy_dir: Path, config: dict) -> pd.Series:
    """Case study: aggregate daily consumption across many meter Parquet files."""
    data_cfg = config["data"]
    ts_col = data_cfg["timestamp_column"]
    logger.info("--- Energy grid case study (%s) ---", energy_dir)
    ddf = dd.read_parquet(str(energy_dir / "*.parquet"))
    ddf[ts_col] = dd.to_datetime(ddf[ts_col])
    daily_energy = ddf.groupby(ddf[ts_col].dt.date)["consumption"].sum().compute()
    logger.info("Daily totals (last 5 days):\n%s", daily_energy.tail(5))
    return daily_energy


def run_distributed_demo(energy_dir: Path, config: dict) -> None:
    from dask.distributed import Client

    n_workers = int(config["dask"].get("n_workers", 4))
    logger.info("--- Dask Distributed (%s workers) ---", n_workers)
    with Client(n_workers=n_workers) as client:
        logger.info("Dashboard: %s", client.dashboard_link)
        data_cfg = config["data"]
        ts_col = data_cfg["timestamp_column"]
        ddf = dd.read_parquet(str(energy_dir / "*.parquet"))
        ddf[ts_col] = dd.to_datetime(ddf[ts_col])
        result = ddf.groupby(ddf[ts_col].dt.date)["consumption"].sum().compute()
        logger.info("Distributed daily sum (last 3): %s", result.tail(3).to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaling time series analysis with Dask (article companion)."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller synthetic dataset for a fast smoke run",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run the energy case study on a local Dask Distributed cluster",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated Parquet files (default: config output.data_dir)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config["output"]["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 50_000 if args.quick else None
    pdf = build_synthetic_series(config, n_rows=n_rows)
    ddf = to_dask(pdf, int(config["data"]["npartitions"]))
    data_cfg = config["data"]
    ts_col = data_cfg["timestamp_column"]
    val_col = data_cfg["value_column"]
    logger.info("Synthetic series: %s rows, %s partitions", len(pdf), ddf.npartitions)
    logger.info("Dask preview:\n%s", ddf.head(3, compute=True))
    demo_pandas_vs_dask(pdf, ddf, ts_col, val_col)
    demo_resample(ddf, config)
    demo_rolling(ddf, config)
    demo_groupby(ddf, config)
    demo_missing_data(ddf, config)
    parquet_path = output_dir / "timeseries.parquet"
    reloaded = demo_parquet(ddf, parquet_path)
    logger.info("Reloaded head:\n%s", reloaded.head(3, compute=True))
    energy_dir = write_energy_parquet(output_dir, config)
    demo_energy_case_study(energy_dir, config)
    use_distributed = args.distributed or config["dask"].get("use_distributed", False)
    if use_distributed:
        run_distributed_demo(energy_dir, config)

    logger.info("Done. Generated data under %s", output_dir.resolve())


if __name__ == "__main__":
    main()
