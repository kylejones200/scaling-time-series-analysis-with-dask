# Description: Short example for Scaling Time Series Analysis with Dask.


import logging

import dask.dataframe as dd
import dask_ml.xgboost as dxgb
import pandas as pd
from dask.distributed import Client


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


    # pip install dask[complete]


    # Load a large time series dataset
    df = dd.read_csv("large_timeseries_data.csv", parse_dates=["timestamp"])

    # Display the first few rows
    logger.info(df.head())

    # Convert pandas DataFrame to Dask DataFrame

    pdf = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=10**7, freq="T"),
            "value": range(10**7),
        }
    )

    # Convert to Dask DataFrame
    ddf = dd.from_pandas(pdf, npartitions=10)

    logger.info(ddf.npartitions)  # Shows how many partitions the data has

    # Resample to daily frequency
    df_resampled = df.set_index("timestamp").resample("D").mean()

    # Compute results
    df_resampled.compute()

    # Compute a rolling mean with a 7-day window
    df["rolling_mean"] = df["value"].rolling(window=7).mean()

    # Compute results
    df.compute()

    # Group data by day and calculate mean
    daily_means = df.groupby(df.timestamp.dt.date).value.mean()

    # Compute results
    daily_means.compute()

    # Fill missing values with the previous value
    df_filled = df.ffill()

    # Compute results
    df_filled.compute()

    # Save as a partitioned Parquet file
    df.to_parquet("large_timeseries.parquet", engine="pyarrow")

    # Load back the Parquet file
    df = dd.read_parquet("large_timeseries.parquet")


    # Prepare data for training
    X = df[["value"]]
    y = df["value"].shift(-1)  # Target variable

    # Train a Dask-based XGBoost model
    model = dxgb.train({"objective": "reg:squarederror"}, X, y)


    # Start a distributed cluster
    client = Client(n_workers=4)
    logger.info(client)

    # Load dataset from multiple Parquet files
    df = dd.read_parquet("energy_data/*.parquet")

    # Convert timestamp column to datetime
    df["timestamp"] = dd.to_datetime(df["timestamp"])

    # Aggregate daily energy consumption
    daily_energy = df.groupby(df.timestamp.dt.date)["consumption"].sum()

    # Compute the result
    result = daily_energy.compute()
    logger.info(result)


if __name__ == "__main__":
    main()
