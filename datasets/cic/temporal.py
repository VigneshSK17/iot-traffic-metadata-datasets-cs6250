"""Prepare CIC packet-window histories for chronological device identification."""

import argparse
import hashlib
from pathlib import Path

import polars as pl

from datasets.history import HISTORY_LENGTH, order_flows


CIC_SHA256 = "fe42c53e4090a60aca6217e45b152fe98a3dcaf397eaded56ea54762beea6524"
CIC_SESSION_GAP_SECONDS = 40


def scan_cic(path):
    with path.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != CIC_SHA256:
            raise ValueError("CIC Parquet SHA-256 differs from the pinned source")
    return pl.scan_parquet(path, row_index_name="flow_id")


def window_end_us():
    return (pl.col("epoch_timestamp") * 1_000_000).round().cast(pl.Int64)


def order_cic(flows):
    return order_flows(
        flows, CIC_SESSION_GAP_SECONDS,
        capture_column="source_file", timestamp_filter=pl.col("timestamp").is_not_null(),
    )


def prepare_cic(flows):
    timestamp = pl.col("timestamp")
    cutoffs = flows.group_by("device_id").agg(
        timestamp.quantile(0.6, interpolation="higher").alias("validation_start"),
        timestamp.quantile(0.8, interpolation="higher").alias("test_start"),
    )
    assigned = (
        flows.join(cutoffs, on="device_id")
        .with_columns(
            pl.when(timestamp < pl.col("validation_start"))
            .then(pl.lit("train"))
            .when(timestamp < pl.col("test_start"))
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
        .filter(
            ((pl.col("split") == "train") & (window_end_us() < pl.col("validation_start").cast(pl.Int64)))
            | ((pl.col("split") == "validation") & (window_end_us() < pl.col("test_start").cast(pl.Int64)))
            | (pl.col("split") == "test")
        )
        .drop("validation_start", "test_start")
    )
    return order_cic(assigned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.parquet.resolve() == args.output.resolve():
        parser.error("The output must differ from the input Parquet")
    flows = scan_cic(args.parquet)
    ordered = prepare_cic(flows)
    summary = ordered.group_by("split").agg(
        pl.len().alias("rows"),
        pl.col("device_id").n_unique().alias("devices"),
        (pl.col("context_length") == HISTORY_LENGTH).mean().alias("full_history_fraction"),
    ).sort("split").collect()
    if summary.height != 3 or not (summary["devices"] == 30).all():
        parser.error("Expected all 30 CIC devices in every split")
    ordered.sink_parquet(args.output)
    print(summary)


if __name__ == "__main__":
    main()
