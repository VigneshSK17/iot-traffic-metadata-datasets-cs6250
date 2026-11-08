"""Order split-tagged UNSW flows and annotate causal history within each session."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from datasets.history import HISTORY_LENGTH, order_flows


CAPTURE_START = datetime(2016, 1, 1, tzinfo=timezone.utc)
CAPTURE_END = datetime(2018, 1, 1, tzinfo=timezone.utc)
SESSION_GAP_SECONDS = 600


def capture_timestamp():
    timestamp = pl.col("timestamp")
    return timestamp.is_between(CAPTURE_START, CAPTURE_END, closed="left").fill_null(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_parquet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.split_parquet.resolve():
        parser.error("The output must differ from the input Parquet")

    flows = pl.scan_parquet(args.split_parquet)
    ordered = order_flows(flows, SESSION_GAP_SECONDS, timestamp_filter=capture_timestamp())
    ordered.sink_parquet(args.output)
    print(flows.group_by("split").agg((~capture_timestamp()).sum().alias("excluded")).collect())
    print(
        pl.scan_parquet(args.output)
        .group_by("split")
        .agg(pl.len().alias("flows"), (pl.col("context_length") == HISTORY_LENGTH).mean().alias("full_context"))
        .sort("split")
        .collect()
    )


if __name__ == "__main__":
    main()
