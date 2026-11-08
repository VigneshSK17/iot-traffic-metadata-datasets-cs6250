"""Create a shared chronological split for UNSW known-device classification."""

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from .temporal import capture_timestamp


def split_flows(flows):
    eligible = flows.filter(capture_timestamp())
    time = pl.col("timestamp")
    cutoffs = eligible.group_by("device_id").agg(
        time.quantile(0.6, interpolation="higher").alias("validation_start"),
        time.quantile(0.8, interpolation="higher").alias("test_start"),
    )
    assigned = (
        eligible.join(cutoffs, on="device_id")
        .with_columns(
            pl.when(time < pl.col("validation_start"))
            .then(pl.lit("train"))
            .when(time < pl.col("test_start"))
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
        .drop("validation_start", "test_start")
        .sort("flow_id")
    )
    return assigned, cutoffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if len({p.resolve() for p in (args.parquet, args.output, args.manifest)}) != 3:
        parser.error("Input and output paths must be distinct")

    with args.parquet.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    flows = pl.scan_parquet(args.parquet, row_index_name="flow_id")
    assigned, cutoffs = split_flows(flows)
    counts = assigned.group_by("device_id", "device_type", "split").len().collect()
    counts = counts.pivot(on="split", index=["device_id", "device_type"], values="len")
    if not all(split in counts.columns for split in ("train", "validation", "test")):
        parser.error("All three splits must be present")
    if counts.null_count().row(0) != (0,) * counts.width:
        parser.error("Every device needs flows in all three chronological splits")
    if counts["device_id"].n_unique() != counts.height:
        parser.error("Expected a single device type per device")
    source_count, excluded = flows.select(
        pl.len(), (~capture_timestamp()).sum().alias("excluded")
    ).collect().row(0)
    if counts.height != flows.select(pl.col("device_id").n_unique()).collect().item():
        parser.error("At least one device has no eligible flows")
    devices = (
        counts.join(cutoffs.collect(), on="device_id")
        .select("device_id", "device_type", "train", "validation", "test", "validation_start", "test_start")
        .sort("device_id")
    )
    assigned.sink_parquet(args.output)
    manifest = {
        "parquet_sha256": checksum,
        "source_flows": source_count,
        "excluded_timestamps": excluded,
        "devices": devices.to_dicts(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"{devices.height} devices, {excluded} out-of-era or missing timestamps excluded")
    print(devices.select("train", "validation", "test").sum())


if __name__ == "__main__":
    main()
