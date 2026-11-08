"""Build CIC histories for classifying the type of unseen devices."""

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from .temporal import CIC_SHA256, order_cic, scan_cic, window_end_us


DEVICE_KEYS = ["device_type", "device_id"]
TEST_DEVICE_SEED = 42


def split_devices(flows):
    timestamp = pl.col("timestamp")
    cutoffs = flows.group_by(DEVICE_KEYS).agg(
        timestamp.quantile(0.8, interpolation="higher").cast(pl.Int64).alias("cutoff_us")
    )
    eligible = flows.filter(timestamp.is_not_null()).join(cutoffs, on=DEVICE_KEYS)
    cutoff = pl.col("cutoff_us")
    coverage = eligible.group_by(DEVICE_KEYS).agg(
        cutoff.first(),
        pl.len().alias("windows"),
        (window_end_us() < cutoff).sum().alias("train_windows"),
        (timestamp.cast(pl.Int64) >= cutoff).sum().alias("validation_windows"),
    ).sort(DEVICE_KEYS).collect()

    assignments = []
    for group in coverage.partition_by("device_type", maintain_order=True):
        devices = group.to_dicts()
        fit_ids = {
            row["device_id"] for row in devices
            if row["train_windows"] and row["validation_windows"]
        }
        candidates = [row["device_id"] for row in devices if fit_ids - {row["device_id"]}]
        if not candidates:
            continue
        held_out = min(
            candidates,
            key=lambda device: hashlib.sha256(f"{TEST_DEVICE_SEED}:{device}".encode()).digest(),
        )
        for device in devices:
            name = device["device_id"]
            if name == held_out or name in fit_ids:
                assignments.append({
                    **device,
                    "role": "test" if name == held_out else "train_validation",
                })

    if not assignments:
        raise ValueError("No CIC device types have separate training and test devices")
    roles = pl.DataFrame(assignments).lazy().select(*DEVICE_KEYS, "role")
    assigned = (
        eligible.join(roles, on=DEVICE_KEYS)
        .with_columns(
            pl.when(pl.col("role") == "test")
            .then(pl.lit("test"))
            .when(timestamp.cast(pl.Int64) >= cutoff)
            .then(pl.lit("validation"))
            .when(window_end_us() < cutoff)
            .then(pl.lit("train"))
            .alias("split")
        )
        .filter(pl.col("split").is_not_null())
        .drop("cutoff_us", "role")
    )
    return order_cic(assigned), assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if len({path.resolve() for path in (args.parquet, args.output, args.manifest)}) != 3:
        parser.error("Input and output paths must be distinct")
    flows, devices = split_devices(scan_cic(args.parquet))
    summary = flows.group_by("split", "device_type").agg(
        pl.len().alias("windows"),
        pl.col("device_id").n_unique().alias("devices"),
    ).sort("split", "device_type").collect()
    flows.sink_parquet(args.output)
    manifest = {
        "parquet_sha256": CIC_SHA256,
        "test_assignment_seed": TEST_DEVICE_SEED,
        "devices": devices,
        "counts": summary.to_dicts(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    with pl.Config(tbl_rows=-1):
        print(summary)


if __name__ == "__main__":
    main()
