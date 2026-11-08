"""Build or apply the shared per-flow UNSW device-type split."""

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl


KEYS = ["device_type", "device_id"]
SEED = 42


def inspect_flows(path):
    flows = pl.scan_parquet(path, row_index_name="flow_id")
    timestamps = flows.collect_schema()["timestamp"]
    if not isinstance(timestamps, pl.Datetime):
        raise ValueError(f"Expected datetime timestamps, found {timestamps}")
    flows = flows.with_columns(pl.col("timestamp").cast(pl.Int64).alias("_time"))

    time = pl.col("_time")
    coverage = flows.group_by(KEYS).agg(
        pl.len().alias("rows"),
        time.count().alias("usable"),
        time.drop_nulls().n_unique().alias("distinct_timestamps"),
        time.min().alias("first_timestamp"),
        time.max().alias("last_timestamp"),
        time.quantile(0.8, interpolation="higher").cast(pl.Int64).alias("cutoff"),
    )
    coverage = (
        flows.join(coverage, on=KEYS)
        .group_by(KEYS)
        .agg(
            pl.col(
                "rows", "usable", "distinct_timestamps",
                "first_timestamp", "last_timestamp", "cutoff",
            ).first(),
            (time < pl.col("cutoff")).sum().alias("train_rows"),
            (time >= pl.col("cutoff")).sum().alias("validation_rows"),
        )
        .sort(KEYS)
        .collect()
    )
    labels = flows.select(KEYS).unique().collect()
    if labels.null_count().row(0) != (0, 0) or labels["device_id"].n_unique() != labels.height:
        raise ValueError("Expected exactly one non-null type per non-null device ID")
    return flows, coverage, timestamps.time_unit


def make_manifest(coverage, checksum, time_unit):
    devices = []
    for group in coverage.partition_by("device_type", maintain_order=True):
        rows = group.to_dicts()
        usable = [row for row in rows if row["usable"]]
        can_train = [
            row for row in usable if row["train_rows"] and row["validation_rows"]
        ]
        candidates = [
            row["device_id"]
            for row in usable
            if any(other["device_id"] != row["device_id"] for other in can_train)
        ]
        test_id = min(
            candidates,
            key=lambda device: hashlib.sha256(f"{SEED}:{device}".encode()).digest(),
            default=None,
        )
        train_ids = {row["device_id"] for row in can_train}
        for row in rows:
            device_id = row["device_id"]
            if device_id == test_id:
                role = "test"
            elif test_id is not None and device_id in train_ids:
                role = "train_validation"
            else:
                role = "excluded"
            devices.append({**row, "role": role})
    return {
        "parquet_sha256": checksum,
        "timestamp_unit": time_unit,
        "test_assignment_seed": SEED,
        "devices": devices,
    }


def assign_flows(flows, manifest):
    roles = pl.DataFrame(manifest["devices"]).select(*KEYS, "role", "cutoff")
    return (
        flows.join(roles.lazy(), on=KEYS)
        .filter(pl.col("_time").is_not_null() & (pl.col("role") != "excluded"))
        .with_columns(
            pl.when(pl.col("role") == "test")
            .then(pl.lit("test"))
            .when(pl.col("_time") < pl.col("cutoff"))
            .then(pl.lit("train"))
            .otherwise(pl.lit("validation"))
            .alias("split")
        )
        .drop("_time", "role", "cutoff")
        .sort("flow_id")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("held_out_split.json"))
    parser.add_argument("--generate", action="store_true", help="Generate the manifest")
    parser.add_argument("--output", type=Path, help="Write split-tagged flows to Parquet")
    args = parser.parse_args()
    with args.parquet.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()

    if args.generate:
        flows, coverage, time_unit = inspect_flows(args.parquet)
        manifest = make_manifest(coverage, checksum, time_unit)
        if not any(row["role"] == "test" for row in manifest["devices"]):
            parser.error("No types have usable devices for a held-out-device split")
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        manifest = json.loads(args.manifest.read_text())
        if checksum != manifest["parquet_sha256"]:
            parser.error("Input Parquet SHA-256 differs from the split manifest")
        flows = pl.scan_parquet(args.parquet, row_index_name="flow_id").with_columns(
            pl.col("timestamp").cast(pl.Int64).alias("_time")
        )

    assigned = assign_flows(flows, manifest)
    if args.output:
        if args.output.resolve() == args.parquet.resolve():
            parser.error("The output must differ from the input Parquet")
        assigned.sink_parquet(args.output)
    summary = assigned.group_by("split", "device_type").agg(
        pl.len().alias("flows"),
        pl.col("device_id").n_unique().alias("devices"),
    ).sort("split", "device_type").collect()
    with pl.Config(tbl_rows=-1):
        print(summary)


if __name__ == "__main__":
    main()
