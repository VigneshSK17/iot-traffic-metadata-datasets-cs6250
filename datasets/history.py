"""Session and history annotations shared by the dataset pipelines."""

import polars as pl


TIMELINE = ["split", "device_id"]
HISTORY_LENGTH = 20


def order_flows(
    flows, gap_seconds, *, timestamp_filter, capture_column=None
):
    """Add session IDs, elapsed seconds, and available context length."""
    if gap_seconds <= 0:
        raise ValueError("gap_seconds must be positive")

    capture_keys = [capture_column] if capture_column else []
    gap = pl.col("_gap_us")
    starts_session = gap.is_null() | (gap > gap_seconds * 1_000_000)
    return (
        flows.filter(timestamp_filter)
        .sort(*TIMELINE, *capture_keys, "timestamp", "flow_id")
        .with_columns(
            pl.col("timestamp").diff().over([*TIMELINE, *capture_keys]).dt.total_microseconds().alias("_gap_us")
        )
        .with_columns(
            starts_session.cast(pl.UInt32).cum_sum().over(TIMELINE).alias("session_id"),
            pl.when(starts_session)
            .then(0.0)
            .otherwise(gap / 1_000_000)
            .alias("gap_seconds"),
        )
        .with_columns(
            pl.col("flow_id")
            .cum_count()
            .over([*TIMELINE, "session_id"])
            .clip(upper_bound=HISTORY_LENGTH)
            .alias("context_length")
        )
        .drop("_gap_us")
    )
