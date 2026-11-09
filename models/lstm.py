"""Train ordered and shuffled-history LSTMs on UNSW flows or CIC windows."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from datasets.history import HISTORY_LENGTH


NUMERIC = ("total_packets", "total_bytes", "avg_packet_size", "gap_seconds")
PROTOCOLS = ("TCP", "UDP", "ICMP", "OTHER")
PORTS = ("src_port", "dst_port")
SEED = 42


@dataclass
class Flows:
    features: np.ndarray
    labels: np.ndarray
    lengths: np.ndarray
    flow_ids: np.ndarray


def load_split(path, split, target, *, numeric_columns=NUMERIC, eligible_labels=None):
    rows = pl.scan_parquet(path).filter(pl.col("split") == split)
    if eligible_labels is not None:
        rows = rows.filter(pl.col(target).is_in(eligible_labels))
    return (
        rows.sort("device_id", "session_id", "timestamp", "flow_id")
        .select("flow_id", pl.col(target).alias("label"), "context_length", *numeric_columns, *PORTS, "transport_protocol")
        .collect()
    )


def log_numeric(frame, numeric_columns=NUMERIC):
    if not numeric_columns:
        return np.empty((frame.height, 0), dtype=np.float32)
    raw = frame.select(numeric_columns).to_numpy().astype(np.float32)
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Expected finite, nonnegative traffic measurements")
    return np.log1p(raw)


def encode(frame, numeric, mean, scale, classes, *, unordered_ports=False):
    columns = [(numeric - mean) / scale]
    port_buckets = []
    for name in PORTS:
        port = frame[name].fill_null(-1).to_numpy()
        port_buckets.append(np.column_stack([
            port < 0,
            (port >= 0) & (port < 1024),
            (port >= 1024) & (port < 49152),
            port >= 49152,
        ]))
    if unordered_ports:
        columns.append(np.sum(port_buckets, axis=0, dtype=np.float32))
    else:
        columns.extend(port_buckets)
    protocol = frame["transport_protocol"].fill_null("OTHER").to_numpy()
    protocol = np.where(np.isin(protocol, PROTOCOLS), protocol, "OTHER")
    columns.append(np.column_stack([protocol == value for value in PROTOCOLS]))
    lengths = frame["context_length"].to_numpy().astype(np.int64)
    if (lengths < 1).any() or (lengths > HISTORY_LENGTH).any():
        raise ValueError("Invalid context length")
    if (lengths > np.arange(1, len(lengths) + 1)).any():
        raise ValueError("Context precedes the start of the split")
    labels = np.searchsorted(classes, frame["label"].to_numpy())
    if not np.array_equal(np.take(classes, labels, mode="clip"), frame["label"].to_numpy()):
        raise ValueError("Found a label absent from training")
    return Flows(
        np.concatenate(columns, axis=1, dtype=np.float32),
        labels,
        lengths,
        frame["flow_id"].to_numpy(),
    )


def sample_by_class(labels, n_classes, limit, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(n_classes):
        candidates = np.flatnonzero(labels == label)
        if limit and len(candidates) > limit:
            candidates = rng.choice(candidates, size=limit, replace=False)
        selected.append(candidates)
    return np.sort(np.concatenate(selected))


def make_batch(flows, endpoints, shuffle_rng, device):
    lengths = flows.lengths[endpoints]
    offsets = np.arange(HISTORY_LENGTH)
    indices = endpoints[:, None] + 1 - lengths[:, None] + offsets
    values = flows.features[np.minimum(indices, endpoints[:, None])]
    values[offsets >= lengths[:, None]] = 0
    if shuffle_rng is not None:
        keys = shuffle_rng.random(values.shape[:2])
        # Fixed order for the target and padding. Randomize only earlier flows.
        fixed = offsets >= lengths[:, None] - 1
        keys = np.where(fixed, 1 + offsets, keys)
        values = np.take_along_axis(values, np.argsort(keys, axis=1)[:, :, None], axis=1)
    return (
        torch.from_numpy(values).to(device),
        torch.from_numpy(lengths).to(device),
        torch.from_numpy(flows.labels[endpoints]).to(device),
    )


class FlowLSTM(nn.Module):
    def __init__(self, feature_count, class_count):
        super().__init__()
        self.lstm = nn.LSTM(feature_count, 64, batch_first=True)
        self.head = nn.Linear(64, class_count)

    def forward(self, features, lengths):
        states, _ = self.lstm(features)
        # The causal state at the target precedes all right padding.
        target = states[torch.arange(len(lengths), device=lengths.device), lengths - 1]
        return self.head(target)


def scores(labels, predictions, n_classes):
    matrix = np.bincount(
        labels * n_classes + predictions, minlength=n_classes**2
    ).reshape(n_classes, n_classes)
    correct = matrix.diagonal()
    denom = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * correct, denom, out=np.zeros(n_classes), where=denom > 0)
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(correct.sum() / matrix.sum()),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def predict(model, flows, ids, shuffled, batch_size, device, seed):
    model.eval()
    shuffle_rng = np.random.default_rng(seed) if shuffled else None
    predictions = []
    for start in range(0, len(ids), batch_size):
        features, lengths, _ = make_batch(
            flows, ids[start:start + batch_size], shuffle_rng, device
        )
        predictions.append(model(features, lengths).argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def train_model(train, validation, train_ids, val_ids, n_classes, mode, args, device, extra_ids=None, *, model_factory=FlowLSTM):
    torch.manual_seed(args.seed)
    model = model_factory(train.features.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    batch_rng = np.random.default_rng(args.seed)
    shuffle_rng = np.random.default_rng(args.seed + 10) if mode == "shuffled" else None
    extra_rng = np.random.default_rng(args.seed + 40)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = batch_rng.permutation(train_ids)
        loss_sum = 0.0
        starts = range(0, len(order), args.batch_size)
        extra_batches = (
            np.array_split(extra_rng.permutation(extra_ids), len(starts))
            if extra_ids is not None else [None] * len(starts)
        )
        endpoint_count = len(order) + (len(extra_ids) if extra_ids is not None else 0)
        for start, extra in zip(starts, extra_batches):
            endpoints = order[start:start + args.batch_size]
            if extra is not None:
                endpoints = np.concatenate((endpoints, extra))
            features, lengths, labels = make_batch(train, endpoints, shuffle_rng, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features, lengths), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)

        predicted = predict(
            model, validation, val_ids, mode == "shuffled", args.batch_size, device, args.seed + 20
        )
        metrics = scores(validation.labels[val_ids], predicted, n_classes)
        f1 = metrics["macro_f1"]
        history.append({
            "epoch": epoch, "loss": loss_sum / endpoint_count,
            "validation_macro_f1": f1, "validation_accuracy": metrics["accuracy"],
        })
        print(f"{mode} epoch {epoch}: loss {loss_sum / endpoint_count:.4f}, val macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            torch.save(model.state_dict(), args.output / f"{mode}.pt")
        if args.cic_reference is None and epoch - best_epoch >= 2:
            break
    return {"best_epoch": best_epoch, "validation_macro_f1": best_f1, "epochs": history}


def main(*, model_factory=FlowLSTM, model_metadata=None, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("ordered_parquet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cic-reference", type=Path, help="CIC ordered file for shared-label comparison")
    parser.add_argument("--augment", action="store_true", help="Include the reference CIC training rows")
    parser.add_argument("--ordered-only", action="store_true")
    parser.add_argument("--target", choices=("device_type", "device_id"), default="device_type")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-per-class", "--train-per-type", type=int, default=20000, help="0 uses all flows")
    parser.add_argument("--validation-per-class", "--validation-per-type", type=int, default=4000, help="0 uses all flows")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch size, and threads must be positive")
    if min(args.train_per_class, args.validation_per_class, args.seed) < 0:
        parser.error("sampling caps and seed must be nonnegative")
    if args.augment and args.cic_reference is None:
        parser.error("--augment requires --cic-reference")
    if args.cic_reference and args.target != "device_type":
        parser.error("The pooled comparison requires --target device_type")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    with args.ordered_parquet.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()

    shared = args.cic_reference is not None
    numeric_columns = () if shared else NUMERIC
    train_frame = load_split(args.ordered_parquet, "train", args.target, numeric_columns=numeric_columns)
    classes = sorted(train_frame["label"].unique().to_list())
    cic_frame = None
    if shared:
        cic_frame = load_split(args.cic_reference, "train", args.target, numeric_columns=numeric_columns)
        classes = sorted(set(classes).intersection(cic_frame["label"].unique().to_list()))
        if not classes:
            parser.error("No shared training labels")
        train_frame = train_frame.filter(pl.col("label").is_in(classes))
        cic_frame = cic_frame.filter(pl.col("label").is_in(classes))
    validation_frame = load_split(
        args.ordered_parquet, "validation", args.target,
        numeric_columns=numeric_columns, eligible_labels=classes if shared else None,
    )
    numeric = log_numeric(train_frame, numeric_columns)
    mean = numeric.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = numeric.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0] = 1

    def encode_split(frame):
        return encode(
            frame, log_numeric(frame, numeric_columns), mean, scale, classes,
            unordered_ports=shared,
        )

    train = encode(train_frame, numeric, mean, scale, classes, unordered_ports=shared)
    validation = encode_split(validation_frame)
    del train_frame, validation_frame, numeric

    train_ids = sample_by_class(train.labels, len(classes), args.train_per_class, args.seed)
    val_ids = sample_by_class(validation.labels, len(classes), args.validation_per_class, args.seed + 1)
    extra_ids = None
    if args.augment:
        cic = encode_split(cic_frame)
        extra_ids = np.arange(len(train.labels), len(train.labels) + len(cic.labels))
        train = Flows(*(
            np.concatenate((getattr(train, field), getattr(cic, field)))
            for field in ("features", "labels", "lengths", "flow_ids")
        ))
    report = {
        "ordered_parquet_sha256": checksum, "classes": classes, "target": args.target, "seed": args.seed,
        "device": args.device, "threads": args.threads,
        "epochs_limit": args.epochs, "batch_size": args.batch_size,
        "optimizer": "Adam", "learning_rate": 1e-3, "early_stopping_patience": None if shared else 2,
        "versions": {"numpy": np.__version__, "polars": pl.__version__, "torch": torch.__version__},
        "history_length_including_target": HISTORY_LENGTH,
        **({"hidden_size": 64} if model_metadata is None else model_metadata),
        "numeric_features": numeric_columns, "numeric_log_mean": mean.tolist(),
        "numeric_log_scale": scale.tolist(), "port_features": PORTS, "unordered_port_bucket_counts": shared, "protocols": PROTOCOLS,
        "train_per_class_limit": args.train_per_class,
        "validation_per_class_limit": args.validation_per_class,
        "train_endpoint_counts": np.bincount(train.labels[train_ids]).tolist(),
        "validation_endpoint_counts": np.bincount(validation.labels[val_ids]).tolist(),
        "models": {},
    }
    if shared:
        with args.cic_reference.open("rb") as reference:
            report["cic_reference_sha256"] = hashlib.file_digest(reference, "sha256").hexdigest()
        report["cic_training_endpoint_counts"] = (
            np.bincount(train.labels[extra_ids], minlength=len(classes)).tolist()
            if extra_ids is not None else [0] * len(classes)
        )
    modes = ("ordered",) if args.ordered_only else ("ordered", "shuffled")
    print(f"Training on {len(train_ids):,} endpoints, validating on {len(val_ids):,}", flush=True)
    for mode in modes:
        report["models"][mode] = train_model(
            train, validation, train_ids, val_ids, len(classes), mode, args, device, extra_ids,
            model_factory=model_factory,
        )
        (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n")

    del train, validation
    # Load test only after selecting the validation checkpoints.
    test_frame = load_split(
        args.ordered_parquet, "test", args.target,
        numeric_columns=numeric_columns, eligible_labels=classes if shared else None,
    )
    test = encode_split(test_frame)
    del test_frame
    test_ids = np.arange(len(test.labels))
    results = {"flow_id": test.flow_ids, "actual": np.asarray(classes)[test.labels]}
    for mode in modes:
        model = model_factory(test.features.shape[1], len(classes)).to(device)
        model.load_state_dict(torch.load(args.output / f"{mode}.pt", map_location=device, weights_only=True))
        predicted = predict(
            model, test, test_ids, mode == "shuffled", args.batch_size, device, args.seed + 30
        )
        metrics = scores(test.labels, predicted, len(classes))
        report["models"][mode]["test"] = metrics
        results[mode] = np.asarray(classes)[predicted]
        print(f"{mode} test: macro-F1 {metrics['macro_f1']:.4f}, accuracy {metrics['accuracy']:.4f}", flush=True)
    report["test_endpoint_counts"] = np.bincount(test.labels).tolist()
    pl.DataFrame(results).sort("flow_id").write_parquet(args.output / "test_predictions.parquet")
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
