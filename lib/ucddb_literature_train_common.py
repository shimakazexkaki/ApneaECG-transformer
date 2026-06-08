import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import ucddb_literature_features as litfeat


RANDOM_STATE = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class CNNBiGRUAttention(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=96, dropout=0.35):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=96,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pool = AttentionPool(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 2),
        )

    def forward(self, features, raw=None):
        x = features.permute(0, 2, 1)
        x = self.cnn(x).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.pool(x)
        return self.classifier(x)


class RRITransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=96, nhead=4, layers=3, dropout=0.25):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, 900, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool = AttentionPool(d_model)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, features, raw=None):
        x = self.proj(features)
        x = x + self.pos[:, : x.size(1), :]
        x = self.encoder(x)
        return self.classifier(self.pool(x))


class CNNTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=96, nhead=4, layers=3, dropout=0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, d_model, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.pos = nn.Parameter(torch.randn(1, 225, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool = AttentionPool(d_model)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, features, raw=None):
        x = self.cnn(features.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + self.pos[:, : x.size(1), :]
        x = self.encoder(x)
        return self.classifier(self.pool(x))


class HybridRawRRITransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=96, nhead=4, layers=3, dropout=0.25, token_count=90):
        super().__init__()
        self.token_count = token_count
        self.rri_proj = nn.Linear(input_dim, d_model)
        self.rri_pool = nn.AdaptiveAvgPool1d(token_count)
        self.raw_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, d_model, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.raw_pool = nn.AdaptiveAvgPool1d(token_count)
        self.type_embedding = nn.Parameter(torch.randn(1, token_count * 2, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, token_count * 2, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool = AttentionPool(d_model)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, features, raw=None):
        if raw is None:
            raise ValueError("HybridRawRRITransformer requires raw ECG input.")
        rri = self.rri_proj(features).permute(0, 2, 1)
        rri = self.rri_pool(rri).permute(0, 2, 1)
        raw_tokens = self.raw_pool(self.raw_cnn(raw.unsqueeze(1))).permute(0, 2, 1)
        x = torch.cat([rri, raw_tokens], dim=1)
        x = x + self.pos[:, : x.size(1), :] + self.type_embedding[:, : x.size(1), :]
        x = self.encoder(x)
        return self.classifier(self.pool(x))


def make_model(args, model_kind):
    if model_kind == "bigru":
        return CNNBiGRUAttention(hidden_dim=args.hidden_dim, dropout=args.dropout)
    if model_kind == "rri_transformer":
        return RRITransformer(d_model=args.d_model, nhead=args.nhead, layers=args.layers, dropout=args.dropout)
    if model_kind == "cnn_transformer":
        return CNNTransformer(d_model=args.d_model, nhead=args.nhead, layers=args.layers, dropout=args.dropout)
    if model_kind == "hybrid_transformer":
        return HybridRawRRITransformer(
            d_model=args.d_model,
            nhead=args.nhead,
            layers=args.layers,
            dropout=args.dropout,
            token_count=args.hybrid_tokens,
        )
    raise ValueError(f"Unknown model kind: {model_kind}")


def records_to_arrays(records):
    features = []
    labels = []
    record_ids = []
    channels = []
    minutes = []
    signals = {}
    for record in records:
        key = (record.record_id, int(record.channel))
        signals[key] = record.signal
        features.append(record.features)
        labels.append(record.labels)
        record_ids.extend([record.record_id] * len(record.labels))
        channels.extend([int(record.channel)] * len(record.labels))
        minutes.extend(record.minute_indices.tolist())
    return {
        "features": np.concatenate(features, axis=0).astype(np.float32),
        "labels": np.concatenate(labels, axis=0).astype(np.int64),
        "record_ids": np.asarray(record_ids),
        "channels": np.asarray(channels, dtype=np.int16),
        "minutes": np.asarray(minutes, dtype=np.int32),
        "signals": signals,
    }


class LiteratureDataset(Dataset):
    def __init__(self, arrays, indices=None, include_raw=False, context_minutes=5, raw_length=3000):
        self.arrays = arrays
        self.indices = np.arange(len(arrays["labels"]), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self.include_raw = include_raw
        self.context_minutes = context_minutes
        self.raw_length = raw_length

    def __len__(self):
        return len(self.indices)

    def _raw_context(self, idx):
        record_id = str(self.arrays["record_ids"][idx])
        channel = int(self.arrays["channels"][idx])
        minute = int(self.arrays["minutes"][idx])
        signal = self.arrays["signals"][(record_id, channel)]
        half = self.context_minutes // 2
        start = (minute - half) * 60 * litfeat.FS
        end = start + self.context_minutes * 60 * litfeat.FS
        raw = signal[start:end]
        if len(raw) == 0:
            raw = np.zeros((self.raw_length,), dtype=np.float32)
        else:
            xp = np.linspace(0.0, 1.0, len(raw), endpoint=False)
            xq = np.linspace(0.0, 1.0, self.raw_length, endpoint=False)
            raw = np.interp(xq, xp, raw).astype(np.float32)
            raw = (raw - raw.mean()) / (raw.std() + 1e-8)
        return raw.astype(np.float32, copy=False)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        features = torch.from_numpy(self.arrays["features"][idx])
        label = torch.tensor(int(self.arrays["labels"][idx]), dtype=torch.long)
        if self.include_raw:
            raw = torch.from_numpy(self._raw_context(idx))
        else:
            raw = torch.empty(0, dtype=torch.float32)
        return {
            "features": features,
            "raw": raw,
            "label": label,
            "record_id": str(self.arrays["record_ids"][idx]),
            "channel": int(self.arrays["channels"][idx]),
            "minute": int(self.arrays["minutes"][idx]),
        }


def evaluate_from_scores(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    pred = (scores >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = recall_score(y_true, pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.0
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }


def best_threshold(y_true, scores, min_specificity=0.0, objective="bacc"):
    """objective="bacc" 以 balanced accuracy 為主、F1 次之挑閾值；
    objective="f1" 反過來以 F1 為主（用來找「更平衡、F1 最佳」的操作點）。"""
    def _key(metrics):
        if objective == "f1":
            return (metrics["f1"], metrics["balanced_accuracy"])
        return (metrics["balanced_accuracy"], metrics["f1"])

    best_any = None
    best_constrained = None
    for threshold in np.linspace(0.02, 0.98, 193):
        metrics = evaluate_from_scores(y_true, scores, threshold)
        candidate = (_key(metrics), float(threshold), metrics)
        if best_any is None or candidate[0] > best_any[0]:
            best_any = candidate
        if metrics["specificity"] >= min_specificity:
            if best_constrained is None or candidate[0] > best_constrained[0]:
                best_constrained = candidate
    selected = best_constrained if best_constrained is not None else best_any
    return selected[1], selected[2]


def record_burden_report(y_true, scores, metas, threshold):
    grouped = {}
    for y, score, record_id, minute in zip(y_true, scores, metas["record_ids"], metas["minutes"]):
        key = (str(record_id), int(minute))
        if key not in grouped:
            grouped[key] = {"label": int(y), "scores": []}
        grouped[key]["label"] = max(grouped[key]["label"], int(y))
        grouped[key]["scores"].append(float(score))

    per_record = {}
    for (record_id, minute), item in grouped.items():
        rec = per_record.setdefault(record_id, {"true": [], "pred": [], "scores": [], "minutes": []})
        rec["true"].append(item["label"])
        avg_score = float(np.mean(item["scores"]))
        rec["scores"].append(avg_score)
        rec["pred"].append(int(avg_score >= threshold))
        rec["minutes"].append(int(minute))

    rows = []
    binary_true = []
    binary_pred = []
    abs_errors = []
    for record_id, item in sorted(per_record.items()):
        duration_hours = max(item["minutes"]) / 60.0 if item["minutes"] else 1.0
        duration_hours = max(duration_hours, 1e-6)
        true_burden = float(np.sum(item["true"]) / duration_hours)
        pred_burden = float(np.sum(item["pred"]) / duration_hours)
        rows.append(
            {
                "record": record_id,
                "true_positive_minutes_per_hour": true_burden,
                "pred_positive_minutes_per_hour": pred_burden,
                "absolute_error": abs(true_burden - pred_burden),
            }
        )
        abs_errors.append(abs(true_burden - pred_burden))
        binary_true.append(int(true_burden >= 5.0))
        binary_pred.append(int(pred_burden >= 5.0))

    return {
        "records": rows,
        "burden_mae_per_hour": float(np.mean(abs_errors)) if abs_errors else 0.0,
        "screening_accuracy_threshold5": float(accuracy_score(binary_true, binary_pred)) if binary_true else 0.0,
    }


def predict_scores(model, dataset, batch_size, device, amp=True, no_progress=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model.eval()
    labels = []
    scores = []
    metas = {"record_ids": [], "channels": [], "minutes": []}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False, disable=no_progress):
            features = batch["features"].to(device, non_blocking=True)
            raw = batch["raw"].to(device, non_blocking=True) if batch["raw"].numel() else None
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(features, raw)
            prob = torch.softmax(logits, dim=1)[:, 1]
            labels.extend(batch["label"].numpy().tolist())
            scores.extend(prob.cpu().numpy().tolist())
            metas["record_ids"].extend(batch["record_id"])
            metas["channels"].extend([int(x) for x in batch["channel"]])
            metas["minutes"].extend([int(x) for x in batch["minute"]])
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32), metas


def make_sampler(dataset, samples_per_epoch=0):
    y = dataset.arrays["labels"][dataset.indices]
    counts = np.bincount(y, minlength=2)
    weights = len(y) / (2.0 * np.maximum(counts, 1))
    sample_weights = weights[y]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=samples_per_epoch if samples_per_epoch > 0 else len(y),
        replacement=True,
    )


def train_one_split(args, model_kind, arrays, train_idx, val_idx, test_idx, tag, include_raw=False):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_dataset = LiteratureDataset(
        arrays,
        train_idx,
        include_raw=include_raw,
        context_minutes=args.context_minutes,
        raw_length=args.raw_length,
    )
    val_dataset = LiteratureDataset(
        arrays,
        val_idx,
        include_raw=include_raw,
        context_minutes=args.context_minutes,
        raw_length=args.raw_length,
    )
    test_dataset = LiteratureDataset(
        arrays,
        test_idx,
        include_raw=include_raw,
        context_minutes=args.context_minutes,
        raw_length=args.raw_length,
    )

    model = make_model(args, model_kind).to(device)
    sampler = make_sampler(train_dataset, args.samples_per_epoch) if args.oversample else None
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.4, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{tag}_model.pth"

    best_score = -1.0
    best_threshold_value = 0.5
    best_epoch = 0
    patience = 0
    denom = args.samples_per_epoch if args.samples_per_epoch > 0 else len(train_dataset)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(loader, desc=f"{tag} Epoch {epoch}/{args.epochs}", leave=False, disable=args.no_progress):
            features = batch["features"].to(device, non_blocking=True)
            raw = batch["raw"].to(device, non_blocking=True) if include_raw else None
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(features, raw)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * labels.size(0)

        val_y, val_scores, _ = predict_scores(model, val_dataset, args.batch_size, device, args.amp, args.no_progress)
        threshold, val_metrics = best_threshold(val_y, val_scores, args.min_specificity)
        scheduler.step(val_metrics["balanced_accuracy"])
        print(
            f"{tag} Epoch {epoch:02d} | Loss={running_loss / denom:.4f} | "
            f"Val BAcc={val_metrics['balanced_accuracy']:.4f} F1={val_metrics['f1']:.4f} "
            f"Rec={val_metrics['recall']:.4f} Spec={val_metrics['specificity']:.4f} "
            f"AUC={val_metrics['roc_auc']:.4f} thr={threshold:.3f}"
        )
        if val_metrics["balanced_accuracy"] > best_score:
            best_score = val_metrics["balanced_accuracy"]
            best_threshold_value = threshold
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"{tag} early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    test_y, test_scores, test_metas = predict_scores(model, test_dataset, args.batch_size, device, args.amp, args.no_progress)
    oracle_threshold, oracle_metrics = best_threshold(test_y, test_scores, 0.0)
    # F1 為目標的操作點：在 val 上挑讓 F1 最大的閾值再套到 test（誠實），另存 test 上的 F1-oracle 上限。
    val_y_f1, val_scores_f1, _ = predict_scores(model, val_dataset, args.batch_size, device, args.amp, args.no_progress)
    f1_threshold_val, _ = best_threshold(val_y_f1, val_scores_f1, 0.0, objective="f1")
    f1_oracle_threshold, f1_oracle_metrics = best_threshold(test_y, test_scores, 0.0, objective="f1")
    result = {
        "tag": tag,
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_score),
        "best_threshold": float(best_threshold_value),
        "model_path": str(model_path),
        "sample_counts": {
            "train": int(len(train_dataset)),
            "val": int(len(val_dataset)),
            "test": int(len(test_dataset)),
            "train_positive": int(np.sum(arrays["labels"][train_idx] == 1)),
            "val_positive": int(np.sum(arrays["labels"][val_idx] == 1)),
            "test_positive": int(np.sum(arrays["labels"][test_idx] == 1)),
        },
        "test": {
            "threshold_0_5": evaluate_from_scores(test_y, test_scores, 0.5),
            "threshold_val": evaluate_from_scores(test_y, test_scores, best_threshold_value),
            "threshold_oracle": oracle_metrics,
            "oracle_threshold": float(oracle_threshold),
            "threshold_f1_val": evaluate_from_scores(test_y, test_scores, f1_threshold_val),
            "threshold_f1_oracle": f1_oracle_metrics,
            "f1_threshold_val": float(f1_threshold_val),
            "f1_oracle_threshold": float(f1_oracle_threshold),
            "record_burden": record_burden_report(test_y, test_scores, test_metas, best_threshold_value),
        },
    }
    return result


def balanced_folds(records, n_splits, seed):
    summaries = []
    for record_id in sorted({record.record_id for record in records}):
        labels = [record.labels for record in records if record.record_id == record_id]
        y = np.concatenate(labels)
        summaries.append({"record": record_id, "samples": int(len(y)), "positive": int(y.sum())})
    rng = random.Random(seed)
    rng.shuffle(summaries)
    summaries.sort(key=lambda item: (item["positive"], item["samples"]), reverse=True)
    folds = [{"records": [], "samples": 0, "positive": 0} for _ in range(n_splits)]
    for summary in summaries:
        fold = min(folds, key=lambda item: (item["positive"], item["samples"], len(item["records"])))
        fold["records"].append(summary["record"])
        fold["samples"] += summary["samples"]
        fold["positive"] += summary["positive"]
    for fold in folds:
        fold["records"].sort()
        fold["normal"] = fold["samples"] - fold["positive"]
        fold["positive_ratio"] = float(fold["positive"] / fold["samples"]) if fold["samples"] else 0.0
    return folds


def aggregate_results(results):
    keys = ["accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1", "roc_auc"]
    aggregate = {}
    for threshold_key in ["threshold_0_5", "threshold_val", "threshold_oracle"]:
        aggregate[threshold_key] = {}
        for key in keys:
            values = np.asarray([r["test"][threshold_key][key] for r in results], dtype=np.float32)
            aggregate[threshold_key][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "values": values.tolist(),
            }
    return aggregate


def markdown_summary(summary):
    lines = [
        f"# {summary['experiment_name']}",
        "",
        f"Protocol: `{summary['settings']['protocol']}`",
        f"Model: `{summary['model_kind']}`",
        "",
        "## Test Metrics",
        "",
    ]
    if "aggregate" in summary:
        lines.append("| Threshold | BAcc | AUC | F1 | Recall | Specificity |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for threshold, metrics in summary["aggregate"].items():
            lines.append(
                f"| {threshold} | {metrics['balanced_accuracy']['mean']:.4f} +/- {metrics['balanced_accuracy']['std']:.4f} "
                f"| {metrics['roc_auc']['mean']:.4f} +/- {metrics['roc_auc']['std']:.4f} "
                f"| {metrics['f1']['mean']:.4f} +/- {metrics['f1']['std']:.4f} "
                f"| {metrics['recall']['mean']:.4f} +/- {metrics['recall']['std']:.4f} "
                f"| {metrics['specificity']['mean']:.4f} +/- {metrics['specificity']['std']:.4f} |"
            )
    else:
        result = summary["results"][0]
        lines.append("| Threshold | BAcc | AUC | F1 | Recall | Specificity |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for threshold, metrics in result["test"].items():
            if not isinstance(metrics, dict) or "balanced_accuracy" not in metrics:
                continue
            lines.append(
                f"| {threshold} | {metrics['balanced_accuracy']:.4f} | {metrics['roc_auc']:.4f} "
                f"| {metrics['f1']:.4f} | {metrics['recall']:.4f} | {metrics['specificity']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def run_training(args, model_kind, include_raw=False):
    set_seed(args.seed)
    records_all = litfeat.load_records(args, litfeat.available_records(args))
    record_ids = sorted({record.record_id for record in records_all})
    results = []

    if args.protocol == "literature":
        arrays = records_to_arrays(records_all)
        indices = np.arange(len(arrays["labels"]))
        train_val, test = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=arrays["labels"],
        )
        train, val = train_test_split(
            train_val,
            test_size=args.val_size / (1.0 - args.test_size),
            random_state=args.seed,
            stratify=arrays["labels"][train_val],
        )
        results.append(train_one_split(args, model_kind, arrays, train, val, test, "literature_split", include_raw))
        split_info = {"mode": "segment_level_literature_comparable", "records": record_ids}
    elif args.protocol == "holdout":
        train_val_records, test_records = train_test_split(record_ids, test_size=args.test_size, random_state=args.seed)
        train_records, val_records = train_test_split(
            train_val_records,
            test_size=args.val_size / (1.0 - args.test_size),
            random_state=args.seed,
        )
        arrays = records_to_arrays(records_all)
        train = np.flatnonzero(np.isin(arrays["record_ids"], train_records))
        val = np.flatnonzero(np.isin(arrays["record_ids"], val_records))
        test = np.flatnonzero(np.isin(arrays["record_ids"], test_records))
        results.append(train_one_split(args, model_kind, arrays, train, val, test, "record_holdout", include_raw))
        split_info = {"mode": "record_holdout", "train": sorted(train_records), "val": sorted(val_records), "test": sorted(test_records)}
    elif args.protocol == "cv":
        folds = balanced_folds(records_all, args.n_splits, args.seed)
        arrays = records_to_arrays(records_all)
        selected_folds = args.folds if args.folds is not None else list(range(args.n_splits))
        for fold_idx in selected_folds:
            test_records = folds[fold_idx]["records"]
            train_val_records = [record for i, fold in enumerate(folds) if i != fold_idx for record in fold["records"]]
            train_records, val_records = train_test_split(
                train_val_records,
                test_size=args.cv_val_fraction,
                random_state=args.seed + fold_idx,
            )
            train = np.flatnonzero(np.isin(arrays["record_ids"], train_records))
            val = np.flatnonzero(np.isin(arrays["record_ids"], val_records))
            test = np.flatnonzero(np.isin(arrays["record_ids"], test_records))
            results.append(train_one_split(args, model_kind, arrays, train, val, test, f"fold{fold_idx}", include_raw))
        split_info = {"mode": "grouped_cv", "folds": folds, "selected_folds": selected_folds}
    else:
        raise ValueError(f"Unknown protocol: {args.protocol}")

    summary = {
        "experiment_name": args.experiment_name,
        "model_kind": model_kind,
        "settings": vars(args),
        "split_info": split_info,
        "feature_record_stats": [record.stats for record in records_all],
        "results": results,
    }
    if len(results) > 1:
        summary["aggregate"] = aggregate_results(results)

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_summary(summary), encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    print(f"Saved markdown: {markdown_path}")
    return summary


def add_train_args(parser):
    litfeat.add_feature_args(parser)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--protocol", choices=["literature", "holdout", "cv"], default="literature")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hybrid-tokens", type=int, default=90)
    parser.add_argument("--raw-length", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--oversample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--min-specificity", type=float, default=0.0)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--cv-val-fraction", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--cpu", action="store_true")
