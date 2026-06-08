import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyedflib
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import resample_poly
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

import apnea_trainer
import mixed_trainer
import ucddb_runner


FS = 100
RANDOM_STATE = 42


@dataclass
class LoadedRecord:
    record_id: str
    channel: int
    signal: np.ndarray
    second_labels: np.ndarray

    @property
    def duration_sec(self):
        return int(len(self.second_labels))


class SleepLiteCNN(nn.Module):
    """Small 1D CNN for 11-second ECG windows."""

    def __init__(self, dropout=0.35):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 64, kernel_size=45, padding=22, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(64, 96, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(96, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class HighResCNNTransformer(nn.Module):
    """CNN front-end + Transformer encoder for 11-second UCDDB windows."""

    def __init__(self, d_model=96, nhead=4, layers=2, dropout=0.30, max_tokens=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, d_model, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pos = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
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
        self.attention_score = nn.Linear(d_model, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        if x.size(1) > self.pos.size(1):
            raise ValueError(f"Transformer token length {x.size(1)} exceeds max_tokens={self.pos.size(1)}")
        x = x + self.pos[:, : x.size(1), :]
        x = self.encoder(x)
        weights = torch.softmax(self.attention_score(x).squeeze(-1), dim=1)
        x = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.classifier(x)


class HighResWindowDataset(Dataset):
    def __init__(
        self,
        records,
        window_sec,
        stride_sec,
        label_second,
        label_mode="second",
        min_event_overlap_sec=5,
        augment=False,
        aug_args=None,
        seed=RANDOM_STATE,
        max_normal_ratio=0.0,
        max_windows=0,
        return_record_index=False,
    ):
        if window_sec <= label_second:
            raise ValueError("--label-second must be inside the window.")

        self.records = records
        self.window_sec = int(window_sec)
        self.window_samples = int(window_sec * FS)
        self.label_second = int(label_second)
        self.label_mode = label_mode
        self.min_event_overlap_sec = int(min_event_overlap_sec)
        self.augment = augment
        self.aug_args = aug_args
        self.return_record_index = return_record_index
        self.rng = np.random.default_rng(seed)

        record_indices = []
        start_seconds = []
        labels = []
        for rec_idx, record in enumerate(records):
            max_start = record.duration_sec - window_sec
            if max_start < 0:
                continue
            starts = np.arange(0, max_start + 1, stride_sec, dtype=np.int32)
            if label_mode == "second":
                label_positions = starts + label_second
                y = record.second_labels[label_positions].astype(np.int64, copy=False)
            elif label_mode == "overlap":
                padded = np.concatenate([[0], np.cumsum(record.second_labels, dtype=np.int32)])
                event_seconds = padded[starts + window_sec] - padded[starts]
                y = (event_seconds >= self.min_event_overlap_sec).astype(np.int64)
            else:
                raise ValueError("--label-mode must be 'second' or 'overlap'.")
            record_indices.append(np.full(len(starts), rec_idx, dtype=np.int16))
            start_seconds.append(starts)
            labels.append(y)

        if not labels:
            raise RuntimeError("No high-resolution windows were created.")

        self.record_indices = np.concatenate(record_indices)
        self.start_seconds = np.concatenate(start_seconds)
        self.labels = np.concatenate(labels)

        if max_normal_ratio > 0:
            self._undersample_normals(max_normal_ratio)
        if max_windows > 0 and len(self.labels) > max_windows:
            self._cap_windows(max_windows)

    def _select(self, selected):
        selected = np.asarray(selected, dtype=np.int64)
        self.record_indices = self.record_indices[selected]
        self.start_seconds = self.start_seconds[selected]
        self.labels = self.labels[selected]

    def _undersample_normals(self, max_normal_ratio):
        pos = np.flatnonzero(self.labels == 1)
        neg = np.flatnonzero(self.labels == 0)
        if len(pos) == 0 or len(neg) == 0:
            return
        max_neg = min(len(neg), int(round(len(pos) * max_normal_ratio)))
        keep_neg = self.rng.choice(neg, size=max_neg, replace=False)
        selected = np.concatenate([pos, keep_neg])
        self.rng.shuffle(selected)
        self._select(selected)

    def _cap_windows(self, max_windows):
        selected = self.rng.choice(len(self.labels), size=max_windows, replace=False)
        self._select(selected)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        record = self.records[int(self.record_indices[idx])]
        start = int(self.start_seconds[idx]) * FS
        end = start + self.window_samples
        x = record.signal[start:end].astype(np.float32, copy=True)
        x = mixed_trainer.normalize_segment(x)
        if self.augment:
            x = mixed_trainer.apply_ecg_augmentation(x, self.rng, self.aug_args)
        y = int(self.labels[idx])
        if self.return_record_index:
            return (
                torch.from_numpy(x).unsqueeze(0),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(int(self.record_indices[idx]), dtype=torch.long),
            )
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(y, dtype=torch.long)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cache_path(cache_dir, record_id, channel, include_hypopnea):
    hyp = "hyp" if include_hypopnea else "apneaonly"
    return cache_dir / f"{record_id}_ch{channel}_{hyp}_fs100_bp0p5_45_highres.npz"


def second_labels(duration_sec, events):
    labels = np.zeros(int(duration_sec), dtype=np.int64)
    for start, end, _ in events:
        start = max(0, int(start))
        end = min(int(duration_sec), int(np.ceil(end)))
        if end > start:
            labels[start:end] = 1
    return labels


def read_ucddb_signal(data_dir, record_id, channel):
    edf_path = data_dir / f"{record_id}_lifecard.edf"
    ucddb_runner.patch_edf_start_time(edf_path)
    with pyedflib.EdfReader(str(edf_path)) as edf:
        fs = float(edf.getSampleFrequency(channel))
        duration_sec = float(edf.file_duration)
        signal = edf.readSignal(channel).astype(np.float32)

    if int(round(fs)) != FS:
        signal = resample_poly(signal, FS, int(round(fs))).astype(np.float32)
    usable_samples = int(duration_sec) * FS
    signal = signal[:usable_samples]
    signal = apnea_trainer.butter_bandpass_filter(signal, 0.5, 45.0, FS, order=3).astype(np.float32)
    return signal


def load_record(data_dir, cache_dir, record_id, channel, include_hypopnea):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, record_id, channel, include_hypopnea)
    if path.exists():
        cached = np.load(path)
        return LoadedRecord(
            record_id=record_id,
            channel=channel,
            signal=cached["signal"].astype(np.float32, copy=False),
            second_labels=cached["second_labels"].astype(np.int64, copy=False),
        )

    signal = read_ucddb_signal(data_dir, record_id, channel)
    duration_sec = len(signal) // FS
    event_path = data_dir / f"{record_id}_respevt.txt"
    events = ucddb_runner.parse_respiratory_events(event_path, include_hypopnea=include_hypopnea)
    labels = second_labels(duration_sec, events)
    np.savez_compressed(path, signal=signal, second_labels=labels)
    return LoadedRecord(record_id=record_id, channel=channel, signal=signal, second_labels=labels)


def record_has_positive(data_dir, record_id, include_hypopnea):
    events = ucddb_runner.parse_respiratory_events(
        data_dir / f"{record_id}_respevt.txt",
        include_hypopnea=include_hypopnea,
    )
    return len(events) > 0


def load_records(args, record_ids):
    data_dir = Path(args.ucddb_dir)
    cache_dir = Path(args.cache_dir)
    records = []
    stats = []
    for record_id in tqdm(record_ids, desc="Loading UCDDB high-res records", disable=args.no_progress):
        for channel in args.channels:
            record = load_record(data_dir, cache_dir, record_id, channel, include_hypopnea=not args.apnea_only)
            records.append(record)
            stats.append(
                {
                    "record": record_id,
                    "channel": int(channel),
                    "duration_sec": int(record.duration_sec),
                    "positive_seconds": int(record.second_labels.sum()),
                    "normal_seconds": int(record.duration_sec - record.second_labels.sum()),
                }
            )
    return records, stats


def split_records(record_ids, val_size, test_size, seed):
    train_val, test = train_test_split(record_ids, test_size=test_size, random_state=seed, shuffle=True)
    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=adjusted_val, random_state=seed, shuffle=True)
    return sorted(train), sorted(val), sorted(test)


def evaluate_from_scores(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    y_pred = (scores >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }


def best_threshold_balanced(y_true, scores, min_specificity=0.0):
    best_any = None
    best_constrained = None
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        raise ValueError("Cannot tune a threshold on empty scores.")
    y_true = np.asarray(y_true, dtype=np.int64)
    if float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0:
        thresholds = np.linspace(0.02, 0.98, 193)
    else:
        thresholds = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 193)))
    for threshold in thresholds:
        metrics = evaluate_from_scores(y_true, scores, threshold)
        candidate = (metrics["balanced_accuracy"], metrics["f1"], float(threshold), metrics)
        if best_any is None or candidate[:2] > best_any[:2]:
            best_any = candidate
        if metrics["specificity"] >= min_specificity:
            if best_constrained is None or candidate[:2] > best_constrained[:2]:
                best_constrained = candidate
    selected = best_constrained if best_constrained is not None else best_any
    return selected[2], selected[3], best_constrained is not None


def best_threshold_prevalence_match(y_true, scores, min_specificity=0.0):
    best_any = None
    best_constrained = None
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        raise ValueError("Cannot tune a threshold on empty scores.")
    if float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0:
        thresholds = np.linspace(0.02, 0.98, 193)
    else:
        thresholds = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 193)))
    target_prevalence = float(y_true.mean()) if len(y_true) else 0.0
    for threshold in thresholds:
        metrics = evaluate_from_scores(y_true, scores, threshold)
        predicted_prevalence = float((scores >= threshold).mean())
        prevalence_error = abs(predicted_prevalence - target_prevalence)
        candidate = (
            -prevalence_error,
            metrics["balanced_accuracy"],
            metrics["f1"],
            float(threshold),
            metrics,
        )
        if best_any is None or candidate[:3] > best_any[:3]:
            best_any = candidate
        if metrics["specificity"] >= min_specificity:
            if best_constrained is None or candidate[:3] > best_constrained[:3]:
                best_constrained = candidate
    selected = best_constrained if best_constrained is not None else best_any
    return selected[3], selected[4], best_constrained is not None


def choose_threshold(y_true, scores, min_specificity=0.0, strategy="balanced"):
    if strategy == "balanced":
        return best_threshold_balanced(y_true, scores, min_specificity)
    if strategy == "prevalence_match":
        return best_threshold_prevalence_match(y_true, scores, min_specificity)
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def make_sampler(dataset, samples_per_epoch, record_balanced=False):
    y = dataset.labels
    counts = np.bincount(y, minlength=2)
    weights = len(y) / (2.0 * np.maximum(counts, 1))
    sample_weights = weights[y]
    if record_balanced:
        record_counts = np.bincount(dataset.record_indices.astype(np.int64))
        record_weights = len(y) / (len(record_counts) * np.maximum(record_counts, 1))
        sample_weights = sample_weights * record_weights[dataset.record_indices.astype(np.int64)]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=samples_per_epoch if samples_per_epoch > 0 else len(y),
        replacement=True,
    )


def class_weights_from_dataset(dataset, device):
    counts = np.bincount(dataset.labels, minlength=2)
    weights = len(dataset.labels) / (2.0 * np.maximum(counts, 1))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def per_sample_training_loss(args, logits, targets, class_weights):
    ce = nn.functional.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
        label_smoothing=args.label_smoothing,
    )
    if getattr(args, "focal_gamma", 0.0) > 0:
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** args.focal_gamma) * ce
    return ce


def init_group_weights(args, dataset, device):
    if getattr(args, "record_aware_loss", "none") != "group_dro":
        return None
    n_groups = int(dataset.record_indices.max()) + 1
    return torch.full((n_groups,), 1.0 / max(n_groups, 1), dtype=torch.float32, device=device)


def compute_training_loss(args, logits, targets, record_indices, criterion, class_weights, group_weights=None):
    mode = getattr(args, "record_aware_loss", "none")
    if mode == "none":
        return criterion(logits, targets)
    if record_indices is None:
        raise ValueError("record_indices are required when --record-aware-loss is enabled.")

    sample_loss = per_sample_training_loss(args, logits, targets, class_weights)
    group_ids = torch.unique(record_indices)
    group_losses = torch.stack([sample_loss[record_indices == group_id].mean() for group_id in group_ids])
    if mode == "group_max":
        return group_losses.max()
    if mode == "group_mean":
        return group_losses.mean()
    if mode == "group_dro":
        if group_weights is None:
            raise ValueError("group_weights are required for group_dro.")
        with torch.no_grad():
            group_weights[group_ids] *= torch.exp(args.groupdro_eta * group_losses.detach())
            group_weights /= group_weights.sum().clamp_min(1e-8)
        batch_weights = group_weights[group_ids] / group_weights[group_ids].sum().clamp_min(1e-8)
        return torch.sum(batch_weights.detach() * group_losses)
    raise ValueError(f"Unknown record-aware loss mode: {mode}")


def make_model(args):
    if args.model == "sleeplite":
        return SleepLiteCNN(dropout=args.dropout)
    if args.model == "cnn_transformer":
        return HighResCNNTransformer(
            d_model=args.d_model,
            nhead=args.nhead,
            layers=args.layers,
            dropout=args.dropout,
            max_tokens=args.max_tokens,
        )
    if args.model == "parallel_cnn_transformer":
        return apnea_trainer.ParallelCNNTransformer()
    raise ValueError(f"Unknown model: {args.model}")


def predict_scores(model, dataset, batch_size, device, no_progress=False, amp=True):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Predicting", leave=False, disable=no_progress):
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(batch_x.to(device))
            prob = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(prob.cpu().numpy())
            labels.extend(batch_y.numpy())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def normalize_scores_by_record(dataset, scores, mode="none", group="subject"):
    scores = np.asarray(scores, dtype=np.float32)
    if mode == "none":
        return scores
    normalized = scores.copy()
    groups = {}
    for rec_idx, record in enumerate(dataset.records):
        if group == "record-channel":
            key = (record.record_id, int(record.channel))
        else:
            key = record.record_id
        mask = dataset.record_indices == rec_idx
        if np.any(mask):
            groups.setdefault(key, []).append(np.flatnonzero(mask))

    for indices_parts in groups.values():
        indices = np.concatenate(indices_parts)
        values = scores[indices]
        if len(values) == 0:
            continue
        if mode == "record_center":
            normalized[indices] = values - float(np.median(values))
        elif mode == "record_zscore":
            normalized[indices] = (values - float(values.mean())) / (float(values.std()) + 1e-6)
        elif mode == "record_minmax":
            normalized[indices] = (values - float(values.min())) / (float(values.max() - values.min()) + 1e-6)
        elif mode == "record_rank":
            if len(values) == 1:
                normalized[indices] = 0.5
            else:
                order = np.argsort(values, kind="mergesort")
                ranks = np.empty(len(values), dtype=np.float32)
                ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
                normalized[indices] = ranks
        else:
            raise ValueError(f"Unknown score normalization mode: {mode}")
    return normalized.astype(np.float32)


def normalize_scores_from_args(dataset, scores, args):
    return normalize_scores_by_record(
        dataset,
        scores,
        mode=getattr(args, "score_normalization", "none"),
        group=getattr(args, "score_normalization_group", "subject"),
    )


def aggregate_minutes(dataset, scores, reducer="max"):
    minute_y = []
    minute_scores = []
    scores = np.asarray(scores, dtype=np.float32)
    for rec_idx, record in enumerate(dataset.records):
        mask = dataset.record_indices == rec_idx
        if not np.any(mask):
            continue
        starts = dataset.start_seconds[mask]
        labels = dataset.labels[mask]
        rec_scores = scores[mask]
        minute_ids = (starts + dataset.label_second) // 60
        for minute in np.unique(minute_ids):
            in_minute = minute_ids == minute
            minute_y.append(int(labels[in_minute].max()))
            if reducer == "mean":
                minute_scores.append(float(rec_scores[in_minute].mean()))
            else:
                minute_scores.append(float(rec_scores[in_minute].max()))
    return np.asarray(minute_y, dtype=np.int64), np.asarray(minute_scores, dtype=np.float32)


def aggregate_minute_rows(dataset, scores, reducer="max", combine_channels=False):
    scores = np.asarray(scores, dtype=np.float32)
    grouped = {}
    for rec_idx, record in enumerate(dataset.records):
        mask = dataset.record_indices == rec_idx
        if not np.any(mask):
            continue
        starts = dataset.start_seconds[mask]
        labels = dataset.labels[mask]
        rec_scores = scores[mask]
        minute_ids = (starts + dataset.label_second) // 60
        for minute in np.unique(minute_ids):
            in_minute = minute_ids == minute
            minute_score = float(rec_scores[in_minute].mean()) if reducer == "mean" else float(rec_scores[in_minute].max())
            if combine_channels:
                key = (record.record_id, int(minute))
                channel = "all"
            else:
                key = (record.record_id, int(record.channel), int(minute))
                channel = int(record.channel)
            item = grouped.setdefault(
                key,
                {
                    "record": record.record_id,
                    "channel": channel,
                    "minute": int(minute),
                    "labels": [],
                    "scores": [],
                },
            )
            item["labels"].append(int(labels[in_minute].max()))
            item["scores"].append(minute_score)

    rows = []
    for item in grouped.values():
        item_scores = item.pop("scores")
        item_labels = item.pop("labels")
        item["label"] = int(max(item_labels))
        item["score"] = float(np.mean(item_scores) if reducer == "mean" else np.max(item_scores))
        rows.append(item)
    rows.sort(key=lambda row: (row["record"], str(row["channel"]), row["minute"]))
    return rows


def _safe_corr(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(x) < 2 or float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _burden_rows(minute_rows, threshold):
    grouped = {}
    for row in minute_rows:
        key = (row["record"], row["channel"])
        item = grouped.setdefault(key, {"record": row["record"], "channel": row["channel"], "labels": [], "scores": []})
        item["labels"].append(int(row["label"]))
        item["scores"].append(float(row["score"]))

    rows = []
    for item in grouped.values():
        labels = np.asarray(item["labels"], dtype=np.int64)
        scores = np.asarray(item["scores"], dtype=np.float32)
        pred = scores >= threshold
        minutes = int(len(labels))
        if minutes == 0:
            continue
        true_apnea_minutes = int(labels.sum())
        pred_apnea_minutes = int(pred.sum())
        true_apnea_minutes_per_hour = float(true_apnea_minutes * 60.0 / minutes)
        pred_apnea_minutes_per_hour = float(pred_apnea_minutes * 60.0 / minutes)
        rows.append(
            {
                "record": item["record"],
                "channel": item["channel"],
                "scored_minutes": minutes,
                "true_apnea_minutes": true_apnea_minutes,
                "pred_apnea_minutes": pred_apnea_minutes,
                "true_apnea_fraction": float(true_apnea_minutes / minutes),
                "pred_apnea_fraction": float(pred_apnea_minutes / minutes),
                "true_apnea_minutes_per_hour": true_apnea_minutes_per_hour,
                "pred_apnea_minutes_per_hour": pred_apnea_minutes_per_hour,
                "abs_error_apnea_minutes_per_hour": float(
                    abs(pred_apnea_minutes_per_hour - true_apnea_minutes_per_hour)
                ),
                "mean_score": float(scores.mean()),
            }
        )
    rows.sort(key=lambda row: (row["record"], str(row["channel"])))
    return rows


def _burden_summary(rows):
    if not rows:
        return {
            "records": 0,
            "mae_apnea_minutes_per_hour": 0.0,
            "mean_true_apnea_minutes_per_hour": 0.0,
            "mean_pred_apnea_minutes_per_hour": 0.0,
            "corr_true_pred_apnea_minutes_per_hour": 0.0,
        }
    true_values = [row["true_apnea_minutes_per_hour"] for row in rows]
    pred_values = [row["pred_apnea_minutes_per_hour"] for row in rows]
    abs_errors = [row["abs_error_apnea_minutes_per_hour"] for row in rows]
    return {
        "records": int(len(rows)),
        "mae_apnea_minutes_per_hour": float(np.mean(abs_errors)),
        "mean_true_apnea_minutes_per_hour": float(np.mean(true_values)),
        "mean_pred_apnea_minutes_per_hour": float(np.mean(pred_values)),
        "corr_true_pred_apnea_minutes_per_hour": _safe_corr(true_values, pred_values),
    }


def record_burden_report(dataset, scores, threshold, reducer="max"):
    record_channel_rows = _burden_rows(
        aggregate_minute_rows(dataset, scores, reducer=reducer, combine_channels=False),
        threshold,
    )
    subject_rows = _burden_rows(
        aggregate_minute_rows(dataset, scores, reducer=reducer, combine_channels=True),
        threshold,
    )
    return {
        "definition": "AHI-like burden is apnea-positive minutes per hour, not clinical event-based AHI.",
        "threshold": float(threshold),
        "minute_reducer": reducer,
        "record_channel_summary": _burden_summary(record_channel_rows),
        "subject_summary": _burden_summary(subject_rows),
        "record_channel_rows": record_channel_rows,
        "subject_rows": subject_rows,
    }


def dataset_summary(dataset):
    y = dataset.labels
    return {
        "windows": int(len(y)),
        "positive": int((y == 1).sum()),
        "normal": int((y == 0).sum()),
        "positive_ratio": float((y == 1).mean()) if len(y) else 0.0,
    }


def threshold_report(y, scores, tuned_threshold):
    oracle_threshold, oracle_metrics, _ = best_threshold_balanced(y, scores, min_specificity=0.0)
    return {
        "threshold_0_5": evaluate_from_scores(y, scores, 0.5),
        "threshold_val": evaluate_from_scores(y, scores, tuned_threshold),
        "threshold_oracle": oracle_metrics,
        "oracle_threshold": float(oracle_threshold),
    }


def evaluate_model_dataset(model, dataset, threshold, minute_threshold, args, device):
    y, scores = predict_scores(model, dataset, args.batch_size, device, args.no_progress, args.amp)
    scores = normalize_scores_from_args(dataset, scores, args)
    min_y, min_scores = aggregate_minutes(dataset, scores, reducer=args.minute_reducer)
    return {
        "window": threshold_report(y, scores, threshold),
        "minute": threshold_report(min_y, min_scores, minute_threshold),
        "record_burden": record_burden_report(dataset, scores, minute_threshold, args.minute_reducer),
    }


def train(args):
    set_seed(args.seed)
    mixed_trainer.SHOW_PROGRESS = not args.no_progress
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.ucddb_dir)
    record_ids = ucddb_runner.available_record_ids(data_dir)
    if args.exclude_no_positive:
        record_ids = [
            record_id
            for record_id in record_ids
            if record_has_positive(data_dir, record_id, include_hypopnea=not args.apnea_only)
        ]

    train_ids, val_ids, test_ids = split_records(record_ids, args.val_size, args.test_size, args.seed)
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Records: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"Train records: {train_ids}")
    print(f"Val records:   {val_ids}")
    print(f"Test records:  {test_ids}")
    print(
        "Preprocess: resample to 100 Hz, bandpass 0.5-45 Hz, "
        "per-window z-score normalization."
    )
    print(
        f"Windows: {args.window_sec}s, train stride={args.train_stride_sec}s, "
        f"eval stride={args.eval_stride_sec}s, label second={args.label_second}"
    )

    train_records, train_stats = load_records(args, train_ids)
    val_records, val_stats = load_records(args, val_ids)
    test_records, test_stats = load_records(args, test_ids)

    train_dataset = HighResWindowDataset(
        train_records,
        args.window_sec,
        args.train_stride_sec,
        args.label_second,
        label_mode=args.label_mode,
        min_event_overlap_sec=args.min_event_overlap_sec,
        augment=True,
        aug_args=args,
        seed=args.seed,
        max_normal_ratio=args.max_normal_ratio,
        max_windows=args.max_train_windows,
        return_record_index=args.record_aware_loss != "none",
    )
    val_dataset = HighResWindowDataset(
        val_records,
        args.window_sec,
        args.eval_stride_sec,
        args.label_second,
        label_mode=args.label_mode,
        min_event_overlap_sec=args.min_event_overlap_sec,
        augment=False,
        seed=args.seed,
    )
    test_dataset = HighResWindowDataset(
        test_records,
        args.window_sec,
        args.eval_stride_sec,
        args.label_second,
        label_mode=args.label_mode,
        min_event_overlap_sec=args.min_event_overlap_sec,
        augment=False,
        seed=args.seed,
    )

    print("Window summaries:")
    print(f"  train: {dataset_summary(train_dataset)}")
    print(f"  val:   {dataset_summary(val_dataset)}")
    print(f"  test:  {dataset_summary(test_dataset)}")

    sampler = (
        make_sampler(train_dataset, args.samples_per_epoch, args.record_balanced_sampler)
        if args.weighted_sampler
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = make_model(args).to(device)
    if args.pretrained:
        print(f"Loading pretrained weights: {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device, weights_only=False))

    model_path = output_dir / args.model_name

    if args.eval_only:
        print("Eval-only mode: skipping training and tuning thresholds on the validation split.")
        final_val_dataset = HighResWindowDataset(
            val_records,
            args.window_sec,
            args.final_eval_stride_sec,
            args.label_second,
            label_mode=args.label_mode,
            min_event_overlap_sec=args.min_event_overlap_sec,
            augment=False,
            seed=args.seed,
        )
        final_test_dataset = HighResWindowDataset(
            test_records,
            args.window_sec,
            args.final_eval_stride_sec,
            args.label_second,
            label_mode=args.label_mode,
            min_event_overlap_sec=args.min_event_overlap_sec,
            augment=False,
            seed=args.seed,
        )
        val_y, val_scores = predict_scores(
            model, final_val_dataset, args.batch_size, device, args.no_progress, args.amp
        )
        val_scores = normalize_scores_from_args(final_val_dataset, val_scores, args)
        best_threshold, _, _ = choose_threshold(
            val_y,
            val_scores,
            args.min_specificity,
            args.threshold_strategy,
        )
        val_min_y, val_min_scores = aggregate_minutes(final_val_dataset, val_scores, reducer=args.minute_reducer)
        best_minute_threshold, _, _ = choose_threshold(
            val_min_y,
            val_min_scores,
            args.min_specificity,
            args.threshold_strategy,
        )
        best_epoch = 0
        best_score = 0.0

        results = {
            "settings": vars(args),
            "records": {"train": train_ids, "val": val_ids, "test": test_ids},
            "record_stats": {"train": train_stats, "val": val_stats, "test": test_stats},
            "window_summaries": {
                "train": dataset_summary(train_dataset),
                "val": dataset_summary(final_val_dataset),
                "test": dataset_summary(final_test_dataset),
            },
            "best_epoch": int(best_epoch),
            "best_selection_score": float(best_score),
            "best_threshold_window": float(best_threshold),
            "best_threshold_minute": float(best_minute_threshold),
            "val": evaluate_model_dataset(model, final_val_dataset, best_threshold, best_minute_threshold, args, device),
            "test": evaluate_model_dataset(model, final_test_dataset, best_threshold, best_minute_threshold, args, device),
            "model_path": args.pretrained,
        }

        result_path = output_dir / args.result_name
        result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results: {result_path}")
        print("\nFinal test metrics")
        for level, report in results["test"].items():
            if level == "record_burden":
                continue
            for key, metrics in report.items():
                if key == "oracle_threshold":
                    continue
                print(
                    f"  {level}_{key}: Acc={metrics['accuracy']:.4f} "
                    f"BAcc={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f} "
                    f"Rec={metrics['recall']:.4f} Spec={metrics['specificity']:.4f} "
                    f"AUC={metrics['roc_auc']:.4f}"
                )
        return

    class_weights = class_weights_from_dataset(train_dataset, device)
    if args.record_aware_loss != "none":
        criterion = None
        print(f"Loss: record-aware {args.record_aware_loss} focal_gamma={args.focal_gamma}")
    elif args.focal_gamma > 0:
        criterion = mixed_trainer.SoftmaxFocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
        print(f"Loss: focal gamma={args.focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
        print(f"Loss: weighted cross entropy weights={class_weights.detach().cpu().tolist()}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.4, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_minute_threshold = 0.5
    patience = 0
    group_weights = init_group_weights(args, train_dataset, device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False,
            disable=args.no_progress,
        ):
            if args.record_aware_loss == "none":
                batch_x, batch_y = batch
                batch_record = None
            else:
                batch_x, batch_y, batch_record = batch
                batch_record = batch_record.to(device, non_blocking=True)
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(batch_x)
                loss = compute_training_loss(
                    args,
                    logits,
                    batch_y,
                    batch_record,
                    criterion,
                    class_weights,
                    group_weights,
                )
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / (args.samples_per_epoch if args.samples_per_epoch > 0 else len(train_dataset))
        val_y, val_scores = predict_scores(model, val_dataset, args.batch_size, device, args.no_progress, args.amp)
        val_scores = normalize_scores_from_args(val_dataset, val_scores, args)
        threshold, val_metrics, constrained = choose_threshold(
            val_y,
            val_scores,
            min_specificity=args.min_specificity,
            strategy=args.threshold_strategy,
        )
        val_min_y, val_min_scores = aggregate_minutes(val_dataset, val_scores, reducer=args.minute_reducer)
        minute_threshold, val_min_metrics, _ = choose_threshold(
            val_min_y,
            val_min_scores,
            min_specificity=args.min_specificity,
            strategy=args.threshold_strategy,
        )
        selection_score = (
            args.window_val_weight * val_metrics["balanced_accuracy"]
            + (1.0 - args.window_val_weight) * val_min_metrics["balanced_accuracy"]
        )
        scheduler.step(selection_score)

        print(
            f"Epoch {epoch:02d} | Loss={train_loss:.4f} | thr={threshold:.3f}"
            f"{'' if constrained else '*'} | "
            f"Win BAcc={val_metrics['balanced_accuracy']:.4f} F1={val_metrics['f1']:.4f} "
            f"Rec={val_metrics['recall']:.4f} Spec={val_metrics['specificity']:.4f} "
            f"AUC={val_metrics['roc_auc']:.4f} | "
            f"Min BAcc={val_min_metrics['balanced_accuracy']:.4f} "
            f"AUC={val_min_metrics['roc_auc']:.4f}"
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_threshold = threshold
            best_minute_threshold = minute_threshold
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best epoch={best_epoch} score={best_score:.4f}")
    print(f"Saved model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

    final_val_dataset = HighResWindowDataset(
        val_records,
        args.window_sec,
        args.final_eval_stride_sec,
        args.label_second,
        label_mode=args.label_mode,
        min_event_overlap_sec=args.min_event_overlap_sec,
        augment=False,
        seed=args.seed,
    )
    final_test_dataset = HighResWindowDataset(
        test_records,
        args.window_sec,
        args.final_eval_stride_sec,
        args.label_second,
        label_mode=args.label_mode,
        min_event_overlap_sec=args.min_event_overlap_sec,
        augment=False,
        seed=args.seed,
    )

    final_val_y, final_val_scores = predict_scores(
        model, final_val_dataset, args.batch_size, device, args.no_progress, args.amp
    )
    final_val_scores = normalize_scores_from_args(final_val_dataset, final_val_scores, args)
    final_threshold, _, _ = choose_threshold(
        final_val_y,
        final_val_scores,
        args.min_specificity,
        args.threshold_strategy,
    )
    final_val_min_y, final_val_min_scores = aggregate_minutes(
        final_val_dataset, final_val_scores, reducer=args.minute_reducer
    )
    final_minute_threshold, _, _ = choose_threshold(
        final_val_min_y,
        final_val_min_scores,
        args.min_specificity,
        args.threshold_strategy,
    )

    results = {
        "settings": vars(args),
        "records": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "record_stats": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "window_summaries": {
            "train": dataset_summary(train_dataset),
            "val": dataset_summary(final_val_dataset),
            "test": dataset_summary(final_test_dataset),
        },
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "best_threshold_window": float(best_threshold),
        "best_threshold_minute": float(best_minute_threshold),
        "final_eval_threshold_window": float(final_threshold),
        "final_eval_threshold_minute": float(final_minute_threshold),
        "val": evaluate_model_dataset(model, final_val_dataset, final_threshold, final_minute_threshold, args, device),
        "test": evaluate_model_dataset(model, final_test_dataset, final_threshold, final_minute_threshold, args, device),
        "model_path": str(model_path),
    }

    result_path = output_dir / args.result_name
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results: {result_path}")

    print("\nFinal test metrics")
    for level, report in results["test"].items():
        if level == "record_burden":
            continue
        for key, metrics in report.items():
            if key == "oracle_threshold":
                continue
            print(
                f"  {level}_{key}: Acc={metrics['accuracy']:.4f} BAcc={metrics['balanced_accuracy']:.4f} "
                f"F1={metrics['f1']:.4f} Rec={metrics['recall']:.4f} "
                f"Spec={metrics['specificity']:.4f} AUC={metrics['roc_auc']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="High-resolution UCDDB ECG apnea trainer.")
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--cache-dir", default="aligned_data/ucddb_highres")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-name", default="ucddb_highres_sleeplite.pth")
    parser.add_argument("--result-name", default="ucddb_highres_sleeplite_results.json")
    parser.add_argument("--model", choices=["sleeplite", "cnn_transformer", "parallel_cnn_transformer"], default="sleeplite")
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--window-sec", type=int, default=11)
    parser.add_argument("--train-stride-sec", type=int, default=1)
    parser.add_argument("--eval-stride-sec", type=int, default=1)
    parser.add_argument("--final-eval-stride-sec", type=int, default=1)
    parser.add_argument("--label-second", type=int, default=1)
    parser.add_argument("--label-mode", choices=["second", "overlap"], default="second")
    parser.add_argument("--min-event-overlap-sec", type=int, default=5)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--exclude-no-positive", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--record-aware-loss", choices=["none", "group_mean", "group_max", "group_dro"], default="none")
    parser.add_argument("--groupdro-eta", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--weighted-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-balanced-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--max-normal-ratio", type=float, default=3.0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--min-specificity", type=float, default=0.0)
    parser.add_argument("--window-val-weight", type=float, default=0.7)
    parser.add_argument("--minute-reducer", choices=["max", "mean"], default="max")
    parser.add_argument("--threshold-strategy", choices=["balanced", "prevalence_match"], default="balanced")
    parser.add_argument(
        "--score-normalization",
        choices=["none", "record_center", "record_zscore", "record_minmax", "record_rank"],
        default="none",
    )
    parser.add_argument("--score-normalization-group", choices=["subject", "record-channel"], default="subject")
    parser.add_argument("--flip-prob", type=float, default=0.25)
    parser.add_argument("--max-shift-sec", type=float, default=0.5)
    parser.add_argument("--scale-min", type=float, default=0.80)
    parser.add_argument("--scale-max", type=float, default=1.25)
    parser.add_argument("--noise-std-max", type=float, default=0.04)
    parser.add_argument("--baseline-amp-max", type=float, default=0.05)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mask-max-sec", type=float, default=0.25)
    parser.add_argument("--time-warp-prob", type=float, default=0.0)
    parser.add_argument("--time-warp-min", type=float, default=0.95)
    parser.add_argument("--time-warp-max", type=float, default=1.05)
    parser.add_argument("--smooth-prob", type=float, default=0.0)
    parser.add_argument("--smooth-max-kernel", type=int, default=9)
    parser.add_argument("--spike-prob", type=float, default=0.0)
    parser.add_argument("--spike-amp-max", type=float, default=0.5)
    parser.add_argument("--spike-max-count", type=int, default=3)
    parser.add_argument("--quantize-prob", type=float, default=0.0)
    parser.add_argument("--quantize-levels", type=int, default=64)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
