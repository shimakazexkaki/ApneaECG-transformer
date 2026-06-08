import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wfdb
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
import ucddb_runner
import ucddb_trainer


FS = 100
MINUTE_SAMPLES = FS * 60
RANDOM_STATE = 42
SHOW_PROGRESS = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize_segment(segment):
    segment = segment.astype(np.float32, copy=False)
    return (segment - np.mean(segment)) / (np.std(segment) + 1e-8)


def normalize_matrix(x):
    x = x.astype(np.float32, copy=False)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def make_context_windows(x, y, context_minutes):
    if context_minutes == 1:
        return x, y
    if context_minutes < 1 or context_minutes % 2 == 0:
        raise ValueError("--context-minutes must be a positive odd number.")

    half = context_minutes // 2
    usable = len(y) - 2 * half
    if usable <= 0:
        return (
            np.empty((0, MINUTE_SAMPLES * context_minutes), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    windows = []
    for center in range(half, len(y) - half):
        windows.append(x[center - half : center + half + 1].reshape(-1))

    context_x = normalize_matrix(np.stack(windows))
    context_y = y[half : len(y) - half].astype(np.int64, copy=False)
    return context_x, context_y


def apply_ecg_augmentation(segment, rng, args):
    x = segment.astype(np.float32, copy=True)

    if rng.random() < args.flip_prob:
        x = -x

    if args.max_shift_sec > 0:
        max_shift = int(args.max_shift_sec * FS)
        shift = rng.integers(-max_shift, max_shift + 1)
        if shift != 0:
            x = np.roll(x, shift)

    if getattr(args, "time_warp_prob", 0.0) > 0 and rng.random() < args.time_warp_prob:
        factor = rng.uniform(getattr(args, "time_warp_min", 0.95), getattr(args, "time_warp_max", 1.05))
        original_len = len(x)
        warped_len = max(8, int(round(original_len * factor)))
        old_grid = np.linspace(0.0, 1.0, original_len, dtype=np.float32)
        warped_grid = np.linspace(0.0, 1.0, warped_len, dtype=np.float32)
        warped = np.interp(warped_grid, old_grid, x).astype(np.float32)
        if warped_len >= original_len:
            start = int(rng.integers(0, warped_len - original_len + 1))
            x = warped[start : start + original_len]
        else:
            pad_left = int(rng.integers(0, original_len - warped_len + 1))
            pad_right = original_len - warped_len - pad_left
            x = np.pad(warped, (pad_left, pad_right), mode="edge").astype(np.float32)

    if args.scale_min != 1.0 or args.scale_max != 1.0:
        x *= rng.uniform(args.scale_min, args.scale_max)

    if getattr(args, "smooth_prob", 0.0) > 0 and rng.random() < args.smooth_prob:
        max_kernel = max(3, int(getattr(args, "smooth_max_kernel", 9)))
        if max_kernel % 2 == 0:
            max_kernel += 1
        kernel_size = int(rng.integers(3, max_kernel + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        x = np.convolve(x, kernel, mode="same").astype(np.float32)

    if args.noise_std_max > 0:
        x += rng.normal(0.0, rng.uniform(0.0, args.noise_std_max), size=x.shape).astype(np.float32)

    if getattr(args, "spike_prob", 0.0) > 0 and rng.random() < args.spike_prob:
        max_count = max(1, int(getattr(args, "spike_max_count", 3)))
        count = int(rng.integers(1, max_count + 1))
        amp_max = float(getattr(args, "spike_amp_max", 0.5))
        for _ in range(count):
            center = int(rng.integers(0, len(x)))
            width = int(rng.integers(1, max(2, int(0.04 * FS))))
            start = max(0, center - width)
            end = min(len(x), center + width + 1)
            if end > start:
                taper = np.hanning(end - start + 2)[1:-1].astype(np.float32)
                x[start:end] += rng.uniform(-amp_max, amp_max) * taper

    if args.baseline_amp_max > 0:
        t = np.arange(len(x), dtype=np.float32) / FS
        freq = rng.uniform(0.03, 0.30)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.0, args.baseline_amp_max)
        x += amp * np.sin(2.0 * np.pi * freq * t + phase).astype(np.float32)

    if getattr(args, "quantize_prob", 0.0) > 0 and rng.random() < args.quantize_prob:
        levels = max(8, int(getattr(args, "quantize_levels", 64)))
        lo, hi = np.percentile(x, [1, 99])
        if hi > lo:
            clipped = np.clip(x, lo, hi)
            x = (np.round((clipped - lo) / (hi - lo) * (levels - 1)) / (levels - 1) * (hi - lo) + lo).astype(
                np.float32
            )

    if args.mask_prob > 0 and rng.random() < args.mask_prob:
        mask_len = int(rng.uniform(0.05, args.mask_max_sec) * FS)
        mask_len = max(1, min(mask_len, len(x)))
        start = rng.integers(0, len(x) - mask_len + 1)
        x[start : start + mask_len] = 0.0

    return normalize_segment(x)


class ECGDataset(Dataset):
    def __init__(self, x, y, augment=False, aug_args=None, seed=RANDOM_STATE):
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment
        self.aug_args = aug_args
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augment:
            x = apply_ecg_augmentation(x, self.rng, self.aug_args)
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(self.y[idx], dtype=torch.long)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_probs).clamp(min=1e-6).pow(self.gamma)

        if self.label_smoothing > 0:
            n_classes = logits.size(1)
            smooth_loss = -log_probs.mean(dim=1)
            nll_loss = -target_log_probs
            ce_loss = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            ce_loss = -target_log_probs

        return (focal_weight * ce_loss).mean()


def available_apnea_records(data_dir):
    list_path = data_dir / "list"
    records = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    return [r for r in records if (data_dir / f"{r}.apn").exists()]


def split_records(record_ids, val_size, test_size, seed):
    train_val, test = train_test_split(record_ids, test_size=test_size, random_state=seed, shuffle=True)
    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=adjusted_val, random_state=seed, shuffle=True)
    return sorted(train), sorted(val), sorted(test)


def apnea_cache_path(cache_dir, record_id):
    return cache_dir / f"{record_id}_ch0_zscore.npz"


def load_apnea_record(data_dir, cache_dir, record_id):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = apnea_cache_path(cache_dir, record_id)
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["x"], cached["y"]

    signal, _ = wfdb.rdsamp(str(data_dir / record_id))
    annotation = wfdb.rdann(str(data_dir / record_id), "apn")
    ecg = signal[:, 0].astype(np.float32)

    xs = []
    ys = []
    for sample, label in zip(annotation.sample, annotation.symbol):
        if label not in ("A", "N"):
            continue
        start = int(sample)
        end = start + MINUTE_SAMPLES
        if start < 0 or end > len(ecg):
            continue
        segment = ecg[start:end]
        segment = apnea_trainer.butter_bandpass_filter(segment, 0.5, 45.0, FS, order=3)
        xs.append(normalize_segment(segment))
        ys.append(1 if label == "A" else 0)

    if not xs:
        x = np.empty((0, MINUTE_SAMPLES), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    else:
        x = np.stack(xs).astype(np.float32)
        y = np.asarray(ys, dtype=np.int64)

    np.savez_compressed(cache_path, x=x, y=y)
    return x, y


def load_apnea_records(data_dir, cache_dir, record_ids, context_minutes):
    xs = []
    ys = []
    stats = []
    for record_id in tqdm(record_ids, desc="Loading Apnea-ECG records", disable=not SHOW_PROGRESS):
        x, y = load_apnea_record(data_dir, cache_dir, record_id)
        if len(y) == 0:
            continue
        x = normalize_matrix(x)
        original_samples = len(y)
        x, y = make_context_windows(x, y, context_minutes)
        if len(y) == 0:
            continue
        xs.append(x)
        ys.append(y)
        stats.append(
            {
                "record": record_id,
                "original_minute_samples": int(original_samples),
                "samples": int(len(y)),
                "positive": int(y.sum()),
                "normal": int((y == 0).sum()),
            }
        )
    if not xs:
        raise RuntimeError("No Apnea-ECG samples were loaded.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), stats


def load_ucddb_records(data_dir, cache_dir, record_ids, channels, include_hypopnea, min_overlap_sec, context_minutes):
    xs = []
    ys = []
    stats = []
    for record_id in tqdm(record_ids, desc="Loading UCDDB records", disable=not SHOW_PROGRESS):
        for channel in channels:
            x, y = ucddb_trainer.load_ucddb_record(
                data_dir,
                cache_dir,
                record_id,
                channel,
                include_hypopnea=include_hypopnea,
                min_overlap_sec=min_overlap_sec,
            )
            if len(y) == 0:
                continue
            x = normalize_matrix(x)
            original_samples = len(y)
            x, y = make_context_windows(x, y, context_minutes)
            if len(y) == 0:
                continue
            xs.append(x)
            ys.append(y)
            stats.append(
                {
                    "record": record_id,
                    "channel": channel,
                    "original_minute_samples": int(original_samples),
                    "samples": int(len(y)),
                    "positive": int(y.sum()),
                    "normal": int((y == 0).sum()),
                }
            )
    if not xs:
        raise RuntimeError("No UCDDB samples were loaded.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), stats


def source_labels(n_ucddb, n_apnea):
    return np.concatenate(
        [
            np.zeros(n_ucddb, dtype=np.int64),
            np.ones(n_apnea, dtype=np.int64),
        ]
    )


def make_sampler(y, source, ucddb_weight, apnea_weight):
    y = np.asarray(y, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    class_counts = np.bincount(y, minlength=2)
    class_weights = len(y) / (2.0 * np.maximum(class_counts, 1))
    domain_weights = np.where(source == 0, ucddb_weight, apnea_weight)
    sample_weights = class_weights[y] * domain_weights
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


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


def best_threshold_balanced(y_true, scores, min_specificity):
    best_any = None
    best_constrained = None
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = evaluate_from_scores(y_true, scores, threshold=threshold)
        candidate = (metrics["balanced_accuracy"], metrics["f1"], float(threshold), metrics)
        if best_any is None or candidate[:2] > best_any[:2]:
            best_any = candidate
        if metrics["specificity"] >= min_specificity:
            if best_constrained is None or candidate[:2] > best_constrained[:2]:
                best_constrained = candidate

    selected = best_constrained if best_constrained is not None else best_any
    return selected[2], selected[3], best_constrained is not None


def predict_scores(model, x, y, batch_size, device):
    loader = DataLoader(ECGDataset(x, y, augment=False), batch_size=batch_size, shuffle=False)
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            scores = torch.softmax(logits, dim=1)[:, 1]
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    return np.asarray(all_labels, dtype=np.int64), np.asarray(all_scores, dtype=np.float32)


def dataset_summary(name, y):
    return {
        "name": name,
        "samples": int(len(y)),
        "positive": int(np.sum(y == 1)),
        "normal": int(np.sum(y == 0)),
        "positive_ratio": float(np.mean(y == 1)) if len(y) else 0.0,
    }


def train(args):
    global SHOW_PROGRESS
    SHOW_PROGRESS = not args.no_progress

    set_seed(args.seed)
    if args.context_minutes < 1 or args.context_minutes % 2 == 0:
        raise ValueError("--context-minutes must be a positive odd number.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ucddb_data_dir = Path(args.ucddb_dir)
    apnea_data_dir = Path(args.apnea_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ucddb_records = ucddb_runner.available_record_ids(ucddb_data_dir)
    apnea_records = available_apnea_records(apnea_data_dir)
    if not ucddb_records:
        raise RuntimeError("No UCDDB records found.")
    if not apnea_records:
        raise RuntimeError("No Apnea-ECG records found.")

    ucddb_train_records, ucddb_val_records, ucddb_test_records = split_records(
        ucddb_records, args.val_size, args.test_size, args.seed
    )
    apnea_train_records, apnea_val_records, apnea_test_records = split_records(
        apnea_records, args.val_size, args.test_size, args.seed
    )

    print(f"Using device: {device}")
    print(f"UCDDB records: train={len(ucddb_train_records)} val={len(ucddb_val_records)} test={len(ucddb_test_records)}")
    print(f"Apnea records: train={len(apnea_train_records)} val={len(apnea_val_records)} test={len(apnea_test_records)}")
    print("Normalization: per-minute z-score is applied to both datasets after bandpass filtering.")
    print(f"Context: {args.context_minutes} minute(s); label is the center minute.")

    x_u_train, y_u_train, u_train_stats = load_ucddb_records(
        ucddb_data_dir,
        Path(args.ucddb_cache_dir),
        ucddb_train_records,
        args.channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
        context_minutes=args.context_minutes,
    )
    x_u_val, y_u_val, u_val_stats = load_ucddb_records(
        ucddb_data_dir,
        Path(args.ucddb_cache_dir),
        ucddb_val_records,
        args.channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
        context_minutes=args.context_minutes,
    )
    x_u_test, y_u_test, u_test_stats = load_ucddb_records(
        ucddb_data_dir,
        Path(args.ucddb_cache_dir),
        ucddb_test_records,
        args.channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
        context_minutes=args.context_minutes,
    )

    x_a_train, y_a_train, a_train_stats = load_apnea_records(
        apnea_data_dir, Path(args.apnea_cache_dir), apnea_train_records, args.context_minutes
    )
    x_a_val, y_a_val, a_val_stats = load_apnea_records(
        apnea_data_dir, Path(args.apnea_cache_dir), apnea_val_records, args.context_minutes
    )
    x_a_test, y_a_test, a_test_stats = load_apnea_records(
        apnea_data_dir, Path(args.apnea_cache_dir), apnea_test_records, args.context_minutes
    )

    x_train = np.concatenate([x_u_train, x_a_train], axis=0)
    y_train = np.concatenate([y_u_train, y_a_train], axis=0)
    train_source = source_labels(len(y_u_train), len(y_a_train))

    print("Train/val/test sample summaries:")
    for summary in [
        dataset_summary("ucddb_train", y_u_train),
        dataset_summary("ucddb_val", y_u_val),
        dataset_summary("ucddb_test", y_u_test),
        dataset_summary("apnea_train", y_a_train),
        dataset_summary("apnea_val", y_a_val),
        dataset_summary("apnea_test", y_a_test),
        dataset_summary("combined_train", y_train),
    ]:
        print(
            f"  {summary['name']}: n={summary['samples']} pos={summary['positive']} "
            f"normal={summary['normal']} pos_ratio={summary['positive_ratio']:.3f}"
        )

    sampler = make_sampler(y_train, train_source, args.ucddb_sample_weight, args.apnea_sample_weight)
    train_loader = DataLoader(
        ECGDataset(x_train, y_train, augment=True, aug_args=args, seed=args.seed),
        batch_size=args.batch_size,
        sampler=sampler,
    )

    model = apnea_trainer.ParallelCNNTransformer().to(device)
    if args.pretrained:
        print(f"Loading pretrained weights: {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))

    if args.focal_gamma > 0:
        criterion = SoftmaxFocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
        print(f"Loss: focal loss gamma={args.focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print("Loss: cross entropy")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=3)

    best_score = -1.0
    best_threshold = 0.5
    best_ucddb_threshold = 0.5
    best_epoch = 0
    patience = 0
    model_path = output_dir / args.model_name

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False,
            disable=not SHOW_PROGRESS,
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / len(y_train)

        u_val_true, u_val_scores = predict_scores(model, x_u_val, y_u_val, args.batch_size, device)
        a_val_true, a_val_scores = predict_scores(model, x_a_val, y_a_val, args.batch_size, device)
        combined_val_true = np.concatenate([u_val_true, a_val_true])
        combined_val_scores = np.concatenate([u_val_scores, a_val_scores])

        threshold, combined_metrics, constrained = best_threshold_balanced(
            combined_val_true, combined_val_scores, args.min_specificity
        )
        ucddb_threshold, _, _ = best_threshold_balanced(u_val_true, u_val_scores, args.min_specificity)

        u_metrics = evaluate_from_scores(u_val_true, u_val_scores, threshold=threshold)
        a_metrics = evaluate_from_scores(a_val_true, a_val_scores, threshold=threshold)
        selection_score = (
            args.ucddb_val_weight * u_metrics["balanced_accuracy"]
            + (1.0 - args.ucddb_val_weight) * a_metrics["balanced_accuracy"]
        )
        scheduler.step(selection_score)

        print(
            f"Epoch {epoch:02d} | Loss={train_loss:.4f} | thr={threshold:.3f}"
            f"{'' if constrained else '*'} | "
            f"UCDDB Val BAcc={u_metrics['balanced_accuracy']:.4f} F1={u_metrics['f1']:.4f} "
            f"Rec={u_metrics['recall']:.4f} Spec={u_metrics['specificity']:.4f} | "
            f"Apnea Val BAcc={a_metrics['balanced_accuracy']:.4f} F1={a_metrics['f1']:.4f} "
            f"Rec={a_metrics['recall']:.4f} Spec={a_metrics['specificity']:.4f} | "
            f"Score={selection_score:.4f}"
        )

        if selection_score > best_score:
            best_score = selection_score
            best_threshold = threshold
            best_ucddb_threshold = ucddb_threshold
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best epoch={best_epoch} score={best_score:.4f} threshold={best_threshold:.3f}")
    print(f"Saved model: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))

    def test_block(name, x, y):
        labels, scores = predict_scores(model, x, y, args.batch_size, device)
        return {
            "threshold_0_5": evaluate_from_scores(labels, scores, threshold=0.5),
            "threshold_combined_val": evaluate_from_scores(labels, scores, threshold=best_threshold),
            "threshold_ucddb_val": evaluate_from_scores(labels, scores, threshold=best_ucddb_threshold),
        }

    results = {
        "settings": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "focal_gamma": args.focal_gamma,
            "channels": args.channels,
            "context_minutes": args.context_minutes,
            "include_hypopnea": not args.apnea_only,
            "min_overlap_sec": args.min_overlap_sec,
            "min_specificity": args.min_specificity,
            "ucddb_sample_weight": args.ucddb_sample_weight,
            "apnea_sample_weight": args.apnea_sample_weight,
            "ucddb_val_weight": args.ucddb_val_weight,
            "pretrained": args.pretrained,
            "normalization": "per-minute z-score after bandpass; repeated after augmentation",
            "augmentation": {
                "flip_prob": args.flip_prob,
                "max_shift_sec": args.max_shift_sec,
                "scale_min": args.scale_min,
                "scale_max": args.scale_max,
                "noise_std_max": args.noise_std_max,
                "baseline_amp_max": args.baseline_amp_max,
                "mask_prob": args.mask_prob,
                "mask_max_sec": args.mask_max_sec,
            },
        },
        "records": {
            "ucddb_train": ucddb_train_records,
            "ucddb_val": ucddb_val_records,
            "ucddb_test": ucddb_test_records,
            "apnea_train": apnea_train_records,
            "apnea_val": apnea_val_records,
            "apnea_test": apnea_test_records,
        },
        "record_stats": {
            "ucddb_train": u_train_stats,
            "ucddb_val": u_val_stats,
            "ucddb_test": u_test_stats,
            "apnea_train": a_train_stats,
            "apnea_val": a_val_stats,
            "apnea_test": a_test_stats,
        },
        "sample_summaries": {
            "ucddb_train": dataset_summary("ucddb_train", y_u_train),
            "ucddb_val": dataset_summary("ucddb_val", y_u_val),
            "ucddb_test": dataset_summary("ucddb_test", y_u_test),
            "apnea_train": dataset_summary("apnea_train", y_a_train),
            "apnea_val": dataset_summary("apnea_val", y_a_val),
            "apnea_test": dataset_summary("apnea_test", y_a_test),
            "combined_train": dataset_summary("combined_train", y_train),
        },
        "best_selection_score": best_score,
        "best_threshold_combined_val": best_threshold,
        "best_threshold_ucddb_val": best_ucddb_threshold,
        "ucddb_test": test_block("ucddb_test", x_u_test, y_u_test),
        "apnea_test": test_block("apnea_test", x_a_test, y_a_test),
        "model_path": str(model_path),
    }

    result_path = output_dir / args.result_name
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results: {result_path}")

    print("\nFinal holdout test metrics")
    for domain in ("ucddb_test", "apnea_test"):
        print(f"{domain}:")
        for threshold_name, metrics in results[domain].items():
            print(
                f"  {threshold_name}: Acc={metrics['accuracy']:.4f} "
                f"BAcc={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f} "
                f"Rec={metrics['recall']:.4f} Spec={metrics['specificity']:.4f} "
                f"AUC={metrics['roc_auc']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="UCDDB-focused mixed training for wearable ECG apnea detection.")
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--apnea-dir", default="apnea-ecg")
    parser.add_argument("--ucddb-cache-dir", default="aligned_data/ucddb")
    parser.add_argument("--apnea-cache-dir", default="aligned_data/apnea_ecg")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-name", default="mixed_ucddb_focused_parallel_cnn_transformer.pth")
    parser.add_argument("--result-name", default="mixed_ucddb_focused_results.json")
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--context-minutes", type=int, default=1)
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ucddb-sample-weight", type=float, default=2.0)
    parser.add_argument("--apnea-sample-weight", type=float, default=1.0)
    parser.add_argument("--ucddb-val-weight", type=float, default=0.7)
    parser.add_argument("--min-specificity", type=float, default=0.65)
    parser.add_argument("--flip-prob", type=float, default=0.35)
    parser.add_argument("--max-shift-sec", type=float, default=2.0)
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.30)
    parser.add_argument("--noise-std-max", type=float, default=0.05)
    parser.add_argument("--baseline-amp-max", type=float, default=0.08)
    parser.add_argument("--mask-prob", type=float, default=0.20)
    parser.add_argument("--mask-max-sec", type=float, default=0.40)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
