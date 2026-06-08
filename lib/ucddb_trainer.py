import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import apnea_trainer
import ucddb_runner


BATCH_SIZE = 64
DEFAULT_EPOCHS = 40
DEFAULT_LR = 1e-4
PATIENCE = 8
RANDOM_STATE = 42


class MinuteECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32)).unsqueeze(1)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_from_scores(y_true, scores, threshold=0.5):
    y_pred = (scores >= threshold).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def best_threshold_for_f1(y_true, scores):
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(y_true, (scores >= threshold).astype(np.int64), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def predict_scores(model, loader, device):
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(probs.cpu().numpy())
            labels.extend(batch_y.numpy())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def cache_path_for_record(cache_dir, record_id, channel, include_hypopnea, min_overlap_sec):
    hyp = "hyp" if include_hypopnea else "apneaonly"
    overlap = str(min_overlap_sec).replace(".", "p")
    return cache_dir / f"{record_id}_ch{channel}_{hyp}_overlap{overlap}.npz"


def load_ucddb_record(data_dir, cache_dir, record_id, channel, include_hypopnea, min_overlap_sec):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for_record(cache_dir, record_id, channel, include_hypopnea, min_overlap_sec)
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["x"], cached["y"]

    x, y = ucddb_runner.load_ucddb_segments(
        data_dir / f"{record_id}_lifecard.edf",
        data_dir / f"{record_id}_respevt.txt",
        channel=channel,
        include_hypopnea=include_hypopnea,
        min_overlap_sec=min_overlap_sec,
    )
    np.savez_compressed(cache_path, x=x, y=y)
    return x, y


def load_ucddb_records(data_dir, cache_dir, record_ids, channels, include_hypopnea, min_overlap_sec):
    xs = []
    ys = []
    stats = []
    for record_id in tqdm(record_ids, desc="Loading UCDDB records"):
        for channel in channels:
            x, y = load_ucddb_record(data_dir, cache_dir, record_id, channel, include_hypopnea, min_overlap_sec)
            if len(y) == 0:
                continue
            xs.append(x)
            ys.append(y)
            stats.append(
                {
                    "record": record_id,
                    "channel": channel,
                    "samples": int(len(y)),
                    "positive": int(y.sum()),
                    "normal": int((y == 0).sum()),
                }
            )
    if not xs:
        raise RuntimeError("No UCDDB samples were loaded.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), stats


def load_apnea_test():
    original_tqdm = apnea_trainer.tqdm
    apnea_trainer.tqdm = lambda x, **kwargs: x
    try:
        x, y = apnea_trainer.load_data()
    finally:
        apnea_trainer.tqdm = original_tqdm
    return x.astype(np.float32), y.astype(np.int64)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = ucddb_runner.available_record_ids(data_dir)
    train_records, val_records = train_test_split(all_records, test_size=args.val_size, random_state=args.seed)
    print(f"Using device: {device}")
    print(f"UCDDB records: train={len(train_records)} val={len(val_records)}")
    print(f"Train records: {train_records}")
    print(f"Val records:   {val_records}")

    channels = args.channels
    x_train, y_train, train_stats = load_ucddb_records(
        data_dir,
        cache_dir,
        train_records,
        channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
    )
    x_val, y_val, val_stats = load_ucddb_records(
        data_dir,
        cache_dir,
        val_records,
        channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
    )

    print(f"UCDDB train samples: {len(y_train)} | A/HYP={int(y_train.sum())} N={int((y_train == 0).sum())}")
    print(f"UCDDB val samples:   {len(y_val)} | A/HYP={int(y_val.sum())} N={int((y_val == 0).sum())}")

    train_loader = DataLoader(MinuteECGDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MinuteECGDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = apnea_trainer.ParallelCNNTransformer().to(device)

    if args.pretrained:
        print(f"Loading pretrained weights: {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))

    class_counts = np.bincount(y_train, minlength=2)
    weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=3)

    best_val_f1 = -1.0
    best_threshold = 0.5
    patience = 0
    model_path = output_dir / args.model_name

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_true, val_scores = predict_scores(model, val_loader, device)
        threshold, threshold_f1 = best_threshold_for_f1(val_true, val_scores)
        val_metrics_default = evaluate_from_scores(val_true, val_scores, threshold=0.5)
        val_metrics_tuned = evaluate_from_scores(val_true, val_scores, threshold=threshold)
        scheduler.step(val_metrics_tuned["f1"])

        print(
            f"Epoch {epoch:02d} | Loss={train_loss:.4f} | "
            f"ValF1@0.5={val_metrics_default['f1']:.4f} | "
            f"ValF1@thr={val_metrics_tuned['f1']:.4f} thr={threshold:.3f} | "
            f"ValAcc={val_metrics_tuned['accuracy']:.4f} Rec={val_metrics_tuned['recall']:.4f} "
            f"Spec={val_metrics_tuned['specificity']:.4f}"
        )

        if threshold_f1 > best_val_f1:
            best_val_f1 = threshold_f1
            best_threshold = threshold
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best UCDDB val F1={best_val_f1:.4f} at threshold={best_threshold:.3f}")
    print(f"Saved model: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    val_true, val_scores = predict_scores(model, val_loader, device)
    val_default = evaluate_from_scores(val_true, val_scores, threshold=0.5)
    val_tuned = evaluate_from_scores(val_true, val_scores, threshold=best_threshold)

    print("Loading Apnea-ECG external test set...")
    x_apnea, y_apnea = load_apnea_test()
    apnea_loader = DataLoader(MinuteECGDataset(x_apnea, y_apnea), batch_size=args.batch_size, shuffle=False)
    apnea_true, apnea_scores = predict_scores(model, apnea_loader, device)
    apnea_default = evaluate_from_scores(apnea_true, apnea_scores, threshold=0.5)
    apnea_tuned = evaluate_from_scores(apnea_true, apnea_scores, threshold=best_threshold)

    result = {
        "settings": {
            "data_dir": str(data_dir),
            "channels": channels,
            "include_hypopnea": not args.apnea_only,
            "min_overlap_sec": args.min_overlap_sec,
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "class_weights": weights.tolist(),
            "pretrained": args.pretrained,
        },
        "records": {"train": train_records, "val": val_records},
        "train_stats": train_stats,
        "val_stats": val_stats,
        "best_threshold": best_threshold,
        "ucddb_val_threshold_0_5": val_default,
        "ucddb_val_tuned_threshold": val_tuned,
        "apnea_ecg_test_threshold_0_5": apnea_default,
        "apnea_ecg_test_tuned_threshold": apnea_tuned,
        "model_path": str(model_path),
    }

    result_path = output_dir / args.result_name
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved results: {result_path}")

    print("\nApnea-ECG External Test Metrics")
    print("Threshold 0.5:")
    for key, value in apnea_default.items():
        print(f"  {key}: {value}")
    print(f"Tuned threshold {best_threshold:.3f}:")
    for key, value in apnea_tuned.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Train on aligned UCDDB and test on Apnea-ECG.")
    parser.add_argument("--data-dir", default="ucddb")
    parser.add_argument("--cache-dir", default="aligned_data/ucddb")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-name", default="ucddb_parallel_cnn_transformer.pth")
    parser.add_argument("--result-name", default="ucddb_train_apnea_test_results.json")
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
