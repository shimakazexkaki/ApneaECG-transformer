"""
訓練腳本 (v2 — 修正類別不平衡問題)

改進:
  - 移除雙重加權 (僅使用 Focal Loss + class weight)
  - 加入 Focal Loss (gamma=2)
  - 加入驗證集 threshold tuning (最佳化 F1)
  - 加入 ECG 數據增強
  - 降低預設學習率至 0.0003

用法:
  # 5-Fold CV (推薦，適合小資料集)
  python train.py --mode cv --folds 5 --feature-type raw --dataset ucddb

  # Hold-out
  python train.py --mode holdout --feature-type raw --dataset ucddb
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, cohen_kappa_score,
)

from models import CNNTransformerLSTM, build_model, count_parameters
from data_preprocessing import (
    available_ucddb_records, available_apnea_ecg_records,
    load_dataset, MINUTE_SAMPLES, RRI_RESAMPLE_LEN,
)


# ============================================================
# 隨機種子
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# Focal Loss (處理類別不平衡)
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    對易分類樣本降低損失權重，聚焦於難分類樣本。
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_onehot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        log_probs = F.log_softmax(logits, dim=1)
        ce = -(targets_onehot * log_probs).sum(dim=1)

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = (targets_onehot * alpha).sum(dim=1)
            loss = alpha_t * focal_weight * ce
        else:
            loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================
# ECG 數據增強
# ============================================================
class ECGAugmentation:
    """ECG 信號增強: 時間偏移、振幅縮放、高斯噪聲。"""

    def __init__(self, shift_range=200, scale_range=(0.9, 1.1), noise_std=0.05):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 隨機時間偏移
        if self.shift_range > 0 and random.random() < 0.5:
            shift = random.randint(-self.shift_range, self.shift_range)
            x = torch.roll(x, shifts=shift, dims=-1)

        # 隨機振幅縮放
        if random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            x = x * scale

        # 高斯噪聲
        if random.random() < 0.5:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x


# ============================================================
# Dataset
# ============================================================
class ECGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False):
        if x.ndim == 2:
            self.x = torch.from_numpy(x.astype(np.float32)).unsqueeze(1)
        else:
            self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.augmentation = ECGAugmentation() if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augmentation is not None:
            x = self.augmentation(x)
        return x, self.y[idx]


# ============================================================
# 評估指標 (論文 Section 3.6)
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    auc_val = 0.0
    if y_scores is not None:
        try:
            auc_val = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc_val = 0.0

    return {
        "accuracy": float(acc),
        "sensitivity": float(sen),
        "specificity": float(spe),
        "precision": float(pre),
        "f1_score": float(f1),
        "auc": float(auc_val),
        "kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
    }


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    在驗證集上搜尋最佳閾值，最大化 F1-score。
    """
    best_f1 = -1.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (y_scores >= thresh).astype(np.int64)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def compute_per_recording_metrics(
    record_ids: list,
    segment_record_ids: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray = None,
) -> dict:
    actual_ahi = {}
    predicted_ahi = {}

    for rid in record_ids:
        mask = np.array([s == rid for s in segment_record_ids])
        if mask.sum() == 0:
            continue
        total = mask.sum()
        actual_apnea = y_true[mask].sum()
        predicted_apnea = y_pred[mask].sum()
        actual_ahi[rid] = (actual_apnea / total) * 60
        predicted_ahi[rid] = (predicted_apnea / total) * 60

    if not actual_ahi:
        return {"error": "No per-recording data available"}

    rids = sorted(actual_ahi.keys())
    actual_labels = np.array([1 if actual_ahi[r] > 5 else 0 for r in rids])
    pred_labels = np.array([1 if predicted_ahi[r] > 5 else 0 for r in rids])
    actual_values = np.array([actual_ahi[r] for r in rids])
    pred_values = np.array([predicted_ahi[r] for r in rids])

    rec_acc = accuracy_score(actual_labels, pred_labels)
    rec_sen = recall_score(actual_labels, pred_labels, zero_division=0)
    cm = confusion_matrix(actual_labels, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    rec_spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(actual_values) > 1 and np.std(actual_values) > 0 and np.std(pred_values) > 0:
        corr = float(np.corrcoef(actual_values, pred_values)[0, 1])
    else:
        corr = 0.0

    rec_auc = 0.0
    try:
        rec_auc = roc_auc_score(actual_labels, pred_values)
    except ValueError:
        pass

    return {
        "accuracy": float(rec_acc),
        "sensitivity": float(rec_sen),
        "specificity": float(rec_spe),
        "auc": float(rec_auc),
        "pearson_corr": corr,
        "confusion_matrix": cm.tolist(),
        "n_recordings": len(rids),
        "n_sa": int(actual_labels.sum()),
        "n_normal": int((actual_labels == 0).sum()),
        "per_recording_details": {
            rid: {
                "actual_ahi": float(actual_ahi[rid]),
                "predicted_ahi": float(predicted_ahi[rid]),
                "actual_label": "SA" if actual_ahi[rid] > 5 else "Normal",
                "predicted_label": "SA" if predicted_ahi[rid] > 5 else "Normal",
            }
            for rid in rids
        },
    }


# ============================================================
# 訓練一個 epoch
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
        n_samples += batch_x.size(0)
    return running_loss / max(n_samples, 1)


# ============================================================
# 預測
# ============================================================
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_scores = []
    all_preds = []
    all_labels = []
    for batch_x, batch_y in loader:
        logits = model(batch_x.to(device))
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_scores.extend(scores)
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())
    return (
        np.array(all_labels, dtype=np.int64),
        np.array(all_preds, dtype=np.int64),
        np.array(all_scores, dtype=np.float32),
    )


@torch.no_grad()
def predict_with_threshold(model, loader, device, threshold=0.5):
    """使用自訂閾值進行預測。"""
    model.eval()
    all_scores = []
    all_labels = []
    for batch_x, batch_y in loader:
        logits = model(batch_x.to(device))
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(batch_y.numpy())
    scores = np.array(all_scores, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    preds = (scores >= threshold).astype(np.int64)
    return labels, preds, scores


# ============================================================
# 建構 loss 和 dataloader
# ============================================================
def build_criterion(y_train, args, device):
    """建構損失函數 (Focal Loss + class weights)。"""
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * np.maximum(class_counts, 1))

    if args.loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights.tolist(), gamma=args.focal_gamma)
    else:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=args.label_smoothing)

    return criterion, class_weights


def build_train_loader(x_train, y_train, args, use_sampler=True):
    """建構訓練 DataLoader (帶加權取樣)。"""
    train_ds = ECGDataset(x_train, y_train, augment=args.augment)

    if use_sampler:
        class_counts = np.bincount(y_train, minlength=2)
        sample_w = len(y_train) / (2.0 * np.maximum(class_counts, 1))
        weights = sample_w[y_train]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights), replacement=True,
        )
        return DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        return DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)


# ============================================================
# 訓練核心迴圈
# ============================================================
def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler,
               device, args, model_path, y_val):
    """通用訓練迴圈。返回 (best_epoch, history, best_threshold)。"""
    best_val_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)

        val_labels, val_preds, val_scores = predict(model, val_loader, device)

        # Threshold tuning
        best_thresh = find_best_threshold(val_labels, val_scores)
        val_preds_tuned = (val_scores >= best_thresh).astype(np.int64)
        val_m = compute_metrics(val_labels, val_preds_tuned, val_scores)

        # Val loss (unweighted for stable monitoring)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx.to(device))
                val_loss += F.cross_entropy(logits, by.to(device)).item() * bx.size(0)
        val_loss /= max(len(y_val), 1)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_m["accuracy"])
        history["val_f1"].append(val_m["f1_score"])
        history["val_auc"].append(val_m["auc"])

        if epoch % 5 == 0 or epoch <= 3:
            print(
                f"  Epoch {epoch:03d} | TrLoss={train_loss:.4f} VaLoss={val_loss:.4f} | "
                f"Acc={val_m['accuracy']:.3f} Sen={val_m['sensitivity']:.3f} "
                f"Spe={val_m['specificity']:.3f} F1={val_m['f1_score']:.3f} "
                f"AUC={val_m['auc']:.3f} | Thr={best_thresh:.2f}"
            )

        # Early stopping (基於 AUC，更穩定)
        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "threshold": best_thresh,
                "val_auc": val_m["auc"],
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  >> Early Stop at Epoch {epoch} (best={best_epoch}, AUC={best_val_auc:.4f})")
                break

    return best_epoch, history, best_val_auc


# ============================================================
# Hold-out 訓練
# ============================================================
def train_holdout(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"CNN-Transformer-LSTM -- Hold-out Training")
    print(f"Dataset: {args.dataset} | Feature: {args.feature_type} | Device: {device}")
    print(f"Loss: {args.loss_type} | LR: {args.lr} | Augment: {args.augment}")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    if args.dataset == "ucddb":
        all_records = available_ucddb_records(data_dir)
    else:
        all_records = available_apnea_ecg_records(data_dir)

    if not all_records:
        raise RuntimeError(f"No records found in {data_dir}")

    print(f"\nRecords: {len(all_records)}")

    # 切分 (8:1:1)
    train_val_records, test_records = train_test_split(
        all_records, test_size=0.1, random_state=args.seed, shuffle=True
    )
    train_records, val_records = train_test_split(
        train_val_records, test_size=1/9, random_state=args.seed, shuffle=True
    )

    print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

    in_channels = 1 if args.feature_type == "raw" else 2
    pool_size = 4 if args.feature_type == "raw" else 2

    # 載入資料
    print(f"\nLoading data ({args.feature_type})...")
    x_train, y_train, train_stats = load_dataset(
        data_dir, train_records, args.dataset, args.feature_type,
        channel=args.channel, include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
    )
    x_val, y_val, val_stats = load_dataset(
        data_dir, val_records, args.dataset, args.feature_type,
        channel=args.channel, include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
    )
    x_test, y_test, test_stats = load_dataset(
        data_dir, test_records, args.dataset, args.feature_type,
        channel=args.channel, include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
    )

    test_segment_rids = []
    for stat in test_stats:
        test_segment_rids.extend([stat["record_id"]] * stat["total_segments"])

    print(f"\nData stats:")
    print(f"  Train: {len(y_train)} (A={np.sum(y_train==1)}, N={np.sum(y_train==0)}, ratio={np.mean(y_train==1):.3f})")
    print(f"  Val:   {len(y_val)} (A={np.sum(y_val==1)}, N={np.sum(y_val==0)}, ratio={np.mean(y_val==1):.3f})")
    print(f"  Test:  {len(y_test)} (A={np.sum(y_test==1)}, N={np.sum(y_test==0)}, ratio={np.mean(y_test==1):.3f})")

    # DataLoader (使用 sampler 平衡批次)
    train_loader = build_train_loader(x_train, y_train, args, use_sampler=True)
    val_loader = DataLoader(ECGDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ECGDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    # 模型
    model = build_model(
        args.model_type, in_channels=in_channels,
        cnn_filters=(64, 128, 128), pool_size=pool_size,
    ).to(device)
    n_params = count_parameters(model)
    print(f"\nModel: {args.model_type} | Params: {n_params:,}")

    # Loss (僅使用 Focal Loss 的 alpha，不再雙重加權)
    criterion, class_weights = build_criterion(y_train, args, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    model_path = output_dir / f"{args.experiment_name}.pth"

    print(f"\n{'='*70}")
    print(f"Training (max_epochs={args.epochs}, patience={args.patience})")
    print(f"{'='*70}")

    start_time = time.time()
    best_epoch, history, best_val_auc = train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args, model_path, y_val
    )
    elapsed = time.time() - start_time

    print(f"\nDone! Time: {elapsed:.1f}s | Best Epoch: {best_epoch} | Best Val AUC: {best_val_auc:.4f}")

    # 載入最佳模型
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    best_threshold = ckpt["threshold"]

    # 先在驗證集上 tune threshold
    val_labels, _, val_scores = predict(model, val_loader, device)
    best_threshold = find_best_threshold(val_labels, val_scores)
    print(f"Best threshold (tuned on val): {best_threshold:.3f}")

    # 測試集評估
    print(f"\n{'='*70}")
    print(f"Test Set Evaluation (threshold={best_threshold:.3f})")
    print(f"{'='*70}")

    test_labels, test_preds, test_scores = predict_with_threshold(
        model, test_loader, device, threshold=best_threshold
    )
    test_seg_m = compute_metrics(test_labels, test_preds, test_scores)

    for key, val in test_seg_m.items():
        if key != "confusion_matrix":
            print(f"  {key}: {val:.4f}")
    print(f"  CM: {test_seg_m['confusion_matrix']}")

    # Per-recording
    print(f"\nPer-Recording:")
    test_rec_m = compute_per_recording_metrics(
        test_records, test_segment_rids, test_labels, test_preds, test_scores
    )
    for key, val in test_rec_m.items():
        if key not in ("per_recording_details", "confusion_matrix"):
            print(f"  {key}: {val}")

    # 也用 default 0.5 threshold 評估作比較
    _, test_preds_05, _ = predict(model, test_loader, device)
    test_seg_m_05 = compute_metrics(test_labels, test_preds_05, test_scores)
    print(f"\n  (Comparison @0.5: Acc={test_seg_m_05['accuracy']:.4f} "
          f"Sen={test_seg_m_05['sensitivity']:.4f} Spe={test_seg_m_05['specificity']:.4f} "
          f"F1={test_seg_m_05['f1_score']:.4f})")

    # 儲存結果
    results = {
        "experiment_name": args.experiment_name,
        "settings": {
            "dataset": args.dataset,
            "feature_type": args.feature_type,
            "model_type": args.model_type,
            "loss_type": args.loss_type,
            "n_params": n_params,
            "epochs_trained": best_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "augment": args.augment,
            "threshold": best_threshold,
            "training_time_sec": elapsed,
        },
        "records": {"train": train_records, "val": val_records, "test": test_records},
        "data_stats": {
            "train": {"total": len(y_train), "apnea": int(np.sum(y_train==1)), "normal": int(np.sum(y_train==0))},
            "val": {"total": len(y_val), "apnea": int(np.sum(y_val==1)), "normal": int(np.sum(y_val==0))},
            "test": {"total": len(y_test), "apnea": int(np.sum(y_test==1)), "normal": int(np.sum(y_test==0))},
        },
        "history": history,
        "test_per_segment": test_seg_m,
        "test_per_segment_threshold_05": test_seg_m_05,
        "test_per_recording": test_rec_m,
        "model_path": str(model_path),
    }

    result_path = output_dir / f"{args.experiment_name}_results.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved: {result_path}")

    return results


# ============================================================
# 5-Fold Cross-Validation
# ============================================================
def train_cv(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"CNN-Transformer-LSTM -- {args.folds}-Fold Cross-Validation")
    print(f"Dataset: {args.dataset} | Feature: {args.feature_type} | Device: {device}")
    print(f"Loss: {args.loss_type} | LR: {args.lr} | Augment: {args.augment}")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    if args.dataset == "ucddb":
        all_records = available_ucddb_records(data_dir)
    else:
        all_records = available_apnea_ecg_records(data_dir)

    if not all_records:
        raise RuntimeError(f"No records found in {data_dir}")

    print(f"Records: {len(all_records)}")

    in_channels = 1 if args.feature_type == "raw" else 2
    pool_size = 4 if args.feature_type == "raw" else 2

    print(f"\nLoading all data ({args.feature_type})...")
    all_x, all_y, all_stats = load_dataset(
        data_dir, all_records, args.dataset, args.feature_type,
        channel=args.channel, include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
    )

    all_segment_rids = []
    for stat in all_stats:
        all_segment_rids.extend([stat["record_id"]] * stat["total_segments"])
    all_segment_rids = np.array(all_segment_rids)

    print(f"\nTotal: {len(all_y)} segs (A={np.sum(all_y==1)}, N={np.sum(all_y==0)}, "
          f"ratio={np.mean(all_y==1):.3f})")

    # Patient-level stratified K-Fold
    record_ids_array = np.array(all_records)
    record_labels = []
    for rid in all_records:
        mask = all_segment_rids == rid
        record_labels.append(1 if mask.sum() > 0 and all_y[mask].mean() > 0.1 else 0)
    record_labels = np.array(record_labels)

    print(f"Record-level: {np.sum(record_labels==1)} SA, {np.sum(record_labels==0)} Normal")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    total_start = time.time()

    for fold_idx, (train_record_idx, test_record_idx) in enumerate(skf.split(record_ids_array, record_labels)):
        fold_start = time.time()
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx + 1}/{args.folds}")
        print(f"{'='*70}")

        train_rids = record_ids_array[train_record_idx].tolist()
        test_rids = record_ids_array[test_record_idx].tolist()

        # Val split from train
        if len(train_rids) > 3:
            train_rids_actual, val_rids = train_test_split(
                train_rids, test_size=0.15, random_state=args.seed + fold_idx
            )
        else:
            train_rids_actual = train_rids
            val_rids = test_rids

        print(f"  Train: {len(train_rids_actual)} | Val: {len(val_rids)} | Test: {len(test_rids)}")
        print(f"  Test records: {test_rids}")

        # Get segments
        train_mask = np.isin(all_segment_rids, train_rids_actual)
        val_mask = np.isin(all_segment_rids, val_rids)
        test_mask = np.isin(all_segment_rids, test_rids)

        x_train, y_train = all_x[train_mask], all_y[train_mask]
        x_val, y_val = all_x[val_mask], all_y[val_mask]
        x_test, y_test = all_x[test_mask], all_y[test_mask]
        test_seg_rids = all_segment_rids[test_mask].tolist()

        print(f"  Segments -- Train: {len(y_train)} (A={np.sum(y_train==1)}) | "
              f"Val: {len(y_val)} (A={np.sum(y_val==1)}) | "
              f"Test: {len(y_test)} (A={np.sum(y_test==1)})")

        # DataLoader
        train_loader = build_train_loader(x_train, y_train, args, use_sampler=True)
        val_loader = DataLoader(ECGDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(ECGDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

        # Model (每折重新初始化)
        set_seed(args.seed + fold_idx)  # 每折不同 seed
        model = build_model(
            args.model_type, in_channels=in_channels,
            cnn_filters=(64, 128, 128), pool_size=pool_size,
        ).to(device)

        criterion, _ = build_criterion(y_train, args, device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

        fold_model_path = output_dir / f"{args.experiment_name}_fold{fold_idx}.pth"

        best_epoch, fold_history, best_val_auc = train_loop(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, args, fold_model_path, y_val
        )

        # 載入最佳模型
        ckpt = torch.load(fold_model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        # Threshold tuning on val
        val_labels, _, val_scores = predict(model, val_loader, device)
        best_thresh = find_best_threshold(val_labels, val_scores)

        # Test
        test_labels, test_preds, test_scores = predict_with_threshold(
            model, test_loader, device, threshold=best_thresh
        )
        test_m = compute_metrics(test_labels, test_preds, test_scores)

        # Also test with default 0.5
        _, test_preds_05, _ = predict(model, test_loader, device)
        test_m_05 = compute_metrics(test_labels, test_preds_05, test_scores)

        # Per-recording
        rec_m = compute_per_recording_metrics(test_rids, test_seg_rids, test_labels, test_preds, test_scores)

        fold_elapsed = time.time() - fold_start
        print(f"\n  Fold {fold_idx+1} Results (thr={best_thresh:.2f}, {fold_elapsed:.0f}s):")
        print(f"    Acc={test_m['accuracy']:.4f} Sen={test_m['sensitivity']:.4f} "
              f"Spe={test_m['specificity']:.4f} F1={test_m['f1_score']:.4f} "
              f"AUC={test_m['auc']:.4f} K={test_m['kappa']:.4f}")
        print(f"    (@0.5: Acc={test_m_05['accuracy']:.4f} Sen={test_m_05['sensitivity']:.4f} "
              f"Spe={test_m_05['specificity']:.4f} F1={test_m_05['f1_score']:.4f})")
        if "accuracy" in rec_m:
            print(f"    Per-Recording: Acc={rec_m['accuracy']:.4f} Corr={rec_m.get('pearson_corr', 0):.4f}")

        fold_results.append({
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "threshold": best_thresh,
            "test_records": test_rids,
            "per_segment": test_m,
            "per_segment_threshold_05": test_m_05,
            "per_recording": rec_m,
            "training_time_sec": fold_elapsed,
        })

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"{args.folds}-Fold CV Summary (total: {total_elapsed:.0f}s)")
    print(f"{'='*70}")

    metrics_keys = ["accuracy", "sensitivity", "specificity", "f1_score", "auc", "kappa"]
    summary = {}
    for key in metrics_keys:
        values = [r["per_segment"][key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        print(f"  {key}: {np.mean(values)*100:.1f}% +/- {np.std(values)*100:.1f}%")

    summary_05 = {}
    for key in metrics_keys:
        values = [r["per_segment_threshold_05"][key] for r in fold_results]
        summary_05[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    print(f"\n  (Comparison @0.5 threshold):")
    for key in metrics_keys:
        s = summary_05[key]
        print(f"  {key}: {s['mean']*100:.1f}% +/- {s['std']*100:.1f}%")

    # Save
    cv_results = {
        "experiment_name": args.experiment_name,
        "settings": {
            "dataset": args.dataset,
            "feature_type": args.feature_type,
            "model_type": args.model_type,
            "loss_type": args.loss_type,
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "augment": args.augment,
            "total_time_sec": total_elapsed,
        },
        "fold_results": fold_results,
        "summary": summary,
        "summary_threshold_05": summary_05,
    }

    result_path = output_dir / f"{args.experiment_name}_cv_results.json"
    result_path.write_text(json.dumps(cv_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCV results saved: {result_path}")

    return cv_results


# ============================================================
# 主程式
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="CNN-Transformer-LSTM Training (Sleep Apnea Detection)"
    )

    parser.add_argument("--mode", choices=["holdout", "cv"], default="cv")
    parser.add_argument("--folds", type=int, default=5)

    parser.add_argument("--dataset", choices=["ucddb", "apnea_ecg"], default="ucddb")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--feature-type", choices=["raw", "rri"], default="raw")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)

    parser.add_argument("--model-type", default="cnn_transformer_lstm",
                        choices=["cnn_transformer_lstm", "cnn_only", "cnn_transformer", "cnn_lstm"])

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate (default: 0.0003)")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--loss-type", choices=["focal", "ce"], default="focal", help="Loss: focal or ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true", help="Enable ECG data augmentation")

    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--experiment-name", default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    if args.data_dir is None:
        if args.dataset == "ucddb":
            args.data_dir = str(Path(__file__).parent.parent / "ucddb")
        else:
            args.data_dir = str(Path(__file__).parent.parent / "apnea-ecg")

    if args.experiment_name is None:
        args.experiment_name = f"{args.model_type}_{args.dataset}_{args.feature_type}_{args.mode}"

    if args.mode == "holdout":
        train_holdout(args)
    else:
        train_cv(args)


if __name__ == "__main__":
    main()
