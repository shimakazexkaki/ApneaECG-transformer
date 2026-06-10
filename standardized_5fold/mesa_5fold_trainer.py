"""
三路平行 1D CNN + Transformer — MESA Stratified 5-Fold Cross Validation

From scratch training on MESA (no pre-training).
Label: Apnea + Hypopnea events from NSRR XML annotations.
Split: Segment-level Stratified 5-Fold CV.
Preprocessing: resample 256->100Hz, Butterworth 0.5-45Hz, per-segment Z-score.
"""

import os
import sys
import json
import time
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyedflib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.signal import butter, lfilter, resample_poly
from tqdm import tqdm

# ==========================================
# 1. Parameters
# ==========================================
MESA_DIR = r"D:\mesa"
EDF_DIR = os.path.join(MESA_DIR, "mesa", "polysomnography", "edfs")
XML_DIR = os.path.join(MESA_DIR, "mesa", "polysomnography", "annotations-events-nsrr")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', '5fold_mesa_parallel_cnn_transformer')

FS_TARGET = 100
MINUTE_SAMPLES = FS_TARGET * 60  # 6000
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
PATIENCE = 10
N_FOLDS = 5
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_ECG_LABELS = ("ekg", "ecg", "ecg1", "ecgl", "ekg1")

# ==========================================
# 2. MESA Data Loading
# ==========================================
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def find_ecg_channel(edf):
    labels = [str(lab).strip().lower() for lab in edf.getSignalLabels()]
    for i, lab in enumerate(labels):
        if lab in _ECG_LABELS:
            return i
    for i, lab in enumerate(labels):
        if ("ekg" in lab or "ecg" in lab) and not lab.endswith("_off"):
            return i
    raise RuntimeError(f"No ECG/EKG channel found, labels={edf.getSignalLabels()}")

def parse_mesa_events(xml_path, include_hypopnea=True):
    events = []
    root = ET.parse(str(xml_path)).getroot()
    for se in root.findall(".//ScoredEvent"):
        concept = (se.findtext("EventConcept") or "").lower()
        is_apnea = "apnea" in concept
        is_hyp = "hypopnea" in concept
        if not is_apnea and not (include_hypopnea and is_hyp):
            continue
        try:
            start = float(se.findtext("Start"))
            dur = float(se.findtext("Duration"))
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        events.append((start, start + dur, concept))
    return events

def labels_for_minutes(duration_sec, events, min_overlap_sec=1.0):
    n_minutes = int(duration_sec // 60)
    y = np.zeros(n_minutes, dtype=np.int64)
    for minute in range(n_minutes):
        start = minute * 60
        end = start + 60
        for event_start, event_end, _ in events:
            overlap = min(end, event_end) - max(start, event_start)
            if overlap >= min_overlap_sec:
                y[minute] = 1
                break
    return y

def get_available_records():
    edf_dir = Path(EDF_DIR)
    xml_dir = Path(XML_DIR)
    records = []
    for edf_path in sorted(edf_dir.glob("mesa-sleep-*.edf")):
        record_id = edf_path.stem
        xml_path = xml_dir / f"{record_id}-nsrr.xml"
        if xml_path.exists():
            records.append((record_id, str(edf_path), str(xml_path)))
    return records

def load_mesa_all(include_hypopnea=True, min_overlap_sec=1.0):
    print(f"Loading MESA from {EDF_DIR}...")
    X_all, y_all = [], []
    record_stats = []

    available = get_available_records()
    print(f"Found {len(available)} records with both EDF + XML annotations")

    for record_id, edf_path, xml_path in tqdm(available, desc="Loading MESA records"):
        try:
            with pyedflib.EdfReader(edf_path) as edf:
                ch = find_ecg_channel(edf)
                fs = float(edf.getSampleFrequency(ch))
                duration_sec = float(edf.file_duration)
                signal = edf.readSignal(ch).astype(np.float32)

            if int(round(fs)) != FS_TARGET:
                signal = resample_poly(signal, FS_TARGET, int(round(fs))).astype(np.float32)

            usable_samples = int(duration_sec) * FS_TARGET
            signal = signal[:usable_samples]

            events = parse_mesa_events(xml_path, include_hypopnea=include_hypopnea)
            y = labels_for_minutes(duration_sec, events, min_overlap_sec=min_overlap_sec)

            usable_minutes = min(len(y), len(signal) // MINUTE_SAMPLES)
            if usable_minutes == 0:
                continue

            signal = signal[:usable_minutes * MINUTE_SAMPLES]
            y = y[:usable_minutes]
            x = signal.reshape(usable_minutes, MINUTE_SAMPLES)

            processed = []
            for segment in x:
                segment = butter_bandpass_filter(segment, 0.5, 45.0, FS_TARGET, order=3)
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                processed.append(segment.astype(np.float32))

            X_record = np.stack(processed)
            X_all.append(X_record)
            y_all.append(y)

            n_pos = int(y.sum())
            n_neg = int((y == 0).sum())
            record_stats.append({
                'record': record_id,
                'total': len(y),
                'apnea': n_pos,
                'normal': n_neg
            })

        except Exception as e:
            print(f"  Error loading {record_id}: {e}")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    print(f"Loaded {len(record_stats)} records, {len(X)} total segments")
    return X, y, record_stats

# ==========================================
# 3. Model Architecture (same as apnea_5fold_trainer.py)
# ==========================================
class ParallelCNNBlock(nn.Module):
    def __init__(self):
        super(ParallelCNNBlock, self).__init__()
        self.path1 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=300, stride=1, padding=150),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.path2 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=30, stride=1, padding=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.path3 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=10, stride=1, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        min_len = min(p1.size(2), p2.size(2), p3.size(2))
        p1, p2, p3 = p1[:,:,:min_len], p2[:,:,:min_len], p3[:,:,:min_len]
        return torch.cat([p1, p2, p3], dim=1)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class ParallelCNNTransformer(nn.Module):
    def __init__(self):
        super(ParallelCNNTransformer, self).__init__()
        self.parallel_cnn = ParallelCNNBlock()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.channel_reduce = nn.Sequential(
            nn.Conv1d(144, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2)
        )
        self.dense_block = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        self.gap = nn.AdaptiveAvgPool1d(64)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, 64) * 0.02)
        self.transformer = TransformerBlock(d_model=64, nhead=8, dim_feedforward=256, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.parallel_cnn(x)
        x = self.pool(x)
        x = self.channel_reduce(x)
        identity = x
        x = self.dense_block(x)
        if x.shape == identity.shape:
            x = x + identity
        x = self.gap(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

# ==========================================
# 4. Dataset & Evaluation
# ==========================================
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spec = 0.0
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(spec),
        'f1': float(f1)
    }

def evaluate_on_loader(model, loader, device):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            outputs = model(batch_X.to(device))
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(batch_y.numpy())
    return evaluate_metrics(np.array(all_trues), np.array(all_preds))

# ==========================================
# 5. Single Fold Training
# ==========================================
def train_single_fold(fold_idx, X_train, y_train, X_val, y_val, output_dir):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx + 1}/{N_FOLDS}")
    print(f"  Train: {len(X_train)} (Apnea/Hyp={sum(y_train==1)}, Normal={sum(y_train==0)})")
    print(f"  Val:   {len(X_val)} (Apnea/Hyp={sum(y_val==1)}, Normal={sum(y_val==0)})")
    print(f"{'='*60}")

    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = ParallelCNNTransformer().to(DEVICE)

    class_counts = np.bincount(y_train, minlength=2)
    weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    best_val_f1 = 0
    best_val_metrics = None
    best_train_metrics = None
    patience_counter = 0
    fold_history = []
    model_save_path = os.path.join(output_dir, f'best_model_fold{fold_idx}.pth')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch}/{EPOCHS}", leave=False):
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step(train_loss)

        train_metrics = evaluate_on_loader(model, train_loader, DEVICE)
        val_metrics = evaluate_on_loader(model, val_loader, DEVICE)

        epoch_record = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train': train_metrics,
            'val': val_metrics
        }
        fold_history.append(epoch_record)

        print(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | "
              f"Train F1: {train_metrics['f1']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val Prec: {val_metrics['precision']:.4f} | "
              f"Val Rec: {val_metrics['recall']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_metrics = val_metrics.copy()
            best_train_metrics = train_metrics.copy()
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  [BEST] New best Val F1: {best_val_f1:.4f} - model saved")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [STOP] Early Stopping at Epoch {epoch} (patience={PATIENCE})")
                break

    return {
        'fold': fold_idx,
        'best_epoch': fold_history[np.argmax([h['val']['f1'] for h in fold_history])]['epoch'],
        'total_epochs': len(fold_history),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'train_apnea': int(sum(y_train == 1)),
        'train_normal': int(sum(y_train == 0)),
        'val_apnea': int(sum(y_val == 1)),
        'val_normal': int(sum(y_val == 0)),
        'best_train_metrics': best_train_metrics,
        'best_val_metrics': best_val_metrics,
        'history': fold_history
    }

# ==========================================
# 6. Main 5-Fold CV Training
# ==========================================
def train_5fold():
    print(f"Device: {DEVICE}")
    print(f"Random seed: {RANDOM_SEED}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y, record_stats = load_mesa_all(include_hypopnea=True, min_overlap_sec=1.0)
    total = len(X)
    n_apnea = int(sum(y == 1))
    n_normal = int(sum(y == 0))
    print(f"\nMESA Total: {total} segments | Apnea/Hyp: {n_apnea} | Normal: {n_normal}")
    print(f"Positive ratio: {n_apnea/total*100:.1f}%")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_fold_results = []
    start_time = time.time()

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        fold_result = train_single_fold(fold_idx, X_train, y_train, X_val, y_val, OUTPUT_DIR)
        all_fold_results.append(fold_result)

    elapsed = time.time() - start_time

    val_metrics_keys = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    avg_val, std_val = {}, {}
    avg_train, std_train = {}, {}

    for key in val_metrics_keys:
        vals = [r['best_val_metrics'][key] for r in all_fold_results]
        avg_val[key] = float(np.mean(vals))
        std_val[key] = float(np.std(vals))
        trains = [r['best_train_metrics'][key] for r in all_fold_results]
        avg_train[key] = float(np.mean(trains))
        std_train[key] = float(np.std(trains))

    print(f"\n{'='*70}")
    print(f"  MESA Stratified 5-Fold Cross Validation Results")
    print(f"{'='*70}")
    print(f"\nDataset: MESA | Total {total} segments (Apnea/Hyp {n_apnea}, Normal {n_normal})")
    print(f"Model: ParallelCNNTransformer (from scratch, no pre-training)")
    print(f"Training time: {elapsed/60:.1f} min")

    print(f"\n--- Per-Fold Best Validation Metrics ---")
    print(f"{'Fold':>6} | {'Acc':>8} | {'Precision':>10} | {'Recall':>8} | {'Spec':>8} | {'F1':>8} | {'Train F1':>10} | {'Epoch':>6}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")
    for r in all_fold_results:
        vm = r['best_val_metrics']
        tm = r['best_train_metrics']
        print(f"  {r['fold']+1:>4} | {vm['accuracy']:>8.4f} | {vm['precision']:>10.4f} | "
              f"{vm['recall']:>8.4f} | {vm['specificity']:>8.4f} | {vm['f1']:>8.4f} | "
              f"{tm['f1']:>10.4f} | {r['best_epoch']:>6}")

    print(f"\n--- 5-Fold Mean +/- Std ---")
    print(f"  Validation:")
    for key in val_metrics_keys:
        print(f"    {key:>12}: {avg_val[key]:.4f} +/- {std_val[key]:.4f}")
    print(f"  Training:")
    for key in val_metrics_keys:
        print(f"    {key:>12}: {avg_train[key]:.4f} +/- {std_train[key]:.4f}")

    results_summary = {
        'model': 'ParallelCNNTransformer',
        'architecture': '3-path parallel 1D CNN (k=300/30/10) + Transformer',
        'dataset': 'MESA',
        'pretrained': False,
        'total_samples': total,
        'total_apnea_hyp': n_apnea,
        'total_normal': n_normal,
        'n_records': len(record_stats),
        'record_stats': record_stats,
        'preprocessing': {
            'segment_length': '1 minute (6000 samples @ 100Hz)',
            'resample': '256 Hz -> 100 Hz (resample_poly)',
            'bandpass': 'Butterworth 0.5-45 Hz, order=3',
            'normalization': 'per-segment Z-score',
            'label_rule': 'Apnea + Hypopnea events from NSRR XML, min overlap 1s with 60s window'
        },
        'training': {
            'split_method': 'Stratified 5-Fold Cross Validation (segment-level)',
            'n_folds': N_FOLDS,
            'epochs_max': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'loss': 'Weighted CrossEntropyLoss (class-balanced)',
            'scheduler': 'ReduceLROnPlateau(factor=0.2, patience=5)',
            'early_stopping_patience': PATIENCE,
            'best_model_criterion': 'val F1-score',
            'random_seed': RANDOM_SEED
        },
        'elapsed_seconds': float(elapsed),
        'device': str(DEVICE),
        'avg_val_metrics': avg_val,
        'std_val_metrics': std_val,
        'avg_train_metrics': avg_train,
        'std_train_metrics': std_train,
        'per_fold': [{
            'fold': r['fold'],
            'best_epoch': r['best_epoch'],
            'total_epochs': r['total_epochs'],
            'train_samples': r['train_samples'],
            'val_samples': r['val_samples'],
            'train_apnea': r['train_apnea'],
            'train_normal': r['train_normal'],
            'val_apnea': r['val_apnea'],
            'val_normal': r['val_normal'],
            'best_train_metrics': r['best_train_metrics'],
            'best_val_metrics': r['best_val_metrics']
        } for r in all_fold_results]
    }

    results_path = os.path.join(OUTPUT_DIR, '5fold_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")

    full_history_path = os.path.join(OUTPUT_DIR, '5fold_full_history.json')
    full_history = {
        'folds': [{
            'fold': r['fold'],
            'history': r['history']
        } for r in all_fold_results]
    }
    with open(full_history_path, 'w', encoding='utf-8') as f:
        json.dump(full_history, f, indent=2, ensure_ascii=False)
    print(f"Full history saved to: {full_history_path}")

    return results_summary

if __name__ == '__main__':
    train_5fold()
