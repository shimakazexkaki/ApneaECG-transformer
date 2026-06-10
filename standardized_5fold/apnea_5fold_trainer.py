"""
三路平行 1D CNN + Transformer — Stratified 5-Fold Cross Validation 訓練腳本

基於 apnea_trainer.py 的 ParallelCNNTransformer 模型，
改用 Stratified 5-Fold CV 進行訓練與驗證。

架構: 三平行卷積路徑 (Parallel CNN) + Transformer
資料切分: Stratified 5-Fold Cross Validation
前處理: Butterworth 0.5-45Hz 帶通 + per-segment Z-score
"""

import os
import sys
import json
import time
import wfdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from scipy.signal import butter, lfilter

# ==========================================
# 1. 參數設定
# ==========================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'apnea-ecg')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', '5fold_parallel_cnn_transformer')
FS = 100           # 採樣率 100 Hz
MINUTE_SAMPLES = FS * 60  # 每分鐘 6000 個取樣點
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
PATIENCE = 10      # Early Stopping patience
N_FOLDS = 5
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 資料集準備與濾波 (與 apnea_trainer.py 一致)
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

class ApneaECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # shape: (N, 1, 6000)
        self.y = torch.LongTensor(y)                 # shape: (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    """讀取 Apnea-ECG 資料, 濾波 0.5-45Hz, Z-score 正規化"""
    print("正在載入 Apnea-ECG 資料集 (Butterworth 0.5-45Hz)...")
    X_all, y_all = [], []

    list_path = os.path.join(DATA_DIR, 'list')
    with open(list_path, 'r') as f:
        records = [line.strip() for line in f.readlines() if line.strip()]

    # 只保留有標註呼吸中止的紀錄 (有 .apn 檔案)
    annotated_records = [r for r in records if os.path.exists(os.path.join(DATA_DIR, f"{r}.apn"))]
    print(f"找到 {len(annotated_records)} 筆具有標記的紀錄。")

    for record in tqdm(annotated_records, desc="資料處理進度"):
        record_path = os.path.join(DATA_DIR, record)
        try:
            signal, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'apn')

            ecg_signal = signal[:, 0]
            sample_pts = annotation.sample
            labels = annotation.symbol

            for pt, label in zip(sample_pts, labels):
                if label not in ['A', 'N']:
                    continue
                if pt < 0 or pt + MINUTE_SAMPLES > len(ecg_signal):
                    continue

                segment = ecg_signal[pt : pt + MINUTE_SAMPLES]
                # 帶通濾波 0.5-45Hz
                segment = butter_bandpass_filter(segment, 0.5, 45.0, FS, order=3)
                # Z-score 正規化
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)

                X_all.append(segment)
                y_all.append(1 if label == 'A' else 0)

        except Exception as e:
            print(f"處理 {record} 時發生錯誤: {e}")

    return np.array(X_all), np.array(y_all)

# ==========================================
# 3. 模型架構 (直接從 apnea_trainer.py 匯入，保持一致)
# ==========================================
# 為了獨立執行，這裡重新定義（與 apnea_trainer.py 完全相同）

class ParallelCNNBlock(nn.Module):
    """三條平行卷積路徑，分別擷取不同尺度的 ECG 特徵"""
    def __init__(self):
        super(ParallelCNNBlock, self).__init__()
        # 路徑 1: 大 kernel (捕捉長期心率趨勢, ~3秒)
        self.path1 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=300, stride=1, padding=150),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 路徑 2: 中 kernel (捕捉中期特徵, ~0.3秒)
        self.path2 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=30, stride=1, padding=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 路徑 3: 小 kernel (捕捉 R 波細節, ~0.1秒)
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
        p1 = p1[:, :, :min_len]
        p2 = p2[:, :, :min_len]
        p3 = p3[:, :, :min_len]

        out = torch.cat([p1, p2, p3], dim=1)  # [batch, 144, min_len]
        return out

class TransformerBlock(nn.Module):
    """Transformer Block"""
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
    """
    完整架構:
    Parallel CNN (3 paths) → Concatenation → MaxPool →
    降維卷積 → 殘差 Dense Block → GAP →
    Transformer → 二分類輸出
    """
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
# 4. 評估函數
# ==========================================
def evaluate_metrics(y_true, y_pred):
    """計算所有指標"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
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
    """對一個 DataLoader 跑完整評估"""
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
# 5. 單 Fold 訓練
# ==========================================
def train_single_fold(fold_idx, X_train, y_train, X_val, y_val, output_dir):
    """訓練單一 fold，回傳最佳 validation 指標"""
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx + 1}/{N_FOLDS}")
    print(f"  Train: {len(X_train)} (Apnea={sum(y_train==1)}, Normal={sum(y_train==0)})")
    print(f"  Val:   {len(X_val)} (Apnea={sum(y_val==1)}, Normal={sum(y_val==0)})")
    print(f"{'='*60}")

    train_loader = DataLoader(ApneaECGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ApneaECGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = ParallelCNNTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    best_val_f1 = 0
    best_val_metrics = None
    best_train_metrics = None
    patience_counter = 0
    fold_history = []

    model_save_path = os.path.join(output_dir, f'best_model_fold{fold_idx}.pth')

    for epoch in range(1, EPOCHS + 1):
        # --- Training ---
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

        # --- Evaluation on training set ---
        train_metrics = evaluate_on_loader(model, train_loader, DEVICE)

        # --- Evaluation on validation set ---
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

        # --- Best model selection (by val F1) ---
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
# 6. 主訓練流程 (5-Fold CV)
# ==========================================
def train_5fold():
    print(f"使用裝置: {DEVICE}")
    print(f"隨機種子: {RANDOM_SEED}")

    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 載入全部資料
    X, y = load_data()
    total = len(X)
    n_apnea = int(sum(y == 1))
    n_normal = int(sum(y == 0))
    print(f"\n資料總計: {total} 段 | Apnea: {n_apnea} | Normal: {n_normal}")
    print(f"Apnea 比例: {n_apnea/total*100:.1f}%")

    # 2. Stratified 5-Fold CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_fold_results = []
    start_time = time.time()

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        fold_result = train_single_fold(fold_idx, X_train, y_train, X_val, y_val, OUTPUT_DIR)
        all_fold_results.append(fold_result)

    elapsed = time.time() - start_time

    # 3. 計算 5-Fold 平均與標準差
    val_metrics_keys = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    avg_val = {}
    std_val = {}
    avg_train = {}
    std_train = {}

    for key in val_metrics_keys:
        vals = [r['best_val_metrics'][key] for r in all_fold_results]
        avg_val[key] = float(np.mean(vals))
        std_val[key] = float(np.std(vals))

        trains = [r['best_train_metrics'][key] for r in all_fold_results]
        avg_train[key] = float(np.mean(trains))
        std_train[key] = float(np.std(trains))

    # 4. 印出結果
    print(f"\n{'='*70}")
    print(f"  Stratified 5-Fold Cross Validation 結果")
    print(f"{'='*70}")

    print(f"\n資料集: Apnea-ECG | 總共 {total} 段 (Apnea {n_apnea}, Normal {n_normal})")
    print(f"模型: ParallelCNNTransformer (三路平行 1D CNN + Transformer)")
    print(f"訓練時間: {elapsed/60:.1f} 分鐘")

    print(f"\n--- 每 Fold 最佳 Validation 指標 ---")
    print(f"{'Fold':>6} | {'Acc':>8} | {'Precision':>10} | {'Recall':>8} | {'Spec':>8} | {'F1':>8} | {'Train F1':>10} | {'Epoch':>6}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")
    for r in all_fold_results:
        vm = r['best_val_metrics']
        tm = r['best_train_metrics']
        print(f"  {r['fold']+1:>4} | {vm['accuracy']:>8.4f} | {vm['precision']:>10.4f} | "
              f"{vm['recall']:>8.4f} | {vm['specificity']:>8.4f} | {vm['f1']:>8.4f} | "
              f"{tm['f1']:>10.4f} | {r['best_epoch']:>6}")

    print(f"\n--- 5-Fold 平均 ± 標準差 ---")
    print(f"  Validation:")
    for key in val_metrics_keys:
        print(f"    {key:>12}: {avg_val[key]:.4f} ± {std_val[key]:.4f}")
    print(f"  Training:")
    for key in val_metrics_keys:
        print(f"    {key:>12}: {avg_train[key]:.4f} ± {std_train[key]:.4f}")

    # 5. 存結果到 JSON
    results_summary = {
        'model': 'ParallelCNNTransformer',
        'architecture': '三路平行 1D CNN (k=300/30/10, 48ch each) → Concat → MaxPool → Conv1d(144→64) → ResidualDense → GAP → Transformer(d=64,nhead=8) → Linear(2)',
        'dataset': 'Apnea-ECG',
        'total_samples': total,
        'total_apnea': n_apnea,
        'total_normal': n_normal,
        'preprocessing': {
            'segment_length': '1 minute (6000 samples @ 100Hz)',
            'bandpass': 'Butterworth 0.5-45 Hz, order=3',
            'normalization': 'per-segment Z-score',
            'only_annotated': True
        },
        'training': {
            'split_method': 'Stratified 5-Fold Cross Validation',
            'n_folds': N_FOLDS,
            'epochs_max': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'optimizer': 'Adam',
            'loss': 'CrossEntropyLoss',
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
    print(f"\n結果已儲存至: {results_path}")

    # 6. 也存完整 history (含每 epoch 的紀錄)
    full_history_path = os.path.join(OUTPUT_DIR, '5fold_full_history.json')
    full_history = {
        'folds': [{
            'fold': r['fold'],
            'history': r['history']
        } for r in all_fold_results]
    }
    with open(full_history_path, 'w', encoding='utf-8') as f:
        json.dump(full_history, f, indent=2, ensure_ascii=False)
    print(f"完整訓練歷程已儲存至: {full_history_path}")

    return results_summary

if __name__ == '__main__':
    train_5fold()
