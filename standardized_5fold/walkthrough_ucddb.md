# 三路平行 1D CNN + Transformer — UCDDB Stratified 5-Fold CV 訓練結果

## 訓練完成

- **模型**: ParallelCNNTransformer（三路平行 1D CNN + Transformer）
- **資料集**: UCDDB（從頭訓練，無預訓練）
- **訓練時間**: 14.9 分鐘（CUDA GPU）
- **裝置**: PyTorch + CUDA

---

## 一、資料集確認

| 項目 | 數值 |
|---|--:|
| **總樣本數** | 12,206 段 |
| **Apnea/Hypopnea (陽性)** | 2,677 |
| **Normal (陰性)** | 9,529 |
| **陽性比例** | 21.9% |
| **每段長度** | 1 分鐘 (6000 點 @ 100Hz) |
| **受試者數** | 25 人 |

### 各受試者統計

| 受試者 | 總段數 | Apnea/Hyp | Normal |
|---|--:|--:|--:|
| ucddb002 | 459 | 110 | 349 |
| ucddb003 | 485 | 214 | 271 |
| ucddb005 | 486 | 51 | 435 |
| ucddb006 | 507 | 130 | 377 |
| ucddb007 | 495 | 79 | 416 |
| ucddb008 | 502 | 17 | 485 |
| ucddb009 | 483 | 70 | 413 |
| ucddb010 | 489 | 127 | 362 |
| ucddb011 | 490 | 37 | 453 |
| ucddb012 | 495 | 143 | 352 |
| ucddb013 | 511 | 70 | 441 |
| ucddb014 | 451 | 187 | 264 |
| ucddb015 | 517 | 41 | 476 |
| ucddb017 | 458 | 68 | 390 |
| ucddb018 | 504 | 8 | 496 |
| ucddb019 | 497 | 90 | 407 |
| ucddb020 | 471 | 75 | 396 |
| ucddb021 | 506 | 80 | 426 |
| ucddb022 | 470 | 35 | 435 |
| ucddb023 | 521 | 152 | 369 |
| ucddb024 | 501 | 138 | 363 |
| ucddb025 | 493 | 315 | 178 |
| ucddb026 | 469 | 90 | 379 |
| ucddb027 | 495 | 180 | 315 |
| ucddb028 | 451 | 170 | 281 |

### 前處理流程

1. 從 UCDDB Lifecard EDF 讀取 ECG 通道（ch0）
2. 重新取樣：128 Hz → 100 Hz（resample_poly）
3. 每分鐘為單位切分（6000 samples）
4. Butterworth 帶通濾波 0.5–45 Hz（去除範圍外雜訊）
5. per-segment Z-score 正規化：`segment = (segment - mean) / std`
6. 標籤規則：APNEA + HYP 事件，與 60 秒窗口重疊 ≥ 1 秒 → 陽性

### 資料切分（Stratified 5-Fold CV，Segment-level）

每 fold 的 train/val 分配：

| Fold | Train 總數 | Train Apnea/Hyp | Train Normal | Val 總數 | Val Apnea/Hyp | Val Normal |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 9,764 | 2,141 | 7,623 | 2,442 | 536 | 1,906 |
| 2 | 9,765 | 2,142 | 7,623 | 2,441 | 535 | 1,906 |
| 3 | 9,765 | 2,142 | 7,623 | 2,441 | 535 | 1,906 |
| 4 | 9,765 | 2,142 | 7,623 | 2,441 | 535 | 1,906 |
| 5 | 9,765 | 2,141 | 7,624 | 2,441 | 536 | 1,905 |

> 每 fold 的陽性比例都維持在 ~21.9%，Stratified 分層抽樣正確。

---

## 二、5-Fold 訓練成果

### 每 Fold 最佳 Validation 指標

| Fold | **F1** | **Accuracy** | **Precision** | **Recall** | Specificity | Best Epoch | Total Epochs |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | **0.5300** | 0.7305 | 0.4294 | 0.6922 | 0.7413 | 8 | 18 |
| 2 | **0.5130** | 0.7091 | 0.4052 | 0.6991 | 0.7120 | 12 | 22 |
| 3 | **0.4911** | 0.6731 | 0.3727 | 0.7196 | 0.6600 | 13 | 23 |
| 4 | **0.4970** | 0.6899 | 0.3856 | 0.6991 | 0.6873 | 10 | 20 |
| 5 | **0.5160** | 0.7517 | 0.4511 | 0.6026 | 0.7937 | 14 | 24 |

### 5-Fold 平均 ± 標準差

| 指標 | **Validation** | **Training** |
|---|---|---|
| **F1** | **0.5094 ± 0.0139** | 0.5524 ± 0.0219 |
| **Accuracy** | **0.7109 ± 0.0280** | 0.7361 ± 0.0284 |
| **Precision** | **0.4088 ± 0.0285** | 0.4437 ± 0.0371 |
| **Recall** | **0.6825 ± 0.0410** | 0.7403 ± 0.0560 |
| Specificity | 0.7189 ± 0.0461 | 0.7349 ± 0.0479 |

> **F1 僅 0.509** — UCDDB 信號品質較差（Holter 等級，基線飄移、QRS 畸形），且陽性比例僅 21.9%（嚴重不平衡），即使使用 Weighted CE 仍難以有效學習。Precision 僅 ~0.41，表示模型傾向過度預測陽性以換取 Recall。

---

## 三、模型架構確認

```
輸入 (1, 6000) — 原始 ECG 波形，1 分鐘 @ 100Hz
  │
  ├── Path 1: Conv1d(1→48, k=300) + ReLU + MaxPool2   ← 長期心率趨勢 (~3秒)
  ├── Path 2: Conv1d(1→48, k=30)  + ReLU + MaxPool2   ← 中期特徵 (~0.3秒)
  └── Path 3: Conv1d(1→48, k=10)  + ReLU + MaxPool2   ← R 波細節 (~0.1秒)
       │
       Concatenation → (144, ~3000)
       │
       MaxPool(2)
       │
       Conv1d(144→64, k=3) + BN + LeakyReLU + MaxPool(2)  ← 降維卷積
       │
       Conv1d(64→64, k=3) + BN + LeakyReLU + Dropout(0.3) + Skip Connection  ← 殘差 Dense Block
       │
       Global Average Pooling → (64, 64)
       │
       + Positional Embedding
       │
       Transformer Block (d=64, nhead=8, ff=256)
       │
       Mean Pooling → (64,)
       │
       Dropout(0.3) → Linear(64→2) → 二分類輸出
```

### 訓練參數

| 參數 | 值 |
|---|---|
| Optimizer | Adam, lr=0.001 |
| Loss | **Weighted CrossEntropyLoss**（類別平衡加權） |
| Scheduler | ReduceLROnPlateau(factor=0.2, patience=5) |
| Early Stopping | patience=10, 基於 val F1 |
| Batch Size | 64 |
| Max Epochs | 60 |

> 因 UCDDB 陽性比例僅 21.9%，使用 class-balanced weighted CE 來處理類別不平衡。

---

## 四、產出檔案

| 檔案 | 路徑 |
|---|---|
| 訓練腳本 | `lib/ucddb_5fold_trainer.py` |
| 結果 JSON | `outputs/5fold_ucddb_parallel_cnn_transformer/5fold_results.json` |
| Fold 1 模型 | `outputs/5fold_ucddb_parallel_cnn_transformer/best_model_fold0.pth` |
| Fold 2 模型 | `outputs/5fold_ucddb_parallel_cnn_transformer/best_model_fold1.pth` |
| Fold 3 模型 | `outputs/5fold_ucddb_parallel_cnn_transformer/best_model_fold2.pth` |
| Fold 4 模型 | `outputs/5fold_ucddb_parallel_cnn_transformer/best_model_fold3.pth` |
| Fold 5 模型 | `outputs/5fold_ucddb_parallel_cnn_transformer/best_model_fold4.pth` |

---

## 五、與 Apnea-ECG 比較

| 資料集 | 切分方式 | F1 | Acc | Precision | Recall |
|---|---|--:|--:|--:|--:|
| **Apnea-ECG** | Stratified 5-Fold | **0.8867** | **0.9126** | **0.8800** | **0.8943** |
| **UCDDB** | Stratified 5-Fold | 0.5094 | 0.7109 | 0.4088 | 0.6825 |

> UCDDB 在相同模型架構下 F1 僅 0.509，遠低於 Apnea-ECG 的 0.887。主要原因：
> 1. **信號品質差異**：UCDDB 來自 Holter 記錄器，基線飄移嚴重（~5mV DC offset）、QRS 波形畸變；Apnea-ECG 來自實驗室 PSG，信號乾淨。
> 2. **類別不平衡更嚴重**：UCDDB 陽性僅 21.9%（Apnea-ECG 為 38.2%）。
> 3. **資料量較少**：UCDDB 僅 12,206 段（Apnea-ECG 有 17,023 段），25 位受試者的個體差異更難被模型學習。
