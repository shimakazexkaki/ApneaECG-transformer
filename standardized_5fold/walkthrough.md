# 三路平行 1D CNN + Transformer — Stratified 5-Fold CV 訓練結果

## 訓練完成 ✅

- **模型**: ParallelCNNTransformer（三路平行 1D CNN + Transformer）
- **資料集**: Apnea-ECG
- **訓練時間**: 27.9 分鐘（CUDA GPU）
- **裝置**: PyTorch 2.6.0 + CUDA

---

## 一、資料集確認

| 項目 | 數值 |
|---|--:|
| **總樣本數** | 17,023 段 |
| **Apnea (陽性)** | 6,511 |
| **Normal (陰性)** | 10,512 |
| **陽性比例** | 38.2% |
| **每段長度** | 1 分鐘 (6000 點 @ 100Hz) |

### 前處理流程（已確認符合要求）
1. ✅ 只保留有標註呼吸中止的紀錄
2. ✅ 每分鐘為單位切分（6000 samples）
3. ✅ Butterworth 帶通濾波 0.5–45 Hz（去除範圍外雜訊）
4. ✅ per-segment Z-score 正規化：`segment = (segment - mean) / std`

### 資料切分（Stratified 5-Fold CV）

每 fold 的 train/val 分配：

| Fold | Train 總數 | Train Apnea | Train Normal | Val 總數 | Val Apnea | Val Normal |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 13,618 | 5,209 | 8,409 | 3,405 | 1,302 | 2,103 |
| 2 | 13,618 | 5,209 | 8,409 | 3,405 | 1,302 | 2,103 |
| 3 | 13,618 | 5,208 | 8,410 | 3,405 | 1,303 | 2,102 |
| 4 | 13,619 | 5,209 | 8,410 | 3,404 | 1,302 | 2,102 |
| 5 | 13,619 | 5,209 | 8,410 | 3,404 | 1,302 | 2,102 |

> 每 fold 的陽性比例都維持在 ~38.2%，Stratified 分層抽樣正確。

---

## 二、5-Fold 訓練成果

### 每 Fold 最佳 Validation 指標

| Fold | **F1** | **Accuracy** | **Precision** | **Recall** | Specificity | Best Epoch | Total Epochs |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | **0.8761** | 0.9028 | 0.8546 | 0.8986 | 0.9054 | 13 | 23 |
| 2 | **0.8859** | 0.9134 | 0.8924 | 0.8794 | 0.9344 | 3 | 13 |
| 3 | **0.8973** | 0.9201 | 0.8833 | 0.9117 | 0.9253 | 13 | 23 |
| 4 | **0.8929** | 0.9154 | 0.8653 | 0.9224 | 0.9110 | 10 | 20 |
| 5 | **0.8814** | 0.9116 | 0.9046 | 0.8594 | 0.9439 | 8 | 18 |

### 5-Fold 平均 ± 標準差

| 指標 | **Validation** | **Training** |
|---|---|---|
| **F1** | **0.8867 ± 0.0076** | 0.9311 ± 0.0227 |
| **Accuracy** | **0.9126 ± 0.0057** | 0.9471 ± 0.0173 |
| **Precision** | **0.8800 ± 0.0181** | 0.9263 ± 0.0255 |
| **Recall** | **0.8943 ± 0.0226** | 0.9366 ± 0.0307 |
| Specificity | 0.9240 ± 0.0143 | 0.9535 ± 0.0169 |

> [!TIP]
> **F1 的標準差僅 0.0076**，代表模型在 5 個 fold 上表現非常穩定，泛化能力良好。

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
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau(factor=0.2, patience=5) |
| Early Stopping | patience=10, 基於 val F1 |
| Batch Size | 64 |
| Max Epochs | 60 |

---

## 四、產出檔案

| 檔案 | 路徑 |
|---|---|
| 訓練腳本 | [apnea_5fold_trainer.py](file:///c:/Users/a2003/Desktop/Code/apnea/lib/apnea_5fold_trainer.py) |
| 結果 JSON | [5fold_results.json](file:///c:/Users/a2003/Desktop/Code/apnea/outputs/5fold_parallel_cnn_transformer/5fold_results.json) |
| 完整歷程 | [5fold_full_history.json](file:///c:/Users/a2003/Desktop/Code/apnea/outputs/5fold_parallel_cnn_transformer/5fold_full_history.json) |
| Fold 1 模型 | `outputs/5fold_parallel_cnn_transformer/best_model_fold0.pth` |
| Fold 2 模型 | `outputs/5fold_parallel_cnn_transformer/best_model_fold1.pth` |
| Fold 3 模型 | `outputs/5fold_parallel_cnn_transformer/best_model_fold2.pth` |
| Fold 4 模型 | `outputs/5fold_parallel_cnn_transformer/best_model_fold3.pth` |
| Fold 5 模型 | `outputs/5fold_parallel_cnn_transformer/best_model_fold4.pth` |

---

## 五、與先前版本比較

| 版本 | 切分方式 | F1 | Acc | Precision | Recall |
|---|---|--:|--:|--:|--:|
| **本次 5-Fold CV** | Stratified 5-Fold | **0.8867** | **0.9126** | **0.8800** | **0.8943** |
| 期中 80/20 split | Random 80/20 | (未保存) | (未保存) | (未保存) | (未保存) |
| 期末主模型 (CNN+Transformer, RRI特徵) | Withheld test | 0.820 | 0.868 | 0.851 | 0.790 |

> [!IMPORTANT]
> 本次 5-Fold CV 的 F1 (0.887) 高於期末主模型 (0.820)，但需注意兩者**不可直接比較**：
> - 本次是 **segment-level split**（同受試者的片段可能同時出現在 train/val）
> - 期末主模型是 **subject-level withheld test**（整位受試者 held out）
> - Segment split 通常會因受試者洩漏而虛高
