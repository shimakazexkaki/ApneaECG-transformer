# 完整結果總表:架構 × 流程 × 指標(AUC / F1 / Acc / Precision / Recall)

> ⚠️ **比較鐵則**:數字只能在**相同評估協定**內比。
> - **segment split / holdout**:同受試者片段同時進 train/test → 洩漏、虛高。
> - **grouped CV / 外部測試**:整位受試者 held out 或完全不同資料集 → 誠實。
> `*` 標示的 Precision 由 `P = F1·R/(2R−F1)` 從 F1 與 Recall 精確反推。

---

## 0. 共用流程與模型架構(下面各表共用,不重複)

### 共用前處理 → 特徵
1. 讀 ECG(UCDDB Lifecard ch0/2;Apnea-ECG lead V2;MESA EKG)→ **重採樣 100 Hz** → **Butterworth 帶通 0.5–45 Hz**。
2. **R-peak**:BioSPPy **Hamilton** + correct_rpeaks。
3. 由 R-peak 算 **RRI(clip 0.30–2.50 s)** 與 **R-peak 振幅**。
4. 每個目標分鐘取 **5 分鐘 context**,RRI 與振幅各內插成 **900 點** → 輸入 **(900, 2)**。
5. **正規化**:per-window z-score(預設)或 per-recording z-score(MESA 篩檢用,保留個體水準)。
6. **標籤**:該分鐘與呼吸事件重疊 **>5 s** 為陽性(MESA-internal 最佳版改用「10 秒段完整落在事件內=陽性、零重疊=陰性、部分丟棄」)。

### 模型架構
- **CNN-BiGRU(FNINS baseline)**:Conv1d(2→64,k7)+BN+ReLU+MaxPool2 → Conv1d(64→96,k5)+BN+ReLU+MaxPool2 → Dropout → **BiGRU(hidden 96)** → Attention pooling → Linear(192→2)。
- **CNN+Transformer(提案主模型)**:Conv1d(2→64,k7)+BN+GELU+MaxPool2 → Conv1d(64→96,k5)+BN+GELU+MaxPool2(900→225 token)→ 可學習位置編碼 → **TransformerEncoder(d=96,nhead,layers,GELU,norm_first)** → Attention pooling → Linear(96→2)。
- **M11 CNN-Transformer-LSTM(Pham&Moucek 2025 複現)**:Conv1d 64→128→128(k7,各 BN+ReLU+MaxPool4)+Dropout → 正弦位置編碼 → **1 層 Transformer(nhead8,ff256)** → **LSTM(hidden128)** → Linear(128→2)。
- **HRV+CVHR 古典 ensemble**:每分鐘 **41 個手工 HRV/CVHR/EDR 標量特徵**(5 分鐘 context)+ 受試者層級 z-score → **LogReg + HistGradientBoosting 軟投票**。
- **HRV ExtraTrees**:HRV 特徵 → 600 樹 ExtraTrees。

訓練:AdamW、AMP、loss∈{CE / weighted-CE / focal}、`WeightedRandomSampler` 過採樣、ReduceLROnPlateau、early stopping、grad-clip 1.0。

---

## 1. Apnea-ECG(乾淨 benchmark;release 訓練 / withheld x01–x35 測試)

| 模型 | Acc | Recall | Precision | F1 | Spec | AUC |
|---|--:|--:|--:|--:|--:|--:|
| **CNN-BiGRU** | **0.887** | 0.782 | 0.907 | **0.840** | 0.951 | **0.946** |
| **CNN+Transformer** | 0.868 | 0.790 | 0.851 | 0.820 | 0.916 | 0.931 |
| HRV ExtraTrees | 0.774 | 0.517 | 0.752 | 0.613 | 0.910 | 0.831 |

**per-recording AHI 篩檢(CNN+Transformer,專題主結果):**
- per-minute pooled **AUC 0.931**;apnea-minute **Pearson r 0.914 / Spearman 0.929**;**AHI index MAE 5.37/h**
- **apneic 判定準確率:AI≥5 門檻 97.1% / ≥100min 門檻 94.3%**(對照 Liu 2023:100% / MAE 4.33)
→ 深度學習在乾淨資料明顯優於 HRV;貼合 Polar H10 穿戴式篩檢。

---

## 2. UCDDB — segment split(⚠️ 同受試者洩漏,數字虛高)

### 2a. 有 Apnea-ECG 預訓練 → UCDDB fine-tune(8:1:1)
| 模型 | 標籤 | Acc | Recall | Precision\* | F1 | Spec | AUC |
|---|---|--:|--:|--:|--:|--:|--:|
| CNN-BiGRU | apnea+hyp | 0.769 | 0.563 | 0.512\* | 0.536 | 0.833 | 0.776 |
| CNN+Transformer | apnea+hyp | 0.774 | 0.446 | 0.527\* | 0.483 | 0.876 | 0.751 |
| CNN-BiGRU | apnea-only | 0.910 | 0.627 | 0.350\* | 0.449 | 0.928 | 0.858 |
| CNN+Transformer | apnea-only | 0.934 | 0.492 | 0.439\* | 0.464 | 0.961 | 0.884 |

### 2b. UCDDB-only 從頭訓練(無預訓練)
| 模型 | Loss | Acc | Recall | Precision\* | F1 | Spec | AUC |
|---|---|--:|--:|--:|--:|--:|--:|
| **M11 CNN-Transformer-LSTM(5分鐘 RRI)** | CE | 0.794 | 0.599 | 0.509\* | 0.550 | 0.846 | **0.790** |
| M11(11秒視窗,論文協定,30萬密集) | CE | 0.750 | 0.764 | 0.238\* | 0.363 | 0.749 | 0.838 |
| CNN+Transformer | plain CE | — | 0.171 | 0.308\* | 0.220 | 0.882 | 0.575 |
| CNN+Transformer | weighted CE | — | 0.683 | 0.312\* | 0.428 | 0.531 | 0.676 |
| CNN+Transformer | focal γ2 | — | 1.000 | 0.238\* | 0.385 | 0.006 | 0.677 |
| CNN-BiGRU | plain CE | — | 0.000 | 0.000 | 0.000 | 1.000 | 0.481(崩潰) |
| CNN-BiGRU | weighted CE | — | 0.583 | 0.358\* | 0.444 | 0.676 | 0.680 |
| CNN-BiGRU | focal γ2 | — | 0.921 | 0.280\* | 0.429 | 0.262 | 0.729 |

> 重點:plain CE 讓從頭深度模型崩潰(全判正常);weighted-CE/focal 救回。論文 98% 來自 11秒視窗+10秒重疊的洩漏。

---

## 3. UCDDB — grouped CV(✅ 誠實,整位受試者 held out)

> 此協定只存了 AUC / BAcc / F1(per-fold 平均);Recall/Precision 當時未逐列保存(標 —,需重跑補)。

| 模型 | 標籤/方法 | BAcc | F1 | AUC |
|---|---|--:|--:|--:|
| **CNN+Transformer(提案,調參後最佳)** | ctx5, ch0+2, focal γ2 | 0.556 | 0.373 | 0.552 |
| 5-min CNN-Transformer | context3, focal γ2 | 0.535 | 0.251 | 0.526 |
| Literature BiGRU | RRI/amp 序列 | 0.532 | 0.250 | 0.552 |
| M11 CNN-Transformer-LSTM(5分鐘) | grouped 5-fold | 0.513 | 0.265 | 0.537 |
| M11(11秒視窗,論文協定) | grouped 5-fold | 0.485 | 0.129 | 0.511(近隨機) |
| High-res 11s CNN-Transformer | minute 聚合 | 0.536 | 0.370 | 0.552 |
| **HRV+CVHR Ensemble(誠實基準最佳)** | LogReg+HGB 軟投票 | 0.542 | 0.303 | **0.570** |
| HRV+CVHR HGB | 梯度提升 | 0.535 | 0.290 | 0.564 |
| HRV+CVHR LogReg | 受試者正規化 | 0.536 | 0.333 | 0.553 |
| HRV ExtraTrees | 手工 HRV | 0.540 | — | 0.556 |

**含 Recall/Precision 的不平衡對照(HGB, grouped CV):**
| 方法 | Recall | Precision\* | F1 | Spec | AUC |
|---|--:|--:|--:|--:|--:|
| 類別加權 | 0.362 | 0.242\* | 0.290 | 0.708 | 0.564 |
| SMOTE | 0.466 | 0.233\* | 0.311 | 0.600 | 0.551 |
| 欠採樣 | 0.354 | 0.240\* | 0.286 | 0.718 | 0.563 |
| EasyEnsemble | 0.349 | 0.243\* | 0.287 | 0.728 | 0.562 |

> **誠實天花板 ~0.55–0.57**:所有深度與古典方法都卡在此;調參、loss、不平衡處理皆動不了 AUC(只移動操作點)。

---

## 4. MESA → UCDDB 跨資料集(train on MESA / test on 全 UCDDB,零洩漏)

| MESA 訓練人數 N | Acc | Recall | Precision\* | F1 | Spec | AUC |
|--:|--:|--:|--:|--:|--:|--:|
| 14 | 0.519 | 0.577 | 0.237\* | 0.336 | 0.503 | 0.552 |
| 44 | 0.740 | 0.211 | 0.322\* | 0.255 | 0.881 | 0.561 |
| 74 | 0.366 | 0.784 | 0.220\* | 0.343 | 0.255 | 0.547 |

> 跨資料集 AUC 抖動於 0.55,未隨 N 上升 → **domain shift(不同儀器/族群/導程)為主因**。

---

## 5. MESA-internal(同資料集,grouped CV;方法改進 + 規模)

模型:CNN+Transformer(消融證明 LSTM 多餘);標籤:10秒乾淨段。

| 設定 | per-segment AUC | F1 | 備註 |
|---|--:|--:|---|
| minute 標籤(原 M11) | 0.631 | 0.265 | 基準 |
| 乾淨 10s-segment 標籤 | 0.678 | ~0.13 | +0.047 |
| + RRID 第三通道 | 0.665 | — | 無效 |
| 乾淨 + recording-norm + CNN+Transformer,**74 人** | 0.709 | 0.131 | |
| 乾淨 + recording-norm + CNN+Transformer,**149 人** | **0.735** | ~0.16 | **規模有效 +0.026** |

> F1 偏低(0.13–0.16)是因陽性率僅 ~4%(乾淨段),屬結構性;AUC 是鑑別力指標。

**per-subject AHI 篩檢(Olsen-style,149 人):** 嚴重度準確率 **0.40**、負荷 **Spearman 0.185(p=0.024 顯著)**、pooled AUC 0.731。
**層級消融(clean-seg,3-fold):** full 0.646 / no_lstm(CNN+T)0.631 / no_transformer(CNN+LSTM)0.626 / cnn_only 0.620 → 各層僅 1–2%,**LSTM 多餘**。
**文獻天花板:** Olsen 2020(MESA+SHHS 近萬筆,RR+EDR+BiGRU)= per-subject AHI 嚴重度 **Acc 0.849 / R² 0.83 / per-event F1 0.67**。

---

## 6. 一句話總結

| 情境 | 最佳模型 | 關鍵指標 |
|---|---|---|
| **乾淨 benchmark(Apnea-ECG / Polar H10 相似)** | CNN+Transformer | per-minute AUC **0.931** / F1 0.820;**per-recording 篩檢 97.1%** |
| **誠實小世代(UCDDB grouped CV)** | HRV+CVHR Ensemble | AUC **0.570**(深度模型 ~0.55,HRV 略勝且可解釋) |
| **大世代同資料(MESA-internal)** | CNN+Transformer(clean-seg+rec-norm) | per-segment AUC **0.735**(149 人,規模有效) |
| **跨資料集(MESA→UCDDB)** | — | AUC ~0.55(domain shift 限制) |

**核心結論**:資料乾淨/量大 → 深度學習勝(Apnea-ECG 0.93–0.95);資料小/雜/跨人 → 深度優勢消失、HRV 特徵追平且更可解釋可部署(見 `experiments/work_ucddb_hrv/hrv_vs_deeplearning.md`)。

詳見:`experiments/work_ucddb_hrv/model_comparison.md`、`experiments/work_mesa_transfer/mesa_transfer_results.md`。
