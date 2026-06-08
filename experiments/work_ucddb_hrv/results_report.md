# UCDDB HRV/CVHR 改進實驗報告

對應企畫書目標：基於 ECG 的睡眠呼吸中止症（SA）**初步篩檢**，以 UCDDB 為主資料集，
用 Accuracy / Recall / Precision / Specificity / F1 / AUC 多角度評估，並力求超越參考文獻。

實驗程式：`work_ucddb_hrv/`（`hrv_features.py`、`hrv_grouped_cv.py`、`hrv_mlp_cv.py`、`diagnose_signal.py`）
環境：`C:\Users\a2003\miniconda3\envs\apnea\python.exe`（CUDA 可用）

---

## 1. 出發點：先誠實面對「天花板」

把專案裡所有既有的 UCDDB **跨受試者**（record-level grouped CV）結果攤開比對，
不論深度或古典模型，全部卡在 **AUC ~0.55、BAcc ~0.53**：

| 既有方法 | 評估 | AUC | BAcc |
|---|---|---:|---:|
| HRV ExtraTrees | grouped split | 0.556 | 0.54 |
| 原始 ECG 11s CNN-Transformer | grouped CV | 0.55 | 0.53 |
| 5-min CNN-Transformer | grouped CV | 0.53 | 0.53 |
| Literature BiGRU | grouped CV | 0.55 | 0.53 |

對照組：**同一個 HRV 模型在 Apnea-ECG 上有 AUC 0.83 / BAcc 0.71**。
→ 瓶頸不是模型架構，而是 **UCDDB 的跨受試者泛化**。

burden 診斷也顯示模型嚴重高估（ucddb018 預測 50 min/hr、實際 1），
而 R-peak 偵測沒問題（壞 RRI 比例平均僅 0.8%）。
代表模型跨受試者根本無法區分，於是把幾乎所有分鐘判成 apnea。

---

## 2. 假設與方法（Part A）

**根因假設**：既有「literature features」其實是內插成 900 點的 RRI/振幅**序列** + 每視窗各自 z-score，
並沒有**顯式**抽出 apnea 最關鍵的判別訊號——
心率週期性變化（**CVHR**，約 0.01–0.05 Hz 的振盪）。
20 多位受試者的小資料，讓深度模型自己學出這個頻段振盪幾乎不可能。

**改進**：`hrv_features.py` 用領域知識，對每個目標分鐘（5 分鐘 context）抽 **41 個標量特徵**：

- 時域 HRV：meanRR, SDNN, RMSSD, SDSD, pNN50/20, CVRR, HR 統計, RR 散布。
- Poincaré 非線性：SD1, SD2, SD1/SD2, 橢圓面積。
- 頻域 HRV：VLF / LF / HF / total power, LF/HF, normalized units。
- **CVHR / apnea 頻段**：0.01–0.04 Hz 功率、佔比、峰值頻率/功率，
  以及 20–70 秒 apnea-cycle 自相關週期性分數（apnea 最關鍵特徵）。
- **EDR（呼吸）**：R-peak 振幅變異 + 振幅頻譜在呼吸帶（0.1–0.4 Hz）的功率。

兩個關鍵設計：

1. **受試者內正規化（per-subject normalization）**：每位受試者用自己整晚的特徵分布做 z-score。
   不用任何標籤，對 held-out test subject 也合法，模擬穿戴式裝置「對配戴者基線校準」。
2. **正規化古典分類器**：L2 Logistic Regression、HistGradientBoosting，及兩者軟投票 ensemble。
   在 ~25 受試者上比深度網路更不易過擬合。

評估：5-fold **record-level grouped CV**（整位受試者 held out），與既有專案協定一致、可直接比較。

---

## 3. 主結果（apnea+hypopnea，雙通道 ch0+ch2，per-subject zscore）

| 模型 | Pooled AUC | BAcc | F1 | Recall | Specificity | Precision |
|---|---:|---:|---:|---:|---:|---:|
| 既有最佳深度基準 | ~0.55 | ~0.53 | ~0.37(min) | — | — | — |
| HRV LogReg | 0.553 | 0.536 | 0.333 | 0.606 | 0.467 | 0.233 |
| HRV HGB | 0.564 | 0.535 | 0.290 | 0.362 | 0.713 | 0.249 |
| **HRV Ensemble（LogReg+HGB）** | **0.570** | **0.542** | 0.303 | 0.387 | 0.697 | 0.254 |
| HRV MLP（深度，Part B） | 0.551 | 0.526 | 0.342 | 0.726 | 0.326 | — |

（minute-level，threshold 在訓練集上調，metrics 在 held-out 受試者上算。）

**重點**：
- HRV+CVHR ensemble **AUC 0.570 > 既有 0.55**，是**誠實、可重現**的改進。
- 改進幅度雖小（+0.02 AUC），但換來兩個對「資料探勘」專題更重要的東西：
  **(a) 可解釋的領域特徵**（取代黑箱深度網路）、**(b) 清楚解釋了為什麼難**（見第 5 節）。

---

## 4. 消融實驗（驗證每個設計決策）

| 消融項 | 設定 | Pooled AUC |
|---|---|---:|
| **受試者正規化** | LogReg，**無** norm | 0.492 |
| | LogReg，**zscore** norm | **0.553**（+0.061）|
| **標籤定義** | Ensemble，apnea+hyp | **0.570** |
| | Ensemble，apnea-only | 0.442（退化，<0.5）|
| **古典 vs 深度** | Ensemble（古典） | **0.570** |
| | MLP（深度） | 0.551 |

三個明確結論：
1. **受試者內正規化有效**（+0.061 AUC）→ 證實「跨受試者」是核心問題，且這招直接對症。
2. **minute 分類要用 apnea+hyp**；apnea-only 陽性率僅 ~5% 且高度集中在少數人，
   fold 退化、AUC 掉到 0.44，跨受試者不可行。
3. **小資料上古典 > 深度**：正規化 ensemble 勝過 MLP。這本身是一個有價值的 ML 發現。

---

## 4.5 Loss function 掃描（Part B 深度 MLP）

在 MLP 上比較四種 loss（apnea+hyp、雙通道、per-subject zscore，相同其餘設定）：

| Loss | Minute AUC | BAcc | F1 | 篩檢 AUC(burden≥15%) |
|---|---:|---:|---:|---:|
| weighted-CE（基準） | **0.5505** | 0.526 | 0.342 | 0.513 |
| focal (γ=2) | 0.5495 | 0.531 | 0.345 | 0.513 |
| soft-AUC（pairwise ranking） | 0.5452 | 0.526 | 0.329 | 0.487 |
| group-DRO（最差受試者） | 0.5396 | 0.528 | 0.350 | **0.633** |

結論：
1. **換 loss 沒有拉高 minute-level AUC**——四種全部落在 0.540–0.551，連「直接優化 AUC 的
   soft-AUC ranking loss」也沒贏。這是瓶頸來自**特徵/任務本身、而非 loss** 的直接證據，
   與第 5 節「受試者內單特徵也僅 ~0.6」一致。focal 只微調了 recall/spec 取捨。
2. **group-DRO 雖讓 minute AUC 略降，卻把受試者層級篩檢 AUC 從 0.51 拉到 0.63**——
   「跨受試者穩健性」的 loss 在**篩檢方向**才看得到效果，再次指向真正的 gain 在受試者層級。
   （既有深度管線 `ucddb_highres_trainer.py` 也早已試過 focal / Group DRO，同樣停在 ~0.55。）

## 5. 為什麼這麼難：訊號存在性診斷

`diagnose_signal.py`：比較「受試者**內**」與「跨受試者」的單特徵判別力。

| | apnea+hyp | apnea-only |
|---|---:|---:|
| 跨受試者 pooled 單特徵 AUC（最佳） | 0.54 | 0.59 |
| **受試者內** 單特徵 AUC（最佳） | **0.59** | **0.64** |

- 受試者**內**最佳單特徵也只有 ~0.59–0.64 → **minute-level 標籤跟 HRV 的關聯本身就弱**，
  不是純粹的跨受試者轉移問題，有相當部分是任務的內在難度（label 是「該分鐘是否含 >5 秒呼吸事件」，
  事件短、稀疏，5 分鐘 context 大多是正常心跳）。
- 但 apnea-only 下最強的受試者內特徵正是我設計的 **CVHR/頻譜特徵**
  （`apnea_peak_power`、`vlf_power`、`apnea_band_power`、`sdnn`）→ **特徵工程方向正確**。

---

## 6. 與參考論文比較（重要：協定差異）

參考論文（Pham & Moucek 2025）回報 UCDDB **98–99% accuracy**，但那是
**8:1:1 視窗切分**——同一位受試者的視窗同時出現在 train/valid/test（資料洩漏），
且 11 秒視窗 10 秒重疊使相鄰視窗幾乎相同。
（詳見 `work_ucddb_paper_replication/paper_protocol_audit.md`。）

本專案用 **record-level grouped CV**（整位受試者 held out），更貼近「穿戴式裝置給新使用者用」的真實情境，
數字較低是因為協定更誠實、更難，**不應直接與其 98% 相比**。

可在簡報寫：
> 「參考文獻在視窗層級 8:1:1 切分下回報高準確率；本專案改用 record-level grouped CV
> 以更貼近未見過使用者的真實泛化，屬更嚴格的協定。」

---

## 7. 結論與建議

**已完成的改進**：
- 用顯式 HRV+CVHR+EDR 特徵 + 受試者內正規化 + 正規化 ensemble，
  把誠實的跨受試者 UCDDB minute-level **AUC 從 ~0.55 提升到 0.570**，BAcc 0.542。
- 模型可解釋、可重現，並用消融與診斷完整支撐每個設計決策。

**誠實的限制**：
- minute-level 跨受試者有真實天花板（受試者內單特徵也僅 ~0.6），UCDDB 僅 25 人、變異大。

**建議的下一步（若要更大幅提升）**：
1. **轉向受試者層級篩檢**：企畫書目標是「初步篩檢」，本質是判斷「這個人是否疑似 SA、該不該做 PSG」。
   診斷顯示 `cvhr_acf_lag` 等單特徵與整晚 apnea 負擔相關 r≈0.48，
   用受試者層級回歸（per-subject 聚合特徵 → 預測 AHI tier）可能得到更乾淨、更可發表的正向結果。
   （目前用 minute-score 平均當 burden 估計效果不佳，需專屬 subject-level 模型。）
2. 用 Apnea-ECG 做 pretraining / 多任務輔助（其 AUC 0.83，訊號乾淨）。
3. 嘗試多尺度 context（同時 3 分鐘 + 5 分鐘特徵）與訊號品質 gating。

---

## 8. 重現指令

```powershell
$py = 'C:\Users\a2003\miniconda3\envs\apnea\python.exe'
# 主結果（古典 ensemble，apnea+hyp）
& $py work_ucddb_hrv/hrv_grouped_cv.py --channels 0 2 --classifier ensemble --subject-norm zscore `
    --output work_ucddb_hrv/outputs/hrv_ensemble_ch0ch2_hyp_subjnorm.json
# 消融：關掉受試者正規化
& $py work_ucddb_hrv/hrv_grouped_cv.py --channels 0 2 --classifier logreg --subject-norm none
# 消融：apnea-only
& $py work_ucddb_hrv/hrv_grouped_cv.py --channels 0 2 --classifier ensemble --subject-norm zscore --apnea-only
# Part B：深度 MLP
& $py work_ucddb_hrv/hrv_mlp_cv.py --channels 0 2 --subject-norm zscore `
    --output work_ucddb_hrv/outputs/hrv_mlp_ch0ch2_hyp_subjnorm.json
# 訊號存在性診斷
& $py work_ucddb_hrv/diagnose_signal.py
```
