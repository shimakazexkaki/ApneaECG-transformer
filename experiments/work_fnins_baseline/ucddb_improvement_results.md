# UCDDB CNN+Transformer 改進結果（P0 調參 / P3 訊號品質 / P1 篩檢）

主角：提案的 **CNN+Transformer**（RRI+R 振幅 900×2 → CNN 下採樣 → Transformer → 分類）。
評估：**record-level grouped CV**（整位受試者 held out，誠實），與 HRV ensemble（0.570）同協定可比。
程式：`cnn_transformer_grouped_cv.py`（grouped CV + 調參 + 篩檢）、`cnn_transformer_tune.py`、`models/screening.py`。

## 誠實基準（最佳組態，30 epoch）
- `ctx5, ch0+2, d96/l2/h4/drop0.3/lr5e-4/focal γ2`
- **Pooled AUC 0.552、per-fold AUC 0.572、val-threshold BAcc 0.556**（fold 3 穩定失敗，AUC≈0.49）。

## P0 — 系統性調參（15 組，grouped CV）
**結論：預設超參就是最佳（~0.554），沒有任何一組能贏；調參無法突破 ~0.55。**
單導程（ch0）與短 context（3min）都更差。詳見 `cnn_transformer_tuning_report.md`。
→ 瓶頸是輸入特徵/表徵與跨受試者泛化，不是超參數。

## P3 — R-peak 偵測 + 訊號品質 gating
- 全資料視窗壞-RR 分布：中位數 0、平均 0.057；門檻 0.2 會丟 ~12% 的噪音分鐘。
- gate 0.2 grouped CV：**Pooled AUC 0.548**（vs 基準 0.552）→ **無增益**。
- **結論：訊號品質不是瓶頸**（與既有診斷一致：壞-RRI 比例低且與誤差不相關）。已加 `--max-bad-rr-fraction` 參數備用。

## P1 — 受試者層級 AHI 篩檢 + 校準
把逐分鐘分數聚合成整晚 apnea 負擔，評估能否分流高 AHI 受試者。

| 設定 | minute AUC | screening AUC(≥10%/15%/20%) | Pearson | Spearman |
|---|---:|---|---:|---:|
| 21 人（排除無 SA 記錄） | 0.552 | 0.35 / 0.41 / 0.39 | 0.108 | -0.039 |
| **全 25 人（含低負擔負樣本）** | 0.541 | **0.57 / 0.53 / 0.35** | 0.077 | 0.064 |

- 修正了「排除 4 個無 SA 記錄會讓篩檢無負樣本」的問題；納入全 25 人後篩檢略好但仍弱。
- **結論：CNN+Transformer 的逐分鐘分數無法聚合成有用的 AHI 估計**（0.5 門檻下幾乎全判正、mean score 跨受試者幾乎不變）。
  對照：先前 HRV+group-DRO 的篩檢 AUC 0.63 較佳，但仍未達 0.70 可用門檻。

## 總結（誠實）
- **CNN+Transformer 在 UCDDB 跨受試者有穩固的 ~0.55 天花板**；
  系統性調參、訊號品質 gating、受試者層級篩檢重構，**三者都無法突破**。
- 最佳誠實 minute-level 模型仍是可解釋的 **HRV ensemble（0.570）**，CNN+Transformer（0.552）略低。
- 瓶頸是**內在的**：UCDDB 僅 25 人、跨受試者變異大，且 minute 標籤與 ECG-derived 特徵的關聯本就弱
  （受試者內單特徵 AUC 也僅 ~0.6）。這不是調參/loss/不平衡/訊號品質能解的。
- 對專題的正面價值：(1) 誠實的 grouped-CV 協定（vs 文獻洩漏式 98%）；(2) 完整的調參與消融證明已達架構上限；
  (3) 清楚定位瓶頸在資料/任務本質，而非工程細節。

## 重現
```
python experiments/work_fnins_baseline/cnn_transformer_grouped_cv.py --channels 0 2 --epochs 30   # 基準
python experiments/work_fnins_baseline/cnn_transformer_tune.py --search-epochs 15                 # 調參
python experiments/work_fnins_baseline/cnn_transformer_grouped_cv.py --channels 0 2 --epochs 30 --max-bad-rr-fraction 0.2   # P3
python models/screening.py --include-all-records                                                  # P1（全25人）
```
