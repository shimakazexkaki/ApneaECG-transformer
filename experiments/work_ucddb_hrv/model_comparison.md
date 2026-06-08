# 模型總覽：訓練方法與各資料集表現

> ⚠️ **最重要原則**：數字只能在**相同評估協定**內比較。
> - **Holdout / segment split**：同一受試者的片段同時出現在 train/test → **資料洩漏、數字虛高**。
> - **Grouped CV**：整位受試者 held out → **誠實**、貼近真實穿戴式裝置給新使用者用。
> 不要把這兩欄的數字直接相比。

---

## 1. Apnea-ECG（資料乾淨，模型表現好）

評估：release 記錄訓練、withheld `x01–x35` 測試（標準 PhysioNet 協定）。

| 模型 | 訓練方法 | Acc | Recall | Spec | Prec | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| **FNINS CNN-BiGRU**（baseline） | RRI+amp→CNN-BiGRU+attention, CE, 40ep | **0.887** | 0.782 | 0.951 | 0.907 | **0.840** | **0.946** |
| CNN+Transformer+temporal | 同前處理, transformer 分支 | 0.868 | 0.790 | 0.916 | 0.851 | 0.820 | 0.931 |
| HRV ExtraTrees | 手工 HRV 特徵, 600 樹 | 0.774 | 0.517 | 0.910 | 0.752 | 0.613 | 0.831 |

→ 在 Apnea-ECG 上 AUC 0.83–0.95，**訊號乾淨、模型有效**。這是後續 UCDDB 失敗的對照組。

**Apnea-ECG per-recording AHI 篩檢(CNN+Transformer,withheld x01-x35,專題主結果):**
per-minute pooled AUC **0.931**;apnea-minute 相關 **r=0.914 / Spearman 0.929**;**AHI index MAE 5.37/h**;
**apneic 判定準確率 AI≥5 門檻 97.1%、≥100min 門檻 94.3%**(對照 Liu 2023:per-recording 100% / AHI MAE 4.33)。
→ 貼合 Polar H10 穿戴式篩檢情境(波形與 Apnea-ECG 相似);這是比逐分鐘更有臨床意義的交付。

---

## 2. UCDDB — Holdout / segment split（⚠️ 樂觀，有同受試者洩漏）

| 模型 | 標籤 | 訓練方法 | Acc | Recall | Spec | F1 | AUC |
|---|---|---|---:|---:|---:|---:|---:|
| FNINS CNN-BiGRU | apnea+hyp | Apnea 預訓練→UCDDB fine-tune, 8:1:1 | 0.769 | 0.563 | 0.833 | 0.536 | 0.776 |
| CNN+Transformer | apnea+hyp | 同上 | 0.774 | 0.446 | 0.876 | 0.483 | 0.751 |
| FNINS CNN-BiGRU | apnea-only | 同上 | 0.910 | 0.627 | 0.928 | 0.449 | 0.858 |
| CNN+Transformer | apnea-only | 同上 | 0.934 | 0.492 | 0.961 | 0.464 | 0.884 |
| **M11 CNN-Transformer-LSTM（5分鐘 RRI，UCDDB-only 從頭，40ep）** | apnea+hyp | segment split 8:1:1 | 0.794 | 0.599 | 0.846 | 0.550 | **0.790** |
| **M11 CNN-Transformer-LSTM（11秒視窗，論文協定，30萬密集）** | second-level | segment split 8:1:1 | 0.750 | 0.764 | 0.749 | 0.363 | **0.838** |
| hybrid_transformer | apnea+hyp | segment split | — | — | — | — | 0.806 |

> ⚠️ 論文 Pham&Moucek 回報 UCDDB **98–99%**，是因為他們用 **11 秒視窗 + 10 秒重疊**，視窗化後才 8:1:1 →
> 相鄰視窗近重複、同病患近重複樣本同時進 train/test，洩漏更嚴重。本版用 5 分鐘 RRI（較不重複）→ segment split 0.79。

→ AUC 0.70–0.88 看似不錯，但**因為 train/test 同受試者洩漏而虛高**，不能代表真實泛化。

---

## 2.5 UCDDB-only 從頭訓練（零 Apnea 預訓練）— 證明預訓練的必要性

同樣 segment split（8:1:1），但**不**用 Apnea 預訓練、完整在 UCDDB 從頭訓練 40 epoch（plain CE）：

同樣 segment split（8:1:1），但**不**用 Apnea 預訓練、完整在 UCDDB 從頭訓練 40 epoch，**比較三種 loss**：

| 模型 | Loss | AUC | F1 | Recall | Spec |
|---|---|---:|---:|---:|---:|
| CNN+Transformer | plain CE | 0.575 | 0.220 | 0.171 | 0.882 |
| CNN+Transformer | weighted CE | **0.676** | 0.428 | 0.683 | 0.531 |
| CNN+Transformer | focal γ2 | 0.677 | 0.385 | 1.000 | 0.006 |
| CNN-BiGRU | plain CE | **0.481**（崩潰） | 0.000 | 0.000 | 1.000 |
| CNN-BiGRU | weighted CE | 0.680 | 0.444 | 0.583 | 0.676 |
| CNN-BiGRU | focal γ2 | **0.729** | 0.429 | 0.921 | 0.262 |

對照（有 Apnea 預訓練，同 split，見第 2 節）：CNN-BiGRU AUC 0.776、CNN+Transformer 0.751。

**關鍵發現：**
1. **plain CE 是讓從頭訓練崩潰的元兇**；換 weighted CE / focal 大幅救回——
   CNN-BiGRU 從 AUC 0.48（全判正常）→ focal **0.729**；CNN+Transformer 0.575 → 0.68。
2. **這與「HRV 模型換 loss 沒用」互補不矛盾**：差別在基準有沒有處理不平衡。
   HRV 古典/MLP 本來就加權 → 換 loss 不動天花板；從頭深度模型用 plain CE 會崩 → 正確 loss 是必需品。
   **loss 的價值取決於起點。**
3. **加上正確 loss 後，從頭訓練（0.68–0.73）幾乎追平有 Apnea 預訓練的版本（0.75–0.78）**——
   在此 split 上正確 loss 大致可取代預訓練。
4. ⚠️ 以上為 segment split（同受試者洩漏），數字樂觀；誠實 grouped CV 仍約 0.55–0.57。
   focal 兩列 recall 0.92–1.0 / spec 很低，是 0.5 門檻操作點問題（AUC 不受影響）。

## 2.6 標註門檻 × 操作點 消融（M11, UCDDB-only, segment split）

問題：(a) 調權重/閾值能不能更平衡 F1？(b)「一分鐘 ≥5 秒重疊就標 apnea」這個規則該不該改？

| overlap | 陽性率 | AUC | 閾值 | Acc | Rec | Prec | F1 | Spec |
|---|--:|--:|---|--:|--:|--:|--:|--:|
| 5s | 0.210 | 0.789 | 預設 | 0.747 | 0.686 | 0.435 | 0.532 | 0.763 |
| 5s | | | F1-opt(0.60) | 0.801 | 0.564 | 0.524 | **0.543** | 0.864 |
| 15s | 0.128 | 0.803 | 預設 | 0.823 | 0.624 | 0.383 | 0.475 | 0.852 |
| 30s | 0.010 | 0.850 | 預設 | 0.872 | 0.553 | 0.042 | 0.078 | 0.875 |
| 30s | | | F1-opt(0.97) | 0.965 | 0.362 | 0.110 | 0.168 | 0.971 |

**結論：**
1. **操作點(閾值/權重)只能微調 F1**：overlap 5s 下 F1-最佳閾值把 precision 0.435→0.524、F1 0.532→**0.543**，更平衡但幅度小；AUC 不受閾值影響。
2. **標註越嚴 → 陽性率暴跌、AUC 微升、F1 崩潰**：5→15→30s 陽性率 21%→13%→**1%**，AUC 0.79→0.80→0.85，F1 0.53→0.48→**0.08**。
3. **30s 病態**(1% 正樣本,F1 0.08;AUC 0.85 只是剩餘正樣本好認)。
4. **「high AUC / low F1」是低陽性率的必然,不是標註 bug**。現用的 **5 秒已是 F1 最佳點**；要動應往更寬鬆(3 秒)試,不是收緊。

## 3. UCDDB — Grouped CV（✅ 誠實，整位受試者 held out）

這是專案的**核心誠實基準**。所有方法（深度＋古典）都卡在 **AUC ~0.52–0.57、BAcc ~0.53**。

| 模型 | 標籤 | 訓練方法 | BAcc | AUC | F1 |
|---|---|---|---:|---:|---:|
| **CNN+Transformer（提案模型，調參後最佳）** | apnea+hyp | ctx5, ch0+2, focal γ2, 30ep（誠實 grouped CV）| **0.556** | 0.552(pooled)/0.572(per-fold) | 0.373 |
| 5-min CNN-Transformer | apnea+hyp | context3, focal γ2, 5-fold | 0.535 | 0.526 | 0.251 |
| Literature BiGRU | apnea+hyp | RRI/amp 序列, 5-fold | 0.532 | 0.552 | 0.250 |
| hybrid_transformer | apnea+hyp | RRI+raw, grouped CV | 0.512 | 0.501 | — |
| High-res 11s CNN-Transformer | apnea+hyp | 原始 ECG 視窗, minute 聚合 | 0.536 | 0.552 | 0.370 |
| **M11 CNN-Transformer-LSTM（5分鐘 RRI，40ep）** | apnea+hyp | grouped 5-fold | **0.513** | **0.537**（per-fold 0.41–0.63）| 0.265 |
| **M11 CNN-Transformer-LSTM（11秒視窗，論文協定）** | second-level | grouped 5-fold | 0.485 | **0.511**（近隨機）| 0.129 |
| HRV ExtraTrees | apnea+hyp | 手工 HRV, 單一 grouped split | 0.540 | 0.556 | — |
| **HRV+CVHR LogReg**【新】 | apnea+hyp | 41 特徵+受試者正規化 | 0.536 | 0.553 | 0.333 |
| **HRV+CVHR HGB**【新】 | apnea+hyp | 同上, 梯度提升 | 0.535 | 0.564 | 0.290 |
| **HRV+CVHR Ensemble**【新,最佳】 | apnea+hyp | LogReg+HGB 軟投票 | **0.542** | **0.570** | 0.303 |
| HRV+CVHR MLP【新】 | apnea+hyp | 深度 MLP, weighted-CE | 0.526 | 0.551 | 0.342 |
| HRV+CVHR Ensemble | apnea-only | 同上（退化） | 0.48 | 0.442 | 0.05 |

**我的新工作把誠實基準從 ~0.55 → 0.570（AUC），並換成可解釋的領域特徵。**

---

## 4. Loss function 對照（MLP, apnea+hyp, grouped CV）

| Loss | Minute AUC | BAcc | F1 | 篩檢 AUC(≥15%) |
|---|---:|---:|---:|---:|
| 純 CE（未加權） | 0.5259 | 0.524 | 0.349 | — |
| weighted-CE | **0.5505** | 0.526 | 0.342 | 0.513 |
| focal γ2 | 0.5495 | 0.531 | 0.345 | 0.513 |
| soft-AUC ranking | 0.5452 | 0.526 | 0.329 | 0.487 |
| group-DRO | 0.5396 | 0.528 | 0.350 | **0.633** |

→ 純 CE 最差（未處理不平衡）；**加權 CE ≈ focal**（~0.55），真正有用的是「為不平衡加權」而非 focal 本身。
換 loss 動不了 minute AUC（瓶頸是特徵）；group-DRO 只在**篩檢**方向有效。

---

## 4.6 類別不平衡處理對照（HGB, apnea+hyp, grouped CV）

| 方法 | AUC | BAcc | F1 | Recall | Spec |
|---|---:|---:|---:|---:|---:|
| 類別加權（基準） | **0.564** | 0.535 | 0.290 | 0.362 | 0.708 |
| SMOTE（合成少數類） | 0.551 | 0.533 | **0.311** | 0.466 | 0.600 |
| 欠採樣（undersample） | 0.563 | 0.536 | 0.286 | 0.354 | 0.718 |
| EasyEnsemble（10×平衡集成） | 0.562 | 0.538 | 0.287 | 0.349 | 0.728 |

已試過的不平衡處理：類別加權、focal、過採樣（`WeightedRandomSampler`）、欠採樣、
threshold moving / prevalence matching、label smoothing、SMOTE、EasyEnsemble。
結論：**所有方法 AUC 都不動（0.55–0.56）**，只在 recall↔specificity 間移動操作點。
SMOTE 偏 recall 取得最佳 F1（0.311）。想突破 AUC，不平衡處理是錯的槓桿——瓶頸在特徵可分性。

## 5. 消融重點（HRV+CVHR，grouped CV）

| 消融 | 設定 | AUC |
|---|---|---:|
| 受試者正規化 | 無 → zscore（LogReg） | 0.492 → **0.553**（+0.061） |
| 標籤定義 | apnea-only → apnea+hyp（Ensemble） | 0.442 → **0.570** |
| 古典 vs 深度 | MLP → Ensemble | 0.551 → **0.570** |

---

## 6. 為什麼 UCDDB 這麼難（訊號診斷）

| | 跨受試者單特徵 AUC | 受試者**內**單特徵 AUC |
|---|---:|---:|
| apnea+hyp | 0.54 | 0.59 |
| apnea-only | 0.59 | **0.64** |

受試者**內**最佳單特徵也僅 ~0.6–0.64 → minute-level 標籤與 HRV 的關聯本身就弱，
不只是跨受試者轉移問題。UCDDB 僅 25 人、變異大。

---

## 7. 與參考文獻比較

| 來源 | 資料集 | 回報 | 協定 | 註 |
|---|---|---|---|---|
| Chen 2022 (FNINS) | Apnea-ECG | Acc ~0.89 | 標準 holdout | 我們複現 F1 0.840 ✅ |
| Chen 2022 (FNINS) | UCDDB | Acc 0.923 | 非受試者層級 | 接近我們 apnea-only holdout |
| Pham & Moucek 2025 | UCDDB | **Acc 98–99%** | **8:1:1 視窗洩漏** | 不可與 grouped CV 相比 |
| **本專案（誠實基準）** | UCDDB | **AUC 0.570 / BAcc 0.542** | **record-level grouped CV** | 更嚴格、更真實 |

> 簡報可寫：「參考文獻在視窗層級切分下回報 98% 準確率；本專案改用 record-level grouped CV
> 以更貼近未見過使用者的真實泛化，屬更嚴格的協定。」

---

## 8. 一句話總結

- **Apnea-ECG**：模型有效，F1 0.84 / AUC 0.95，成功複現文獻。
- **UCDDB holdout**：AUC 0.70–0.88，但有洩漏、虛高。
- **UCDDB grouped CV（誠實）**：所有方法 ~0.55；我的 HRV+CVHR ensemble 做到 **0.570**（最佳）+ 可解釋。
- **瓶頸**：跨受試者 + minute 標籤內在弱訊號，換模型/loss 都救不了；**真正的出路是受試者層級篩檢**（group-DRO 已顯示篩檢 AUC 0.51→0.63）。
