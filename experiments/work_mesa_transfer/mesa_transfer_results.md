# MESA → UCDDB 跨資料集:用大世代測試 UCDDB ~0.55 天花板

## 動機
前一階段已確立 **UCDDB 的誠實天花板 ~0.55**(record-level grouped CV;HRV ensemble 0.570、M11 CNN-Transformer-LSTM 0.54、系統性調參 14 組最佳 0.558 皆打不破),並判定瓶頸是**資料量**(UCDDB 僅 25 受試者)。

本實驗用 **MESA(NSRR,2000+ 受試者整夜 PSG)** 做最乾淨的跨資料集測試:
**train on MESA → test on 全 UCDDB(完全外部、零洩漏)**。模型 = M11 CNN-Transformer-LSTM(classifier_dropout=0.2,調參最佳),特徵與 UCDDB **以相同管線建構**(EKG 256→100Hz、Hamilton R-peak、5 分鐘 context、900 點 RRI/振幅內插、逐窗 z-score),確保跨資料集可比。

## 方法重點
- MESA reader:`lib/mesa_features.py`(讀 EDF 的 EKG channel + 解析 NSRR XML 的 apnea/hypopnea 事件)。
- 訓練/評估:重用 `lib/ucddb_literature_train_common.train_one_split`(MESA 內依受試者切 train/val 挑閾值;test = 全 UCDDB)。
- driver:`experiments/work_mesa_transfer/mesa_to_ucddb_trainer.py`。
- 每窗自身 z-score → 自動消除 MESA/UCDDB 振幅尺度差;RRI 同單位(秒)。

## 結果:AUC vs 訓練世代規模 N

| MESA 訓練人數 N | UCDDB 外部 AUC | BAcc | F1 | Recall | Spec | 備註 |
|---:|---:|---:|---:|---:|---:|---|
| 14 | 0.552 | 0.540 | 0.336 | 0.577 | 0.503 | N 太小,≈ 天花板 |
| 44 | 0.561 | 0.546 | 0.255 | 0.211 | 0.881 | 微升 |
| 74 | 0.547 | 0.520 | 0.343 | 0.784 | 0.255 | **回落**,非單調 |

**結論(N=14/44/74):AUC 在 0.547–0.561 之間抖動,沒有隨 N 上升 → 大世代訓練在此規模下未突破 UCDDB ~0.55 天花板。**
判讀:**跨資料集 domain shift(MESA↔UCDDB 不同儀器/族群/導程)為主因**,而非單純人數不足;且 minute 標籤↔ECG 關聯本就弱(受試者內單特徵 AUC 僅 ~0.6–0.64)。增加 MESA 人數(14→74)無助於 UCDDB 外部泛化。N=149 因趨勢已平、不再續推。

對照(UCDDB-internal,誠實天花板):HRV ensemble 0.570、M11 grouped-CV 0.54、調參最佳 0.558。

## MESA-internal:方法改進實驗(74 人,grouped CV,同資料集無 domain shift)

目的:判斷「UCDDB 太難 vs 訊號天花板」,並對標文獻把 CNN+Transformer 在 MESA 上做到最好。

**文獻定位(誠實基準):**
- **Olsen 2020(《Sleep》,MESA+SHHS 近萬筆,RR+EDR+BiGRU)= per-subject AHI 嚴重度 Acc 0.849、R²=0.83、per-event F1 0.67。** 這是 MESA 等級資料的真實天花板。
- 那些 AUC 0.95–0.98 的論文(Liu CNN-Transformer、時空模型、SNN)**全部在 Apnea-ECG**(精選簡單資料),不是 MESA。我們在 Apnea-ECG 也是 0.93–0.95。

**per-segment AUC 進展:**
| 設定 | AUC | 備註 |
|---|---:|---|
| minute 標籤(原始 M11) | 0.631 | 基準 |
| 乾淨 10s-segment 標籤(完整落事件內=陽性) | 0.678 | +0.047 ✅ |
| + RRID 第三通道 | 0.665 | 無效 ❌ |
| **乾淨 + recording-norm + CNN+Transformer** | **0.709** | 最佳 ✅ |

**層級消融(clean-seg, grouped 3-fold):** full(CNN+T+LSTM)0.646 / no_lstm(CNN+T)0.631 / no_transformer(CNN+LSTM)0.626 / cnn_only 0.620。
→ 各層只貢獻 ~0.01–0.02(在 std 內);**LSTM 疊在 Transformer 後多餘** → 以後預設 CNN+Transformer。

**per-subject AHI 篩檢(Olsen 的 headline 指標):**
| 設定 | 嚴重度準確率(tertile) | 負荷 Spearman |
|---|---:|---:|
| window-norm(逐窗 z-score) | 0.26(比亂猜 0.33 差) | 0.04 |
| **recording-norm(整夜統計量),74 人** | 0.41 | 0.15(p=0.20,不顯著) |
| **recording-norm,149 人** | 0.40 | **0.185(p=0.024,顯著)** |
→ 逐窗正規化會洗掉個體水準、害了篩檢;改 recording-norm 後篩檢從「壞掉」變「弱可用」。

**規模測試(clean-seg + recording-norm + CNN+Transformer,per-segment AUC):**
| 受試者數 N | per-segment AUC | 篩檢 Spearman |
|---:|---:|---:|
| 74 | 0.709 | 0.15(不顯著) |
| **149** | **0.735** | **0.185(p=0.024 顯著)** |
→ **規模有效**:74→149 人,per-segment +0.026、篩檢相關性首次達統計顯著。**證實 Olsen「人多有用」**;趨勢指向更多人(朝近萬)會再升 → 下載續推 ~310 人可再測。

**Olsen 同協定較量(per-minute 標籤,prevalence 20.6%,74 人,CNN+Transformer):**
| | Recall | Precision | F1 | AUC | Spec |
|---|--:|--:|--:|--:|--:|
| 我們(74 人) | 0.631 | 0.279 | **0.385** | 0.635 | 0.573 |
| Olsen(MESA+SHHS ~10000) | 0.687 | 0.691 | **0.666** | — | ~0.92 |
→ ① F1 從 per-10秒段的 0.16 → per-minute 的 **0.385**(證實低 F1 是 4.3% prevalence 的假象,非 bug)。
② 仍輸 Olsen(0.385 vs 0.666),**差距在 precision(0.28 vs 0.69)= 誤報多**,主因規模(74 vs 近萬)+ 缺 proper EDR。
③ 標籤取捨:clean-10秒段 AUC 高(0.71)F1 低(0.16);per-minute F1 高(0.385)AUC 低(0.635)。

**結論:**
1. CNN+Transformer + 乾淨標籤 + recording-norm 把 per-segment 從 **0.63 → 0.71**。
2. per-subject 篩檢仍弱(Spearman 0.15),離 Olsen 0.85 還遠 → 差距主因 **世代規模(74 vs 近萬)** + 缺 proper EDR / 假影處理。
3. 架構不是瓶頸(消融每層只 1–2%);**訊號 × 共病族群 × 小 N 才是**。

## 目前判讀(隨 N 增加更新)
- **N=14:AUC 0.552**,等於天花板,**尚無突破——但在預期內**:14 人比 UCDDB 的 25 人還少,根本未進入「大世代」範圍。訓練中 val AUC 數個 epoch 達 0.60–0.64,顯示**有訊號但不足以穩定外部泛化**。
- 本實驗的關鍵不是單一數字,而是 **AUC 是否隨 N 單調上升**:若 40→100 明顯爬升,即證實「資料量是瓶頸、MESA 大世代是解法」;若持平於 ~0.55,則顯示跨資料集 domain shift 才是主因。

## 重現
```
python experiments/work_mesa_transfer/download_mesa.py --pattern "mesa-sleep-0[0-2]*" --data-dir D:/mesa
python experiments/work_mesa_transfer/mesa_to_ucddb_trainer.py --mesa-limit 40  --experiment-name mesa2ucddb_n40  --no-progress
python experiments/work_mesa_transfer/mesa_to_ucddb_trainer.py --mesa-limit 100 --experiment-name mesa2ucddb_n100 --no-progress
```

## 注意
- MESA EDF 每人 ~300MB(全通道),下載量大;特徵建構 ~160s/人(一次性,有 npz 快取)。
- 跨資料集 domain shift 真實存在;務實期望若 scale 有效,落點 ~0.65–0.75,不會到洩漏式 98%。
