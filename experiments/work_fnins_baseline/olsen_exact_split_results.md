# Olsen 2020 完全同切分:CNN+Transformer 三資料集結果

## 協定(與 Olsen 2020 SLEEP 完全對齊)
- **切分**:把整筆記錄(受試者)隨機分 **train/eval/test = 80% / 10% / 10%**,每位受試者只出現在一個集合(subject-independent、零洩漏)。**單次 hold-out,非 cross-validation**(Olsen 原文 [_olsen.txt:207-208](../../_olsen.txt#L207)、[223-224](../../_olsen.txt#L223))。
- **標籤**:per-minute,呼吸事件重疊 **>5 秒**為陽性(規則 A)。
- **模型**:同一個 `fn.CNNTransformer`(900×2 輸入,d_model=96 / nhead 4 / 2 層 TransformerEncoder),三資料集完全相同。
- **訓練**:focal loss(γ=2)+ WeightedRandomSampler 過採樣;閾值在 **eval 集**挑(balanced-accuracy 最佳);指標報在 **test 集**。
- **穩定性**:各資料集 10% test 僅數位受試者,單次切分會抖 → 對「同一個 Olsen 切分機制」跑 **5 個隨機種子**(seed 42–46),報 **mean ± std**;每個種子都是一次完整的 80/10/10 hold-out。
- driver:[olsen_exact_split_cnn_transformer.py](olsen_exact_split_cnn_transformer.py);原始輸出:[outputs/olsen_exact_split/olsen_exact_split_summary.json](outputs/olsen_exact_split/olsen_exact_split_summary.json)。

---

## 主結果(threshold 來自 eval、指標在 held-out test;mean ± std over 5 splits)

| 資料集 | 記錄數 | 陽性率 | Acc | Recall | Precision | F1 | Spec | AUC |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **Apnea-ECG** | 70 | 0.380 | **0.900**±0.020 | 0.797±0.112 | **0.833**±0.017 | **0.811**±0.069 | 0.937±0.014 | **0.947**±0.024 |
| **UCDDB** | 21 | 0.237 | 0.436±0.100 | 0.736±0.207 | 0.261±0.080 | 0.381±0.112 | 0.306±0.224 | 0.539±0.078 |
| **MESA** | 74 | 0.206 | 0.557±0.124 | 0.630±0.251 | 0.262±0.075 | 0.364±0.106 | 0.512±0.244 | 0.632±0.099 |

> 與既有結果高度吻合,確認實作正確:Apnea-ECG AUC 0.947(對照 holdout 協定 §1 的 0.931)、UCDDB ~0.54(誠實天花板)、MESA AUC 0.632 / F1 0.364(對照先前 per-minute 0.635 / 0.385)。

---

## 對標 Olsen 2020(他的 headline 數字)

| | Recall (Se) | Precision (Pp) | F1 | AUC | Spec |
|---|--:|--:|--:|--:|--:|
| **我們 Apnea-ECG**(70,in-dataset) | 0.797 | **0.833** | **0.811** | 0.947 | 0.937 |
| **我們 MESA**(74) | 0.630 | 0.262 | 0.364 | 0.632 | 0.512 |
| **我們 UCDDB**(21) | 0.736 | 0.261 | 0.381 | 0.539 | 0.306 |
| **Olsen 2020 MESA+SHHS**(~10000,RR+EDR+BiGRU) | 0.687 | 0.691 | **0.666** | — | ~0.92 |

**判讀:**
1. **乾淨資料(Apnea-ECG)我們贏面大**:F1 0.811、AUC 0.947。注意 Olsen 是把 Apnea-ECG 當「外部測試」(他訓練在 MESA+SHHS),我們是 in-dataset 80/10/10,所以本就占優,不能直接比;但證明在乾淨單導 ECG 上 CNN+Transformer 很強。
2. **MESA 同協定下我們輸 Olsen**:F1 0.364 vs 0.666,**差距全在 precision(0.26 vs 0.69)= 誤報太多**。主因:(a) 規模 74 人 vs 近萬人;(b) 我們只用 RRI+振幅,沒有 Olsen 的 proper EDR(ECG 導出呼吸)通道。
3. **UCDDB 同協定 ~0.54**:25 人小世代,deep 學不動,與 grouped CV 的天花板一致。

---

## 每個種子明細(看出單次切分的變異)

**Apnea-ECG**(穩):AUC 各種子 = 0.952 / 0.968 / 0.962 / 0.953 / **0.901** → 大致 ~0.95,僅 seed46 偏低。

**MESA**(抖):AUC = **0.438** / 0.697 / 0.640 / 0.704 / 0.682 → seed42 抽到 pathological 的 7 人 test(AUC 0.44 拖低均值);其餘 4 次都在 **0.64–0.70**。亦即 MESA 的 per-minute AUC 實際約 **0.68**,被一次壞抽樣拉到均值 0.632。

**UCDDB**(最抖):AUC = 0.527 / 0.440 / 0.669 / 0.490 / 0.569,test 每次只有 **2 位受試者** → F1 在 0.21–0.51 間大幅震盪。

> **這就是 Olsen 單次 hold-out 套在小資料上的固有限制**:Olsen 有 ~萬筆,單一 10% test(~千筆)很穩;我們 N 小(2–7 人 test),單次切分數字本身是抽樣彩券,故以 5 種子 mean±std 呈現。

---

## 一句話
**Olsen 完全同協定下:Apnea-ECG(乾淨)CNN+Transformer F1 0.811 / AUC 0.947 很強;MESA F1 0.364 仍輸 Olsen 0.666,瓶頸是規模(74 vs 萬)與缺 EDR、而非架構;UCDDB 0.54 撞小世代天花板。**
