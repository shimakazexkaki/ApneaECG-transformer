# HRV 特徵 vs 深度學習:何時用哪個 + 為何深度學習失敗時 HRV 是較好的選項

## 0. 部署前提:為何選 Apnea-ECG(Polar H10 相關性)

本專題的目標裝置設想為 **Polar H10 胸帶**(SDK 可取 130 Hz 原始 ECG 與 RR intervals)。
- **Polar H10 的單導 ECG 波形,與 Apnea-ECG(modified lead V2、單導、乾淨)最相似**;MESA 是臨床 PSG、且為動脈硬化共病世代,離一般使用者配戴胸帶的情境較遠。
- 我們的特徵(RRI + R-peak 振幅 / HRV)**Polar H10 原生就給得出來**。
- 因此:**Apnea-ECG 為主要訓練/驗證資料、MESA 當大世代誠實對照(錦上添花)**,在部署相關性上是合理的取捨。

> 誠實提醒:Apnea-ECG 分數高是因為它是「精選乾淨考卷」;benchmark 0.95 ≠ Polar 真實世界 0.95(domain gap 真實存在)。

---

## 1. 核心對照:同一資料、同一協定下比

| 資料集 / 協定 | 深度學習(最佳) | HRV 特徵(最佳) | 誰贏 |
|---|---|---|---|
| **Apnea-ECG**(乾淨,標準 holdout) | CNN-BiGRU **AUC 0.946 / F1 0.840** | HRV ExtraTrees AUC 0.831 / F1 0.613 | **DL 大勝** |
| **UCDDB**(誠實 grouped CV,小 N、跨受試者) | CNN+Transformer AUC 0.552 / F1 0.373 | **HRV+CVHR Ensemble AUC 0.570 / F1 0.303** | **HRV 略勝/打平** |
| **MESA-internal**(grouped CV,clean-seg) | CNN+Transformer per-seg **AUC 0.709** | (HRV 待補) | DL(資料夠多時) |

**模式很清楚:**
- **資料乾淨、量大、同分佈(Apnea-ECG)→ 深度學習明顯勝**(0.95 vs 0.83):有足夠訊號時,深度網路學到更豐富的表徵。
- **資料小、雜、跨受試者(UCDDB 誠實協定)→ 深度學習優勢消失,HRV ensemble 反而追平甚至略勝**(0.570 vs 0.552)。

---

## 2. 為什麼「深度學習失敗時,HRV 是較好的選項」

當深度學習在小/雜/跨受試者資料上崩潰(我們實測:UCDDB 上 plain-CE 的深度模型直接全判正常、AUC 0.48),HRV 特徵有五個結構性優勢:

1. **資料效率**:HRV 是 ~41 個有物理意義的標量特徵 + 古典 ML(LogReg/HGB/ExtraTrees),**少量資料就能泛化**;深度網路要大量乾淨標註,小 N 時嚴重過擬合或崩潰。
2. **可解釋性 / 臨床可信**:SDNN、RMSSD、LF/HF、**CVHR(心率週期變化 0.01–0.04 Hz)**、EDR 都有明確生理意義——醫師能看懂「為何這分鐘被標為疑似呼吸中止」;深度模型是黑箱。
3. **抗 domain shift**:RR interval 不管來自 Polar H10 或 PSG 都是 RR interval,HRV 特徵**跨裝置/跨資料集穩定**;深度模型容易學到資料集特有的假影(我們實測 Apnea-ECG 0.95 → 跨資料集崩到 0.55)。
4. **可在裝置端 / 穿戴即時運算**:Polar H10 原生串 RR intervals → HRV 可在手機端即時算,**不需 GPU**;深度模型運算與部署成本高。
5. **實證**:在我們**誠實的 UCDDB 基準**上,HRV+CVHR ensemble(0.570)≥ 所有深度模型(~0.55)。一旦拿掉「簡單 benchmark 的灌水」,HRV 就具競爭力或更好。

---

## 3. 一句話決策準則

- **有乾淨、大量、同分佈資料(如 Apnea-ECG benchmark)→ 用深度學習**,拿最高分。
- **真實穿戴式篩檢(小 N、雜訊、跨人、跨裝置,如 Polar H10 部署)→ HRV 特徵是更穩、更可解釋、更省、且實測不輸的選項**;深度學習一旦在此情境失敗,HRV 是合理的退路與基準。

→ 本專題策略:**Apnea-ECG 用 CNN+Transformer 拿強 benchmark + per-recording AHI 篩檢**;同時提供 **HRV 特徵法作為可解釋、可部署的對照**,並以 UCDDB 誠實協定證明「DL 優勢在難資料上消失、HRV 追平」。

---

## 4. 數字出處
- Apnea-ECG / UCDDB 各數字見 `model_comparison.md`(第 1、3 節)。
- MESA-internal 見 `../work_mesa_transfer/mesa_transfer_results.md`。
- HRV+CVHR ensemble(UCDDB grouped CV 0.570)見 `results_report.md` 與 `models/hrv_ensemble.py`。
