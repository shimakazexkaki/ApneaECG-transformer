# 期末專題更新建議：以 UCDDB 為主的穿戴式 ECG 睡眠呼吸中止偵測

## 研究目標

本專題目標從原本的 Apnea-ECG 分鐘分類，調整為更貼近穿戴式裝置的 UCDDB ECG 睡眠呼吸中止篩檢：

- 使用單導程或少導程 ECG，模擬穿戴式裝置可取得的訊號。
- 以 UCDDB 作為主要資料集，因為它有 PSG 對照的呼吸事件標註。
- 輸出分鐘級 sleep apnea 風險，並進一步估計整晚 apnea burden。

## 參考文獻方法

參考 Pham and Moucek 2025 的 CNN-Transformer-LSTM 做法：

- 使用 Hamilton algorithm 偵測 R peak。
- 從 ECG 萃取 RRI 與 R-peak amplitude。
- 每個樣本使用 5 分鐘上下文，重採樣成 900 個 RRI 點與 900 個 R-amplitude 點。
- 模型結構為 CNN + Transformer + LSTM。
- UCDDB 另可使用 11 秒 window、10 秒 overlap，對齊 apnea 事件至少 10 秒的臨床定義。

## 本次已完成

- 建立獨立 work 資料夾：`work_ucddb_paper_replication`。
- 新增 M11-style `CNN-Transformer-LSTM` trainer。
- 使用指定 conda 環境執行：
  `C:\Users\a2003\miniconda3\envs\apnea\python.exe`
- 在 UCDDB channel 0 + channel 2 上完成 smoke、segment split、grouped CV 初步實驗。

## 初步結果

| 實驗 | 評估方式 | Balanced Accuracy | AUC | F1 |
|---|---|---:|---:|---:|
| M11, full UCDDB, 20 epochs | segment-level split | 0.6389 | 0.6982 | 0.4275 |
| M11, full UCDDB, 5-fold grouped CV, 3 epochs | record-level grouped CV | 0.5318 +/- 0.0139 | 0.5732 +/- 0.0202 | 0.2846 +/- 0.1082 |
| 既有 high-res CNN-Transformer | minute-level grouped CV | 0.5355 +/- 0.0355 | 0.5522 +/- 0.0474 | 0.3697 +/- 0.0663 |

## 實驗解讀

- Segment-level split 下模型能學到 RRI/R-amplitude 中的 apnea pattern，但這種切分可能讓同一位受試者的資料同時出現在 train/test，因此不能直接代表穿戴式裝置泛化能力。
- Grouped CV 才是比較可信的結果，因為 test subject 不會出現在 train set。
- 目前結果顯示 UCDDB 跨受試者泛化仍困難，不能直接引用論文 98-99% accuracy 作為本專題結果。
- 高解析度 11 秒 window 方向更符合 UCDDB 呼吸事件標註，也更接近穿戴式裝置即時篩檢。

## 後續建議

- 將 CNN-Transformer-LSTM 加入 11 秒 high-resolution UCDDB pipeline。
- 跑完整 grouped 5-fold CV，並比較 full 25 subjects 與 reduced 21 subjects。
- 報告 Accuracy、Recall、Precision、Specificity、F1、AUC、Balanced Accuracy。
- 加入每位受試者的 AHI-like apnea burden，讓結果更貼近實際篩檢情境。
- Apnea-ECG 可作為 pretraining 或輔助比較，但最終結論以 UCDDB grouped CV 為主。
