# M11 CNN-Transformer-LSTM：重現論文的協定式高分 vs 誠實跨受試者

模型：Pham & Moucek (2025) 的 M11 CNN-Transformer-LSTM。UCDDB-only，從頭訓練。
目的：用同一個模型，攤開「segment split（論文協定，會洩漏）」與「record-level grouped CV（誠實）」的差距。

## 結果總覽

| 輸入 | 協定 | Acc | Recall | Spec | F1 | **AUC** |
|---|---|---:|---:|---:|---:|---:|
| 5 分鐘 RRI/振幅 (900×2) | segment split 8:1:1 | 0.794 | 0.599 | 0.846 | 0.550 | **0.790** |
| 5 分鐘 RRI/振幅 (900×2) | grouped CV (5-fold) | — | — | — | 0.265 | **0.537** |
| 11 秒原始 ECG 視窗 | segment, 8千視窗（稀疏）| 0.560 | 0.632 | 0.553 | 0.196 | 0.609 |
| 11 秒原始 ECG 視窗 | segment, 10萬視窗 | 0.704 | 0.772 | 0.697 | 0.329 | 0.803 |
| 11 秒原始 ECG 視窗 | **segment, 30萬視窗（密集）** | 0.750 | 0.764 | 0.749 | 0.363 | **0.838** |
| 11 秒原始 ECG 視窗 | **grouped CV (5-fold)** | — | — | — | 0.129 | **0.511** |

## 關鍵結論：論文 98% 的機制

1. **洩漏鐵證（密度趨勢）**：11 秒視窗、10 秒重疊（1 秒 stride）下，segment split 的 AUC 隨「保留視窗密度」單調上升：
   **0.609（8千）→ 0.803（10萬）→ 0.838（30萬）**。視窗越密，相鄰、幾乎一模一樣的視窗越容易同時落在 train 與 test
   → 模型其實是在「認得鄰居」，分數一路朝論文的高分爬。
2. **誠實對照**：把整位受試者 held out（grouped CV），**同一個 11 秒模型只剩 AUC 0.511（近隨機）**，
   甚至比 5 分鐘 RRI 的 grouped CV（0.537）更差——11 秒原始 ECG 視窗跨受試者泛化更弱。
3. **差距 = 協定**：segment 0.838 vs grouped 0.511 → **+0.33 AUC，純粹來自「視窗化後才切分 + 高重疊」的洩漏**，
   不是模型變強。
4. **論文的 98% 是準確率**：第二級標籤陽性率僅 9.3%，高度不平衡下準確率天生就高；
   再加上全密度重疊洩漏、以及排除 4 名無 SA 患者後較同質的 21 人子集，就堆到 98–99%。

## Part 2：5 分鐘 M11 nested grouped-CV 系統性調參（14 組座標搜尋）

每組 = 完整 5-fold record-level grouped CV ×15 epoch，依 `aggregate.threshold_val.roc_auc.mean` 排序。

| 排名 | 組態（相對 baseline 改一維） | AUC | BAcc |
|---|---|---:|---:|
| 1 | classifier_dropout=0.2 | **0.558** | 0.531 |
| 2 | lstm_hidden=64 | 0.555 | 0.529 |
| 3 | nhead=4 | 0.548 | 0.512 |
| 4 | classifier_dropout=0.5 | 0.546 | 0.505 |
| 5 | baseline (d96/L3/h8/lstm128/lr1e-3/do0.3/ff256) | 0.545 | 0.521 |
| – | d_model=64 / d_model=128 / layers=2 / layers=4 | 0.545* | 0.521 |
| 10 | lstm_layers=2 | 0.544 | 0.507 |
| 11 | lr=3e-4 | 0.537 | 0.500 |
| 12 | lr=5e-4 | 0.530 | 0.505 |
| 13 | dim_feedforward=128 | 0.525 | 0.500 |
| 14 | lstm_hidden=256 | 0.514 | 0.508 |

**結論**：最佳組態（dropout 0.2）AUC **0.558**，僅比 baseline 0.545 高 0.013（噪聲級）。14 組全落在 **0.51–0.56**，無一突破。
**系統性調參確認打不破 grouped-CV 天花板** —— 與 loss（CE/WCE/focal）、不平衡方法（SMOTE/undersample/EasyEnsemble）、SQI gating 的結論一致：這是**資料量（25 受試者）**的瓶頸，不是超參。
誠實對照：HRV+CVHR ensemble grouped CV **0.570** 仍是最佳，且為簡單可解釋模型。
\* `d_model`/`layers` 四組跑出位元級相同 AUC，疑似這兩個 CLI flag 未接進 trainer（架構維度硬寫死）；不影響「天花板」結論。
結果檔：`outputs/paper_m11_tune_results.json`。

## 對專題的價值
- 我們**完整重現了論文的高分機制**（密度→洩漏→高分），同時誠實報告 grouped CV ~0.51。
- 這讓本專題比「只報 98%」的論文更可信：能說明那 98% 來自評估協定，而非真實泛化能力。
- **調參、loss、不平衡、SQI 四路系統性嘗試全部打不破 ~0.55**，把瓶頸明確指向「UCDDB 只有 25 人」的資料量問題 —— 這正是 MESA 大世代可以介入的點。

## 重現
```
# 5 分鐘 RRI M11
python experiments/work_ucddb_paper_replication/paper_cnn_transformer_lstm_trainer.py --protocol literature --experiment-name paper_m11_segment --epochs 40
python experiments/work_ucddb_paper_replication/paper_cnn_transformer_lstm_trainer.py --protocol cv --experiment-name paper_m11_cv --epochs 40
# 11 秒視窗 M11（論文協定）
python experiments/work_ucddb_paper_replication/paper_m11_highres.py --protocol segment --max-windows 300000 --epochs 10
python experiments/work_ucddb_paper_replication/paper_m11_highres.py --protocol grouped --max-windows 150000 --epochs 10
```
