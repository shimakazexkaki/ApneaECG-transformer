# models/ — 四個模型，一個指令各自跑

每個檔案是一個模型的入口，已內建好預設參數，**直接執行、不用打一堆參數**。
環境：`C:\Users\a2003\miniconda3\envs\apnea\python.exe`（簡寫 `python`）。

| 指令 | 模型 | 資料/訓練方式 | 誠實 grouped CV |
|---|---|---|---|
| `python models/hrv_ensemble.py` | **HRV+CVHR Ensemble**（最佳、可解釋） | UCDDB only，手工特徵+受試者正規化+LogReg/HGB 投票 | **AUC 0.570 / BAcc 0.542** |
| `python models/hrv_mlp.py` | HRV MLP（深度對照） | UCDDB only，同特徵+深度 MLP | AUC 0.551 |
| `python models/cnn_transformer.py` | CNN+Transformer（提案模型） | UCDDB 從頭訓練 + focal loss | （segment split 0.68；CV ~0.55） |
| `python models/cnn_bigru.py` | CNN-BiGRU（FNINS baseline） | UCDDB 從頭訓練 + focal loss | （Apnea-ECG F1 0.84）|

## 常用變化（可選，不打也能跑）

```powershell
# HRV：改成只用 apnea（不含 hypopnea）標籤
python models/hrv_ensemble.py --apnea-only

# HRV MLP：換 loss
python models/hrv_mlp.py --loss focal

# CNN-BiGRU：複現文獻 Apnea-ECG 結果
python models/cnn_bigru.py --dataset apnea_ecg --loss ce

# CNN+Transformer：用 Apnea 預訓練權重微調（需給 .pth）
python models/cnn_transformer.py --pretrained-path <model.pth>
```

## 說明

- 這些入口會呼叫底層已驗證的訓練程式（`work_ucddb_hrv/`、`work_fnins_baseline/`），
  並把結果存到對應的 `outputs/` 資料夾。
- 共用函式庫（訊號處理、特徵、資料讀取）放在專案根目錄的 7 個 `.py`（`ucddb_runner.py` 等），
  是被 import 的程式庫，**不要直接執行**。
- 探索階段的舊實驗腳本都搬到 `archive/`（不影響上面四個模型）。
- 完整結果與分析見 [`model_comparison.md`](../experiments/work_ucddb_hrv/model_comparison.md)
  與 [`results_report.md`](../experiments/work_ucddb_hrv/results_report.md)。
