#!/usr/bin/env python
"""模型 3：CNN + Transformer（企畫書提案模型）

輸入為 5 分鐘 RRI + R-peak 振幅序列（900x2）。預設：只用 UCDDB 從頭訓練、零 Apnea
預訓練，並用 focal loss（已驗證能避免從頭訓練崩潰、AUC 0.575->0.68）。

直接執行（UCDDB 從頭訓練）：
    python models/cnn_transformer.py

可選覆寫：
    python models/cnn_transformer.py --dataset apnea_ecg          # 改在 Apnea-ECG 上訓練
    python models/cnn_transformer.py --loss wce                   # 改用加權 cross-entropy
    python models/cnn_transformer.py --pretrained-path <某個.pth> # 用 Apnea 預訓練權重微調
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "experiments" / "work_fnins_baseline"
SCRIPT = WORK / "fnins_experiment.py"

# 預設值；使用者在命令列給的同名參數會覆寫這些（argparse 取後者）。
defaults = [
    "--dataset", "ucddb",
    "--model-type", "cnn_transformer",
    "--ucddb-channel", "0",
    "--epochs", "40",
    "--loss", "focal",
    "--experiment-name", "cnn_transformer_ucddb",
    "--cache-dir", str(WORK / "cache"),
    "--output-dir", str(WORK / "outputs"),
    "--ucddb-literature-cache-dir", str(WORK / "cache_ucddb_lit"),
]

cmd = [sys.executable, "-u", str(SCRIPT)] + defaults + sys.argv[1:]
print(">> CNN+Transformer | 預設 UCDDB 從頭訓練 + focal loss\n")
sys.exit(subprocess.run(cmd, cwd=str(ROOT)).returncode)
