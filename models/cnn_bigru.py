#!/usr/bin/env python
"""模型 4：CNN-BiGRU（FNINS 文獻 baseline）

CNN + BiGRU + attention，輸入為 5 分鐘 RRI + R-peak 振幅序列（900x2）。
預設：只用 UCDDB 從頭訓練、零 Apnea 預訓練，focal loss（plain CE 會讓它崩潰）。

直接執行（UCDDB 從頭訓練）：
    python models/cnn_bigru.py

複現文獻 Apnea-ECG 結果（F1 0.84）：
    python models/cnn_bigru.py --dataset apnea_ecg --loss ce
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "experiments" / "work_fnins_baseline"
SCRIPT = WORK / "fnins_experiment.py"

defaults = [
    "--dataset", "ucddb",
    "--model-type", "cnn_bigru",
    "--ucddb-channel", "0",
    "--epochs", "40",
    "--loss", "focal",
    "--experiment-name", "cnn_bigru_ucddb",
    "--cache-dir", str(WORK / "cache"),
    "--output-dir", str(WORK / "outputs"),
    "--ucddb-literature-cache-dir", str(WORK / "cache_ucddb_lit"),
]

cmd = [sys.executable, "-u", str(SCRIPT)] + defaults + sys.argv[1:]
print(">> CNN-BiGRU（FNINS baseline）| 預設 UCDDB 從頭訓練 + focal loss\n")
sys.exit(subprocess.run(cmd, cwd=str(ROOT)).returncode)
