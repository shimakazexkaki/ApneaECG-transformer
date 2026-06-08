#!/usr/bin/env python
"""CNN+Transformer（segment-split 版本，文獻協定）

UCDDB 8:1:1 片段切分（train/val/test），與參考論文（FNINS、Pham&Moucek）同協定。
注意：同一受試者的片段會分散在 train/test（leakage），所以數字會比 record-level grouped CV 樂觀。
此版本用來與文獻直接比較；誠實的泛化請看 grouped CV（models/screening.py / cnn_transformer_grouped_cv.py）。

預設：UCDDB 從頭訓練、weighted cross-entropy（給出較平衡的 recall/specificity）。

直接執行：
    python models/cnn_transformer_segment.py
變化：
    python models/cnn_transformer_segment.py --loss focal      # 改 focal loss
    python models/cnn_transformer_segment.py --no-include-hypopnea   # 只用 apnea 標籤
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "experiments" / "work_fnins_baseline"
SCRIPT = WORK / "fnins_experiment.py"

defaults = [
    "--dataset", "ucddb", "--model-type", "cnn_transformer",
    "--ucddb-channel", "0", "--epochs", "40", "--loss", "wce",
    "--experiment-name", "cnn_transformer_ucddb_segment",
    "--cache-dir", str(WORK / "cache"), "--output-dir", str(WORK / "outputs"),
    "--ucddb-literature-cache-dir", str(WORK / "cache_ucddb_lit"),
]
cmd = [sys.executable, "-u", str(SCRIPT)] + defaults + sys.argv[1:]
print(">> CNN+Transformer | segment split 8:1:1（文獻協定，數字較樂觀）| weighted CE\n")
sys.exit(subprocess.run(cmd, cwd=str(ROOT)).returncode)
