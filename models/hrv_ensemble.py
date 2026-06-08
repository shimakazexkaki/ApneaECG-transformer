#!/usr/bin/env python
"""模型 1：HRV+CVHR Ensemble（本專案 UCDDB 誠實基準最佳，AUC 0.570，可解釋）

只用 UCDDB，零 Apnea 預訓練。手工 HRV+CVHR+EDR 特徵 + 受試者內正規化 +
LogReg/HGB 軟投票，5-fold record-level grouped CV（整位受試者 held out）。

直接執行：
    python models/hrv_ensemble.py
可選覆寫（不需要也能跑），例如只用單通道、改 apnea-only：
    python models/hrv_ensemble.py --channels 0 --apnea-only
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "experiments" / "work_ucddb_hrv" / "hrv_grouped_cv.py"
OUT = ROOT / "experiments" / "work_ucddb_hrv" / "outputs" / "hrv_ensemble_result.json"

cmd = [
    sys.executable, "-u", str(SCRIPT),
    "--channels", "0", "2",
    "--classifier", "ensemble",
    "--subject-norm", "zscore",
    "--output", str(OUT),
] + sys.argv[1:]

print(">> HRV+CVHR Ensemble | UCDDB grouped CV | 雙通道 | 受試者正規化\n")
sys.exit(subprocess.run(cmd, cwd=str(ROOT)).returncode)
