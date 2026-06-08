#!/usr/bin/env python
"""模型 2：HRV MLP（深度對照組）

與模型 1 相同的 HRV+CVHR 特徵與受試者正規化，但改用深度 MLP（weighted-CE）。
用來對照「深度 vs 古典」：在 ~25 受試者的小資料上，MLP（AUC 0.551）略輸 ensemble。
只用 UCDDB，零 Apnea 預訓練，5-fold grouped CV。

直接執行：
    python models/hrv_mlp.py
可選覆寫不同 loss：
    python models/hrv_mlp.py --loss focal
    python models/hrv_mlp.py --loss group_dro
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "experiments" / "work_ucddb_hrv" / "hrv_mlp_cv.py"
OUT = ROOT / "experiments" / "work_ucddb_hrv" / "outputs" / "hrv_mlp_result.json"

cmd = [
    sys.executable, "-u", str(SCRIPT),
    "--channels", "0", "2",
    "--subject-norm", "zscore",
    "--loss", "wce",
    "--output", str(OUT),
] + sys.argv[1:]

print(">> HRV MLP（深度對照）| UCDDB grouped CV | weighted-CE\n")
sys.exit(subprocess.run(cmd, cwd=str(ROOT)).returncode)
