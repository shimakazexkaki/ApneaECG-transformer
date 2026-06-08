#!/usr/bin/env python
"""模型 5：受試者層級 AHI 篩檢（P1）

用調好的 CNN+Transformer（最佳組態）在 record-level grouped CV 下，把每位受試者整晚的
逐分鐘分數聚合成 apnea 負擔，評估「能否分流出疑似中重度 SA 的人」。
輸出 minute-level 指標 + subject-level 篩檢（Pearson/Spearman、各 AHI tier 的 screening AUC）。

直接執行：
    python models/screening.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "experiments" / "work_fnins_baseline"
SCRIPT = WORK / "cnn_transformer_grouped_cv.py"

# 沿用 P0 找到的最佳組態。
cmd = [
    sys.executable, "-u", str(SCRIPT),
    "--channels", "0", "2", "--context-minutes", "5",
    "--d-model", "96", "--layers", "2", "--nhead", "4", "--dropout", "0.3",
    "--lr", "5e-4", "--weight-decay", "1e-4", "--epochs", "30",
    "--loss", "focal", "--focal-gamma", "2.0",
    "--output", str(WORK / "outputs" / "cnn_transformer_screening.json"),
] + sys.argv[1:]

print(">> 受試者層級 AHI 篩檢 | CNN+Transformer 最佳組態 | grouped CV\n")
sys.exit(subprocess.run(cmd, cwd=str(ROOT)).returncode)
