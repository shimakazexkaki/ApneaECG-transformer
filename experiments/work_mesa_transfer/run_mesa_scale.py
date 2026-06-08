"""MESA 149 人規模:最佳配置(clean 10s-seg + recording-norm + CNN+Transformer)
train + per-subject 篩檢,序列執行。建 ~75 人新特徵約 3 小時。
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable
NAME = "mesa_internal_n149"
COMMON = ["--label-mode", "segment", "--segment-sec", "10", "--segment-stride-sec", "10",
          "--norm-mode", "recording", "--use-all-available", "--mesa-limit", "149", "--n-splits", "5"]


def main():
    print("=== STAGE 1: train 149 (best config) ===", flush=True)
    subprocess.run([PY, str(HERE / "mesa_internal_trainer.py"), *COMMON,
                    "--epochs", "15", "--experiment-name", NAME, "--no-progress"], cwd=str(ROOT))
    print("=== STAGE 2: per-subject screening ===", flush=True)
    subprocess.run([PY, str(HERE / "mesa_screening.py"), *COMMON,
                    "--experiment-name", NAME], cwd=str(ROOT))
    print("=== MESA 149 scale DONE ===", flush=True)


if __name__ == "__main__":
    main()
