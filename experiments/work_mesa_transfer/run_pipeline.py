"""無人看管編排:下載 MESA 子集 → train-MESA→test-UCDDB → 視結果自動擴大。

階段:
  1. 下載 ~40 人(idempotent,跳過已下載)
  2. mesa_to_ucddb_trainer --mesa-limit 40 → 讀 UCDDB 外部 AUC
  3. 若 AUC >= --scale-threshold(預設 0.58):下載更多 → --mesa-limit <scale-n> 重訓
  4. 印最終摘要

token 由環境變數 NSRR_TOKEN 取(VPN 需保持連線)。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable


def sh(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def download(pattern, data_dir):
    return sh([PY, str(HERE / "download_mesa.py"), "--pattern", pattern, "--data-dir", data_dir])


def train(mesa_limit, name, data_dir):
    return sh([PY, str(HERE / "mesa_to_ucddb_trainer.py"),
               "--mesa-dir", data_dir, "--mesa-limit", str(mesa_limit),
               "--experiment-name", name, "--no-progress"])


def read_auc(name):
    p = HERE / "outputs" / name / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return float(d["ucddb_external"]["threshold_val"]["roc_auc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="D:/mesa")
    ap.add_argument("--small-pattern", default="mesa-sleep-00[0-9]*")  # ~35-40 人
    ap.add_argument("--small-n", type=int, default=40)
    ap.add_argument("--scale-threshold", type=float, default=0.58)
    ap.add_argument("--scale-pattern", default="mesa-sleep-0[01]*")    # ~200 人
    ap.add_argument("--scale-n", type=int, default=200)
    args = ap.parse_args()

    print("=== STAGE 1: download small subset ===", flush=True)
    download(args.small_pattern, args.data_dir)

    print("=== STAGE 2: train-MESA -> test-UCDDB (small) ===", flush=True)
    train(args.small_n, f"mesa2ucddb_n{args.small_n}", args.data_dir)
    auc = read_auc(f"mesa2ucddb_n{args.small_n}")
    print(f"\n[pipeline] small-run UCDDB external AUC = {auc}", flush=True)

    if auc is not None and auc >= args.scale_threshold:
        print(f"=== STAGE 3: AUC {auc:.4f} >= {args.scale_threshold} -> SCALE UP to ~{args.scale_n} ===", flush=True)
        download(args.scale_pattern, args.data_dir)
        train(args.scale_n, f"mesa2ucddb_n{args.scale_n}", args.data_dir)
        auc2 = read_auc(f"mesa2ucddb_n{args.scale_n}")
        print(f"\n[pipeline] scaled-run UCDDB external AUC = {auc2}", flush=True)
    else:
        print(f"[pipeline] AUC {auc} < {args.scale_threshold} -> 停在小規模(誠實回報:大世代亦未明顯超過 0.55)。", flush=True)

    print("\n[pipeline] DONE.", flush=True)


if __name__ == "__main__":
    main()
