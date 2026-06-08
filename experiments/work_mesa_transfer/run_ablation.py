"""層級消融:在 MESA clean-segment grouped CV 上,拆掉 CNN+Transformer+LSTM 的不同層,
比 AUC,看哪一層其實多餘。序列執行(避免搶 GPU)。

  full           = CNN + Transformer + LSTM (M11 原版)
  no_lstm        = CNN + Transformer (Transformer 後直接 mean-pool)
  no_transformer = CNN + LSTM
  cnn_only       = CNN + mean-pool
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable
TRAINER = HERE / "mesa_internal_trainer.py"
ABLATIONS = ["full", "no_lstm", "no_transformer", "cnn_only"]
N_SPLITS = 3
EPOCHS = 15


def run(abl):
    name = f"mesa_abl_{abl}"
    cmd = [PY, str(TRAINER), "--label-mode", "segment", "--segment-sec", "10",
           "--segment-stride-sec", "10", "--ablation", abl, "--n-splits", str(N_SPLITS),
           "--epochs", str(EPOCHS), "--experiment-name", name, "--no-progress"]
    print(f"\n>>> ablation={abl}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT))
    p = HERE / "outputs" / name / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    a = d["aggregate"]["threshold_val"]
    return a["roc_auc"]["mean"], a["roc_auc"]["std"], a["balanced_accuracy"]["mean"], a["f1"]["mean"]


def main():
    rows = []
    for abl in ABLATIONS:
        r = run(abl)
        rows.append((abl, r))
    print("\n========== 層級消融 (MESA clean-seg, grouped 3-fold) ==========", flush=True)
    print(f"{'variant':16s} {'AUC':>16s} {'BAcc':>8s} {'F1':>8s}", flush=True)
    for abl, r in rows:
        if r is None:
            print(f"{abl:16s}  (failed)", flush=True)
        else:
            auc, std, bacc, f1 = r
            print(f"{abl:16s} {auc:.4f}+/-{std:.3f} {bacc:8.4f} {f1:8.4f}", flush=True)
    (HERE / "outputs" / "ablation_results.json").write_text(
        json.dumps({a: r for a, r in rows}, indent=2), encoding="utf-8")
    print("\nSaved: outputs/ablation_results.json", flush=True)


if __name__ == "__main__":
    main()
