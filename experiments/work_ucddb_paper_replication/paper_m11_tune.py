"""Part 2：5 分鐘 RRI M11（CNN-Transformer-LSTM）的 nested grouped-CV 超參數調整。

座標式搜尋：每次只改一個維度，每組跑完整 5-fold record-level grouped CV（誠實），
以 aggregate.threshold_val.roc_auc.mean 排序挑最佳。搜尋階段用較少 epoch 以省時間。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PY = sys.executable
TRAINER = _HERE / "paper_cnn_transformer_lstm_trainer.py"
OUTROOT = _HERE / "outputs"
ROOT = _HERE.parents[1]

BASE = dict(d_model=96, layers=3, nhead=8, lstm_hidden=128, lstm_layers=1,
            lr=1e-3, classifier_dropout=0.3, dim_feedforward=256)

SEARCH = {
    "d_model": [64, 128],
    "layers": [2, 4],
    "nhead": [4],
    "lstm_hidden": [64, 256],
    "lstm_layers": [2],
    "lr": [3e-4, 5e-4],
    "classifier_dropout": [0.2, 0.5],
    "dim_feedforward": [128],
}

ARGFLAG = {  # config key -> CLI flag
    "d_model": "--d-model", "layers": "--layers", "nhead": "--nhead",
    "lstm_hidden": "--lstm-hidden", "lstm_layers": "--lstm-layers", "lr": "--lr",
    "classifier_dropout": "--classifier-dropout", "dim_feedforward": "--dim-feedforward",
}


def run_config(cfg, tag, epochs, n_splits):
    name = f"m11t_{tag}"
    cmd = [PY, "-u", str(TRAINER), "--protocol", "cv", "--epochs", str(epochs),
           "--n-splits", str(n_splits), "--experiment-name", name, "--no-progress"]
    for k, v in cfg.items():
        cmd += [ARGFLAG[k], str(v)]
    subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summ = OUTROOT / name / "summary.json"
    if not summ.exists():
        return None, None
    d = json.loads(summ.read_text(encoding="utf-8"))
    agg = d["aggregate"]["threshold_val"]
    return float(agg["roc_auc"]["mean"]), float(agg["balanced_accuracy"]["mean"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search-epochs", type=int, default=15)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--output", default=str(OUTROOT / "paper_m11_tune_results.json"))
    args = p.parse_args()

    results = []
    print("=== baseline ===", flush=True)
    auc, bacc = run_config(dict(BASE), "base", args.search_epochs, args.n_splits)
    results.append({"config": "baseline", "auc": auc, "bacc": bacc})
    best = {"config": "baseline", "params": dict(BASE), "auc": auc or 0, "bacc": bacc or 0}
    print(f"  baseline AUC={auc} BAcc={bacc}", flush=True)

    for dim, vals in SEARCH.items():
        for v in vals:
            cfg = dict(BASE); cfg[dim] = v
            tag = f"{dim}_{v}".replace(".", "p")
            print(f"=== {dim}={v} ===", flush=True)
            auc, bacc = run_config(cfg, tag, args.search_epochs, args.n_splits)
            results.append({"config": f"{dim}={v}", "auc": auc, "bacc": bacc})
            print(f"  AUC={auc} BAcc={bacc}", flush=True)
            if auc and (auc, bacc) > (best["auc"], best["bacc"]):
                best = {"config": f"{dim}={v}", "params": cfg, "auc": auc, "bacc": bacc}

    results.sort(key=lambda r: (r["auc"] or 0, r["bacc"] or 0), reverse=True)
    Path(args.output).write_text(json.dumps({"ranked": results, "best": best}, indent=2), encoding="utf-8")
    print("\n========== M11 tuning (by grouped-CV mean AUC) ==========", flush=True)
    for r in results:
        print(f"  {r['config']:24s} AUC={r['auc']} BAcc={r['bacc']}", flush=True)
    print(f"\nBest: {best['config']} AUC={best['auc']} BAcc={best['bacc']}", flush=True)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
