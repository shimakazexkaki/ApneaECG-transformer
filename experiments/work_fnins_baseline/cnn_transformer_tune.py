"""P0 調參：CNN+Transformer 的超參數搜尋，全程用 record-level grouped CV（誠實、避免洩漏）。

做法：對候選組態各跑一次 5-fold grouped CV（重用 `cnn_transformer_grouped_cv.run`），
以外層 grouped-CV mean AUC（次看 val-threshold BAcc）排序，存最佳組態。

為控制時間，預設用較少 epoch 的「搜尋模式」，找到最佳後再用完整 epoch 確認。
候選清單刻意精簡（座標式搜尋，不是全網格），逐維比較對天花板的影響。
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cnn_transformer_grouped_cv as cv  # noqa: E402


def base_config(epochs):
    return dict(channels=[0, 2], context_minutes=5, d_model=96, nhead=4, layers=2,
                dropout=0.3, lr=5e-4, weight_decay=1e-4, epochs=epochs, patience=6,
                batch_size=128, samples_per_epoch=None, loss="focal", focal_gamma=2.0,
                n_splits=5, seed=42, cache_dir=str(_HERE / "cache"), cpu=False, output=None)


# 座標式搜尋：每組只改一個維度，和 base 比較，找出有效的方向。
SEARCH = {
    "d_model":   [64, 96, 128],
    "layers":    [1, 2, 3],
    "nhead":     [4, 8],
    "dropout":   [0.2, 0.3, 0.4],
    "lr":        [3e-4, 5e-4, 1e-3],
    "weight_decay": [1e-4, 1e-3],
    "loss":      ["wce", "focal"],
    "focal_gamma": [1.0, 2.0],
    "context_minutes": [3, 5],
    "channels":  [[0], [0, 2]],
}


def run_config(cfg):
    args = SimpleNamespace(**cfg)
    summary = cv.run(args)
    return summary["pooled_auc"], summary["aggregate"]["threshold_val"]["balanced_accuracy"]["mean"]


def main():
    p = argparse.ArgumentParser(description="CNN+Transformer hyperparameter search (grouped CV).")
    p.add_argument("--search-epochs", type=int, default=15, help="每個候選的 epoch 數（搜尋階段壓低以省時間）")
    p.add_argument("--dims", nargs="*", default=list(SEARCH.keys()),
                   help="要搜尋的維度（預設全部）")
    p.add_argument("--output", default=str(_HERE / "outputs" / "cnn_transformer_tune_results.json"))
    args = p.parse_args()

    base = base_config(args.search_epochs)
    results = []

    print("=== Baseline ===")
    auc, bacc = run_config(dict(base))
    results.append({"config": "baseline", "changed": {}, "pooled_auc": auc, "bacc": bacc})
    best = {"config": "baseline", "params": dict(base), "pooled_auc": auc, "bacc": bacc}

    for dim in args.dims:
        for val in SEARCH[dim]:
            if base.get(dim) == val:
                continue  # baseline 已測過此值
            cfg = dict(base)
            cfg[dim] = val
            print(f"\n=== {dim}={val} ===")
            auc, bacc = run_config(cfg)
            results.append({"config": f"{dim}={val}", "changed": {dim: val}, "pooled_auc": auc, "bacc": bacc})
            if (auc, bacc) > (best["pooled_auc"], best["bacc"]):
                best = {"config": f"{dim}={val}", "params": cfg, "pooled_auc": auc, "bacc": bacc}

    results.sort(key=lambda r: (r["pooled_auc"], r["bacc"]), reverse=True)
    out = {"search_epochs": args.search_epochs, "ranked": results, "best": best}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n========== 調參結果（依 pooled AUC 排序）==========")
    for r in results:
        print(f"  {r['config']:22s} AUC={r['pooled_auc']:.4f} BAcc={r['bacc']:.4f}")
    print(f"\nBest: {best['config']}  AUC={best['pooled_auc']:.4f} BAcc={best['bacc']:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
