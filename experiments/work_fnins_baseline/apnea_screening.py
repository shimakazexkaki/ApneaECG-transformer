"""Apnea-ECG per-recording AHI 篩檢(對標 Liu 2023:per-recording 100% / AHI MAE 4.33)。

重用已訓練的 CNN+Transformer:對 withheld x01-x35 每筆逐分鐘預測 apnea →
聚合成整筆的「apnea 分鐘數 / apnea index(分鐘/小時)」,與真實比對:
- 逐分鐘 pooled AUC
- per-recording apnea-minute 的 Pearson r / Spearman / MAE
- 二元 apneic 判定準確率(門檻:apnea 分鐘 >=100,PhysioNet A 類標準;另報 AI>=5)
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fnins_experiment as fe  # noqa: E402

WITHHELD = [f"x{i:02d}" for i in range(1, 36)]


def make_args():
    return Namespace(
        dataset="apnea_ecg", model_type="cnn_transformer",
        apnea_dir=str(ROOT / "apnea-ecg"), ucddb_dir=str(ROOT / "ucddb"),
        cache_dir=str(HERE / "cache"), output_dir=str(HERE / "outputs"),
        apnea_rpeak_source="hamilton", include_hypopnea=True, label_overlap_sec=5.0,
        context_minutes=5, target_length=900, min_beats=4, edge_policy="clamp",
        rebuild_cache=False, num_workers=1, seed=42, batch_size=128,
        loss="ce", focal_gamma=2.0, samples_per_epoch=None, max_records=None,
        experiment_name="apnea_cnntf",
    )


def main():
    args = make_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fe.build_model("cnn_transformer", args.target_length).to(device)
    ckpt = torch.load(Path(args.output_dir) / f"{args.experiment_name}.pth",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    rows, all_y, all_s = [], [], []
    for rid in WITHHELD:
        rec = fe.get_feature_record(args, "apnea_ecg", rid)
        if len(rec.labels) == 0:
            continue
        loader = fe.make_loader(rec.features, rec.labels, args.batch_size, False, False, None)
        y, pred, score = fe.predict(model, loader, device)
        all_y.append(y); all_s.append(score)
        hours = len(y) / 60.0
        rows.append({
            "record": rid, "minutes": int(len(y)),
            "true_min": int(np.sum(y)), "pred_min": int(np.sum(pred)),
            "true_AI": float(np.sum(y) / hours), "pred_AI": float(np.sum(pred) / hours),
        })

    true_min = np.array([r["true_min"] for r in rows], dtype=float)
    pred_min = np.array([r["pred_min"] for r in rows], dtype=float)
    true_ai = np.array([r["true_AI"] for r in rows])
    pred_ai = np.array([r["pred_AI"] for r in rows])

    pooled_auc = roc_auc_score(np.concatenate(all_y), np.concatenate(all_s))
    r_pear, _ = pearsonr(true_min, pred_min)
    rho, _ = spearmanr(true_min, pred_min)
    mae_min = float(np.mean(np.abs(true_min - pred_min)))
    mae_ai = float(np.mean(np.abs(true_ai - pred_ai)))
    # 二元 apneic：apnea 分鐘 >=100(PhysioNet A 類)
    acc_100 = accuracy_score((true_min >= 100).astype(int), (pred_min >= 100).astype(int))
    acc_ai5 = accuracy_score((true_ai >= 5).astype(int), (pred_ai >= 5).astype(int))

    print("\n========== Apnea-ECG per-recording 篩檢 (CNN+Transformer) ==========", flush=True)
    print(f"  recordings = {len(rows)}", flush=True)
    print(f"  per-minute pooled AUC = {pooled_auc:.4f}", flush=True)
    print(f"  apnea-minute Pearson r = {r_pear:.4f} | Spearman = {rho:.4f}", flush=True)
    print(f"  apnea-minute MAE = {mae_min:.2f} min | apnea-index MAE = {mae_ai:.2f} /h", flush=True)
    print(f"  apneic 判定準確率 (>=100 min) = {acc_100:.4f}", flush=True)
    print(f"  apneic 判定準確率 (AI>=5)     = {acc_ai5:.4f}", flush=True)
    print(f"  (Liu 2023 ref: per-recording acc 1.00, AHI MAE 4.33)", flush=True)
    import json
    (Path(args.output_dir) / "apnea_screening.json").write_text(
        json.dumps({"rows": rows, "pooled_auc": pooled_auc, "pearson": r_pear,
                    "mae_min": mae_min, "mae_ai": mae_ai, "acc_100": acc_100,
                    "acc_ai5": acc_ai5}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
