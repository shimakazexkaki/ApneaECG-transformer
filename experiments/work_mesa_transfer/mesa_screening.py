"""Per-subject AHI 篩檢評估(對標 Olsen 2020 的 headline 指標)。

重用已訓練好的 grouped-CV fold 模型:對每位 held-out 受試者預測 → 聚合成
「每位的預測 apnea 負荷」,與真實負荷比對:
- pooled per-segment AUC
- 每位負荷的 Spearman 相關(預測 vs 真實)
- 三級嚴重度(低/中/高,以真實負荷 tertile 切)分類準確率
這對應 Olsen 的 AHI 嚴重度 Acc 84.9% / R²=0.83。

用法(等對應的訓練跑完、模型存好後):
  python mesa_screening.py --experiment-name mesa_internal_seg10 --label-mode segment
  python mesa_screening.py --experiment-name mesa_internal_seg10_rrid --extra-channels rrid
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score

WORK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WORK_DIR.parents[1]
PAPER_DIR = PROJECT_DIR / "experiments" / "work_ucddb_paper_replication"
for _p in (PROJECT_DIR / "lib", PROJECT_DIR, PAPER_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ucddb_literature_train_common as common  # noqa: E402
import mesa_features as mf  # noqa: E402
import paper_cnn_transformer_lstm_trainer as paper  # noqa: E402
import mesa_internal_trainer as mit  # noqa: E402


def tertile_bins(values):
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    return np.digitize(values, [q1, q2])  # 0/1/2


def main():
    paper.patch_model_factory()
    args = mit.build_args()
    args.use_transformer = args.ablation in ("full", "no_lstm")
    args.use_lstm = args.ablation in ("full", "no_transformer")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if getattr(args, "use_all_available", False):
        ids = mf.available_mesa_records(args.mesa_dir)
        if args.mesa_limit:
            ids = ids[: args.mesa_limit]
    else:
        ids = mit.cached_subjects(args.mesa_cache_dir)
    records = mf.load_mesa_segment_records(args, args.mesa_dir, record_ids=ids) \
        if args.label_mode == "segment" else mf.load_mesa_records(args, args.mesa_dir, record_ids=ids)
    empty = np.zeros((1,), dtype=np.float32)
    for r in records:
        r.signal = empty

    folds = common.balanced_folds(records, args.n_splits, args.seed)
    arrays = common.records_to_arrays(records)
    if args.extra_channels == "rrid":
        f = arrays["features"]; rr = f[:, :, 0]
        rrid = np.diff(rr, axis=1, prepend=rr[:, :1])
        rrid = (rrid - rrid.mean(axis=1, keepdims=True)) / (rrid.std(axis=1, keepdims=True) + 1e-8)
        arrays["features"] = np.concatenate([f, rrid[:, :, None]], axis=2).astype(np.float32)
    args.input_dim = int(arrays["features"].shape[-1])
    rec_ids = arrays["record_ids"]

    out_dir = Path(args.output_dir) / args.experiment_name
    per_subj = {}  # subject -> {"y": [], "s": []}
    all_y, all_s = [], []
    for fi in range(args.n_splits):
        test_recs = folds[fi]["records"]
        test_idx = np.flatnonzero(np.isin(rec_ids, test_recs))
        model_path = out_dir / f"mesa_internal_fold{fi}_model.pth"
        model = common.make_model(args, "paper_cnn_transformer_lstm").to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        ds = common.LiteratureDataset(arrays, test_idx, include_raw=False, context_minutes=args.context_minutes)
        y, s, metas = common.predict_scores(model, ds, args.batch_size, device, args.amp, True)
        all_y.append(y); all_s.append(s)
        for yi, si, rid in zip(y, s, metas["record_ids"]):
            d = per_subj.setdefault(str(rid), {"y": [], "s": []})
            d["y"].append(int(yi)); d["s"].append(float(si))
        print(f"[screening] fold {fi}: {len(test_recs)} subjects scored", flush=True)

    pooled_auc = roc_auc_score(np.concatenate(all_y), np.concatenate(all_s))

    subs = sorted(per_subj)
    true_rate = np.array([np.mean(per_subj[s]["y"]) for s in subs])   # 真實 apnea 負荷 ∝ AHI
    pred_rate = np.array([np.mean(per_subj[s]["s"]) for s in subs])   # 預測負荷
    rho, p = spearmanr(true_rate, pred_rate)
    true_bin = tertile_bins(true_rate)
    pred_bin = tertile_bins(pred_rate)
    sev_acc = accuracy_score(true_bin, pred_bin)
    burden_mae = float(np.mean(np.abs(true_rate - pred_rate)))

    print("\n========== MESA per-subject screening (Olsen-style) ==========", flush=True)
    print(f"  subjects = {len(subs)}", flush=True)
    print(f"  pooled per-segment AUC = {pooled_auc:.4f}", flush=True)
    print(f"  per-subject burden Spearman rho = {rho:.4f} (p={p:.2e})", flush=True)
    print(f"  3-tier severity (tertiles) accuracy = {sev_acc:.4f}", flush=True)
    print(f"  burden MAE (rate units) = {burden_mae:.4f}", flush=True)
    print(f"  (Olsen 2020 ref: AHI-severity acc 0.849, R^2 0.83, per-event F1 0.67)", flush=True)


if __name__ == "__main__":
    main()
