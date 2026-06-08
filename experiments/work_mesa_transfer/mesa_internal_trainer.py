"""MESA-internal 評估:train 一部分 MESA → 測 held-out MESA 受試者(record-level grouped CV)。

同資料集、無 domain shift。用來判定:UCDDB 外部只有 ~0.55,到底是
(a) UCDDB 太髒 / 跨資料集 domain shift,還是 (b) minute-ECG 訊號本身的天花板。
- 若 MESA-internal AUC 高(0.75+) → (a),值得猛推 MESA。
- 若 MESA-internal 也 ~0.6 → (b),訊號天花板。

只用「特徵已快取」的受試者,避免重建(維持快)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WORK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WORK_DIR.parents[1]
PAPER_DIR = PROJECT_DIR / "experiments" / "work_ucddb_paper_replication"
for _p in (PROJECT_DIR / "lib", PROJECT_DIR, PAPER_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ucddb_literature_train_common as common  # noqa: E402
import mesa_features as mf  # noqa: E402
import paper_cnn_transformer_lstm_trainer as paper  # noqa: E402


def cached_subjects(cache_dir):
    """只認 per-minute 快取(含 signal+rpeaks),供 segment 版重用、不重跑 R-peak。"""
    ids = []
    for p in sorted(Path(cache_dir).glob("*_ekg_*.npz")):
        if "_seg" in p.name:
            continue
        ids.append(p.name.split("_ekg_")[0])
    return sorted(set(ids))


def build_args():
    parser = argparse.ArgumentParser()
    common.add_train_args(parser)
    parser.add_argument("--cnn-dropout", type=float, default=0.5)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--mesa-dir", default="D:/mesa")
    parser.add_argument("--mesa-cache-dir", default=str(WORK_DIR / "cache" / "mesa_features"))
    parser.add_argument("--mesa-limit", type=int, default=0)
    parser.add_argument("--mesa-protocol", choices=["grouped", "segment"], default="grouped",
                        help="grouped=subject-wise CV(誠實); segment=window 隨機 8:1:1(他們的協定,有洩漏)")
    parser.add_argument("--label-mode", choices=["minute", "segment"], default="minute",
                        help="minute=逐分鐘≥5s重疊; segment=細粒度乾淨標籤(完整落在事件內=陽性)")
    parser.add_argument("--segment-sec", type=float, default=10.0)
    parser.add_argument("--segment-stride-sec", type=float, default=10.0)
    parser.add_argument("--extra-channels", choices=["none", "rrid"], default="none",
                        help="rrid=加 RR 一階差分當第3通道(對標 DSF-SANet/Olsen)")
    parser.add_argument("--ablation", choices=["full", "no_lstm", "no_transformer", "cnn_only"],
                        default="no_lstm", help="預設 no_lstm=CNN+Transformer(消融顯示 LSTM 多餘)")
    parser.add_argument("--norm-mode", choices=["window", "recording"], default="window",
                        help="recording=用整夜統計量正規化(保留個體水準,供 per-subject 篩檢)")
    parser.add_argument("--use-all-available", action="store_true",
                        help="用所有已下載(edf+xml)的受試者(可配 --mesa-limit),而非只用已快取的")
    parser.set_defaults(
        output_dir=str(WORK_DIR / "outputs"),
        channels=[0], detector="biosppy_hamilton",
        lr=1e-3, nhead=8, batch_size=128, epochs=20, patience=6,
        n_splits=5,
    )
    return parser.parse_args()


def main():
    paper.patch_model_factory()
    args = build_args()
    args.use_transformer = args.ablation in ("full", "no_lstm")
    args.use_lstm = args.ablation in ("full", "no_transformer")

    if getattr(args, "use_all_available", False):
        ids = mf.available_mesa_records(args.mesa_dir)
        if args.mesa_limit:
            ids = ids[: args.mesa_limit]
    else:
        ids = cached_subjects(args.mesa_cache_dir)
    print(f"[mesa-internal] subjects = {len(ids)}  label_mode={args.label_mode}  "
          f"use_all_available={getattr(args, 'use_all_available', False)}", flush=True)
    if args.label_mode == "segment":
        records = mf.load_mesa_segment_records(args, args.mesa_dir, record_ids=ids)
    else:
        records = mf.load_mesa_records(args, args.mesa_dir, record_ids=ids)
    pos = sum(int(r.labels.sum()) for r in records)
    tot = sum(int(len(r.labels)) for r in records)
    print(f"[mesa-internal] samples={tot} positive={pos} ({pos/max(tot,1):.3f})", flush=True)
    empty = np.zeros((1,), dtype=np.float32)
    for r in records:
        r.signal = empty

    arrays = common.records_to_arrays(records)
    if args.extra_channels == "rrid":
        f = arrays["features"]  # (N, L, 2): [rr_z, amp_z]
        rr = f[:, :, 0]
        rrid = np.diff(rr, axis=1, prepend=rr[:, :1])
        rrid = (rrid - rrid.mean(axis=1, keepdims=True)) / (rrid.std(axis=1, keepdims=True) + 1e-8)
        arrays["features"] = np.concatenate([f, rrid[:, :, None]], axis=2).astype(np.float32)
        print(f"[mesa-internal] +RRID channel -> features {arrays['features'].shape}", flush=True)
    args.input_dim = int(arrays["features"].shape[-1])
    rec_ids = arrays["record_ids"]
    from sklearn.model_selection import train_test_split

    if args.mesa_protocol == "segment":
        # 他們的協定:把所有窗倒一起,隨機 8:1:1(同受試者的窗會同時進 train/test → 洩漏)
        idx = np.arange(len(arrays["labels"]))
        y = arrays["labels"]
        tv, test_idx = train_test_split(idx, test_size=0.1, random_state=args.seed, stratify=y)
        train_idx, val_idx = train_test_split(tv, test_size=0.111, random_state=args.seed, stratify=y[tv])
        print(f"[segment] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
        res = common.train_one_split(args, "paper_cnn_transformer_lstm", arrays,
                                     train_idx, val_idx, test_idx, "mesa_internal_segment",
                                     include_raw=False)
        tv_m = res["test"]["threshold_val"]; f1_m = res["test"]["threshold_f1_val"]
        out = Path(args.output_dir) / "mesa_internal_segment"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(
            {"n_subjects": len(ids), "protocol": "segment_split_8:1:1",
             "test_default": tv_m, "test_f1opt": f1_m}, indent=2, default=str), encoding="utf-8")
        print("\n========== MESA segment split (他們的協定,window 隨機切) ==========", flush=True)
        print(f"  AUC={tv_m['roc_auc']:.4f}  Acc={tv_m['accuracy']:.3f} Rec={tv_m['recall']:.3f} "
              f"Prec={tv_m['precision']:.3f} F1={tv_m['f1']:.3f} Spec={tv_m['specificity']:.3f}", flush=True)
        print(f"  F1-opt: Acc={f1_m['accuracy']:.3f} Rec={f1_m['recall']:.3f} Prec={f1_m['precision']:.3f} "
              f"F1={f1_m['f1']:.3f}", flush=True)
        return

    folds = common.balanced_folds(records, args.n_splits, args.seed)
    results = []
    for fi in range(args.n_splits):
        test_recs = folds[fi]["records"]
        trainval = [r for j, f in enumerate(folds) if j != fi for r in f["records"]]
        from sklearn.model_selection import train_test_split
        tr_recs, val_recs = train_test_split(trainval, test_size=0.2, random_state=args.seed + fi)
        train_idx = np.flatnonzero(np.isin(rec_ids, tr_recs))
        val_idx = np.flatnonzero(np.isin(rec_ids, val_recs))
        test_idx = np.flatnonzero(np.isin(rec_ids, test_recs))
        print(f"[fold {fi}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
              f"(test subj={len(test_recs)})", flush=True)
        res = common.train_one_split(args, "paper_cnn_transformer_lstm", arrays,
                                     train_idx, val_idx, test_idx, f"mesa_internal_fold{fi}",
                                     include_raw=False)
        tv = res["test"]["threshold_val"]
        print(f"  fold {fi}: AUC={tv['roc_auc']:.4f} BAcc={tv['balanced_accuracy']:.4f} F1={tv['f1']:.4f}", flush=True)
        results.append(res)

    agg = common.aggregate_results(results)
    out = Path(args.output_dir) / "mesa_internal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"n_subjects": len(ids), "subjects": ids, "aggregate": agg,
         "results": [{"fold": i, "test": r["test"]["threshold_val"]} for i, r in enumerate(results)]},
        indent=2, default=str), encoding="utf-8")

    a = agg["threshold_val"]
    print("\n========== MESA-internal (grouped CV, held-out MESA subjects) ==========", flush=True)
    print(f"  AUC={a['roc_auc']['mean']:.4f} +/- {a['roc_auc']['std']:.4f}", flush=True)
    print(f"  BAcc={a['balanced_accuracy']['mean']:.4f}  F1={a['f1']['mean']:.4f}  "
          f"Rec={a['recall']['mean']:.4f}  Spec={a['specificity']['mean']:.4f}", flush=True)
    print(f"  Saved: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
