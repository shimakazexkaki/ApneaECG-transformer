"""P0: 誠實的 record-level grouped CV + 超參數調整，主角是提案的 CNN+Transformer。

為什麼存在：
- `fnins_experiment.py` 的 UCDDB 評估是 8:1:1 segment split（同受試者洩漏、數字虛高）。
- 本檔改用「整位受試者 held out」的 5-fold grouped CV，才能與 HRV ensemble（grouped-CV AUC 0.570）
  公平比較，也才知道調參是否真的有效。

重用：
- 特徵/模型/訓練：`fnins_experiment`（get_feature_records、CNNTransformer、build_criterion、train_epoch...）。
- fold 切分與指標：`hrv_grouped_cv`（make_folds、best_threshold、metrics_at、aggregate）。
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent          # experiments/work_fnins_baseline
_ROOT = _HERE.parents[1]                          # apnea project root
for _p in (_HERE, _ROOT / "lib", _ROOT, _ROOT / "experiments" / "work_ucddb_hrv"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fnins_experiment as fn   # noqa: E402
import hrv_grouped_cv as gcv    # noqa: E402


def feature_args(channel, context, cache_dir, max_bad_rr_fraction=0.0):
    """Minimal Namespace with every field fnins.get_feature_record reads (UCDDB path)."""
    return SimpleNamespace(
        dataset="ucddb", ucddb_dir=str(_ROOT / "ucddb"), ucddb_channel=channel,
        ucddb_literature_cache_dir=str(_HERE / "cache_ucddb_lit"),
        apnea_dir=str(_ROOT / "apnea-ecg"), apnea_rpeak_source="hamilton",
        include_hypopnea=True, label_overlap_sec=5.0,
        context_minutes=context, target_length=900, min_beats=4, edge_policy="clamp",
        cache_dir=str(cache_dir), rebuild_cache=False, num_workers=1,
        max_bad_rr_fraction=max_bad_rr_fraction,  # P3 signal-quality gate (0 = off)
    )


def load_blocks(args):
    """One block per (record, channel): {record, channel, X:(n,900,2), y, centers}."""
    records = fn.available_ucddb_records(_ROOT / "ucddb")
    if not getattr(args, "include_all_records", False):
        records = [r for r in records if r not in fn.UCDDB_EXCLUDED_NO_SA]
    blocks = []
    for ch in args.channels:
        fa = feature_args(ch, args.context_minutes, args.cache_dir, args.max_bad_rr_fraction)
        for rec in fn.get_feature_records(fa, "ucddb", records):
            if len(rec.labels):
                blocks.append({"record": rec.record_id, "channel": ch,
                               "X": rec.features, "y": rec.labels, "centers": rec.centers})
    if not blocks:
        raise RuntimeError("No UCDDB feature blocks loaded.")
    return blocks


def make_model(args, device):
    return fn.CNNTransformer(input_channels=2, d_model=args.d_model, nhead=args.nhead,
                             num_layers=args.layers, dropout=args.dropout).to(device)


def train_one_fold(args, blocks, train_recs, val_recs, test_recs, device, seed):
    fn.set_seed(seed)
    X_tr, y_tr = gcv.stack(blocks, train_recs)
    X_va, y_va = gcv.stack(blocks, val_recs)
    X_te, y_te = gcv.stack(blocks, test_recs)
    if len(np.unique(y_tr)) < 2:
        return None

    train_loader = fn.make_loader(X_tr, y_tr, args.batch_size, shuffle=True,
                                  oversample=True, samples_per_epoch=args.samples_per_epoch)
    val_loader = fn.make_loader(X_va, y_va, args.batch_size, False, False, None)
    test_loader = fn.make_loader(X_te, y_te, args.batch_size, False, False, None)

    model = make_model(args, device)
    crit_args = SimpleNamespace(loss=args.loss, focal_gamma=args.focal_gamma)
    criterion = fn.build_criterion(crit_args, y_tr, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_bacc, best_state, wait = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        fn.train_epoch(model, train_loader, criterion, opt, device)
        vy, _, vs = fn.predict(model, val_loader, device)
        _, m = gcv.best_threshold(vy, vs)
        if m["balanced_accuracy"] > best_bacc:
            best_bacc = m["balanced_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    vy, _, vs = fn.predict(model, val_loader, device)
    thr_val, _ = gcv.best_threshold(vy, vs)
    ty, _, ts = fn.predict(model, test_loader, device)

    # P1: per-test-subject burden rows (predict each record separately to keep identity).
    subject_rows = []
    for rid in sorted(test_recs):
        recs = [b for b in blocks if b["record"] == rid]
        if not recs:
            continue
        Xr = np.concatenate([b["X"] for b in recs])
        yr = np.concatenate([b["y"] for b in recs])
        ry, _, rs = fn.predict(model, fn.make_loader(Xr, yr, args.batch_size, False, False, None), device)
        subject_rows.append({
            "record": rid, "minutes": int(len(yr)),
            "true_apnea_fraction": float(yr.mean()),
            "mean_score": float(rs.mean()),
            "pred_apnea_fraction_at_val": float((rs >= thr_val).mean()),
        })

    return {
        "threshold_0_5": gcv.metrics_at(ty, ts, 0.5),
        "threshold_val": gcv.metrics_at(ty, ts, thr_val),
        "threshold_oracle": gcv.best_threshold(ty, ts)[1],
        "y_te": ty, "scores": ts, "subject_rows": subject_rows,
    }


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    blocks = load_blocks(args)
    folds = gcv.make_folds(blocks, args.n_splits, args.seed)
    all_recs = {b["record"] for b in blocks}
    total = sum(len(b["y"]) for b in blocks)
    pos = sum(int(b["y"].sum()) for b in blocks)
    print(f"[CNN+Transformer grouped CV] device={device} blocks={len(blocks)} minutes={total} "
          f"pos={pos} ({pos/total:.3f}) channels={args.channels} context={args.context_minutes} "
          f"d_model={args.d_model} layers={args.layers} nhead={args.nhead} dropout={args.dropout} "
          f"lr={args.lr} loss={args.loss} gamma={args.focal_gamma}")

    fold_results, pooled_y, pooled_scores, subject_rows = [], [], [], []
    for i, fold in enumerate(folds):
        test_recs = set(fold["records"])
        trainval = sorted(all_recs - test_recs)
        rng = np.random.default_rng(args.seed + i)
        rng.shuffle(trainval)
        n_val = max(1, len(trainval) // 5)
        val_recs, train_recs = set(trainval[:n_val]), set(trainval[n_val:])
        res = train_one_fold(args, blocks, train_recs, val_recs, test_recs, device, args.seed + i)
        if res is None:
            continue
        fold_results.append({k: res[k] for k in ("threshold_0_5", "threshold_val", "threshold_oracle")})
        pooled_y.append(res["y_te"])
        pooled_scores.append(res["scores"])
        subject_rows.extend(res["subject_rows"])
        m = fold_results[-1]["threshold_val"]
        print(f"  fold {i} test={sorted(test_recs)} "
              f"AUC={m['roc_auc']:.4f} BAcc={m['balanced_accuracy']:.4f} F1={m['f1']:.4f}")

    pooled_y = np.concatenate(pooled_y)
    pooled_scores = np.concatenate(pooled_scores)
    pooled_auc = float(gcv.roc_auc_score(pooled_y, pooled_scores)) if len(np.unique(pooled_y)) > 1 else 0.0

    screening = gcv.screening_report(subject_rows)
    summary = {
        "settings": vars(args),
        "pooled_auc": pooled_auc,
        "aggregate": {lv: gcv.aggregate(fold_results, lv)
                      for lv in ("threshold_0_5", "threshold_val", "threshold_oracle")},
        "screening": screening,
        "subject_rows": subject_rows,
    }
    print(f"\n=== Minute-level aggregate ({len(fold_results)} folds) ===")
    print(f"Pooled out-of-fold AUC: {pooled_auc:.4f}   (對照 HRV ensemble 0.570 / 舊從頭 ~0.55)")
    for lv in ("threshold_0_5", "threshold_val", "threshold_oracle"):
        a = summary["aggregate"][lv]
        print(f"  {lv:16s} BAcc={a['balanced_accuracy']['mean']:.4f}+/-{a['balanced_accuracy']['std']:.4f} "
              f"AUC={a['roc_auc']['mean']:.4f} F1={a['f1']['mean']:.4f} "
              f"Rec={a['recall']['mean']:.4f} Spec={a['specificity']['mean']:.4f}")
    print(f"\n=== Subject-level screening ({screening.get('n_subjects', 0)} subjects) ===")
    print(f"  Pearson(score, true burden)  = {screening.get('pearson_score_vs_trueburden', 0):.3f}")
    print(f"  Spearman(score, true burden) = {screening.get('spearman_score_vs_trueburden', 0):.3f}")
    for tier in (10, 15, 20):
        k = f"screening_auc_burden_ge_{tier}pct"
        if k in screening:
            print(f"  Screening AUC (burden>={tier}%): {screening[k]:.3f}  (n_high={screening.get(f'n_high_burden_ge_{tier}pct')})")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved: {args.output}")
    return summary


def main():
    p = argparse.ArgumentParser(description="CNN+Transformer record-level grouped CV (UCDDB).")
    p.add_argument("--channels", nargs="+", type=int, default=[0, 2])
    p.add_argument("--context-minutes", type=int, default=5)
    p.add_argument("--max-bad-rr-fraction", type=float, default=0.0,
                   help="P3 signal-quality gate: drop minutes whose raw-RR bad fraction exceeds this (0 = off)")
    p.add_argument("--include-all-records", action="store_true",
                   help="P1 screening: keep the 4 no-SA records (008/011/013/018) as low-burden negatives")
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--samples-per-epoch", type=int, default=None)
    p.add_argument("--loss", choices=["ce", "wce", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=str(_HERE / "cache"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
