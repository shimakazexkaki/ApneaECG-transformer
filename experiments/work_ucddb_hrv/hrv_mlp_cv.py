"""Part B: a small deep MLP on the same HRV+CVHR features, grouped-CV, per-subject
normalized inputs. This is the deep-learning counterpart to the classical model in
hrv_grouped_cv.py, evaluated under the identical honest protocol so the comparison
is fair. Reuses the data loading, folds, metrics, and screening from that module.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # apnea project root
for _p in (_HERE, _ROOT / "lib", _ROOT):
    sys.path.insert(0, str(_p))

import hrv_features as hf  # noqa: E402
import hrv_grouped_cv as gcv  # noqa: E402


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(128, 64), dropout=0.4):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def compute_loss(loss_name, logits, y, g, class_w, focal_gamma=2.0):
    """logits:(B,2) y:(B,) g:(B,) subject ids. Returns scalar loss."""
    if loss_name == "ce":
        return nn.functional.cross_entropy(logits, y, label_smoothing=0.02)
    if loss_name == "wce":
        return nn.functional.cross_entropy(logits, y, weight=class_w, label_smoothing=0.02)
    if loss_name == "focal":
        ce = nn.functional.cross_entropy(logits, y, weight=class_w, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** focal_gamma) * ce).mean()
    if loss_name == "soft_auc":
        # Pairwise ranking loss: push score(pos) above score(neg). Directly targets AUC.
        s = torch.log_softmax(logits, dim=1)[:, 1]
        pos = s[y == 1]
        neg = s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return nn.functional.cross_entropy(logits, y, weight=class_w)
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)        # (P, N)
        return nn.functional.softplus(-diff).mean()       # logistic surrogate of 1[pos<neg]
    if loss_name in ("group_dro", "group_max"):
        # Per-subject mean CE, then emphasise the worst subjects (cross-subject robustness).
        ce = nn.functional.cross_entropy(logits, y, weight=class_w, reduction="none")
        uniq = torch.unique(g)
        group_losses = torch.stack([ce[g == gid].mean() for gid in uniq])
        if loss_name == "group_max":
            return group_losses.max()
        # softened worst-case: weight groups by their loss
        wts = torch.softmax(group_losses.detach() * 4.0, dim=0)
        return torch.sum(wts * group_losses)
    raise ValueError(f"Unknown loss: {loss_name}")


def train_mlp(X_tr, y_tr, g_tr, X_va, y_va, device, seed, loss_name="wce",
              epochs=120, lr=1e-3, wd=1e-4, patience=15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(X_tr)
    Xtr = torch.tensor(scaler.transform(X_tr), dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    gtr = torch.tensor(g_tr, dtype=torch.long, device=device)
    Xva = torch.tensor(scaler.transform(X_va), dtype=torch.float32, device=device)

    counts = np.bincount(y_tr, minlength=2)
    w = torch.tensor(len(y_tr) / (2.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    n = len(Xtr)
    bs = 256
    best_bacc, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = compute_loss(loss_name, model(Xtr[idx]), ytr[idx], gtr[idx], w)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_scores = torch.softmax(model(Xva), dim=1)[:, 1].cpu().numpy()
        _, m = gcv.best_threshold(y_va, va_scores)
        if m["balanced_accuracy"] > best_bacc:
            best_bacc, best_state, wait = m["balanced_accuracy"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler


def predict(model, scaler, X, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
        return torch.softmax(model(Xt), dim=1)[:, 1].cpu().numpy()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    blocks, _ = gcv.load_all(args)
    folds = gcv.make_folds(blocks, args.n_splits, args.seed)
    total = sum(len(b["y"]) for b in blocks)
    pos = sum(int(b["y"].sum()) for b in blocks)
    print(f"[MLP] device={device} blocks={len(blocks)} minutes={total} pos={pos} ({pos/total:.3f}) "
          f"norm={args.subject_norm} loss={args.loss} apnea_only={args.apnea_only}")

    all_records = {b["record"] for b in blocks}
    fold_results, pooled_y, pooled_scores, subject_rows = [], [], [], []
    for i, fold in enumerate(folds):
        test_records = set(fold["records"])
        trainval = sorted(all_records - test_records)
        rng = np.random.default_rng(args.seed + i)
        rng.shuffle(trainval)
        n_val = max(1, len(trainval) // 5)
        val_records = set(trainval[:n_val])
        train_records = set(trainval[n_val:])

        X_tr, y_tr = gcv.stack(blocks, train_records)
        X_va, y_va = gcv.stack(blocks, val_records)
        X_te, y_te = gcv.stack(blocks, test_records)
        if len(np.unique(y_tr)) < 2:
            continue

        # Subject ids aligned with X_tr rows (for group-robust losses).
        rec_to_id = {r: j for j, r in enumerate(sorted(train_records))}
        g_tr = np.concatenate([
            np.full(len(b["y"]), rec_to_id[b["record"]], dtype=np.int64)
            for b in blocks if b["record"] in train_records
        ])

        model, scaler = train_mlp(X_tr, y_tr, g_tr, X_va, y_va, device, args.seed + i, loss_name=args.loss)
        va_scores = predict(model, scaler, X_va, device)
        te_scores = predict(model, scaler, X_te, device)
        thr_val, _ = gcv.best_threshold(y_va, va_scores)
        thr_oracle, m_oracle = gcv.best_threshold(y_te, te_scores)

        fold_results.append({
            "fold": i, "test_records": sorted(test_records), "n_test": int(len(y_te)),
            "threshold_0_5": gcv.metrics_at(y_te, te_scores, 0.5),
            "threshold_val": gcv.metrics_at(y_te, te_scores, thr_val),
            "threshold_oracle": m_oracle,
        })
        pooled_y.append(y_te)
        pooled_scores.append(te_scores)
        for rid in sorted(test_records):
            recs = [b for b in blocks if b["record"] == rid]
            Xr = np.concatenate([b["X"] for b in recs])
            yr = np.concatenate([b["y"] for b in recs])
            sr = predict(model, scaler, Xr, device)
            subject_rows.append({
                "record": rid, "minutes": int(len(yr)),
                "true_apnea_fraction": float(yr.mean()),
                "mean_score": float(sr.mean()),
                "pred_apnea_fraction_at_val": float((sr >= thr_val).mean()),
            })

    pooled_y = np.concatenate(pooled_y)
    pooled_scores = np.concatenate(pooled_scores)
    pooled_auc = float(gcv.roc_auc_score(pooled_y, pooled_scores))
    screening = gcv.screening_report(subject_rows)

    print(f"\n=== MLP minute-level ({len(fold_results)} folds) ===")
    print(f"Pooled out-of-fold AUC: {pooled_auc:.4f}")
    for level in ["threshold_0_5", "threshold_val", "threshold_oracle"]:
        a = gcv.aggregate(fold_results, level)
        print(f"  {level:16s} BAcc={a['balanced_accuracy']['mean']:.4f} AUC={a['roc_auc']['mean']:.4f} "
              f"F1={a['f1']['mean']:.4f} Rec={a['recall']['mean']:.4f} Spec={a['specificity']['mean']:.4f}")
    print(f"\n=== MLP subject-level screening ===")
    print(f"  Pearson(score, true burden)  = {screening.get('pearson_score_vs_trueburden', 0):.3f}")
    for tier in (10, 15, 20):
        k = f"screening_auc_burden_ge_{tier}pct"
        if k in screening:
            print(f"  Screening AUC (burden>={tier}%): {screening[k]:.3f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "settings": vars(args), "pooled_auc": pooled_auc,
            "aggregate": {lv: gcv.aggregate(fold_results, lv)
                          for lv in ["threshold_0_5", "threshold_val", "threshold_oracle"]},
            "screening": screening, "subject_rows": subject_rows,
        }, indent=2), encoding="utf-8")
        print(f"\nSaved: {out}")


def main():
    p = argparse.ArgumentParser(description="Deep MLP on HRV features, grouped CV (Part B).")
    p.add_argument("--ucddb-dir", default="ucddb")
    p.add_argument("--channels", nargs="+", type=int, default=[0, 2])
    p.add_argument("--apnea-only", action="store_true")
    p.add_argument("--context-minutes", type=int, default=5)
    p.add_argument("--min-overlap-sec", type=float, default=5.0)
    p.add_argument("--min-beats", type=int, default=20)
    p.add_argument("--subject-norm", choices=["none", "zscore", "robust"], default="zscore")
    p.add_argument("--loss", choices=["ce", "wce", "focal", "soft_auc", "group_dro", "group_max"], default="wce")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-no-positive", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
