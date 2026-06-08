"""Record-level grouped cross-validation for UCDDB apnea detection using explicit
HRV + CVHR + EDR scalar features (see hrv_features.py).

Key ideas being tested against the AUC ~0.55 cross-subject baseline:

1. Domain-knowledge features instead of a learned (900, 2) sequence.
2. Per-subject feature normalization (z-score or robust) so that each subject's
   features are expressed relative to their own overnight baseline. This mimics a
   wearable calibrating to its wearer and uses no labels, so it is valid for the
   held-out test subjects too.
3. Well-regularized classifiers (L2 logistic regression, histogram gradient
   boosting) that generalize on ~25 subjects far better than a deep net.

Honest protocol: entire subjects are held out per fold (GroupKFold-style balanced
folds), matching the existing project's grouped-CV reporting.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # apnea project root
for _p in (_HERE, _ROOT / "lib", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hrv_features as hf  # noqa: E402
import ucddb_runner  # noqa: E402

CACHE_DIR = _HERE / "cache"


# --------------------------------------------------------------------------- #
# Data loading (with on-disk feature cache)
# --------------------------------------------------------------------------- #
def feature_cache_path(record_id, channel, apnea_only, context_minutes, overlap):
    tag = "apneaonly" if apnea_only else "hyp"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{record_id}_ch{channel}_{tag}_ctx{context_minutes}_ov{overlap}_hrvfeat.npz"


def load_record(record_id, channel, args):
    path = feature_cache_path(record_id, channel, args.apnea_only, args.context_minutes, args.min_overlap_sec)
    if path.exists() and not args.rebuild_cache:
        c = np.load(path, allow_pickle=False)
        return c["features"].astype(np.float32), c["labels"].astype(np.int64), c["minute_indices"].astype(np.int32)
    rec = hf.extract_record(
        record_id,
        channel=channel,
        apnea_only=args.apnea_only,
        context_minutes=args.context_minutes,
        min_overlap_sec=args.min_overlap_sec,
        min_beats=args.min_beats,
        ucddb_dir=args.ucddb_dir,
    )
    np.savez_compressed(path, features=rec.features, labels=rec.labels, minute_indices=rec.minute_indices)
    return rec.features, rec.labels, rec.minute_indices


def normalize_per_subject(features, mode):
    """Normalize one subject-channel's features by its own distribution."""
    if mode == "none" or len(features) == 0:
        return features
    x = features.astype(np.float64)
    if mode == "zscore":
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        out = (x - mu) / (sd + 1e-6)
    elif mode == "robust":
        med = np.median(x, axis=0, keepdims=True)
        q1 = np.percentile(x, 25, axis=0, keepdims=True)
        q3 = np.percentile(x, 75, axis=0, keepdims=True)
        out = (x - med) / (q3 - q1 + 1e-6)
    else:
        raise ValueError(f"Unknown subject-normalization mode: {mode}")
    return out.astype(np.float32)


def load_all(args):
    """Return list of per-(record,channel) feature blocks plus a record->group map."""
    record_ids = ucddb_runner.available_record_ids(Path(args.ucddb_dir))
    blocks = []
    for rid in record_ids:
        has_pos = False
        rec_blocks = []
        for ch in args.channels:
            feats, labels, mins = load_record(rid, ch, args)
            if len(labels) == 0:
                continue
            feats = normalize_per_subject(feats, args.subject_norm)
            rec_blocks.append({"record": rid, "channel": ch, "X": feats, "y": labels, "minutes": mins})
            has_pos = has_pos or labels.sum() > 0
        if args.exclude_no_positive and not has_pos:
            continue
        blocks.extend(rec_blocks)
    if not blocks:
        raise RuntimeError("No UCDDB HRV features were loaded.")
    return blocks, record_ids


# --------------------------------------------------------------------------- #
# Balanced subject folds (match the existing grouped-CV fold construction)
# --------------------------------------------------------------------------- #
def make_folds(blocks, n_splits, seed):
    per_record = {}
    for b in blocks:
        item = per_record.setdefault(b["record"], {"record": b["record"], "n": 0, "pos": 0})
        item["n"] += len(b["y"])
        item["pos"] += int(b["y"].sum())
    summaries = list(per_record.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(summaries)
    summaries.sort(key=lambda s: (s["pos"], s["n"]), reverse=True)
    folds = [{"records": [], "n": 0, "pos": 0} for _ in range(n_splits)]
    for s in summaries:
        f = min(folds, key=lambda fo: (fo["pos"], fo["n"], len(fo["records"])))
        f["records"].append(s["record"])
        f["n"] += s["n"]
        f["pos"] += s["pos"]
    for f in folds:
        f["records"].sort()
        f["positive_ratio"] = f["pos"] / f["n"] if f["n"] else 0.0
    return folds


def stack(blocks, record_set):
    X = [b["X"] for b in blocks if b["record"] in record_set]
    y = [b["y"] for b in blocks if b["record"] in record_set]
    if not X:
        return np.empty((0, hf.N_FEATURES), np.float32), np.empty(0, np.int64)
    return np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int64)


# --------------------------------------------------------------------------- #
# Metrics + thresholding
# --------------------------------------------------------------------------- #
def metrics_at(y, scores, thr):
    pred = (scores >= thr).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec = recall_score(y, pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        auc = 0.0
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(rec),
        "specificity": float(spec),
        "balanced_accuracy": float((rec + spec) / 2),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }


def best_threshold(y, scores):
    best = None
    for thr in np.linspace(0.02, 0.98, 193):
        m = metrics_at(y, scores, thr)
        key = (m["balanced_accuracy"], m["f1"])
        if best is None or key > best[0]:
            best = (key, float(thr), m)
    return best[1], best[2]


def _logreg(seed):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=seed)),
    ])


def _hgb(seed):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0,
        class_weight="balanced", random_state=seed,
    )


class _Ensemble:
    """Soft-voting average of logistic regression and gradient boosting."""

    def __init__(self, seed):
        self.models = [_logreg(seed), _hgb(seed)]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.stack([1 - p, p], axis=1)


def make_classifier(name, seed):
    if name == "logreg":
        return _logreg(seed)
    if name == "hgb":
        return _hgb(seed)
    if name == "ensemble":
        return _Ensemble(seed)
    raise ValueError(f"Unknown classifier: {name}")


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def run_cv(blocks, folds, args):
    fold_results = []
    pooled_y, pooled_scores = [], []
    subject_rows = []  # one row per held-out subject for screening evaluation
    for i, fold in enumerate(folds):
        test_records = set(fold["records"])
        train_records = {b["record"] for b in blocks} - test_records
        X_tr, y_tr = stack(blocks, train_records)
        X_te, y_te = stack(blocks, test_records)
        if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
            continue

        clf = make_classifier(args.classifier, args.seed + i)
        clf.fit(X_tr, y_tr)
        train_scores = clf.predict_proba(X_tr)[:, 1]
        test_scores = clf.predict_proba(X_te)[:, 1]

        thr_val, _ = best_threshold(y_tr, train_scores)  # tuned on TRAIN only
        thr_oracle, m_oracle = best_threshold(y_te, test_scores)  # upper bound

        fold_results.append({
            "fold": i,
            "test_records": sorted(test_records),
            "n_test": int(len(y_te)),
            "pos_ratio": float(y_te.mean()),
            "threshold_0_5": metrics_at(y_te, test_scores, 0.5),
            "threshold_val": metrics_at(y_te, test_scores, thr_val),
            "threshold_oracle": m_oracle,
            "thr_val": float(thr_val),
            "thr_oracle": float(thr_oracle),
        })
        pooled_y.append(y_te)
        pooled_scores.append(test_scores)

        # Per-subject screening rows (subject-level burden), grouped over channels.
        for rid in sorted(test_records):
            recs = [b for b in blocks if b["record"] == rid]
            if not recs:
                continue
            Xr = np.concatenate([b["X"] for b in recs])
            yr = np.concatenate([b["y"] for b in recs])
            sr = clf.predict_proba(Xr)[:, 1]
            subject_rows.append({
                "record": rid,
                "fold": i,
                "minutes": int(len(yr)),
                "true_apnea_fraction": float(yr.mean()),
                "true_apnea_min_per_hr": float(yr.mean() * 60.0),
                "mean_score": float(sr.mean()),
                "pred_apnea_fraction_at_val": float((sr >= thr_val).mean()),
                "pred_apnea_min_per_hr_at_val": float((sr >= thr_val).mean() * 60.0),
            })

    pooled_y = np.concatenate(pooled_y)
    pooled_scores = np.concatenate(pooled_scores)
    pooled_auc = float(roc_auc_score(pooled_y, pooled_scores)) if len(np.unique(pooled_y)) > 1 else 0.0
    return fold_results, pooled_auc, subject_rows


def screening_report(subject_rows):
    """Subject-level screening: can we rank subjects by apnea burden / flag high-AHI?"""
    if len(subject_rows) < 3:
        return {}
    true_frac = np.array([r["true_apnea_fraction"] for r in subject_rows])
    mean_score = np.array([r["mean_score"] for r in subject_rows])
    pred_frac = np.array([r["pred_apnea_fraction_at_val"] for r in subject_rows])

    def corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return corr(ra, rb)

    report = {
        "n_subjects": len(subject_rows),
        "pearson_score_vs_trueburden": corr(mean_score, true_frac),
        "spearman_score_vs_trueburden": spearman(mean_score, true_frac),
        "pearson_predfrac_vs_truefrac": corr(pred_frac, true_frac),
        "mae_apnea_min_per_hr": float(np.mean(np.abs(pred_frac - true_frac) * 60.0)),
    }
    # Screening AUC: flag subjects above burden tiers using the continuous mean_score.
    for tier in (0.10, 0.15, 0.20):
        high = (true_frac >= tier).astype(int)
        if 0 < high.sum() < len(high):
            report[f"screening_auc_burden_ge_{int(tier*100)}pct"] = float(roc_auc_score(high, mean_score))
            report[f"n_high_burden_ge_{int(tier*100)}pct"] = int(high.sum())
    return report


def aggregate(fold_results, level):
    keys = ["accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1", "roc_auc"]
    agg = {}
    for k in keys:
        vals = [fr[level][k] for fr in fold_results]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "min": float(np.min(vals)), "max": float(np.max(vals))}
    return agg


def main():
    p = argparse.ArgumentParser(description="UCDDB HRV/CVHR grouped-CV apnea detector.")
    p.add_argument("--ucddb-dir", default="ucddb")
    p.add_argument("--channels", nargs="+", type=int, default=[0, 2])
    p.add_argument("--apnea-only", action="store_true")
    p.add_argument("--context-minutes", type=int, default=5)
    p.add_argument("--min-overlap-sec", type=float, default=5.0)
    p.add_argument("--min-beats", type=int, default=20)
    p.add_argument("--subject-norm", choices=["none", "zscore", "robust"], default="zscore")
    p.add_argument("--classifier", choices=["logreg", "hgb", "ensemble"], default="hgb")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-no-positive", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    blocks, record_ids = load_all(args)
    folds = make_folds(blocks, args.n_splits, args.seed)
    total = sum(len(b["y"]) for b in blocks)
    pos = sum(int(b["y"].sum()) for b in blocks)
    print(f"Loaded {len(blocks)} record-channel blocks | minutes={total} pos={pos} ({pos/total:.3f})")
    print(f"Channels={args.channels} subject_norm={args.subject_norm} classifier={args.classifier} "
          f"apnea_only={args.apnea_only}")
    for f in folds:
        print(f"  fold {f['records']}  n={f['n']} posratio={f['positive_ratio']:.3f}")

    fold_results, pooled_auc, subject_rows = run_cv(blocks, folds, args)
    screening = screening_report(subject_rows)

    summary = {
        "settings": vars(args),
        "n_blocks": len(blocks),
        "minutes_total": total,
        "minutes_positive": pos,
        "positive_ratio": pos / total,
        "folds": [{"records": f["records"], "n": f["n"], "positive_ratio": f["positive_ratio"]} for f in folds],
        "fold_results": fold_results,
        "pooled_auc": pooled_auc,
        "aggregate": {
            "threshold_0_5": aggregate(fold_results, "threshold_0_5"),
            "threshold_val": aggregate(fold_results, "threshold_val"),
            "threshold_oracle": aggregate(fold_results, "threshold_oracle"),
        },
        "screening": screening,
        "subject_rows": subject_rows,
    }

    print(f"\n=== Minute-level test metrics ({len(fold_results)} folds) ===")
    print(f"Pooled out-of-fold AUC: {pooled_auc:.4f}")
    for level in ["threshold_0_5", "threshold_val", "threshold_oracle"]:
        a = summary["aggregate"][level]
        print(f"  {level:16s} BAcc={a['balanced_accuracy']['mean']:.4f}+/-{a['balanced_accuracy']['std']:.4f} "
              f"AUC={a['roc_auc']['mean']:.4f} F1={a['f1']['mean']:.4f} "
              f"Rec={a['recall']['mean']:.4f} Spec={a['specificity']['mean']:.4f} "
              f"Prec={a['precision']['mean']:.4f}")

    print(f"\n=== Subject-level screening ({screening.get('n_subjects', 0)} subjects) ===")
    print(f"  Pearson(score, true burden)  = {screening.get('pearson_score_vs_trueburden', 0):.3f}")
    print(f"  Spearman(score, true burden) = {screening.get('spearman_score_vs_trueburden', 0):.3f}")
    for tier in (10, 15, 20):
        k = f"screening_auc_burden_ge_{tier}pct"
        if k in screening:
            print(f"  Screening AUC (burden>={tier}%): {screening[k]:.3f}  "
                  f"(n_high={screening[f'n_high_burden_ge_{tier}pct']})")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
