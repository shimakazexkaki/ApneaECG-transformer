"""Decisive diagnostic: does a per-minute apnea signal exist in the HRV features at all?

Compares three discriminability levels:
  1. Univariate pooled AUC of each feature (cross-subject, raw).
  2. Within-subject AUC: per subject, AUC of each feature using only that subject's
     minutes. If features separate apnea/normal *within* a subject, the signal exists
     and the bottleneck is purely cross-subject transfer.
  3. Subject-level burden: correlation between true apnea-minute fraction and the
     mean of a candidate feature per subject (screening signal).
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # apnea project root
for _p in (_HERE, _ROOT / "lib", _ROOT):
    sys.path.insert(0, str(_p))

import hrv_features as hf
import ucddb_runner


def load(apnea_only, channels=(0, 2), context=5):
    ids = ucddb_runner.available_record_ids(Path("ucddb"))
    per_subject = []
    for rid in ids:
        Xs, ys = [], []
        for ch in channels:
            rec = hf.extract_record(rid, channel=ch, apnea_only=apnea_only, context_minutes=context)
            if len(rec.labels):
                Xs.append(rec.features)
                ys.append(rec.labels)
        if Xs:
            per_subject.append((rid, np.concatenate(Xs), np.concatenate(ys)))
    return per_subject


def safe_auc(y, s):
    if len(np.unique(y)) < 2:
        return None
    try:
        a = roc_auc_score(y, s)
        return max(a, 1 - a)  # feature direction-agnostic
    except ValueError:
        return None


def main():
    for apnea_only in (False, True):
        tag = "apnea-only" if apnea_only else "apnea+hyp"
        data = load(apnea_only)
        Xall = np.concatenate([d[1] for d in data])
        yall = np.concatenate([d[2] for d in data])
        print(f"\n========== {tag} ==========")
        print(f"subjects={len(data)} minutes={len(yall)} pos={int(yall.sum())} ({yall.mean():.3f})")

        # 1. Pooled univariate AUC (cross-subject)
        pooled = []
        for j, name in enumerate(hf.FEATURE_NAMES):
            a = safe_auc(yall, Xall[:, j])
            if a is not None:
                pooled.append((a, name))
        pooled.sort(reverse=True)
        print("\nTop pooled (cross-subject) univariate AUC:")
        for a, name in pooled[:8]:
            print(f"  {name:18s} {a:.3f}")

        # 2. Within-subject AUC (mean over subjects, per feature)
        print("\nTop WITHIN-subject mean univariate AUC (signal-exists test):")
        within = []
        for j, name in enumerate(hf.FEATURE_NAMES):
            aucs = []
            for rid, X, y in data:
                a = safe_auc(y, X[:, j])
                if a is not None:
                    aucs.append(a)
            if aucs:
                within.append((float(np.mean(aucs)), name, len(aucs)))
        within.sort(reverse=True)
        for a, name, n in within[:8]:
            print(f"  {name:18s} {a:.3f}  (n_subj={n})")

        # 3. Subject-level burden correlation
        print("\nSubject-level burden correlation (screening signal):")
        burden = np.array([y.mean() for _, _, y in data])
        for j, name in enumerate(hf.FEATURE_NAMES):
            feat_mean = np.array([X[:, j].mean() for _, X, _ in data])
            if np.std(feat_mean) < 1e-9:
                continue
            r = np.corrcoef(burden, feat_mean)[0, 1]
            if abs(r) > 0.35:
                print(f"  {name:18s} r={r:+.3f}")


if __name__ == "__main__":
    main()
