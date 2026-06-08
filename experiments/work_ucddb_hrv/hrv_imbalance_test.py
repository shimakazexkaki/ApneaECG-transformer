"""Empirical test of additional class-imbalance methods on the HRV features,
under the same record-level grouped CV. Pure sklearn/numpy (no imbalanced-learn).

Compares, against the class-weighted HGB baseline:
  - SMOTE (manual k-NN interpolation of the minority class) + plain HGB
  - Random undersampling of the majority + plain HGB
  - EasyEnsemble: N base HGBs, each on a balanced undersample, averaged

Expectation from theory + the loss sweep: these shuffle the recall/specificity
tradeoff but do not move AUC, which is set by feature separability.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # apnea project root
for _p in (_HERE, _ROOT / "lib", _ROOT):
    sys.path.insert(0, str(_p))
import hrv_grouped_cv as gcv  # noqa: E402


class Args:
    ucddb_dir = "ucddb"
    channels = [0, 2]
    apnea_only = False
    context_minutes = 5
    min_overlap_sec = 5.0
    min_beats = 20
    subject_norm = "zscore"
    exclude_no_positive = False
    rebuild_cache = False


def plain_hgb(seed):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
    )


def smote(X, y, seed, k=5):
    rng = np.random.default_rng(seed)
    Xmin = X[y == 1]
    n_need = int((y == 0).sum() - (y == 1).sum())
    if n_need <= 0 or len(Xmin) <= k:
        return X, y
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xmin)
    _, idx = nn.kneighbors(Xmin)
    base = rng.integers(0, len(Xmin), size=n_need)
    nbr = idx[base, rng.integers(1, k + 1, size=n_need)]
    gap = rng.random((n_need, X.shape[1]), dtype=np.float32)
    synth = Xmin[base] + gap * (Xmin[nbr] - Xmin[base])
    Xnew = np.vstack([X, synth.astype(np.float32)])
    ynew = np.concatenate([y, np.ones(n_need, dtype=np.int64)])
    return Xnew, ynew


def undersample(X, y, seed):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep_neg = rng.choice(neg, size=len(pos), replace=False)
    idx = rng.permutation(np.concatenate([pos, keep_neg]))
    return X[idx], y[idx]


def fit_predict(method, X_tr, y_tr, X_te, seed):
    if method == "weighted":           # baseline: class-weighted HGB
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0,
            class_weight="balanced", random_state=seed)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]
    if method == "smote":
        Xs, ys = smote(X_tr, y_tr, seed)
        m = plain_hgb(seed).fit(Xs, ys)
        return m.predict_proba(X_te)[:, 1]
    if method == "undersample":
        Xu, yu = undersample(X_tr, y_tr, seed)
        m = plain_hgb(seed).fit(Xu, yu)
        return m.predict_proba(X_te)[:, 1]
    if method == "easyensemble":
        scores = []
        for j in range(10):
            Xu, yu = undersample(X_tr, y_tr, seed + j)
            scores.append(plain_hgb(seed + j).fit(Xu, yu).predict_proba(X_te)[:, 1])
        return np.mean(scores, axis=0)
    raise ValueError(method)


def main():
    args = Args()
    blocks, _ = gcv.load_all(args)
    folds = gcv.make_folds(blocks, 5, 42)
    all_records = {b["record"] for b in blocks}

    methods = ["weighted", "smote", "undersample", "easyensemble"]
    print(f"Minutes={sum(len(b['y']) for b in blocks)} "
          f"pos={sum(int(b['y'].sum()) for b in blocks)} | grouped 5-fold CV\n")
    print(f"{'method':14s} {'AUC':>7s} {'BAcc':>7s} {'F1':>7s} {'Rec':>7s} {'Spec':>7s}")
    for method in methods:
        py, ps = [], []
        baccs, f1s, recs, specs = [], [], [], []
        for i, fold in enumerate(folds):
            test_records = set(fold["records"])
            train_records = all_records - test_records
            X_tr, y_tr = gcv.stack(blocks, train_records)
            X_te, y_te = gcv.stack(blocks, test_records)
            if len(np.unique(y_tr)) < 2:
                continue
            te_scores = fit_predict(method, X_tr, y_tr, X_te, 42 + i)
            tr_scores = fit_predict(method, X_tr, y_tr, X_tr, 42 + i)
            thr, _ = gcv.best_threshold(y_tr, tr_scores)
            m = gcv.metrics_at(y_te, te_scores, thr)
            baccs.append(m["balanced_accuracy"]); f1s.append(m["f1"])
            recs.append(m["recall"]); specs.append(m["specificity"])
            py.append(y_te); ps.append(te_scores)
        auc = roc_auc_score(np.concatenate(py), np.concatenate(ps))
        print(f"{method:14s} {auc:7.4f} {np.mean(baccs):7.4f} {np.mean(f1s):7.4f} "
              f"{np.mean(recs):7.4f} {np.mean(specs):7.4f}")


if __name__ == "__main__":
    main()
