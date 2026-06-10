"""Olsen 2020 完全同切分:單次 recording-level 80/10/10 hold-out(非 cross-validation)。

對齊 Olsen 的協定(SLEEP 2020):
  - 切分 = 把「整筆記錄(受試者)」隨機分成 train/eval/test = 80%/10%/10%,
           每位受試者只出現在一個集合(subject-independent、零洩漏)。**單次 hold-out,不是 k-fold。**
  - 標籤 = per-minute,>5s 重疊為陽性。
  - 模型 = fn.CNNTransformer(900x2,d_model=96/nhead4/2層),三個資料集完全相同。
  - 訓練 = focal loss + 過採樣;在 eval 集挑閾值;報 test 集指標。

因為 UCDDB(~21)/ MESA(74)/ Apnea-ECG(70)的 10% test 只有數位受試者,
單一次切分數字會抖,所以對「同一個 Olsen-style 切分機制」重複 `--repeats` 個隨機種子,
報 mean ± std(每個 repeat 都是一次完整的 80/10/10 hold-out)。

用法:
    python experiments/work_fnins_baseline/olsen_exact_split_cnn_transformer.py
    python experiments/work_fnins_baseline/olsen_exact_split_cnn_transformer.py --datasets mesa --repeats 5
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_HERE, _ROOT / "lib", _ROOT, _ROOT / "experiments" / "work_ucddb_hrv"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fnins_experiment as fn               # noqa: E402
import hrv_grouped_cv as gcv                # noqa: E402
import olsen_protocol_cnn_transformer as opc  # noqa: E402  (reuse block loaders)


def load_blocks(dataset, base):
    a = SimpleNamespace(
        dataset=dataset, channels=[0], context_minutes=base.context_minutes,
        cache_dir=base.cache_dir, mesa_dir=base.mesa_dir, mesa_cache_dir=base.mesa_cache_dir,
        max_bad_rr_fraction=0.0, include_all_records=False,
    )
    return opc.unified_load(a)


def subject_split(records, seed, test_frac=0.10, val_frac=0.10):
    recs = sorted(records)
    rng = np.random.default_rng(seed)
    rng.shuffle(recs)
    n = len(recs)
    n_test = max(1, round(test_frac * n))
    n_val = max(1, round(val_frac * n))
    test = set(recs[:n_test])
    val = set(recs[n_test:n_test + n_val])
    train = set(recs[n_test + n_val:])
    return train, val, test


def train_eval_once(base, blocks, train_recs, val_recs, test_recs, device, seed):
    fn.set_seed(seed)
    X_tr, y_tr = gcv.stack(blocks, train_recs)
    X_va, y_va = gcv.stack(blocks, val_recs)
    X_te, y_te = gcv.stack(blocks, test_recs)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2 or len(np.unique(y_va)) < 2:
        return None

    train_loader = fn.make_loader(X_tr, y_tr, base.batch_size, True, oversample=True,
                                  samples_per_epoch=base.samples_per_epoch)
    val_loader = fn.make_loader(X_va, y_va, base.batch_size, False, False, None)
    test_loader = fn.make_loader(X_te, y_te, base.batch_size, False, False, None)

    model = fn.CNNTransformer(input_channels=2, d_model=base.d_model, nhead=base.nhead,
                              num_layers=base.layers, dropout=base.dropout).to(device)
    criterion = fn.build_criterion(SimpleNamespace(loss=base.loss, focal_gamma=base.focal_gamma),
                                   y_tr, device)
    opt = torch.optim.AdamW(model.parameters(), lr=base.lr, weight_decay=base.weight_decay)

    best_bacc, best_state, wait = -1.0, None, 0
    for _ in range(1, base.epochs + 1):
        fn.train_epoch(model, train_loader, criterion, opt, device)
        vy, _, vs = fn.predict(model, val_loader, device)
        _, m = gcv.best_threshold(vy, vs)
        if m["balanced_accuracy"] > best_bacc:
            best_bacc = m["balanced_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= base.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    vy, _, vs = fn.predict(model, val_loader, device)
    thr_val, _ = gcv.best_threshold(vy, vs)
    ty, _, ts = fn.predict(model, test_loader, device)
    mv = gcv.metrics_at(ty, ts, thr_val)
    mo = gcv.best_threshold(ty, ts)[1]
    return {
        "n_train_subj": len(train_recs), "n_val_subj": len(val_recs), "n_test_subj": len(test_recs),
        "test_minutes": int(len(y_te)), "test_pos": int(y_te.sum()),
        "threshold_val": mv, "threshold_oracle": mo,
    }


def run_dataset(ds, base, device):
    blocks = load_blocks(ds, base)
    records = sorted({b["record"] for b in blocks})
    total = sum(len(b["y"]) for b in blocks)
    pos = sum(int(b["y"].sum()) for b in blocks)
    print("\n" + "=" * 78)
    print(f"  Olsen exact split | dataset={ds}  records={len(records)} minutes={total} "
          f"pos={pos} ({pos/total:.3f})")
    print(f"  single 80/10/10 subject-level hold-out x {base.repeats} seeds")
    print("=" * 78, flush=True)

    runs = []
    for r in range(base.repeats):
        seed = base.seed + r
        tr, va, te = subject_split(records, seed)
        res = train_eval_once(base, blocks, tr, va, te, device, seed)
        if res is None:
            print(f"  seed {seed}: degenerate split (single-class), skipped", flush=True)
            continue
        m = res["threshold_val"]
        print(f"  seed {seed}: test_subj={res['n_test_subj']} (pos_min={res['test_pos']}/{res['test_minutes']}) "
              f"AUC={m['roc_auc']:.3f} Acc={m['accuracy']:.3f} Rec={m['recall']:.3f} "
              f"Prec={m['precision']:.3f} F1={m['f1']:.3f} Spec={m['specificity']:.3f}", flush=True)
        runs.append(res)

    def agg(level, key):
        vals = [run[level][key] for run in runs]
        return float(np.mean(vals)), float(np.std(vals))

    keys = ["accuracy", "recall", "precision", "f1", "specificity", "roc_auc", "balanced_accuracy"]
    summary = {
        "dataset": ds, "n_records": len(records), "prevalence": pos / total,
        "repeats": len(runs),
        "threshold_val": {k: {"mean": agg("threshold_val", k)[0], "std": agg("threshold_val", k)[1]} for k in keys},
        "threshold_oracle": {k: {"mean": agg("threshold_oracle", k)[0], "std": agg("threshold_oracle", k)[1]} for k in keys},
        "per_seed": runs,
    }
    return summary


def main():
    p = argparse.ArgumentParser(description="CNN+Transformer under Olsen EXACT single 80/10/10 split.")
    p.add_argument("--datasets", nargs="+", default=["apnea_ecg", "ucddb", "mesa"],
                   choices=["apnea_ecg", "ucddb", "mesa"])
    p.add_argument("--repeats", type=int, default=5, help="number of random Olsen-style splits to average")
    p.add_argument("--context-minutes", type=int, default=5)
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=str(_HERE / "cache"))
    p.add_argument("--mesa-dir", default="D:/mesa")
    p.add_argument("--mesa-cache-dir",
                   default=str(_ROOT / "experiments" / "work_mesa_transfer" / "cache" / "mesa_features"))
    p.add_argument("--output-dir", default=str(_HERE / "outputs" / "olsen_exact_split"))
    p.add_argument("--cpu", action="store_true")
    base = p.parse_args()
    Path(base.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not base.cpu else "cpu")

    summaries = [run_dataset(ds, base, device) for ds in base.datasets]
    Path(base.output_dir, "olsen_exact_split_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")

    print("\n\n" + "#" * 92)
    print(f"#  CNN+Transformer | Olsen 完全同切分 (single 80/10/10 subject-level hold-out, "
          f"mean±std over {base.repeats} seeds)")
    print("#" * 92)
    hdr = (f"{'dataset':<11}{'recs':>5}{'prev':>7}  {'Acc':>12}{'Recall':>12}{'Prec':>12}"
           f"{'F1':>12}{'Spec':>12}{'AUC':>12}")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        a = s["threshold_val"]
        def cell(k):
            return f"{a[k]['mean']:.3f}±{a[k]['std']:.3f}"
        print(f"{s['dataset']:<11}{s['n_records']:>5}{s['prevalence']:>7.3f}  "
              f"{cell('accuracy'):>12}{cell('recall'):>12}{cell('precision'):>12}"
              f"{cell('f1'):>12}{cell('specificity'):>12}{cell('roc_auc'):>12}")
    print(f"\n(threshold chosen on eval split; metrics on held-out test split)")
    print(f"Saved: {Path(base.output_dir, 'olsen_exact_split_summary.json')}")


if __name__ == "__main__":
    main()
