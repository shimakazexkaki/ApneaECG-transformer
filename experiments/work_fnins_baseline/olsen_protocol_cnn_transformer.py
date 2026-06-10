"""Olsen 同協定的 CNN+Transformer:三個資料集一把尺。

協定(對齊 Olsen 2020 的可比設定):
  - 標籤 = per-minute,>5s 重疊為陽性(規則 A,label_overlap_sec=5.0)。
  - 切分 = record-level grouped CV(整位受試者 held out,subject-independent),零洩漏。
  - 模型 = 同一個 fn.CNNTransformer(900x2 輸入,d_model=96/nhead4/2層),三個資料集完全相同。
  - 訓練 = focal loss + 過採樣 + val 挑閾值(同 cnn_transformer_grouped_cv)。

這樣 UCDDB / Apnea-ECG / MESA 三者唯一的差別只剩「資料本身」,可直接比較。
重用既有 harness(cnn_transformer_grouped_cv.run / train_one_fold、hrv_grouped_cv 指標),
只替換「每個資料集怎麼產生 blocks」。

用法:
    python experiments/work_fnins_baseline/olsen_protocol_cnn_transformer.py
    python experiments/work_fnins_baseline/olsen_protocol_cnn_transformer.py --datasets mesa
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_HERE, _ROOT / "lib", _ROOT, _ROOT / "experiments" / "work_ucddb_hrv"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fnins_experiment as fn          # noqa: E402
import hrv_grouped_cv as gcv           # noqa: E402
import cnn_transformer_grouped_cv as cg  # noqa: E402
import mesa_features as mf             # noqa: E402

_ORIG_UCDDB_LOAD = cg.load_blocks      # 原本就是 UCDDB 版


# --------------------------------------------------------------------------- #
# 每個資料集 → blocks: [{record, channel, X:(n,900,2), y, centers}]
# --------------------------------------------------------------------------- #
def _apnea_feature_args(args):
    return SimpleNamespace(
        dataset="apnea_ecg",
        ucddb_dir=str(_ROOT / "ucddb"), ucddb_channel=0,
        ucddb_literature_cache_dir=str(_HERE / "cache_ucddb_lit"),
        apnea_dir=str(_ROOT / "apnea-ecg"), apnea_rpeak_source="hamilton",
        include_hypopnea=True, label_overlap_sec=5.0,
        context_minutes=args.context_minutes, target_length=900,
        min_beats=4, edge_policy="clamp",
        cache_dir=str(args.cache_dir), rebuild_cache=False, num_workers=1,
        max_bad_rr_fraction=0.0,
    )


def load_apnea_blocks(args):
    fa = _apnea_feature_args(args)
    recs = list(fn.APNEA_RELEASE_RECORDS) + list(fn.APNEA_WITHHELD_RECORDS)
    blocks = []
    for rec in fn.get_feature_records(fa, "apnea_ecg", recs):
        if len(rec.labels):
            blocks.append({"record": rec.record_id, "channel": 0,
                           "X": rec.features, "y": rec.labels, "centers": rec.centers})
    if not blocks:
        raise RuntimeError("No Apnea-ECG feature blocks loaded.")
    return blocks


def _mesa_cached_ids(cache_dir):
    ids = []
    for p in sorted(Path(cache_dir).glob("*_ekg_*.npz")):
        if "_seg" in p.name:
            continue
        ids.append(p.name.split("_ekg_")[0])
    return sorted(set(ids))


def load_mesa_blocks(args):
    ma = SimpleNamespace(
        apnea_only=False, label_overlap_sec=5.0,
        context_minutes=args.context_minutes, target_length=900,
        peak_amplitude="prominence",          # 命中快取時不使用
        mesa_cache_dir=str(args.mesa_cache_dir),
        rebuild_cache=False, mesa_limit=0, no_progress=True,
    )
    ids = _mesa_cached_ids(args.mesa_cache_dir)
    if not ids:
        raise RuntimeError(f"No cached MESA per-minute subjects under {args.mesa_cache_dir}")
    recs = mf.load_mesa_records(ma, args.mesa_dir, record_ids=ids)
    blocks = []
    for r in recs:
        if len(r.labels):
            blocks.append({"record": r.record_id, "channel": 0,
                           "X": r.features, "y": r.labels,
                           "centers": getattr(r, "minute_indices", None)})
    if not blocks:
        raise RuntimeError("No MESA feature blocks loaded.")
    return blocks


def unified_load(args):
    if args.dataset == "ucddb":
        return _ORIG_UCDDB_LOAD(args)
    if args.dataset == "apnea_ecg":
        return load_apnea_blocks(args)
    if args.dataset == "mesa":
        return load_mesa_blocks(args)
    raise ValueError(args.dataset)


cg.load_blocks = unified_load          # 讓 cg.run 用我們的 loader


# --------------------------------------------------------------------------- #
def run_dataset(ds, base):
    args = SimpleNamespace(**vars(base))
    args.dataset = ds
    # 為求公平,三者都用單導 ch0(Apnea-ECG/MESA 本就單導;UCDDB 取 ch0)
    args.channels = [0]
    args.output = str(Path(base.output_dir) / f"olsen_{ds}.json")
    print("\n" + "=" * 78)
    print(f"  Olsen protocol | dataset = {ds}  (per-minute >5s, grouped {base.n_splits}-fold, CNN+Transformer)")
    print("=" * 78, flush=True)
    summary = cg.run(args)
    a = summary["aggregate"]["threshold_val"]
    row = {
        "dataset": ds,
        "n_subjects": summary.get("screening", {}).get("n_subjects"),
        "pooled_auc": summary["pooled_auc"],
        "accuracy": a["accuracy"]["mean"], "recall": a["recall"]["mean"],
        "precision": a["precision"]["mean"], "f1": a["f1"]["mean"],
        "specificity": a["specificity"]["mean"], "auc_mean": a["roc_auc"]["mean"],
        "bacc": a["balanced_accuracy"]["mean"],
        "spearman_burden": summary.get("screening", {}).get("spearman_score_vs_trueburden"),
    }
    return row


def main():
    p = argparse.ArgumentParser(description="CNN+Transformer under Olsen protocol on 3 datasets.")
    p.add_argument("--datasets", nargs="+", default=["apnea_ecg", "ucddb", "mesa"],
                   choices=["apnea_ecg", "ucddb", "mesa"])
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
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=str(_HERE / "cache"))
    p.add_argument("--mesa-dir", default="D:/mesa")
    p.add_argument("--mesa-cache-dir",
                   default=str(_ROOT / "experiments" / "work_mesa_transfer" / "cache" / "mesa_features"))
    p.add_argument("--output-dir", default=str(_HERE / "outputs" / "olsen_protocol"))
    p.add_argument("--cpu", action="store_true")
    # cg.load_blocks(ucddb) 需要的欄位
    p.add_argument("--max-bad-rr-fraction", type=float, default=0.0)
    p.add_argument("--include-all-records", action="store_true")
    base = p.parse_args()
    Path(base.output_dir).mkdir(parents=True, exist_ok=True)

    rows = [run_dataset(ds, base) for ds in base.datasets]

    Path(base.output_dir, "olsen_protocol_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    print("\n\n" + "#" * 84)
    print("#  CNN+Transformer | Olsen 同協定 (per-minute >5s, grouped CV, single-lead ch0)")
    print("#" * 84)
    hdr = f"{'dataset':<11}{'N':>4} {'Acc':>7}{'Recall':>8}{'Prec':>7}{'F1':>7}{'Spec':>7}{'AUC':>7}{'AUC*':>7}{'Spear':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        n = r["n_subjects"] if r["n_subjects"] is not None else 0
        sp = r["spearman_burden"]
        sp = f"{sp:.3f}" if sp is not None else "  -  "
        print(f"{r['dataset']:<11}{n:>4} {r['accuracy']:>7.3f}{r['recall']:>8.3f}"
              f"{r['precision']:>7.3f}{r['f1']:>7.3f}{r['specificity']:>7.3f}"
              f"{r['auc_mean']:>7.3f}{r['pooled_auc']:>7.3f}   {sp:>5}")
    print("\nAUC = per-fold mean,  AUC* = pooled out-of-fold,  Spear = subject burden Spearman")
    print(f"Saved: {Path(base.output_dir, 'olsen_protocol_summary.json')}")


if __name__ == "__main__":
    main()
