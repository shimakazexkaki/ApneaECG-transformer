"""跨資料集:train on MESA → test on 全 UCDDB(完全外部、零洩漏)。

重用既有基礎設施:
- 特徵:lib/mesa_features(MESA)與 lib/ucddb_literature_features(UCDDB),兩邊同形狀 (N,900,2)。
- 訓練/評估:ucddb_literature_train_common.train_one_split(val 挑閾值、test 報全套指標)。
- 模型:M11 CNN-Transformer-LSTM(paper_cnn_transformer_lstm_trainer.patch_model_factory)。

切分:test = 全 UCDDB 樣本;MESA 樣本依「受試者」切 train/val(避免 val 洩漏)。
每個 5 分鐘 context 自身 z-score → 自動消除 MESA/UCDDB 振幅尺度差。

範例:
  python experiments/work_mesa_transfer/mesa_to_ucddb_trainer.py --mesa-limit 40 --experiment-name mesa2ucddb_n40
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
import ucddb_literature_features as litfeat  # noqa: E402
import mesa_features as mf  # noqa: E402
import paper_cnn_transformer_lstm_trainer as paper  # noqa: E402


def _drop_signals(records):
    """include_raw=False 時用不到 raw 訊號;清掉以省 RAM(MESA 每人 ~17MB)。"""
    empty = np.zeros((1,), dtype=np.float32)
    for r in records:
        r.signal = empty
    return records


def build_args():
    parser = argparse.ArgumentParser(description="train-MESA -> test-UCDDB cross-dataset (M11).")
    common.add_train_args(parser)
    # M11 超參(同 paper trainer)
    parser.add_argument("--cnn-dropout", type=float, default=0.5)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)  # 調參最佳
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    # MESA 專屬
    parser.add_argument("--mesa-dir", default="D:/mesa")
    parser.add_argument("--mesa-cache-dir", default=str(WORK_DIR / "cache" / "mesa_features"))
    parser.add_argument("--mesa-limit", type=int, default=0)
    parser.add_argument("--mesa-val-fraction", type=float, default=0.15)
    parser.set_defaults(
        output_dir=str(WORK_DIR / "outputs"),
        cache_dir=str(PAPER_DIR / "cache" / "ucddb_literature_features"),
        ucddb_dir=str(PROJECT_DIR / "ucddb"),
        channels=[0],
        detector="biosppy_hamilton",
        lr=1e-3, nhead=8, batch_size=128, epochs=20, patience=6,
    )
    return parser.parse_args()


def main():
    paper.patch_model_factory()
    args = build_args()
    # 消融顯示 LSTM 多餘 → 以後預設用 CNN+Transformer
    args.use_transformer, args.use_lstm = True, False

    # 1) 載入兩邊特徵
    mesa_records = mf.load_mesa_records(args, args.mesa_dir)
    ucddb_records = litfeat.load_records(args, litfeat.available_records(args))
    mesa_ids = sorted({r.record_id for r in mesa_records})
    ucddb_ids = sorted({r.record_id for r in ucddb_records})
    _drop_signals(mesa_records)
    _drop_signals(ucddb_records)

    arrays = common.records_to_arrays(mesa_records + ucddb_records)
    rec_ids = arrays["record_ids"]
    mesa_set = set(mesa_ids)

    # 2) test = 全 UCDDB;MESA 依受試者切 train/val
    test_idx = np.flatnonzero(~np.isin(rec_ids, list(mesa_set)))
    rng = np.random.default_rng(args.seed)
    shuffled = list(mesa_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * args.mesa_val_fraction)))
    val_records = set(shuffled[:n_val])
    train_records = set(shuffled[n_val:])
    train_idx = np.flatnonzero(np.isin(rec_ids, list(train_records)))
    val_idx = np.flatnonzero(np.isin(rec_ids, list(val_records)))

    print(f"[mesa2ucddb] MESA subjects={len(mesa_ids)} (train={len(train_records)} val={len(val_records)}) "
          f"| UCDDB test subjects={len(ucddb_ids)}")
    print(f"[mesa2ucddb] windows: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"| test positives={int(arrays['labels'][test_idx].sum())}")

    # 3) 訓練 M11(MESA)→ 評估 UCDDB(外部)
    result = common.train_one_split(
        args, "paper_cnn_transformer_lstm", arrays,
        train_idx, val_idx, test_idx, "mesa2ucddb", include_raw=False,
    )

    out_dir = Path(args.output_dir) / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": args.experiment_name,
        "protocol": "train-MESA -> test-UCDDB (external)",
        "mesa_subjects": mesa_ids,
        "ucddb_test_subjects": ucddb_ids,
        "counts": {"train_windows": int(len(train_idx)), "val_windows": int(len(val_idx)),
                   "test_windows": int(len(test_idx))},
        "ucddb_external": {
            "threshold_val": result["test"]["threshold_val"],
            "threshold_oracle": result["test"]["threshold_oracle"],
            "record_burden": result["test"]["record_burden"],
        },
        "settings": {k: v for k, v in vars(args).items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    tv = result["test"]["threshold_val"]
    print("\n========== UCDDB external (train on MESA) ==========")
    print(f"  AUC={tv['roc_auc']:.4f}  BAcc={tv['balanced_accuracy']:.4f}  F1={tv['f1']:.4f}  "
          f"Rec={tv['recall']:.4f}  Spec={tv['specificity']:.4f}  Acc={tv['accuracy']:.4f}")
    print(f"  (oracle AUC={result['test']['threshold_oracle']['roc_auc']:.4f})")
    print(f"  Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
