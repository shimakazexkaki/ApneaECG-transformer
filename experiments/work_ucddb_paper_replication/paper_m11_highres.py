"""Pham & Moucek 2025 的 UCDDB 「真實協定」複現：11 秒視窗 + 10 秒重疊 + CNN-Transformer-LSTM。

兩種評估：
- segment：把所有視窗 pool 起來後做 8:1:1 隨機切分（視窗化「之後」才切）。
  因 11s 視窗 10s 重疊 → 相鄰視窗近重複，同病患近重複樣本同時進 train/test → 分數爆高（重現論文 ~90%+）。
- grouped：整位受試者 held out 的 5-fold CV（誠實，預期 ~0.55）。

輸入為 11 秒原始 ECG 視窗（重用 lib/ucddb_highres_trainer 的 dataset），標籤取視窗第 2 秒（論文設定）。
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT / "lib", _ROOT, _ROOT / "experiments" / "work_ucddb_hrv"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ucddb_highres_trainer as hr  # noqa: E402
import ucddb_runner  # noqa: E402
import hrv_grouped_cv as gcv  # noqa: E402


class CNNTransformerLSTM(nn.Module):
    """11 秒視窗：CNN 下採樣 → Transformer encoder → BiLSTM → attention pooling → 分類。"""

    def __init__(self, d_model=96, nhead=4, layers=2, lstm_hidden=128, dropout=0.3, max_tokens=256):
        super().__init__()
        base = hr.HighResCNNTransformer(d_model=d_model, nhead=nhead, layers=layers,
                                        dropout=dropout, max_tokens=max_tokens)
        self.cnn = base.cnn
        self.pos = base.pos
        self.encoder = base.encoder
        self.lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * lstm_hidden, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * lstm_hidden, 2))

    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        x = x + self.pos[:, : x.size(1), :]
        x = self.encoder(x)
        x, _ = self.lstm(x)
        w = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        x = torch.sum(x * w.unsqueeze(-1), dim=1)
        return self.classifier(x)


def feature_args(args):
    return SimpleNamespace(
        ucddb_dir=str(_ROOT / "ucddb"),
        cache_dir=str(_HERE / "cache" / "ucddb_highres"),
        channels=args.channels, apnea_only=False, no_progress=True,
    )


def build_dataset(records, args, stride):
    return hr.HighResWindowDataset(
        records, window_sec=args.window_sec, stride_sec=stride, label_second=args.label_second,
        label_mode="second", augment=False, seed=args.seed,
        max_windows=args.max_windows, return_record_index=True,
    )


def train_eval(model, train_idx, val_idx, test_idx, dataset, args, device):
    y = dataset.labels
    counts = np.bincount(y[train_idx], minlength=2)
    w = torch.tensor(len(train_idx) / (2.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    lossfn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def loader(idx, shuffle):
        return DataLoader(Subset(dataset, idx), batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    def predict(idx):
        model.eval()
        ys, ss = [], []
        with torch.no_grad():
            for bx, by, _ in loader(idx, False):
                logits = model(bx.to(device))
                ss.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                ys.extend(by.numpy())
        return np.array(ys), np.array(ss, dtype=np.float32)

    best_bacc, best_state, wait = -1, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for bx, by, _ in loader(train_idx, True):
            opt.zero_grad()
            loss = lossfn(model(bx.to(device)), by.to(device))
            loss.backward()
            opt.step()
        vy, vs = predict(val_idx)
        _, m = gcv.best_threshold(vy, vs)
        if m["balanced_accuracy"] > best_bacc:
            best_bacc, best_state, wait = m["balanced_accuracy"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    vy, vs = predict(val_idx)
    thr, _ = gcv.best_threshold(vy, vs)
    ty, ts = predict(test_idx)
    return gcv.metrics_at(ty, ts, thr), ty, ts


def new_model(args, device):
    return CNNTransformerLSTM(d_model=args.d_model, nhead=args.nhead, layers=args.layers,
                              lstm_hidden=args.lstm_hidden, dropout=args.dropout).to(device)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    hr.set_seed(args.seed)
    record_ids = ucddb_runner.available_record_ids(_ROOT / "ucddb")
    if not args.include_all_records:
        record_ids = [r for r in record_ids if r not in {"ucddb008", "ucddb011", "ucddb013", "ucddb018"}]
    records, _ = hr.load_records(feature_args(args), record_ids)
    dataset = build_dataset(records, args, args.stride_sec)
    y = dataset.labels
    print(f"[M11 high-res] device={device} protocol={args.protocol} windows={len(y)} "
          f"pos={int(y.sum())} ({y.mean():.3f}) window={args.window_sec}s stride={args.stride_sec}s "
          f"channels={args.channels}")

    if args.protocol == "segment":
        idx = np.arange(len(y))
        tv, te = train_test_split(idx, test_size=0.1, random_state=args.seed, stratify=y)
        tr, va = train_test_split(tv, test_size=1 / 9, random_state=args.seed, stratify=y[tv])
        m, _, _ = train_eval(new_model(args, device), tr, va, te, dataset, args, device)
        print("\n=== SEGMENT split test ===")
        print(f"  Acc={m['accuracy']:.4f} BAcc={m['balanced_accuracy']:.4f} F1={m['f1']:.4f} "
              f"Rec={m['recall']:.4f} Spec={m['specificity']:.4f} AUC={m['roc_auc']:.4f}")
        summary = {"protocol": "segment", "settings": vars(args), "test": m}
    else:
        rec_idx = dataset.record_indices
        rid_of_block = {i: records[i].record_id for i in range(len(records))}
        groups = np.array([rid_of_block[int(b)] for b in rec_idx])
        uniq = sorted(set(groups))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(uniq)
        folds = np.array_split(uniq, args.n_splits)
        fold_metrics, py, ps = [], [], []
        for i, test_recs in enumerate(folds):
            test_recs = set(test_recs)
            trainval_recs = [r for r in uniq if r not in test_recs]
            rng2 = np.random.default_rng(args.seed + i)
            rng2.shuffle(trainval_recs)
            nval = max(1, len(trainval_recs) // 5)
            val_recs = set(trainval_recs[:nval])
            test_idx = np.flatnonzero(np.isin(groups, list(test_recs)))
            val_idx = np.flatnonzero(np.isin(groups, list(val_recs)))
            train_idx = np.flatnonzero(~np.isin(groups, list(test_recs | val_recs)))
            if len(np.unique(y[train_idx])) < 2:
                continue
            m, ty, ts = train_eval(new_model(args, device), train_idx, val_idx, test_idx, dataset, args, device)
            fold_metrics.append(m)
            py.append(ty); ps.append(ts)
            print(f"  fold {i} AUC={m['roc_auc']:.4f} BAcc={m['balanced_accuracy']:.4f} F1={m['f1']:.4f}")
        py = np.concatenate(py); ps = np.concatenate(ps)
        pooled_auc = float(gcv.roc_auc_score(py, ps)) if len(np.unique(py)) > 1 else 0.0
        mean = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in ("balanced_accuracy", "roc_auc", "f1", "recall", "specificity")}
        print("\n=== GROUPED CV ===")
        print(f"  pooled AUC={pooled_auc:.4f} | mean BAcc={mean['balanced_accuracy']:.4f} "
              f"AUC={mean['roc_auc']:.4f} F1={mean['f1']:.4f}")
        summary = {"protocol": "grouped", "settings": vars(args), "pooled_auc": pooled_auc, "mean": mean,
                   "folds": fold_metrics}

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        print(f"Saved: {args.output}")


def main():
    p = argparse.ArgumentParser(description="M11 CNN-Transformer-LSTM on 11s UCDDB windows.")
    p.add_argument("--protocol", choices=["segment", "grouped"], default="segment")
    p.add_argument("--channels", nargs="+", type=int, default=[0])
    p.add_argument("--window-sec", type=int, default=11)
    p.add_argument("--stride-sec", type=int, default=1)
    p.add_argument("--label-second", type=int, default=1)
    p.add_argument("--max-windows", type=int, default=120000)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--include-all-records", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
