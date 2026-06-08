import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import apnea_trainer
import mixed_trainer
import sequence_evaluator
import ucddb_runner
import ucddb_trainer


RANDOM_STATE = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def record_cache_key(record_id, channel, context_minutes, min_overlap_sec, include_hypopnea):
    overlap = str(min_overlap_sec).replace(".", "p")
    hyp = "hyp" if include_hypopnea else "apneaonly"
    return f"{record_id}_ch{channel}_ctx{context_minutes}_{hyp}_overlap{overlap}"


def load_ucddb_record_context(data_dir, cache_dir, record_id, channel, include_hypopnea, min_overlap_sec, context_minutes):
    x, y = ucddb_trainer.load_ucddb_record(
        data_dir,
        cache_dir,
        record_id,
        channel,
        include_hypopnea=include_hypopnea,
        min_overlap_sec=min_overlap_sec,
    )
    x = mixed_trainer.normalize_matrix(x)
    return mixed_trainer.make_context_windows(x, y, context_minutes)


def load_ucddb_records_by_id(args, record_ids):
    xs = []
    ys = []
    stats = []
    data_dir = Path(args.ucddb_dir)
    cache_dir = Path(args.ucddb_cache_dir)
    for record_id in tqdm(record_ids, desc="Loading UCDDB records", disable=args.no_progress):
        for channel in args.channels:
            x, y = load_ucddb_record_context(
                data_dir,
                cache_dir,
                record_id,
                channel,
                include_hypopnea=not args.apnea_only,
                min_overlap_sec=args.min_overlap_sec,
                context_minutes=args.context_minutes,
            )
            if len(y) == 0:
                continue
            xs.append(x)
            ys.append(y)
            stats.append(
                {
                    "record": record_id,
                    "channel": channel,
                    "samples": int(len(y)),
                    "positive": int(y.sum()),
                    "normal": int((y == 0).sum()),
                }
            )
    if not xs:
        raise RuntimeError(f"No UCDDB samples loaded for records: {record_ids}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), stats


def summarize_record(args, record_id):
    total_samples = 0
    total_positive = 0
    for channel in args.channels:
        _, y = load_ucddb_record_context(
            Path(args.ucddb_dir),
            Path(args.ucddb_cache_dir),
            record_id,
            channel,
            include_hypopnea=not args.apnea_only,
            min_overlap_sec=args.min_overlap_sec,
            context_minutes=args.context_minutes,
        )
        total_samples += len(y)
        total_positive += int(y.sum())
    return {
        "record": record_id,
        "samples": int(total_samples),
        "positive": int(total_positive),
        "normal": int(total_samples - total_positive),
        "positive_ratio": float(total_positive / total_samples) if total_samples else 0.0,
    }


def balanced_folds(record_summaries, n_splits, seed):
    rng = random.Random(seed)
    shuffled = list(record_summaries)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda item: (item["positive"], item["samples"]), reverse=True)

    folds = [{"records": [], "samples": 0, "positive": 0} for _ in range(n_splits)]
    for summary in shuffled:
        fold = min(folds, key=lambda item: (item["positive"], item["samples"], len(item["records"])))
        fold["records"].append(summary["record"])
        fold["samples"] += summary["samples"]
        fold["positive"] += summary["positive"]

    for fold in folds:
        fold["records"].sort()
        fold["normal"] = fold["samples"] - fold["positive"]
        fold["positive_ratio"] = float(fold["positive"] / fold["samples"]) if fold["samples"] else 0.0
    return folds


def train_val_split(record_ids, val_fraction, seed):
    rng = random.Random(seed)
    shuffled = sorted(record_ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val = sorted(shuffled[:val_count])
    train = sorted(shuffled[val_count:])
    if not train:
        train, val = val[:-1], val[-1:]
    return train, val


def make_loss(args):
    if args.focal_gamma > 0:
        return mixed_trainer.SoftmaxFocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def predict_record_scores(model, args, record_ids, device):
    y_all = []
    raw_all = []
    smooth_all = []
    for record_id in record_ids:
        for channel in args.channels:
            x, y = load_ucddb_record_context(
                Path(args.ucddb_dir),
                Path(args.ucddb_cache_dir),
                record_id,
                channel,
                include_hypopnea=not args.apnea_only,
                min_overlap_sec=args.min_overlap_sec,
                context_minutes=args.context_minutes,
            )
            if len(y) == 0:
                continue
            labels, scores = mixed_trainer.predict_scores(model, x, y, args.batch_size, device)
            y_all.extend(labels.tolist())
            raw_all.extend(scores.tolist())
            smooth_all.extend(sequence_evaluator.moving_average(scores, args.smooth_width).tolist())
    return (
        np.asarray(y_all, dtype=np.int64),
        np.asarray(raw_all, dtype=np.float32),
        np.asarray(smooth_all, dtype=np.float32),
    )


def train_one_fold(args, fold_idx, folds, apnea_records_split, device):
    set_seed(args.seed + fold_idx)
    test_records = sorted(folds[fold_idx]["records"])
    train_val_records = sorted(
        record
        for idx, fold in enumerate(folds)
        if idx != fold_idx
        for record in fold["records"]
    )
    ucddb_train_records, ucddb_val_records = train_val_split(
        train_val_records, args.ucddb_val_fraction, args.seed + fold_idx
    )
    apnea_train_records, apnea_val_records, apnea_test_records = apnea_records_split

    x_u_train, y_u_train, u_train_stats = load_ucddb_records_by_id(args, ucddb_train_records)
    x_u_val, y_u_val, u_val_stats = load_ucddb_records_by_id(args, ucddb_val_records)
    x_a_train, y_a_train, a_train_stats = mixed_trainer.load_apnea_records(
        Path(args.apnea_dir), Path(args.apnea_cache_dir), apnea_train_records, args.context_minutes
    )
    x_a_val, y_a_val, a_val_stats = mixed_trainer.load_apnea_records(
        Path(args.apnea_dir), Path(args.apnea_cache_dir), apnea_val_records, args.context_minutes
    )

    x_train = np.concatenate([x_u_train, x_a_train], axis=0)
    y_train = np.concatenate([y_u_train, y_a_train], axis=0)
    source = mixed_trainer.source_labels(len(y_u_train), len(y_a_train))
    sampler = mixed_trainer.make_sampler(y_train, source, args.ucddb_sample_weight, args.apnea_sample_weight)
    train_loader = DataLoader(
        mixed_trainer.ECGDataset(x_train, y_train, augment=True, aug_args=args, seed=args.seed + fold_idx),
        batch_size=args.batch_size,
        sampler=sampler,
    )

    model = apnea_trainer.ParallelCNNTransformer().to(device)
    criterion = make_loss(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=3)

    fold_dir = Path(args.output_dir) / f"{args.experiment_name}_fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.pth"

    best_score = -1.0
    best_threshold = 0.5
    best_ucddb_threshold = 0.5
    best_epoch = 0
    patience = 0

    print(f"\nFold {fold_idx}:")
    print(f"  UCDDB train records: {ucddb_train_records}")
    print(f"  UCDDB val records:   {ucddb_val_records}")
    print(f"  UCDDB test records:  {test_records}")
    print(f"  Loss: {'focal gamma=' + str(args.focal_gamma) if args.focal_gamma > 0 else 'cross entropy'}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in tqdm(
            train_loader,
            desc=f"Fold {fold_idx} Epoch {epoch}/{args.epochs}",
            leave=False,
            disable=args.no_progress,
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / len(y_train)
        u_val_true, u_val_scores = mixed_trainer.predict_scores(model, x_u_val, y_u_val, args.batch_size, device)
        a_val_true, a_val_scores = mixed_trainer.predict_scores(model, x_a_val, y_a_val, args.batch_size, device)
        combined_true = np.concatenate([u_val_true, a_val_true])
        combined_scores = np.concatenate([u_val_scores, a_val_scores])

        threshold, _, constrained = mixed_trainer.best_threshold_balanced(
            combined_true, combined_scores, args.min_specificity
        )
        ucddb_threshold, _, _ = mixed_trainer.best_threshold_balanced(
            u_val_true, u_val_scores, args.min_specificity
        )
        u_metrics = mixed_trainer.evaluate_from_scores(u_val_true, u_val_scores, threshold)
        a_metrics = mixed_trainer.evaluate_from_scores(a_val_true, a_val_scores, threshold)
        score = (
            args.ucddb_val_weight * u_metrics["balanced_accuracy"]
            + (1.0 - args.ucddb_val_weight) * a_metrics["balanced_accuracy"]
        )
        scheduler.step(score)

        print(
            f"Fold {fold_idx} Epoch {epoch:02d} | Loss={train_loss:.4f} | thr={threshold:.3f}"
            f"{'' if constrained else '*'} | UVal BAcc={u_metrics['balanced_accuracy']:.4f} "
            f"AVal BAcc={a_metrics['balanced_accuracy']:.4f} Score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_ucddb_threshold = ucddb_threshold
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Fold {fold_idx} early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    y_test, test_scores_raw, test_scores_smooth = predict_record_scores(model, args, test_records, device)
    fold_result = {
        "fold": fold_idx,
        "records": {
            "ucddb_train": ucddb_train_records,
            "ucddb_val": ucddb_val_records,
            "ucddb_test": test_records,
            "apnea_train": apnea_train_records,
            "apnea_val": apnea_val_records,
            "apnea_test": apnea_test_records,
        },
        "record_stats": {
            "ucddb_train": u_train_stats,
            "ucddb_val": u_val_stats,
            "apnea_train": a_train_stats,
            "apnea_val": a_val_stats,
        },
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_threshold_combined_val": best_threshold,
        "best_threshold_ucddb_val": best_ucddb_threshold,
        "ucddb_test": {
            "raw_threshold_0_5": mixed_trainer.evaluate_from_scores(y_test, test_scores_raw, 0.5),
            "raw_threshold_combined_val": mixed_trainer.evaluate_from_scores(y_test, test_scores_raw, best_threshold),
            "raw_threshold_ucddb_val": mixed_trainer.evaluate_from_scores(y_test, test_scores_raw, best_ucddb_threshold),
            "smooth_threshold_0_5": mixed_trainer.evaluate_from_scores(y_test, test_scores_smooth, 0.5),
            "smooth_threshold_combined_val": mixed_trainer.evaluate_from_scores(y_test, test_scores_smooth, best_threshold),
            "smooth_threshold_ucddb_val": mixed_trainer.evaluate_from_scores(y_test, test_scores_smooth, best_ucddb_threshold),
        },
        "model_path": str(model_path),
    }
    (fold_dir / "results.json").write_text(json.dumps(fold_result, indent=2), encoding="utf-8")
    return fold_result


def aggregate_results(results):
    metric_names = ["accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1", "roc_auc"]
    threshold_keys = sorted(results[0]["ucddb_test"].keys())
    aggregate = {}
    for threshold_key in threshold_keys:
        aggregate[threshold_key] = {}
        for metric in metric_names:
            values = np.asarray([result["ucddb_test"][threshold_key][metric] for result in results], dtype=np.float32)
            aggregate[threshold_key][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "values": values.tolist(),
            }
    return aggregate


def run(args):
    set_seed(args.seed)
    mixed_trainer.SHOW_PROGRESS = not args.no_progress
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record_ids = ucddb_runner.available_record_ids(Path(args.ucddb_dir))
    print(f"Using device: {device}")
    print(f"Preparing balanced UCDDB folds for {len(record_ids)} records...")
    record_summaries = [summarize_record(args, record_id) for record_id in record_ids]
    folds = balanced_folds(record_summaries, args.n_splits, args.seed)
    for idx, fold in enumerate(folds):
        print(
            f"Fold {idx}: records={fold['records']} samples={fold['samples']} "
            f"positive={fold['positive']} ratio={fold['positive_ratio']:.3f}"
        )

    apnea_records = mixed_trainer.available_apnea_records(Path(args.apnea_dir))
    apnea_split = mixed_trainer.split_records(apnea_records, args.apnea_val_size, args.apnea_test_size, args.seed)

    selected_folds = args.folds if args.folds is not None else list(range(args.n_splits))
    results = []
    for fold_idx in selected_folds:
        result = train_one_fold(args, fold_idx, folds, apnea_split, device)
        results.append(result)

    aggregate = aggregate_results(results)
    summary = {
        "settings": vars(args),
        "folds": folds,
        "record_summaries": record_summaries,
        "results": results,
        "aggregate": aggregate,
    }
    summary_path = output_dir / f"{args.experiment_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved CV summary: {summary_path}")
    print("Aggregate UCDDB CV metrics:")
    for threshold_key, metrics in aggregate.items():
        bacc = metrics["balanced_accuracy"]
        f1 = metrics["f1"]
        auc = metrics["roc_auc"]
        print(
            f"  {threshold_key}: BAcc={bacc['mean']:.4f}+/-{bacc['std']:.4f} "
            f"F1={f1['mean']:.4f}+/-{f1['std']:.4f} AUC={auc['mean']:.4f}+/-{auc['std']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Grouped UCDDB cross-validation with Apnea-ECG auxiliary training.")
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--apnea-dir", default="apnea-ecg")
    parser.add_argument("--ucddb-cache-dir", default="aligned_data/ucddb")
    parser.add_argument("--apnea-cache-dir", default="aligned_data/apnea_ecg")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--experiment-name", default="ucddb_cv_context3_focal2")
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--context-minutes", type=int, default=3)
    parser.add_argument("--smooth-width", type=int, default=11)
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--ucddb-val-fraction", type=float, default=0.25)
    parser.add_argument("--apnea-val-size", type=float, default=0.2)
    parser.add_argument("--apnea-test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ucddb-sample-weight", type=float, default=2.0)
    parser.add_argument("--apnea-sample-weight", type=float, default=1.0)
    parser.add_argument("--ucddb-val-weight", type=float, default=0.7)
    parser.add_argument("--min-specificity", type=float, default=0.65)
    parser.add_argument("--flip-prob", type=float, default=0.35)
    parser.add_argument("--max-shift-sec", type=float, default=2.0)
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.30)
    parser.add_argument("--noise-std-max", type=float, default=0.05)
    parser.add_argument("--baseline-amp-max", type=float, default=0.08)
    parser.add_argument("--mask-prob", type=float, default=0.20)
    parser.add_argument("--mask-max-sec", type=float, default=0.40)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
