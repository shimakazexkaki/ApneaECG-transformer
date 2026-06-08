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

import ucddb_highres_trainer as highres
import ucddb_runner


RANDOM_STATE = 42


def record_window_summary(args, record_id):
    records, _ = highres.load_records(args, [record_id])
    dataset = highres.HighResWindowDataset(
        records,
        args.window_sec,
        args.eval_stride_sec,
        args.label_second,
        label_mode=args.label_mode,
        min_event_overlap_sec=args.min_event_overlap_sec,
        augment=False,
        seed=args.seed,
    )
    summary = highres.dataset_summary(dataset)
    summary["record"] = record_id
    return summary


def balanced_folds(record_summaries, n_splits, seed):
    rng = random.Random(seed)
    summaries = list(record_summaries)
    rng.shuffle(summaries)
    summaries.sort(key=lambda item: (item["positive"], item["windows"]), reverse=True)
    folds = [{"records": [], "windows": 0, "positive": 0} for _ in range(n_splits)]
    for summary in summaries:
        fold = min(folds, key=lambda item: (item["positive"], item["windows"], len(item["records"])))
        fold["records"].append(summary["record"])
        fold["windows"] += int(summary["windows"])
        fold["positive"] += int(summary["positive"])
    for fold in folds:
        fold["records"].sort()
        fold["normal"] = fold["windows"] - fold["positive"]
        fold["positive_ratio"] = float(fold["positive"] / fold["windows"]) if fold["windows"] else 0.0
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


def make_loss(args, train_dataset, device):
    if args.record_aware_loss != "none":
        return None
    if args.focal_gamma > 0:
        return highres.mixed_trainer.SoftmaxFocalLoss(
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
    counts = np.bincount(train_dataset.labels, minlength=2)
    weights = len(train_dataset.labels) / (2.0 * np.maximum(counts, 1))
    return nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )


def make_dataset(args, records, stride, augment, seed, max_normal_ratio=0.0, max_windows=0, return_record_index=False):
    return highres.HighResWindowDataset(
        records,
        args.window_sec,
        stride,
        args.label_second,
        label_mode=args.label_mode,
        min_event_overlap_sec=args.min_event_overlap_sec,
        augment=augment,
        aug_args=args,
        seed=seed,
        max_normal_ratio=max_normal_ratio,
        max_windows=max_windows,
        return_record_index=return_record_index,
    )


def evaluate_dataset(model, dataset, threshold, minute_threshold, args, device):
    return highres.evaluate_model_dataset(model, dataset, threshold, minute_threshold, args, device)


def train_one_fold(args, fold_idx, folds, device):
    highres.set_seed(args.seed + fold_idx)
    test_ids = sorted(folds[fold_idx]["records"])
    train_val_ids = sorted(
        record
        for idx, fold in enumerate(folds)
        if idx != fold_idx
        for record in fold["records"]
    )
    train_ids, val_ids = train_val_split(train_val_ids, args.val_fraction, args.seed + fold_idx)

    train_records, train_stats = highres.load_records(args, train_ids)
    val_records, val_stats = highres.load_records(args, val_ids)
    test_records, test_stats = highres.load_records(args, test_ids)

    train_dataset = make_dataset(
        args,
        train_records,
        args.train_stride_sec,
        augment=True,
        seed=args.seed + fold_idx,
        max_normal_ratio=args.max_normal_ratio,
        max_windows=args.max_train_windows,
        return_record_index=args.record_aware_loss != "none",
    )
    val_dataset = make_dataset(
        args,
        val_records,
        args.eval_stride_sec,
        augment=False,
        seed=args.seed + fold_idx,
    )

    sampler = (
        highres.make_sampler(train_dataset, args.samples_per_epoch, args.record_balanced_sampler)
        if args.weighted_sampler
        else None
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = highres.make_model(args).to(device)
    criterion = make_loss(args, train_dataset, device)
    class_weights = highres.class_weights_from_dataset(train_dataset, device)
    group_weights = highres.init_group_weights(args, train_dataset, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.4, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    fold_dir = Path(args.output_dir) / f"{args.experiment_name}_fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.pth"
    denom = args.samples_per_epoch if args.samples_per_epoch > 0 else len(train_dataset)
    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_minute_threshold = 0.5
    patience = 0

    print(f"\nFold {fold_idx}")
    print(f"  Train records: {train_ids}")
    print(f"  Val records:   {val_ids}")
    print(f"  Test records:  {test_ids}")
    print(f"  Train summary: {highres.dataset_summary(train_dataset)}")
    print(f"  Val summary:   {highres.dataset_summary(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(
            loader,
            desc=f"Fold {fold_idx} Epoch {epoch}/{args.epochs}",
            leave=False,
            disable=args.no_progress,
        ):
            if args.record_aware_loss == "none":
                batch_x, batch_y = batch
                batch_record = None
            else:
                batch_x, batch_y, batch_record = batch
                batch_record = batch_record.to(device, non_blocking=True)
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(batch_x)
                loss = highres.compute_training_loss(
                    args,
                    logits,
                    batch_y,
                    batch_record,
                    criterion,
                    class_weights,
                    group_weights,
                )
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * batch_y.size(0)

        val_y, val_scores = highres.predict_scores(model, val_dataset, args.batch_size, device, args.no_progress, args.amp)
        val_scores = highres.normalize_scores_from_args(val_dataset, val_scores, args)
        threshold, val_win_metrics, _ = highres.choose_threshold(
            val_y,
            val_scores,
            min_specificity=args.min_specificity,
            strategy=args.threshold_strategy,
        )
        minute_y, minute_scores = highres.aggregate_minutes(val_dataset, val_scores, reducer=args.minute_reducer)
        minute_threshold, val_min_metrics, _ = highres.choose_threshold(
            minute_y,
            minute_scores,
            min_specificity=args.min_specificity,
            strategy=args.threshold_strategy,
        )
        score = (
            args.window_val_weight * val_win_metrics["balanced_accuracy"]
            + (1.0 - args.window_val_weight) * val_min_metrics["balanced_accuracy"]
        )
        scheduler.step(score)
        print(
            f"Fold {fold_idx} Epoch {epoch:02d} | Loss={running_loss / denom:.4f} | "
            f"Win BAcc={val_win_metrics['balanced_accuracy']:.4f} AUC={val_win_metrics['roc_auc']:.4f} "
            f"Min BAcc={val_min_metrics['balanced_accuracy']:.4f} AUC={val_min_metrics['roc_auc']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_minute_threshold = minute_threshold
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Fold {fold_idx} early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    final_val_dataset = make_dataset(args, val_records, args.final_eval_stride_sec, False, args.seed + fold_idx)
    final_test_dataset = make_dataset(args, test_records, args.final_eval_stride_sec, False, args.seed + fold_idx)
    final_val_y, final_val_scores = highres.predict_scores(
        model,
        final_val_dataset,
        args.batch_size,
        device,
        args.no_progress,
        args.amp,
    )
    final_val_scores = highres.normalize_scores_from_args(final_val_dataset, final_val_scores, args)
    final_threshold, _, _ = highres.choose_threshold(
        final_val_y,
        final_val_scores,
        args.min_specificity,
        args.threshold_strategy,
    )
    final_min_y, final_min_scores = highres.aggregate_minutes(
        final_val_dataset,
        final_val_scores,
        reducer=args.minute_reducer,
    )
    final_minute_threshold, _, _ = highres.choose_threshold(
        final_min_y,
        final_min_scores,
        args.min_specificity,
        args.threshold_strategy,
    )
    result = {
        "fold": fold_idx,
        "records": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "record_stats": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_threshold_window_training_eval": float(best_threshold),
        "best_threshold_minute_training_eval": float(best_minute_threshold),
        "final_threshold_window": float(final_threshold),
        "final_threshold_minute": float(final_minute_threshold),
        "summaries": {
            "train": highres.dataset_summary(train_dataset),
            "val": highres.dataset_summary(final_val_dataset),
            "test": highres.dataset_summary(final_test_dataset),
        },
        "test": evaluate_dataset(
            model,
            final_test_dataset,
            final_threshold,
            final_minute_threshold,
            args,
            device,
        ),
        "model_path": str(model_path),
    }
    (fold_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def apply_summary_settings(args, settings):
    for key, value in settings.items():
        if hasattr(args, key):
            setattr(args, key, value)


def run_eval_existing(args):
    summary_path = Path(args.eval_only_summary)
    output_suffix = args.eval_output_suffix
    retune_thresholds = args.eval_retune_thresholds
    eval_min_specificity = args.eval_min_specificity
    score_normalization = args.score_normalization
    score_normalization_group = args.score_normalization_group
    threshold_strategy = args.threshold_strategy
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    apply_summary_settings(args, summary.get("settings", {}))
    args.score_normalization = score_normalization
    args.score_normalization_group = score_normalization_group
    args.threshold_strategy = threshold_strategy
    if eval_min_specificity is not None:
        args.min_specificity = eval_min_specificity
    highres.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Re-evaluating existing CV summary: {summary_path}")
    print(f"Score normalization: {args.score_normalization} group={args.score_normalization_group}")
    if retune_thresholds:
        print(
            f"Retuning thresholds on validation records with "
            f"strategy={args.threshold_strategy} min_specificity={args.min_specificity}"
        )

    updated_results = []
    for result in summary["results"]:
        fold_idx = int(result["fold"])
        test_ids = sorted(result["records"]["test"])
        val_ids = sorted(result["records"]["val"])
        print(f"Fold {fold_idx}: loading model and test records {test_ids}")
        test_records, _ = highres.load_records(args, test_ids)
        final_test_dataset = make_dataset(
            args,
            test_records,
            args.final_eval_stride_sec,
            False,
            args.seed + fold_idx,
        )
        model = highres.make_model(args).to(device)
        model.load_state_dict(torch.load(result["model_path"], map_location=device, weights_only=False))
        final_threshold = float(result["final_threshold_window"])
        final_minute_threshold = float(result["final_threshold_minute"])
        if retune_thresholds:
            val_records, _ = highres.load_records(args, val_ids)
            final_val_dataset = make_dataset(
                args,
                val_records,
                args.final_eval_stride_sec,
                False,
                args.seed + fold_idx,
            )
            final_val_y, final_val_scores = highres.predict_scores(
                model,
                final_val_dataset,
                args.batch_size,
                device,
                args.no_progress,
                args.amp,
            )
            final_val_scores = highres.normalize_scores_from_args(final_val_dataset, final_val_scores, args)
            final_threshold, _, _ = highres.choose_threshold(
                final_val_y,
                final_val_scores,
                args.min_specificity,
                args.threshold_strategy,
            )
            final_min_y, final_min_scores = highres.aggregate_minutes(
                final_val_dataset,
                final_val_scores,
                reducer=args.minute_reducer,
            )
            final_minute_threshold, _, _ = highres.choose_threshold(
                final_min_y,
                final_min_scores,
                args.min_specificity,
                args.threshold_strategy,
            )
        updated = dict(result)
        updated["eval_retuned_thresholds"] = bool(retune_thresholds)
        updated["eval_min_specificity"] = float(args.min_specificity)
        updated["score_normalization"] = args.score_normalization
        updated["score_normalization_group"] = args.score_normalization_group
        updated["threshold_strategy"] = args.threshold_strategy
        updated["final_threshold_window"] = float(final_threshold)
        updated["final_threshold_minute"] = float(final_minute_threshold)
        updated["test"] = evaluate_dataset(
            model,
            final_test_dataset,
            final_threshold,
            final_minute_threshold,
            args,
            device,
        )
        fold_dir = Path(result["model_path"]).parent
        (fold_dir / "results_with_burden.json").write_text(json.dumps(updated, indent=2), encoding="utf-8")
        updated_results.append(updated)

    summary["results"] = updated_results
    summary["aggregate"] = aggregate_results(updated_results)
    summary["record_burden_aggregate"] = aggregate_record_burden(updated_results)
    summary["eval_only"] = {
        "source_summary": str(summary_path),
        "retuned_thresholds": bool(retune_thresholds),
        "min_specificity": float(args.min_specificity),
        "score_normalization": args.score_normalization,
        "score_normalization_group": args.score_normalization_group,
        "threshold_strategy": args.threshold_strategy,
    }
    output_path = summary_path.with_name(f"{summary_path.stem}{output_suffix}{summary_path.suffix}")
    markdown_path = output_path.with_suffix(".md")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_summary(summary), encoding="utf-8")
    print(f"Saved burden summary: {output_path}")
    print(f"Saved burden markdown: {markdown_path}")
    print(markdown_summary(summary))


def aggregate_results(results):
    aggregate = {}
    for level in ["window", "minute"]:
        aggregate[level] = {}
        for threshold_key in ["threshold_0_5", "threshold_val", "threshold_oracle"]:
            aggregate[level][threshold_key] = {}
            for metric in ["accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1", "roc_auc"]:
                values = np.asarray(
                    [result["test"][level][threshold_key][metric] for result in results],
                    dtype=np.float32,
                )
                aggregate[level][threshold_key][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "values": values.tolist(),
                }
    return aggregate


def aggregate_record_burden(results):
    aggregate = {}
    for group_key in ["record_channel_summary", "subject_summary"]:
        aggregate[group_key] = {}
        for metric in [
            "records",
            "mae_apnea_minutes_per_hour",
            "mean_true_apnea_minutes_per_hour",
            "mean_pred_apnea_minutes_per_hour",
            "corr_true_pred_apnea_minutes_per_hour",
        ]:
            values = np.asarray(
                [result["test"]["record_burden"][group_key][metric] for result in results],
                dtype=np.float32,
            )
            aggregate[group_key][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "values": values.tolist(),
            }
    return aggregate


def markdown_summary(summary):
    lines = [
        f"# {summary['settings']['experiment_name']}",
        "",
        f"Model: `{summary['settings']['model']}`",
        f"Label mode: `{summary['settings']['label_mode']}`",
        f"Channels: `{summary['settings']['channels']}`",
        "",
        "## Aggregate Test Metrics",
        "",
        "| Level | Threshold | BAcc | AUC | F1 | Recall | Specificity |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for level, threshold_reports in summary["aggregate"].items():
        for threshold, metrics in threshold_reports.items():
            lines.append(
                f"| {level} | {threshold} | "
                f"{metrics['balanced_accuracy']['mean']:.4f} +/- {metrics['balanced_accuracy']['std']:.4f} | "
                f"{metrics['roc_auc']['mean']:.4f} +/- {metrics['roc_auc']['std']:.4f} | "
                f"{metrics['f1']['mean']:.4f} +/- {metrics['f1']['std']:.4f} | "
                f"{metrics['recall']['mean']:.4f} +/- {metrics['recall']['std']:.4f} | "
                f"{metrics['specificity']['mean']:.4f} +/- {metrics['specificity']['std']:.4f} |"
            )
    if "record_burden_aggregate" in summary:
        lines.extend(
            [
                "",
                "## Record-Level AHI-Like Burden",
                "",
                "AHI-like burden is apnea-positive minutes per hour, not clinical event-based AHI.",
                "",
                "| Group | MAE | True Mean | Pred Mean | Corr |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        labels = {
            "record_channel_summary": "record-channel",
            "subject_summary": "subject",
        }
        for group_key, metrics in summary["record_burden_aggregate"].items():
            lines.append(
                f"| {labels.get(group_key, group_key)} | "
                f"{metrics['mae_apnea_minutes_per_hour']['mean']:.4f} +/- "
                f"{metrics['mae_apnea_minutes_per_hour']['std']:.4f} | "
                f"{metrics['mean_true_apnea_minutes_per_hour']['mean']:.4f} +/- "
                f"{metrics['mean_true_apnea_minutes_per_hour']['std']:.4f} | "
                f"{metrics['mean_pred_apnea_minutes_per_hour']['mean']:.4f} +/- "
                f"{metrics['mean_pred_apnea_minutes_per_hour']['std']:.4f} | "
                f"{metrics['corr_true_pred_apnea_minutes_per_hour']['mean']:.4f} +/- "
                f"{metrics['corr_true_pred_apnea_minutes_per_hour']['std']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def run(args):
    if args.eval_only_summary:
        run_eval_existing(args)
        return

    highres.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_ids = ucddb_runner.available_record_ids(Path(args.ucddb_dir))
    if args.exclude_no_positive:
        record_ids = [
            record_id
            for record_id in record_ids
            if highres.record_has_positive(Path(args.ucddb_dir), record_id, include_hypopnea=not args.apnea_only)
        ]
    print(f"Using device: {device}")
    print(f"Preparing high-res folds for {len(record_ids)} records...")
    summaries = [record_window_summary(args, record_id) for record_id in record_ids]
    folds = balanced_folds(summaries, args.n_splits, args.seed)
    for idx, fold in enumerate(folds):
        print(
            f"Fold {idx}: records={fold['records']} windows={fold['windows']} "
            f"positive={fold['positive']} ratio={fold['positive_ratio']:.3f}"
        )

    selected_folds = args.folds if args.folds is not None else list(range(args.n_splits))
    results = []
    for fold_idx in selected_folds:
        results.append(train_one_fold(args, fold_idx, folds, device))

    summary = {
        "settings": vars(args),
        "record_summaries": summaries,
        "folds": folds,
        "results": results,
        "aggregate": aggregate_results(results),
        "record_burden_aggregate": aggregate_record_burden(results),
    }
    summary_path = output_dir / f"{args.experiment_name}_summary.json"
    markdown_path = output_dir / f"{args.experiment_name}_summary.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_summary(summary), encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    print(f"Saved markdown: {markdown_path}")
    print(markdown_summary(summary))


def main():
    parser = argparse.ArgumentParser(description="Grouped CV for high-resolution UCDDB ECG models.")
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--cache-dir", default="aligned_data/ucddb_highres")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--experiment-name", default="ucddb_highres_cv")
    parser.add_argument("--model", choices=["sleeplite", "cnn_transformer", "parallel_cnn_transformer"], default="cnn_transformer")
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--window-sec", type=int, default=11)
    parser.add_argument("--train-stride-sec", type=int, default=1)
    parser.add_argument("--eval-stride-sec", type=int, default=10)
    parser.add_argument("--final-eval-stride-sec", type=int, default=1)
    parser.add_argument("--label-second", type=int, default=1)
    parser.add_argument("--label-mode", choices=["second", "overlap"], default="overlap")
    parser.add_argument("--min-event-overlap-sec", type=int, default=5)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--exclude-no-positive", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--record-aware-loss", choices=["none", "group_mean", "group_max", "group_dro"], default="none")
    parser.add_argument("--groupdro-eta", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weighted-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-balanced-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--samples-per-epoch", type=int, default=100000)
    parser.add_argument("--max-normal-ratio", type=float, default=3.0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--min-specificity", type=float, default=0.0)
    parser.add_argument("--window-val-weight", type=float, default=0.7)
    parser.add_argument("--minute-reducer", choices=["max", "mean"], default="mean")
    parser.add_argument("--threshold-strategy", choices=["balanced", "prevalence_match"], default="balanced")
    parser.add_argument(
        "--score-normalization",
        choices=["none", "record_center", "record_zscore", "record_minmax", "record_rank"],
        default="none",
    )
    parser.add_argument("--score-normalization-group", choices=["subject", "record-channel"], default="subject")
    parser.add_argument("--flip-prob", type=float, default=0.25)
    parser.add_argument("--max-shift-sec", type=float, default=0.5)
    parser.add_argument("--scale-min", type=float, default=0.80)
    parser.add_argument("--scale-max", type=float, default=1.25)
    parser.add_argument("--noise-std-max", type=float, default=0.04)
    parser.add_argument("--baseline-amp-max", type=float, default=0.05)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mask-max-sec", type=float, default=0.25)
    parser.add_argument("--time-warp-prob", type=float, default=0.0)
    parser.add_argument("--time-warp-min", type=float, default=0.95)
    parser.add_argument("--time-warp-max", type=float, default=1.05)
    parser.add_argument("--smooth-prob", type=float, default=0.0)
    parser.add_argument("--smooth-max-kernel", type=int, default=9)
    parser.add_argument("--spike-prob", type=float, default=0.0)
    parser.add_argument("--spike-amp-max", type=float, default=0.5)
    parser.add_argument("--spike-max-count", type=int, default=3)
    parser.add_argument("--quantize-prob", type=float, default=0.0)
    parser.add_argument("--quantize-levels", type=int, default=64)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-only-summary", default=None, help="Re-evaluate an existing CV summary without retraining.")
    parser.add_argument("--eval-output-suffix", default="_with_burden")
    parser.add_argument("--eval-retune-thresholds", action="store_true")
    parser.add_argument("--eval-min-specificity", type=float, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
