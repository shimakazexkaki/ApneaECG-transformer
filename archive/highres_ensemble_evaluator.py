import argparse
import json
from pathlib import Path

import numpy as np
import torch

import ucddb_highres_trainer as highres
import ucddb_runner


def split_and_load(args):
    data_dir = Path(args.ucddb_dir)
    record_ids = ucddb_runner.available_record_ids(data_dir)
    if args.exclude_no_positive:
        record_ids = [
            record_id
            for record_id in record_ids
            if highres.record_has_positive(data_dir, record_id, include_hypopnea=not args.apnea_only)
        ]

    train_ids, val_ids, test_ids = highres.split_records(
        record_ids,
        args.val_size,
        args.test_size,
        args.seed,
    )

    val_records, val_stats = highres.load_records(args, val_ids)
    test_records, test_stats = highres.load_records(args, test_ids)
    val_dataset = highres.HighResWindowDataset(
        val_records,
        args.window_sec,
        args.eval_stride_sec,
        args.label_second,
        label_mode=args.label_mode,
        min_event_overlap_sec=args.min_event_overlap_sec,
        augment=False,
        seed=args.seed,
    )
    test_dataset = highres.HighResWindowDataset(
        test_records,
        args.window_sec,
        args.eval_stride_sec,
        args.label_second,
        label_mode=args.label_mode,
        min_event_overlap_sec=args.min_event_overlap_sec,
        augment=False,
        seed=args.seed,
    )
    return train_ids, val_ids, test_ids, val_stats, test_stats, val_dataset, test_dataset


def model_scores(args, model_path, dataset, device):
    model = highres.make_model(args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    labels, scores = highres.predict_scores(
        model,
        dataset,
        args.batch_size,
        device,
        no_progress=args.no_progress,
        amp=args.amp,
    )
    return labels, scores


def weighted_average(score_list, weights):
    out = np.zeros_like(score_list[0], dtype=np.float32)
    for score, weight in zip(score_list, weights):
        out += score.astype(np.float32) * float(weight)
    return out


def score_report(dataset, scores, threshold, minute_threshold, minute_reducer):
    minute_y, minute_scores = highres.aggregate_minutes(dataset, scores, reducer=minute_reducer)
    return {
        "window": highres.threshold_report(dataset.labels, scores, threshold),
        "minute": highres.threshold_report(minute_y, minute_scores, minute_threshold),
    }


def run(args):
    highres.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids, val_stats, test_stats, val_dataset, test_dataset = split_and_load(args)
    print(f"Using device: {device}")
    print(f"Records: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"Models: {args.models}")
    print(f"Eval stride: {args.eval_stride_sec}s | label second: {args.label_second}")
    print(f"Val summary:  {highres.dataset_summary(val_dataset)}")
    print(f"Test summary: {highres.dataset_summary(test_dataset)}")

    val_scores_list = []
    test_scores_list = []
    val_labels = None
    test_labels = None
    for model_path in args.models:
        print(f"Scoring validation with {model_path}")
        labels, scores = model_scores(args, model_path, val_dataset, device)
        if val_labels is None:
            val_labels = labels
        elif not np.array_equal(val_labels, labels):
            raise RuntimeError("Validation labels do not match across models.")
        val_scores_list.append(scores)

        print(f"Scoring test with {model_path}")
        labels, scores = model_scores(args, model_path, test_dataset, device)
        if test_labels is None:
            test_labels = labels
        elif not np.array_equal(test_labels, labels):
            raise RuntimeError("Test labels do not match across models.")
        test_scores_list.append(scores)

    candidates = []
    if len(args.models) == 1:
        weight_grid = [[1.0]]
    elif len(args.models) == 2:
        weight_grid = [[w, 1.0 - w] for w in np.linspace(0.0, 1.0, args.weight_steps + 1)]
    else:
        weight_grid = []
        for raw in np.random.default_rng(args.seed).dirichlet(np.ones(len(args.models)), args.random_weights):
            weight_grid.append(raw.tolist())

    val_min_y = None
    for weights in weight_grid:
        val_scores = weighted_average(val_scores_list, weights)
        threshold, win_metrics, _ = highres.best_threshold_balanced(
            val_labels, val_scores, min_specificity=args.min_specificity
        )
        minute_y, minute_scores = highres.aggregate_minutes(
            val_dataset, val_scores, reducer=args.minute_reducer
        )
        if val_min_y is None:
            val_min_y = minute_y
        minute_threshold, min_metrics, _ = highres.best_threshold_balanced(
            minute_y, minute_scores, min_specificity=args.min_specificity
        )
        score = (
            args.window_val_weight * win_metrics["balanced_accuracy"]
            + (1.0 - args.window_val_weight) * min_metrics["balanced_accuracy"]
        )
        candidates.append(
            {
                "weights": [float(w) for w in weights],
                "score": float(score),
                "window_threshold": float(threshold),
                "minute_threshold": float(minute_threshold),
                "val_window": win_metrics,
                "val_minute": min_metrics,
            }
        )

    best = max(candidates, key=lambda item: (item["score"], item["val_minute"]["roc_auc"]))
    test_scores = weighted_average(test_scores_list, best["weights"])
    val_scores = weighted_average(val_scores_list, best["weights"])

    results = {
        "settings": vars(args),
        "records": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "record_stats": {
            "val": val_stats,
            "test": test_stats,
        },
        "summaries": {
            "val": highres.dataset_summary(val_dataset),
            "test": highres.dataset_summary(test_dataset),
        },
        "best": best,
        "all_candidates": candidates,
        "val": score_report(
            val_dataset,
            val_scores,
            best["window_threshold"],
            best["minute_threshold"],
            args.minute_reducer,
        ),
        "test": score_report(
            test_dataset,
            test_scores,
            best["window_threshold"],
            best["minute_threshold"],
            args.minute_reducer,
        ),
    }

    result_path = output_dir / args.result_name
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results: {result_path}")
    print(f"Best weights: {best['weights']}")
    print("\nFinal ensemble test metrics")
    for level, report in results["test"].items():
        for key, metrics in report.items():
            if key == "oracle_threshold":
                continue
            print(
                f"  {level}_{key}: Acc={metrics['accuracy']:.4f} "
                f"BAcc={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f} "
                f"Rec={metrics['recall']:.4f} Spec={metrics['specificity']:.4f} "
                f"AUC={metrics['roc_auc']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate weighted ensembles of UCDDB high-res models.")
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--cache-dir", default="aligned_data/ucddb_highres")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--result-name", default="ucddb_highres_ensemble_results.json")
    parser.add_argument("--model", choices=["sleeplite", "parallel_cnn_transformer"], default="sleeplite")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--window-sec", type=int, default=11)
    parser.add_argument("--eval-stride-sec", type=int, default=1)
    parser.add_argument("--label-second", type=int, default=5)
    parser.add_argument("--label-mode", choices=["second", "overlap"], default="second")
    parser.add_argument("--min-event-overlap-sec", type=int, default=5)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--exclude-no-positive", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-specificity", type=float, default=0.0)
    parser.add_argument("--window-val-weight", type=float, default=0.7)
    parser.add_argument("--minute-reducer", choices=["max", "mean"], default="max")
    parser.add_argument("--weight-steps", type=int, default=20)
    parser.add_argument("--random-weights", type=int, default=100)
    parser.add_argument("--seed", type=int, default=highres.RANDOM_STATE)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
