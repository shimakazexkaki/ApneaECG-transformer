import argparse
import json
from pathlib import Path

import numpy as np
import torch

import apnea_trainer
import mixed_trainer
import ucddb_runner
import ucddb_trainer


def moving_average(scores, width):
    scores = np.asarray(scores, dtype=np.float32)
    if width <= 1 or len(scores) == 0:
        return scores
    if width % 2 == 0:
        raise ValueError("--smooth-width must be odd.")
    pad = width // 2
    padded = np.pad(scores, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def predict_record_scores(model, x, y, batch_size, device):
    labels, scores = mixed_trainer.predict_scores(model, x, y, batch_size, device)
    return labels, scores


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


def load_apnea_record_context(data_dir, cache_dir, record_id, context_minutes):
    x, y = mixed_trainer.load_apnea_record(data_dir, cache_dir, record_id)
    x = mixed_trainer.normalize_matrix(x)
    return mixed_trainer.make_context_windows(x, y, context_minutes)


def collect_scores(model, domain, records, args, device):
    labels_all = []
    raw_all = []
    smooth_all = []

    if domain == "ucddb":
        data_dir = Path(args.ucddb_dir)
        cache_dir = Path(args.ucddb_cache_dir)
        for record_id in records:
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
                labels, scores = predict_record_scores(model, x, y, args.batch_size, device)
                labels_all.extend(labels.tolist())
                raw_all.extend(scores.tolist())
                smooth_all.extend(moving_average(scores, args.smooth_width).tolist())
    else:
        data_dir = Path(args.apnea_dir)
        cache_dir = Path(args.apnea_cache_dir)
        for record_id in records:
            x, y = load_apnea_record_context(data_dir, cache_dir, record_id, args.context_minutes)
            if len(y) == 0:
                continue
            labels, scores = predict_record_scores(model, x, y, args.batch_size, device)
            labels_all.extend(labels.tolist())
            raw_all.extend(scores.tolist())
            smooth_all.extend(moving_average(scores, args.smooth_width).tolist())

    return (
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(raw_all, dtype=np.float32),
        np.asarray(smooth_all, dtype=np.float32),
    )


def metrics_block(y, raw_scores, smooth_scores, threshold_raw, threshold_smooth, ucddb_threshold_smooth):
    return {
        "raw_threshold_0_5": mixed_trainer.evaluate_from_scores(y, raw_scores, 0.5),
        "raw_threshold_val": mixed_trainer.evaluate_from_scores(y, raw_scores, threshold_raw),
        "smooth_threshold_0_5": mixed_trainer.evaluate_from_scores(y, smooth_scores, 0.5),
        "smooth_threshold_val": mixed_trainer.evaluate_from_scores(y, smooth_scores, threshold_smooth),
        "smooth_threshold_ucddb_val": mixed_trainer.evaluate_from_scores(y, smooth_scores, ucddb_threshold_smooth),
    }


def run(args):
    mixed_trainer.SHOW_PROGRESS = False
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ucddb_records = ucddb_runner.available_record_ids(Path(args.ucddb_dir))
    apnea_records = mixed_trainer.available_apnea_records(Path(args.apnea_dir))
    ucddb_train_records, ucddb_val_records, ucddb_test_records = mixed_trainer.split_records(
        ucddb_records, args.val_size, args.test_size, args.seed
    )
    apnea_train_records, apnea_val_records, apnea_test_records = mixed_trainer.split_records(
        apnea_records, args.val_size, args.test_size, args.seed
    )

    model = apnea_trainer.ParallelCNNTransformer().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print(f"Using device: {device}")
    print(f"Evaluating model: {args.model}")
    print(f"Context={args.context_minutes} smooth_width={args.smooth_width}")

    y_u_val, u_val_raw, u_val_smooth = collect_scores(model, "ucddb", ucddb_val_records, args, device)
    y_a_val, a_val_raw, a_val_smooth = collect_scores(model, "apnea", apnea_val_records, args, device)
    y_u_test, u_test_raw, u_test_smooth = collect_scores(model, "ucddb", ucddb_test_records, args, device)
    y_a_test, a_test_raw, a_test_smooth = collect_scores(model, "apnea", apnea_test_records, args, device)

    raw_val_y = np.concatenate([y_u_val, y_a_val])
    raw_val_scores = np.concatenate([u_val_raw, a_val_raw])
    smooth_val_scores = np.concatenate([u_val_smooth, a_val_smooth])

    threshold_raw, raw_val_metrics, raw_constrained = mixed_trainer.best_threshold_balanced(
        raw_val_y, raw_val_scores, args.min_specificity
    )
    threshold_smooth, smooth_val_metrics, smooth_constrained = mixed_trainer.best_threshold_balanced(
        raw_val_y, smooth_val_scores, args.min_specificity
    )
    ucddb_threshold_smooth, ucddb_smooth_val_metrics, _ = mixed_trainer.best_threshold_balanced(
        y_u_val, u_val_smooth, args.min_specificity
    )

    results = {
        "settings": vars(args),
        "records": {
            "ucddb_val": ucddb_val_records,
            "ucddb_test": ucddb_test_records,
            "apnea_val": apnea_val_records,
            "apnea_test": apnea_test_records,
        },
        "thresholds": {
            "raw_combined_val": threshold_raw,
            "smooth_combined_val": threshold_smooth,
            "smooth_ucddb_val": ucddb_threshold_smooth,
            "raw_constrained": raw_constrained,
            "smooth_constrained": smooth_constrained,
        },
        "validation": {
            "raw_combined_val": raw_val_metrics,
            "smooth_combined_val": smooth_val_metrics,
            "smooth_ucddb_val": ucddb_smooth_val_metrics,
        },
        "ucddb_test": metrics_block(
            y_u_test, u_test_raw, u_test_smooth, threshold_raw, threshold_smooth, ucddb_threshold_smooth
        ),
        "apnea_test": metrics_block(
            y_a_test, a_test_raw, a_test_smooth, threshold_raw, threshold_smooth, ucddb_threshold_smooth
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / args.result_name
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Saved results: {result_path}")
    print("\nFinal holdout test metrics")
    for domain in ("ucddb_test", "apnea_test"):
        print(f"{domain}:")
        for name, metrics in results[domain].items():
            print(
                f"  {name}: Acc={metrics['accuracy']:.4f} "
                f"BAcc={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f} "
                f"Rec={metrics['recall']:.4f} Spec={metrics['specificity']:.4f} "
                f"AUC={metrics['roc_auc']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal smoothing for mixed CNN apnea models.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--apnea-dir", default="apnea-ecg")
    parser.add_argument("--ucddb-cache-dir", default="aligned_data/ucddb")
    parser.add_argument("--apnea-cache-dir", default="aligned_data/apnea_ecg")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--result-name", default="sequence_smoothing_results.json")
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--context-minutes", type=int, default=5)
    parser.add_argument("--smooth-width", type=int, default=5)
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-specificity", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
