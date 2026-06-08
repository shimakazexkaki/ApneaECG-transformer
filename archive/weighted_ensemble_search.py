import argparse
import json
from pathlib import Path

import numpy as np
import torch

import apnea_trainer
import mixed_trainer
import sequence_evaluator
import ucddb_runner
import ucddb_trainer


def load_models(model_paths, device):
    models = []
    for path in model_paths:
        model = apnea_trainer.ParallelCNNTransformer().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model.eval()
        models.append(model)
    return models


def load_ucddb_record_context(args, record_id, channel):
    x, y = ucddb_trainer.load_ucddb_record(
        Path(args.ucddb_dir),
        Path(args.ucddb_cache_dir),
        record_id,
        channel,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
    )
    x = mixed_trainer.normalize_matrix(x)
    return mixed_trainer.make_context_windows(x, y, args.context_minutes)


def load_apnea_record_context(args, record_id):
    x, y = mixed_trainer.load_apnea_record(Path(args.apnea_dir), Path(args.apnea_cache_dir), record_id)
    x = mixed_trainer.normalize_matrix(x)
    return mixed_trainer.make_context_windows(x, y, args.context_minutes)


def predict_score_matrix(models, x, y, batch_size, device):
    scores = []
    labels = None
    for model in models:
        labels, model_scores = mixed_trainer.predict_scores(model, x, y, batch_size, device)
        scores.append(model_scores)
    return labels, np.stack(scores, axis=1).astype(np.float32)


def collect_score_matrix(models, domain, records, args, device):
    labels_all = []
    raw_rows = []
    smooth_rows = []
    for record_id in records:
        channel_list = args.channels if domain == "ucddb" else [None]
        for channel in channel_list:
            if domain == "ucddb":
                x, y = load_ucddb_record_context(args, record_id, channel)
            else:
                x, y = load_apnea_record_context(args, record_id)
            if len(y) == 0:
                continue
            labels, score_matrix = predict_score_matrix(models, x, y, args.batch_size, device)
            smooth_matrix = np.stack(
                [sequence_evaluator.moving_average(score_matrix[:, i], args.smooth_width) for i in range(score_matrix.shape[1])],
                axis=1,
            )
            labels_all.extend(labels.tolist())
            raw_rows.append(score_matrix)
            smooth_rows.append(smooth_matrix)
    return (
        np.asarray(labels_all, dtype=np.int64),
        np.concatenate(raw_rows, axis=0),
        np.concatenate(smooth_rows, axis=0),
    )


def weight_candidates(n_models, step):
    if n_models == 2:
        values = np.arange(0.0, 1.0 + 1e-9, step)
        for w0 in values:
            yield np.asarray([w0, 1.0 - w0], dtype=np.float32)
        return

    if n_models == 3:
        values = np.arange(0.0, 1.0 + 1e-9, step)
        for w0 in values:
            for w1 in values:
                w2 = 1.0 - w0 - w1
                if w2 >= -1e-9:
                    yield np.asarray([w0, w1, max(0.0, w2)], dtype=np.float32)
        return

    raise ValueError("Weight search currently supports 2 or 3 models.")


def search_weights(y, score_matrix, min_specificity, step):
    best = None
    for weights in weight_candidates(score_matrix.shape[1], step):
        scores = score_matrix @ weights
        threshold, metrics, constrained = mixed_trainer.best_threshold_balanced(y, scores, min_specificity)
        candidate = (
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
            threshold,
            weights,
            metrics,
            constrained,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return {
        "balanced_accuracy": best[0],
        "f1": best[1],
        "roc_auc": best[2],
        "threshold": best[3],
        "weights": best[4],
        "metrics": best[5],
        "constrained": best[6],
    }


def metrics_for(y, score_matrix, weights, threshold):
    scores = score_matrix @ weights
    return mixed_trainer.evaluate_from_scores(y, scores, threshold)


def run(args):
    mixed_trainer.SHOW_PROGRESS = False
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    models = load_models(args.models, device)

    ucddb_records = ucddb_runner.available_record_ids(Path(args.ucddb_dir))
    apnea_records = mixed_trainer.available_apnea_records(Path(args.apnea_dir))
    _, ucddb_val_records, ucddb_test_records = mixed_trainer.split_records(
        ucddb_records, args.val_size, args.test_size, args.seed
    )
    _, apnea_val_records, apnea_test_records = mixed_trainer.split_records(
        apnea_records, args.val_size, args.test_size, args.seed
    )

    print(f"Using device: {device}")
    print("Searching weighted ensemble:")
    for model_path in args.models:
        print(f"  {model_path}")

    y_u_val, u_val_raw, u_val_smooth = collect_score_matrix(models, "ucddb", ucddb_val_records, args, device)
    y_a_val, a_val_raw, a_val_smooth = collect_score_matrix(models, "apnea", apnea_val_records, args, device)
    y_u_test, u_test_raw, u_test_smooth = collect_score_matrix(models, "ucddb", ucddb_test_records, args, device)
    y_a_test, a_test_raw, a_test_smooth = collect_score_matrix(models, "apnea", apnea_test_records, args, device)

    combined_y = np.concatenate([y_u_val, y_a_val])
    combined_raw = np.concatenate([u_val_raw, a_val_raw], axis=0)
    combined_smooth = np.concatenate([u_val_smooth, a_val_smooth], axis=0)

    searches = {
        "ucddb_raw": search_weights(y_u_val, u_val_raw, args.min_specificity, args.weight_step),
        "ucddb_smooth": search_weights(y_u_val, u_val_smooth, args.min_specificity, args.weight_step),
        "combined_raw": search_weights(combined_y, combined_raw, args.min_specificity, args.weight_step),
        "combined_smooth": search_weights(combined_y, combined_smooth, args.min_specificity, args.weight_step),
    }

    results = {
        "settings": vars(args),
        "records": {
            "ucddb_val": ucddb_val_records,
            "ucddb_test": ucddb_test_records,
            "apnea_val": apnea_val_records,
            "apnea_test": apnea_test_records,
        },
        "searches": {
            name: {
                "weights": search["weights"].tolist(),
                "threshold": float(search["threshold"]),
                "validation_metrics": search["metrics"],
                "constrained": bool(search["constrained"]),
            }
            for name, search in searches.items()
        },
        "ucddb_test": {},
        "apnea_test": {},
    }

    for name, search in searches.items():
        matrix_kind = "smooth" if "smooth" in name else "raw"
        u_matrix = u_test_smooth if matrix_kind == "smooth" else u_test_raw
        a_matrix = a_test_smooth if matrix_kind == "smooth" else a_test_raw
        weights = search["weights"]
        threshold = search["threshold"]
        results["ucddb_test"][name] = metrics_for(y_u_test, u_matrix, weights, threshold)
        results["apnea_test"][name] = metrics_for(y_a_test, a_matrix, weights, threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / args.result_name
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Saved results: {result_path}")
    print("\nFinal holdout test metrics")
    for search_name in searches:
        weights = searches[search_name]["weights"]
        threshold = searches[search_name]["threshold"]
        u_metrics = results["ucddb_test"][search_name]
        a_metrics = results["apnea_test"][search_name]
        print(f"{search_name}: weights={weights.tolist()} threshold={threshold:.3f}")
        print(
            f"  UCDDB: Acc={u_metrics['accuracy']:.4f} BAcc={u_metrics['balanced_accuracy']:.4f} "
            f"F1={u_metrics['f1']:.4f} Rec={u_metrics['recall']:.4f} Spec={u_metrics['specificity']:.4f} "
            f"AUC={u_metrics['roc_auc']:.4f}"
        )
        print(
            f"  Apnea: Acc={a_metrics['accuracy']:.4f} BAcc={a_metrics['balanced_accuracy']:.4f} "
            f"F1={a_metrics['f1']:.4f} Rec={a_metrics['recall']:.4f} Spec={a_metrics['specificity']:.4f} "
            f"AUC={a_metrics['roc_auc']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Search weights for same-context raw ECG model ensembles.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--apnea-dir", default="apnea-ecg")
    parser.add_argument("--ucddb-cache-dir", default="aligned_data/ucddb")
    parser.add_argument("--apnea-cache-dir", default="aligned_data/apnea_ecg")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--result-name", default="weighted_ensemble_search_results.json")
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--context-minutes", type=int, default=3)
    parser.add_argument("--smooth-width", type=int, default=11)
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-specificity", type=float, default=0.65)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
