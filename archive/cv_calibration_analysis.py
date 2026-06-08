import argparse
import json
from pathlib import Path

import numpy as np
import torch

import apnea_trainer
import mixed_trainer
import sequence_evaluator
import ucddb_trainer


def load_model(model_path, device):
    model = apnea_trainer.ParallelCNNTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model


def load_ucddb_record_context(settings, record_id, channel):
    x, y = ucddb_trainer.load_ucddb_record(
        Path(settings["ucddb_dir"]),
        Path(settings["ucddb_cache_dir"]),
        record_id,
        channel,
        include_hypopnea=not settings.get("apnea_only", False),
        min_overlap_sec=settings.get("min_overlap_sec", 1.0),
    )
    x = mixed_trainer.normalize_matrix(x)
    return mixed_trainer.make_context_windows(x, y, settings.get("context_minutes", 3))


def collect_scores_for_records(model, settings, record_ids, device):
    y_all = []
    raw_all = []
    smooth_all = []
    smooth_width = settings.get("smooth_width", 11)
    batch_size = settings.get("batch_size", 64)
    for record_id in record_ids:
        for channel in settings.get("channels", [0]):
            x, y = load_ucddb_record_context(settings, record_id, channel)
            if len(y) == 0:
                continue
            labels, scores = mixed_trainer.predict_scores(model, x, y, batch_size, device)
            y_all.extend(labels.tolist())
            raw_all.extend(scores.tolist())
            smooth_all.extend(sequence_evaluator.moving_average(scores, smooth_width).tolist())
    return (
        np.asarray(y_all, dtype=np.int64),
        np.asarray(raw_all, dtype=np.float32),
        np.asarray(smooth_all, dtype=np.float32),
    )


def best_threshold(y, scores, min_specificity):
    threshold, metrics, constrained = mixed_trainer.best_threshold_balanced(y, scores, min_specificity)
    return threshold, metrics, constrained


def oracle_threshold(y, scores):
    best = None
    for threshold in np.linspace(0.01, 0.99, 197):
        metrics = mixed_trainer.evaluate_from_scores(y, scores, threshold)
        candidate = (metrics["balanced_accuracy"], metrics["f1"], float(threshold), metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2], best[3]


def threshold_for_prevalence(scores, prevalence):
    scores = np.asarray(scores, dtype=np.float32)
    prevalence = float(np.clip(prevalence, 1e-6, 1.0 - 1e-6))
    return float(np.quantile(scores, 1.0 - prevalence))


def aggregate_metric(fold_metrics):
    metric_names = ["accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1", "roc_auc"]
    aggregate = {}
    for metric in metric_names:
        values = np.asarray([item[metric] for item in fold_metrics], dtype=np.float32)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "values": values.tolist(),
        }
    return aggregate


def evaluate_protocols(fold_data, score_key, min_specificity):
    val_y = np.concatenate([fold["val_y"] for fold in fold_data])
    val_scores = np.concatenate([fold[f"val_{score_key}"] for fold in fold_data])
    pooled_threshold, pooled_val_metrics, pooled_constrained = best_threshold(
        val_y, val_scores, min_specificity
    )
    pooled_prevalence_threshold = threshold_for_prevalence(val_scores, float(val_y.mean()))

    protocol_metrics = {
        "fixed_0_5": [],
        "fold_val_best": [],
        "fold_val_prevalence": [],
        "pooled_val_best": [],
        "pooled_val_prevalence": [],
        "test_oracle": [],
    }
    protocol_thresholds = {
        "pooled_val_best": pooled_threshold,
        "pooled_val_prevalence": pooled_prevalence_threshold,
        "pooled_val_best_constrained": pooled_constrained,
        "pooled_val_best_validation_metrics": pooled_val_metrics,
    }

    per_fold = []
    for fold in fold_data:
        test_y = fold["test_y"]
        test_scores = fold[f"test_{score_key}"]
        val_y_fold = fold["val_y"]
        val_scores_fold = fold[f"val_{score_key}"]

        fold_threshold, fold_val_metrics, fold_constrained = best_threshold(
            val_y_fold, val_scores_fold, min_specificity
        )
        fold_prevalence_threshold = threshold_for_prevalence(val_scores_fold, float(val_y_fold.mean()))
        oracle_t, oracle_metrics = oracle_threshold(test_y, test_scores)

        fold_eval = {
            "fold": fold["fold"],
            "records": fold["test_records"],
            "thresholds": {
                "fixed_0_5": 0.5,
                "fold_val_best": fold_threshold,
                "fold_val_prevalence": fold_prevalence_threshold,
                "pooled_val_best": pooled_threshold,
                "pooled_val_prevalence": pooled_prevalence_threshold,
                "test_oracle": oracle_t,
                "fold_val_best_constrained": fold_constrained,
                "fold_val_best_validation_metrics": fold_val_metrics,
            },
            "metrics": {
                "fixed_0_5": mixed_trainer.evaluate_from_scores(test_y, test_scores, 0.5),
                "fold_val_best": mixed_trainer.evaluate_from_scores(test_y, test_scores, fold_threshold),
                "fold_val_prevalence": mixed_trainer.evaluate_from_scores(
                    test_y, test_scores, fold_prevalence_threshold
                ),
                "pooled_val_best": mixed_trainer.evaluate_from_scores(test_y, test_scores, pooled_threshold),
                "pooled_val_prevalence": mixed_trainer.evaluate_from_scores(
                    test_y, test_scores, pooled_prevalence_threshold
                ),
                "test_oracle": oracle_metrics,
            },
        }
        for protocol, metrics in fold_eval["metrics"].items():
            protocol_metrics[protocol].append(metrics)
        per_fold.append(fold_eval)

    return {
        "score_key": score_key,
        "pooled_thresholds": protocol_thresholds,
        "per_fold": per_fold,
        "aggregate": {
            protocol: aggregate_metric(metrics)
            for protocol, metrics in protocol_metrics.items()
        },
    }


def default_summary_files(output_dir):
    return [
        output_dir / f"ucddb_cv_context3_focal2_fold{i}_summary.json"
        for i in range(5)
    ]


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    summary_files = [Path(p) for p in args.summary_files] if args.summary_files else default_summary_files(output_dir)

    fold_data = []
    print(f"Using device: {device}")
    for summary_path in summary_files:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = summary["results"][0]
        settings = {**summary["settings"], **vars(args)}
        settings["channels"] = summary["settings"].get("channels", args.channels)
        settings["context_minutes"] = summary["settings"].get("context_minutes", args.context_minutes)
        settings["smooth_width"] = args.smooth_width
        settings["batch_size"] = args.batch_size
        model_path = result["model_path"]
        model = load_model(model_path, device)

        val_records = result["records"]["ucddb_val"]
        test_records = result["records"]["ucddb_test"]
        print(f"Collecting fold {result['fold']} from {summary_path.name}")
        val_y, val_raw, val_smooth = collect_scores_for_records(model, settings, val_records, device)
        test_y, test_raw, test_smooth = collect_scores_for_records(model, settings, test_records, device)
        fold_data.append(
            {
                "fold": int(result["fold"]),
                "summary_path": str(summary_path),
                "model_path": model_path,
                "val_records": val_records,
                "test_records": test_records,
                "val_y": val_y,
                "val_raw": val_raw,
                "val_smooth": val_smooth,
                "test_y": test_y,
                "test_raw": test_raw,
                "test_smooth": test_smooth,
            }
        )

    analyses = {
        "raw": evaluate_protocols(fold_data, "raw", args.min_specificity),
        "smooth": evaluate_protocols(fold_data, "smooth", args.min_specificity),
    }
    serializable = {
        "settings": {
            **vars(args),
            "summary_files": [str(path) for path in summary_files],
        },
        "folds": [
            {
                "fold": item["fold"],
                "summary_path": item["summary_path"],
                "model_path": item["model_path"],
                "val_records": item["val_records"],
                "test_records": item["test_records"],
                "val_samples": int(len(item["val_y"])),
                "val_positive": int(item["val_y"].sum()),
                "test_samples": int(len(item["test_y"])),
                "test_positive": int(item["test_y"].sum()),
            }
            for item in fold_data
        ],
        "analyses": analyses,
    }

    output_path = output_dir / args.result_name
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Saved calibration analysis: {output_path}")

    for score_key, analysis in analyses.items():
        print(f"\n{score_key} scores")
        for protocol, aggregate in analysis["aggregate"].items():
            bacc = aggregate["balanced_accuracy"]
            f1 = aggregate["f1"]
            auc = aggregate["roc_auc"]
            print(
                f"  {protocol}: BAcc={bacc['mean']:.4f}+/-{bacc['std']:.4f} "
                f"F1={f1['mean']:.4f}+/-{f1['std']:.4f} "
                f"AUC={auc['mean']:.4f}+/-{auc['std']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Analyze threshold calibration across saved UCDDB CV folds.")
    parser.add_argument("--summary-files", nargs="+", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--result-name", default="ucddb_cv_calibration_analysis.json")
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--ucddb-cache-dir", default="aligned_data/ucddb")
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--context-minutes", type=int, default=3)
    parser.add_argument("--smooth-width", type=int, default=11)
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--min-specificity", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
