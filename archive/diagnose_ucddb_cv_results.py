import argparse
import json
from pathlib import Path

import numpy as np


def pick_metrics(result, level):
    test = result["test"]
    if level in test:
        return test[level]
    return test


def metric_value(metrics, threshold, name):
    return float(metrics[threshold][name])


def fold_positive_ratio(result, level):
    summaries = result.get("summaries", {})
    test_summary = summaries.get("test")
    if test_summary:
        denom_key = "windows" if "windows" in test_summary else "samples"
        denom = test_summary.get(denom_key, 0)
        if denom:
            return float(test_summary.get("positive", 0) / denom)

    record_stats = result.get("record_stats", {}).get("test", [])
    positive = sum(int(item.get("positive_seconds", item.get("positive", 0))) for item in record_stats)
    total = sum(
        int(item.get("duration_sec", item.get("samples", 0)))
        for item in record_stats
    )
    return float(positive / total) if total else 0.0


def fold_records(result):
    records = result.get("records", {}).get("test", [])
    if records:
        return ", ".join(records)
    return "-"


def summarize(summary, level):
    rows = []
    for result in summary["results"]:
        metrics = pick_metrics(result, level)
        bacc_val = metric_value(metrics, "threshold_val", "balanced_accuracy")
        bacc_oracle = metric_value(metrics, "threshold_oracle", "balanced_accuracy")
        auc = metric_value(metrics, "threshold_val", "roc_auc")
        recall = metric_value(metrics, "threshold_val", "recall")
        spec = metric_value(metrics, "threshold_val", "specificity")
        rows.append(
            {
                "fold": result.get("fold", result.get("tag", len(rows))),
                "test_records": fold_records(result),
                "positive_ratio": fold_positive_ratio(result, level),
                "best_epoch": result.get("best_epoch", "-"),
                "threshold": result.get("final_threshold_minute")
                if level == "minute"
                else result.get("final_threshold_window", result.get("best_threshold", "-")),
                "auc": auc,
                "bacc_val": bacc_val,
                "bacc_oracle": bacc_oracle,
                "oracle_gap": bacc_oracle - bacc_val,
                "recall": recall,
                "specificity": spec,
            }
        )
    return rows


def correlation(rows, x_key, y_key):
    if len(rows) < 2:
        return 0.0
    x = np.asarray([row[x_key] for row in rows], dtype=np.float32)
    y = np.asarray([row[y_key] for row in rows], dtype=np.float32)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def render_markdown(summary, rows, level, source_path):
    lines = [
        f"# CV Diagnostics: {summary.get('settings', {}).get('experiment_name', Path(source_path).stem)}",
        "",
        f"Source: `{source_path}`",
        f"Level: `{level}`",
        "",
        "| Fold | Test Records | Pos Ratio | Threshold | AUC | BAcc Val | BAcc Oracle | Oracle Gap | Recall | Specificity |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        threshold = row["threshold"]
        threshold_text = f"{threshold:.3f}" if isinstance(threshold, (float, int)) else str(threshold)
        lines.append(
            f"| {row['fold']} | {row['test_records']} | {row['positive_ratio']:.3f} | {threshold_text} | "
            f"{row['auc']:.4f} | {row['bacc_val']:.4f} | {row['bacc_oracle']:.4f} | "
            f"{row['oracle_gap']:.4f} | {row['recall']:.4f} | {row['specificity']:.4f} |"
        )

    auc_values = np.asarray([row["auc"] for row in rows], dtype=np.float32)
    bacc_values = np.asarray([row["bacc_val"] for row in rows], dtype=np.float32)
    gap_values = np.asarray([row["oracle_gap"] for row in rows], dtype=np.float32)
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Mean AUC: {auc_values.mean():.4f} +/- {auc_values.std(ddof=0):.4f}",
            f"- Mean validation-threshold BAcc: {bacc_values.mean():.4f} +/- {bacc_values.std(ddof=0):.4f}",
            f"- Mean oracle gap: {gap_values.mean():.4f} +/- {gap_values.std(ddof=0):.4f}",
            f"- Corr(pos ratio, AUC): {correlation(rows, 'positive_ratio', 'auc'):.4f}",
            f"- Corr(pos ratio, BAcc): {correlation(rows, 'positive_ratio', 'bacc_val'):.4f}",
            "",
            "## Interpretation Guide",
            "",
            "- Low AUC means the model ranking is weak; threshold calibration cannot fully fix it.",
            "- Large oracle gap means ranking has some useful signal, but validation threshold does not transfer.",
            "- High specificity with very low recall suggests the fold threshold is too conservative.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summarize(summary, args.level)
    output = Path(args.output) if args.output else summary_path.with_name(summary_path.stem + f"_{args.level}_diagnostics.md")
    output.write_text(render_markdown(summary, rows, args.level, summary_path), encoding="utf-8")
    print(f"Saved diagnostics: {output}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose UCDDB CV fold-level behavior.")
    parser.add_argument("summary", help="Path to a CV summary JSON.")
    parser.add_argument("--level", choices=["window", "minute"], default="minute")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
