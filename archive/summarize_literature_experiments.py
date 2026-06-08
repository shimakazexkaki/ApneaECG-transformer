import argparse
import json
from pathlib import Path


def format_metric(value):
    if isinstance(value, dict):
        return f"{value['mean']:.4f} +/- {value['std']:.4f}"
    return f"{value:.4f}"


def threshold_metrics(summary, level):
    aggregate = summary.get("aggregate")
    if aggregate:
        if "threshold_val" in aggregate:
            return aggregate["threshold_val"]
        if level in aggregate:
            return aggregate[level]["threshold_val"]

    if summary.get("results"):
        metrics = summary["results"][0]["test"]
        if level in metrics:
            return metrics[level]["threshold_val"]
        return metrics["threshold_val"]

    metrics = summary["test"]
    if level in metrics:
        return metrics[level]["threshold_val"]
    return metrics["threshold_val"]


def metric_block(summary, level):
    metrics = threshold_metrics(summary, level)
    return {
        "bacc": format_metric(metrics["balanced_accuracy"]),
        "auc": format_metric(metrics["roc_auc"]),
        "f1": format_metric(metrics["f1"]),
        "recall": format_metric(metrics["recall"]),
        "specificity": format_metric(metrics["specificity"]),
    }


def experiment_name(summary, path):
    settings = summary.get("settings", {})
    name = (
        summary.get("experiment_name")
        or settings.get("experiment_name")
        or settings.get("result_name")
        or path.stem
    )
    eval_only = summary.get("eval_only")
    if eval_only and eval_only.get("retuned_thresholds"):
        spec = str(eval_only.get("min_specificity", "")).replace(".", "p")
        name = f"{name}_retune_spec{spec}"
    if eval_only and eval_only.get("threshold_strategy") and eval_only.get("threshold_strategy") != "balanced":
        name = f"{name}_{eval_only['threshold_strategy']}"
    if eval_only and eval_only.get("score_normalization") and eval_only.get("score_normalization") != "none":
        norm = str(eval_only["score_normalization"]).replace("record_", "record")
        group = str(eval_only.get("score_normalization_group", "subject")).replace("-", "")
        name = f"{name}_{norm}_{group}"
    return name


def model_name(summary):
    settings = summary.get("settings", {})
    return summary.get("model_kind") or settings.get("model") or settings.get("model_name") or "-"


def protocol_name(summary):
    settings = summary.get("settings", {})
    if "protocol" in settings:
        return settings["protocol"]
    if "folds" in summary:
        return "cv"
    return "holdout"


def table_title(protocol, rows):
    if protocol == "literature":
        return "## Literature-Comparable Mode"
    if protocol == "cv":
        return "## Honest Grouped CV Mode"
    if protocol == "holdout":
        return "## Record Holdout / Supporting Runs"
    return f"## {protocol}"


def render_table(rows):
    lines = [
        "| Experiment | Model | Protocol | BAcc | AUC | F1 | Recall | Specificity |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['model']} | {row['protocol']} | "
            f"{row['bacc']} | {row['auc']} | {row['f1']} | {row['recall']} | {row['specificity']} |"
        )
    return lines


def run(args):
    grouped = {}
    for summary_path in args.summaries:
        path = Path(summary_path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        protocol = protocol_name(summary)
        metrics = metric_block(summary, args.level)
        grouped.setdefault(protocol, []).append(
            {
                "experiment": experiment_name(summary, path),
                "model": model_name(summary),
                "protocol": protocol,
                "path": str(path),
                **metrics,
            }
        )

    protocol_order = ["literature", "cv", "holdout"]
    ordered_protocols = [protocol for protocol in protocol_order if protocol in grouped]
    ordered_protocols.extend(protocol for protocol in grouped if protocol not in ordered_protocols)

    lines = [
        "# UCDDB Literature Baseline vs Transformer Experiments",
        "",
        f"Metrics use validation-selected thresholds at `{args.level}` level when the summary has window/minute metrics.",
        "",
    ]
    for protocol in ordered_protocols:
        lines.append(table_title(protocol, grouped[protocol]))
        lines.extend(render_table(grouped[protocol]))
        lines.append("")

    lines.extend(
        [
            "Notes:",
            "- `literature` protocol is segment-level and intended for literature-comparable debugging.",
            "- `cv` protocol is grouped by UCDDB record and should be used for wearable-device generalization claims.",
            "- High-resolution CV rows use minute-level metrics by default because the project goal is minute/apnea-burden detection.",
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparison table: {output}")


def main():
    parser = argparse.ArgumentParser(description="Summarize UCDDB literature baseline and Transformer experiment summaries.")
    parser.add_argument("summaries", nargs="+", help="Paths to experiment summary.json files.")
    parser.add_argument("--output", default="outputs/ucddb_literature_transformer_comparison.md")
    parser.add_argument("--level", choices=["window", "minute"], default="minute")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
