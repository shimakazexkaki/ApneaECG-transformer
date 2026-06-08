import argparse
import json
from pathlib import Path

import numpy as np


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(x) < 2 or float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def collect_rows(summary, group):
    key = "subject_rows" if group == "subject" else "record_channel_rows"
    rows = []
    results = summary.get("results")
    if results:
        for result in results:
            fold = result.get("fold", result.get("tag", "-"))
            for row in result["test"]["record_burden"][key]:
                error = float(row["pred_apnea_minutes_per_hour"] - row["true_apnea_minutes_per_hour"])
                rows.append(
                    {
                        "fold": fold,
                        "record": row["record"],
                        "channel": row.get("channel", "all"),
                        "scored_minutes": int(row["scored_minutes"]),
                        "true": float(row["true_apnea_minutes_per_hour"]),
                        "pred": float(row["pred_apnea_minutes_per_hour"]),
                        "error": error,
                        "abs_error": abs(error),
                        "mean_score": float(row["mean_score"]),
                    }
                )
    else:
        for row in summary["test"]["record_burden"][key]:
            error = float(row["pred_apnea_minutes_per_hour"] - row["true_apnea_minutes_per_hour"])
            rows.append(
                {
                    "fold": "-",
                    "record": row["record"],
                    "channel": row.get("channel", "all"),
                    "scored_minutes": int(row["scored_minutes"]),
                    "true": float(row["true_apnea_minutes_per_hour"]),
                    "pred": float(row["pred_apnea_minutes_per_hour"]),
                    "error": error,
                    "abs_error": abs(error),
                    "mean_score": float(row["mean_score"]),
                }
            )
    rows.sort(key=lambda row: row["abs_error"], reverse=True)
    return rows


def render(summary, rows, group, source, top_n):
    true_values = [row["true"] for row in rows]
    pred_values = [row["pred"] for row in rows]
    errors = [row["error"] for row in rows]
    abs_errors = [row["abs_error"] for row in rows]
    over = [row for row in rows if row["error"] > 0]
    under = [row for row in rows if row["error"] < 0]
    experiment = summary.get("settings", {}).get("experiment_name") or summary.get("settings", {}).get("result_name") or Path(source).stem

    lines = [
        f"# High-Res Burden Diagnostics: {experiment}",
        "",
        f"Source: `{source}`",
        f"Group: `{group}`",
        "",
        "AHI-like burden is apnea-positive minutes per hour, not clinical event-based AHI.",
        "",
        "## Summary",
        "",
        f"- Rows: {len(rows)}",
        f"- MAE: {float(np.mean(abs_errors)):.4f}",
        f"- Mean signed error: {float(np.mean(errors)):.4f}",
        f"- True mean: {float(np.mean(true_values)):.4f}",
        f"- Pred mean: {float(np.mean(pred_values)):.4f}",
        f"- Corr(true, pred): {safe_corr(true_values, pred_values):.4f}",
        f"- Over-predicted rows: {len(over)}",
        f"- Under-predicted rows: {len(under)}",
        "",
        f"## Largest Errors Top {min(top_n, len(rows))}",
        "",
        "| Fold | Record | Channel | Minutes | True | Pred | Error | Mean Score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:top_n]:
        lines.append(
            f"| {row['fold']} | {row['record']} | {row['channel']} | {row['scored_minutes']} | "
            f"{row['true']:.4f} | {row['pred']:.4f} | {row['error']:.4f} | {row['mean_score']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run(args):
    source = Path(args.summary)
    summary = json.loads(source.read_text(encoding="utf-8"))
    rows = collect_rows(summary, args.group)
    output = Path(args.output) if args.output else source.with_name(source.stem + f"_{args.group}_burden_diagnostics.md")
    output.write_text(render(summary, rows, args.group, source, args.top_n), encoding="utf-8")
    print(f"Saved burden diagnostics: {output}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose high-resolution UCDDB record-level burden errors.")
    parser.add_argument("summary", help="High-res holdout result JSON or grouped-CV summary JSON with record_burden.")
    parser.add_argument("--group", choices=["subject", "record-channel"], default="subject")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
