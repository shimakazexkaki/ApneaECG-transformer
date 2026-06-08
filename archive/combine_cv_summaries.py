"""Combine partial UCDDB grouped-CV summary files into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ucddb_highres_grouped_cv import (
    aggregate_record_burden,
    aggregate_results,
    markdown_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", help="Partial grouped-CV summary JSON files.")
    parser.add_argument("--output", required=True, help="Output combined summary JSON path.")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name to store in the combined settings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.summaries]
    if not loaded:
        raise SystemExit("No summaries were provided.")

    results = []
    seen_folds = set()
    for summary in loaded:
        for result in summary.get("results", []):
            fold = result.get("fold")
            if fold in seen_folds:
                raise SystemExit(f"Duplicate fold in input summaries: {fold}")
            seen_folds.add(fold)
            results.append(result)
    results.sort(key=lambda item: item.get("fold", -1))

    settings = dict(loaded[0].get("settings", {}))
    output_path = Path(args.output)
    settings["experiment_name"] = args.experiment_name or output_path.stem.removesuffix("_summary")
    settings["folds"] = [result.get("fold") for result in results]

    combined = {
        "settings": settings,
        "record_summaries": loaded[0].get("record_summaries", []),
        "folds": loaded[0].get("folds", []),
        "results": results,
        "aggregate": aggregate_results(results),
    }
    if all("record_burden" in result.get("test", {}) for result in results):
        combined["record_burden_aggregate"] = aggregate_record_burden(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(markdown_summary(combined), encoding="utf-8")
    print(f"Saved combined summary: {output_path}")
    print(f"Saved combined markdown: {output_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
