import argparse
import json
import re
from pathlib import Path

import numpy as np


HIGHRES_PATTERN = "{record}_ch{channel}_hyp_fs100_bp0p5_45_highres.npz"
LITERATURE_PATTERN = "{record}_ch{channel}_hyp_ctx5_len900_overlap5p0_biosppyhamilton_{amplitude}.npz"


def parse_record_channel(path):
    match = re.match(r"(ucddb\d+)_ch(\d+)_", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def available_record_channels(cache_dir, channels=None):
    found = []
    for path in Path(cache_dir).glob("*_highres.npz"):
        parsed = parse_record_channel(path)
        if not parsed:
            continue
        record, channel = parsed
        if channels is not None and channel not in channels:
            continue
        found.append((record, channel))
    return sorted(found)


def load_burden_errors(summary_path):
    if not summary_path:
        return {}, {}
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    subject = {}
    record_channel = {}

    def add_rows(rows, fold="-"):
        for row in rows:
            error = float(row["pred_apnea_minutes_per_hour"] - row["true_apnea_minutes_per_hour"])
            item = {
                "fold": fold,
                "true": float(row["true_apnea_minutes_per_hour"]),
                "pred": float(row["pred_apnea_minutes_per_hour"]),
                "error": error,
                "abs_error": abs(error),
                "mean_score": float(row["mean_score"]),
            }
            channel = row.get("channel", "all")
            if channel == "all":
                subject[row["record"]] = item
            else:
                record_channel[(row["record"], int(channel))] = item

    if summary.get("results"):
        for result in summary["results"]:
            burden = result["test"]["record_burden"]
            fold = result.get("fold", "-")
            add_rows(burden.get("subject_rows", []), fold)
            add_rows(burden.get("record_channel_rows", []), fold)
    else:
        burden = summary["test"]["record_burden"]
        add_rows(burden.get("subject_rows", []))
        add_rows(burden.get("record_channel_rows", []))
    return subject, record_channel


def load_literature_cache(lit_cache_dir, record, channel):
    for amplitude in ["absolute", "signed"]:
        path = Path(lit_cache_dir) / LITERATURE_PATTERN.format(record=record, channel=channel, amplitude=amplitude)
        if path.exists():
            return np.load(path), path
    candidates = sorted(Path(lit_cache_dir).glob(f"{record}_ch{channel}_*biosppyhamilton*.npz"))
    if candidates:
        path = candidates[0]
        return np.load(path), path
    return None, None


def signal_stats(signal):
    signal = np.asarray(signal, dtype=np.float32)
    finite = np.isfinite(signal)
    finite_signal = signal[finite]
    if finite_signal.size == 0:
        return {
            "finite_ratio": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "mad": 0.0,
            "p01": 0.0,
            "p99": 0.0,
            "range_p01_p99": 0.0,
            "diff_mad_ratio": 0.0,
            "flatline_ratio": 1.0,
        }
    median = float(np.median(finite_signal))
    mad = float(np.median(np.abs(finite_signal - median)))
    p01, p99 = np.percentile(finite_signal, [1, 99])
    diff = np.diff(finite_signal)
    diff_mad = float(np.median(np.abs(diff - np.median(diff)))) if diff.size else 0.0
    eps = max(float(np.std(finite_signal)) * 1e-4, 1e-6)
    flatline_ratio = float(np.mean(np.abs(diff) < eps)) if diff.size else 1.0
    return {
        "finite_ratio": float(finite.mean()),
        "mean": float(finite_signal.mean()),
        "std": float(finite_signal.std()),
        "mad": mad,
        "p01": float(p01),
        "p99": float(p99),
        "range_p01_p99": float(p99 - p01),
        "diff_mad_ratio": float(diff_mad / (mad + 1e-8)),
        "flatline_ratio": flatline_ratio,
    }


def rpeak_stats(signal, rpeaks, fs=100):
    rpeaks = np.asarray(rpeaks, dtype=np.int64)
    duration_min = len(signal) / fs / 60.0 if len(signal) else 0.0
    if rpeaks.size < 2 or duration_min <= 0:
        return {
            "beats": int(rpeaks.size),
            "bpm": 0.0,
            "rri_median": 0.0,
            "rri_iqr": 0.0,
            "rri_p05": 0.0,
            "rri_p95": 0.0,
            "rri_cv": 0.0,
            "invalid_rri_ratio": 1.0,
            "amp_median": 0.0,
            "amp_iqr": 0.0,
        }
    rr = np.diff(rpeaks) / fs
    amps = signal[np.clip(rpeaks, 0, len(signal) - 1)]
    q25, q75 = np.percentile(rr, [25, 75])
    amp_q25, amp_q75 = np.percentile(amps, [25, 75])
    return {
        "beats": int(rpeaks.size),
        "bpm": float(rpeaks.size / duration_min),
        "rri_median": float(np.median(rr)),
        "rri_iqr": float(q75 - q25),
        "rri_p05": float(np.percentile(rr, 5)),
        "rri_p95": float(np.percentile(rr, 95)),
        "rri_cv": float(np.std(rr) / (np.mean(rr) + 1e-8)),
        "invalid_rri_ratio": float(np.mean((rr < 0.3) | (rr > 2.5))),
        "amp_median": float(np.median(amps)),
        "amp_iqr": float(amp_q75 - amp_q25),
    }


def quality_flags(row):
    flags = []
    if row["finite_ratio"] < 0.999:
        flags.append("nonfinite")
    if row["flatline_ratio"] > 0.05:
        flags.append("flatline")
    if row["bpm"] < 35 or row["bpm"] > 180:
        flags.append("bpm_outlier")
    if row["invalid_rri_ratio"] > 0.05:
        flags.append("rri_outlier")
    if row["amp_iqr"] < 0.02:
        flags.append("low_r_amp")
    if row["diff_mad_ratio"] > 1.5:
        flags.append("noisy_diff")
    return ",".join(flags) if flags else "ok"


def build_rows(args):
    subject_errors, record_channel_errors = load_burden_errors(args.burden_summary)
    record_channels = available_record_channels(args.highres_cache, args.channels)
    if args.records:
        wanted = set(args.records)
        record_channels = [(record, channel) for record, channel in record_channels if record in wanted]

    rows = []
    for record, channel in record_channels:
        highres_path = Path(args.highres_cache) / HIGHRES_PATTERN.format(record=record, channel=channel)
        highres = np.load(highres_path)
        signal = highres["signal"].astype(np.float32)
        labels = highres["second_labels"].astype(np.int64)
        lit, lit_path = load_literature_cache(args.literature_cache, record, channel)
        rpeaks = lit["rpeaks"] if lit is not None and "rpeaks" in lit.files else np.asarray([], dtype=np.int64)
        sig = signal_stats(signal)
        rpk = rpeak_stats(signal, rpeaks)
        duration_hr = float(len(labels) / 3600.0) if len(labels) else 0.0
        subject_error = subject_errors.get(record, {})
        channel_error = record_channel_errors.get((record, channel), {})
        second_label_minutes_per_hour = float(labels.sum() / 60.0 / duration_hr) if duration_hr else 0.0
        row = {
            "record": record,
            "channel": int(channel),
            "duration_min": float(len(labels) / 60.0),
            "second_label_minutes_per_hour": second_label_minutes_per_hour,
            "subject_true_burden": float(subject_error.get("true", second_label_minutes_per_hour)),
            "subject_burden_error": float(subject_error.get("error", 0.0)),
            "subject_burden_abs_error": float(subject_error.get("abs_error", 0.0)),
            "subject_pred_burden": float(subject_error.get("pred", 0.0)),
            "fold": subject_error.get("fold", "-"),
            "channel_burden_error": float(channel_error.get("error", 0.0)),
            "channel_burden_abs_error": float(channel_error.get("abs_error", 0.0)),
            "literature_cache": str(lit_path) if lit_path else "",
            **sig,
            **rpk,
        }
        row["flags"] = quality_flags(row)
        rows.append(row)
    rows.sort(key=lambda item: (item["subject_burden_abs_error"], item["record"], item["channel"]), reverse=True)
    return rows


def format_float(value, digits=3):
    return f"{float(value):.{digits}f}"


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(x) < 2 or float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def render_markdown(rows, args):
    hard_records = {}
    for row in rows:
        hard_records.setdefault(row["record"], row)
    hardest = sorted(hard_records.values(), key=lambda row: row["subject_burden_abs_error"], reverse=True)
    flag_counts = {}
    for row in rows:
        for flag in row["flags"].split(","):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    lines = [
        "# UCDDB Signal and R-Peak Quality Diagnostics",
        "",
        f"High-res cache: `{args.highres_cache}`",
        f"Literature cache: `{args.literature_cache}`",
        f"Burden summary: `{args.burden_summary}`" if args.burden_summary else "Burden summary: `none`",
        "",
        "## Summary",
        "",
        f"- Rows: {len(rows)} record-channel pairs",
        f"- Records: {len({row['record'] for row in rows})}",
        f"- Mean BPM: {format_float(np.mean([row['bpm'] for row in rows]))}",
        f"- Mean invalid RRI ratio: {format_float(np.mean([row['invalid_rri_ratio'] for row in rows]), 4)}",
        f"- Corr(abs burden error, invalid RRI ratio): {safe_corr([row['subject_burden_abs_error'] for row in rows], [row['invalid_rri_ratio'] for row in rows]):.4f}",
        f"- Corr(abs burden error, BPM): {safe_corr([row['subject_burden_abs_error'] for row in rows], [row['bpm'] for row in rows]):.4f}",
        f"- Corr(abs burden error, R-peak amp IQR): {safe_corr([row['subject_burden_abs_error'] for row in rows], [row['amp_iqr'] for row in rows]):.4f}",
        f"- Quality flags: {', '.join(f'{key}={value}' for key, value in sorted(flag_counts.items()))}",
        "",
        f"## Hardest Records Top {min(args.top_n, len(hardest))}",
        "",
        "| Record | Fold | Abs Burden Error | Error | Pred Burden | True Burden | ch0 BPM | ch2 BPM | ch0 Bad RRI | ch2 Bad RRI | Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    by_key = {(row["record"], row["channel"]): row for row in rows}
    for record_row in hardest[: args.top_n]:
        record = record_row["record"]
        ch0 = by_key.get((record, 0), {})
        ch2 = by_key.get((record, 2), {})
        flags = ",".join(sorted(set(filter(None, [ch0.get("flags", ""), ch2.get("flags", "")]))))
        lines.append(
            f"| {record} | {record_row['fold']} | {record_row['subject_burden_abs_error']:.3f} | "
            f"{record_row['subject_burden_error']:.3f} | {record_row['subject_pred_burden']:.3f} | "
            f"{record_row['subject_true_burden']:.3f} | "
            f"{ch0.get('bpm', 0.0):.2f} | {ch2.get('bpm', 0.0):.2f} | "
            f"{ch0.get('invalid_rri_ratio', 0.0):.3f} | {ch2.get('invalid_rri_ratio', 0.0):.3f} | {flags} |"
        )

    lines.extend(
        [
            "",
            "## Record-Channel Details",
            "",
            "| Record | Ch | Fold | Burden Err | BPM | RRI Median | RRI IQR | Bad RRI | Amp IQR | Signal STD | Diff/MAD | Flatline | Flags |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['record']} | {row['channel']} | {row['fold']} | {row['subject_burden_error']:.3f} | "
            f"{row['bpm']:.2f} | {row['rri_median']:.3f} | {row['rri_iqr']:.3f} | "
            f"{row['invalid_rri_ratio']:.3f} | {row['amp_iqr']:.3f} | {row['std']:.3f} | "
            f"{row['diff_mad_ratio']:.3f} | {row['flatline_ratio']:.3f} | {row['flags']} |"
        )
    return "\n".join(lines) + "\n"


def run(args):
    rows = build_rows(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(rows, args), encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved diagnostics: {output}")
    print(f"Saved rows: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose UCDDB ECG signal/R-peak quality by record and channel.")
    parser.add_argument("--highres-cache", default="aligned_data/ucddb_highres")
    parser.add_argument("--literature-cache", default="aligned_data/ucddb_literature_features")
    parser.add_argument("--burden-summary", default=None)
    parser.add_argument("--channels", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--records", nargs="*", default=None)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--output", default="outputs/ucddb_signal_rpeak_quality_diagnostics.md")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
