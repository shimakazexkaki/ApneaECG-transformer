import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

import mixed_trainer
import ucddb_highres_trainer
import ucddb_runner


FS = 100


@dataclass
class LiteratureFeatureRecord:
    record_id: str
    channel: int
    features: np.ndarray
    labels: np.ndarray
    minute_indices: np.ndarray
    signal: np.ndarray
    rpeaks: np.ndarray
    detector_used: str
    stats: dict


def minute_labels(duration_sec, events, min_overlap_sec):
    n_minutes = int(duration_sec // 60)
    labels = np.zeros(n_minutes, dtype=np.int64)
    for minute in range(n_minutes):
        start = minute * 60
        end = start + 60
        for event_start, event_end, _ in events:
            overlap = min(end, event_end) - max(start, event_start)
            if overlap > min_overlap_sec:
                labels[minute] = 1
                break
    return labels


def _refine_rpeaks(signal, peaks, fs, search_radius_sec=0.08):
    radius = max(1, int(search_radius_sec * fs))
    refined = []
    n = len(signal)
    for peak in peaks:
        start = max(0, int(peak) - radius)
        end = min(n, int(peak) + radius + 1)
        if end <= start:
            continue
        local = signal[start:end]
        refined.append(start + int(np.argmax(np.abs(local))))
    if not refined:
        return np.empty((0,), dtype=np.int64)
    refined = np.unique(np.asarray(refined, dtype=np.int64))
    if len(refined) <= 1:
        return refined
    keep = [int(refined[0])]
    min_distance = int(0.25 * fs)
    for peak in refined[1:]:
        if int(peak) - keep[-1] >= min_distance:
            keep.append(int(peak))
        else:
            previous = keep[-1]
            if abs(signal[int(peak)]) > abs(signal[previous]):
                keep[-1] = int(peak)
    return np.asarray(keep, dtype=np.int64)


def detect_rpeaks(signal, fs=FS, detector="auto"):
    detector = detector.lower()
    if detector in ("auto", "biosppy_hamilton"):
        try:
            from biosppy.signals import ecg

            peaks, = ecg.hamilton_segmenter(signal=signal.astype(np.float64), sampling_rate=fs)
            corrected, = ecg.correct_rpeaks(
                signal=signal.astype(np.float64),
                rpeaks=peaks,
                sampling_rate=fs,
                tol=0.05,
            )
            return np.asarray(corrected, dtype=np.int64), "biosppy_hamilton"
        except Exception:
            if detector == "biosppy_hamilton":
                raise

    if detector in ("auto", "wfdb_xqrs"):
        try:
            import wfdb.processing as processing

            peaks = processing.xqrs_detect(sig=signal.astype(np.float64), fs=fs, verbose=False)
            return np.asarray(peaks, dtype=np.int64), "wfdb_xqrs"
        except Exception:
            if detector == "wfdb_xqrs":
                raise

    x = mixed_trainer.normalize_segment(signal)
    diff = np.diff(x, prepend=x[0])
    energy = diff * diff
    ma_width = max(1, int(0.12 * fs))
    kernel = np.ones(ma_width, dtype=np.float32) / ma_width
    integrated = np.convolve(energy, kernel, mode="same")
    threshold = np.median(integrated) + 0.8 * np.std(integrated)
    peaks, _ = find_peaks(
        integrated,
        height=threshold,
        distance=max(1, int(0.25 * fs)),
        prominence=max(1e-6, 0.15 * np.std(integrated)),
    )
    return _refine_rpeaks(signal, peaks, fs), "scipy_hamilton_style"


def cache_path_for_amp(args, record_id, channel, peak_amplitude):
    hyp = "hyp" if not args.apnea_only else "apneaonly"
    overlap = str(args.label_overlap_sec).replace(".", "p")
    detector = args.detector.replace("_", "")
    return (
        Path(args.cache_dir)
        / f"{record_id}_ch{channel}_{hyp}_ctx{args.context_minutes}_len{args.target_length}_"
        f"overlap{overlap}_{detector}_{peak_amplitude}.npz"
    )


def cache_path(args, record_id, channel):
    return cache_path_for_amp(args, record_id, channel, args.peak_amplitude)


def _rri_amplitude_series(signal, rpeaks, fs, peak_amplitude):
    if len(rpeaks) < 2:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    times = rpeaks.astype(np.float32) / fs
    rr = np.diff(times, prepend=times[0])
    rr[0] = rr[1] if len(rr) > 1 else 1.0
    rr = np.clip(rr, 0.30, 2.50)
    amp = signal[rpeaks].astype(np.float32)
    if peak_amplitude == "absolute":
        amp = np.abs(amp)
    return times.astype(np.float32), rr.astype(np.float32), amp.astype(np.float32)


def _interpolate_context(times, rr, amp, start_sec, end_sec, target_length, min_beats, normalize=True):
    mask = (times >= start_sec) & (times < end_sec)
    if int(mask.sum()) < min_beats:
        return None

    local_t = times[mask] - start_sec
    duration = end_sec - start_sec
    grid = np.linspace(0.0, duration, target_length, endpoint=False, dtype=np.float32)
    rr_interp = np.interp(grid, local_t, rr[mask]).astype(np.float32)
    amp_interp = np.interp(grid, local_t, amp[mask]).astype(np.float32)
    features = np.stack([rr_interp, amp_interp], axis=1)
    if normalize:  # per-window z-score；recording-level 正規化時關掉(保留個體水準供篩檢)
        features = mixed_trainer.normalize_matrix(features.T).T
    return features.astype(np.float32)


def build_features_from_signal_rpeaks(args, record_id, channel, signal, rpeaks, detector_used):
    data_dir = Path(args.ucddb_dir)
    duration_sec = len(signal) // FS
    events = ucddb_runner.parse_respiratory_events(
        data_dir / f"{record_id}_respevt.txt",
        include_hypopnea=not args.apnea_only,
    )
    return build_features_from_events(
        args, record_id, channel, signal, rpeaks, events, duration_sec, detector_used
    )


def build_features_from_events(args, record_id, channel, signal, rpeaks, events, duration_sec, detector_used):
    """Event-agnostic feature builder shared by UCDDB and MESA.

    ``events`` is a list of ``(start_sec, end_sec, type)`` tuples (same shape as
    ``ucddb_runner.parse_respiratory_events`` / MESA NSRR XML parsing). Everything
    downstream (RRI/amplitude series, 5-minute context interpolation, per-window
    z-score, minute labels) is identical to the UCDDB pipeline so that cross-dataset
    train/test features are constructed the same way.
    """
    labels_by_minute = minute_labels(duration_sec, events, args.label_overlap_sec)
    times, rr, amp = _rri_amplitude_series(signal, rpeaks, FS, args.peak_amplitude)

    half = args.context_minutes // 2
    context_sec = args.context_minutes * 60
    xs = []
    ys = []
    minute_indices = []
    skipped_low_beats = 0
    for minute in range(half, len(labels_by_minute) - half):
        start = (minute - half) * 60
        end = start + context_sec
        features = _interpolate_context(
            times,
            rr,
            amp,
            start,
            end,
            args.target_length,
            args.min_beats,
        )
        if features is None:
            skipped_low_beats += 1
            continue
        xs.append(features)
        ys.append(int(labels_by_minute[minute]))
        minute_indices.append(int(minute))

    if xs:
        features = np.stack(xs).astype(np.float32)
        labels = np.asarray(ys, dtype=np.int64)
        minute_indices = np.asarray(minute_indices, dtype=np.int32)
    else:
        features = np.empty((0, args.target_length, 2), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)
        minute_indices = np.empty((0,), dtype=np.int32)

    stats = {
        "record": record_id,
        "channel": int(channel),
        "duration_sec": int(duration_sec),
        "minutes_total": int(len(labels_by_minute)),
        "samples": int(len(labels)),
        "positive": int(labels.sum()) if len(labels) else 0,
        "normal": int((labels == 0).sum()) if len(labels) else 0,
        "rpeaks": int(len(rpeaks)),
        "mean_hr_bpm": float(len(rpeaks) / max(duration_sec, 1) * 60.0),
        "detector_used": detector_used,
        "skipped_low_beats": int(skipped_low_beats),
        "label_rule": f"positive if respiratory-event overlap > {args.label_overlap_sec} sec in target minute",
    }
    return LiteratureFeatureRecord(
        record_id=record_id,
        channel=channel,
        features=features,
        labels=labels,
        minute_indices=minute_indices,
        signal=signal.astype(np.float32, copy=False),
        rpeaks=rpeaks.astype(np.int64, copy=False),
        detector_used=detector_used,
        stats=stats,
    )


def build_segment_features_from_events(args, record_id, channel, signal, rpeaks, events,
                                       duration_sec, detector_used,
                                       seg_sec, stride_sec, context_sec, norm_mode="window"):
    """細粒度、乾淨標籤(對標 DSF-SANet, Diagnostics 2024）。

    每個 seg_sec 秒的目標段,以 context_sec 秒脈絡內插成 (target_length,2) 為輸入。
    標籤規則:
      - 目標段「完整落在」某呼吸事件內 → 陽性 (1)
      - 目標段與所有事件「零重疊」      → 陰性 (0)
      - 部分重疊(模糊)                → 丟棄
    輸入表徵與 per-minute 版完全相同(逐窗 z-score 的 900x2),只改標籤粒度/乾淨度。
    """
    times, rr, amp = _rri_amplitude_series(signal, rpeaks, FS, args.peak_amplitude)
    if norm_mode == "recording" and len(rr):
        # 用整夜統計量正規化 → 保留窗與窗之間、相對個體基線的水準差(篩檢需要)
        rr = (rr - rr.mean()) / (rr.std() + 1e-8)
        amp = (amp - amp.mean()) / (amp.std() + 1e-8)
    win_normalize = (norm_mode == "window")
    half_ctx = context_sec / 2.0
    n_seg = max(0, int((duration_sec - seg_sec) // stride_sec) + 1)
    xs, ys, seg_idx = [], [], []
    skipped_low_beats = 0
    skipped_ambiguous = 0
    for k in range(n_seg):
        s0 = k * stride_sec
        s1 = s0 + seg_sec
        fully_in = any(es <= s0 and ee >= s1 for (es, ee, *_) in events)
        any_overlap = any(min(ee, s1) > max(es, s0) for (es, ee, *_) in events)
        if fully_in:
            y = 1
        elif not any_overlap:
            y = 0
        else:
            skipped_ambiguous += 1
            continue
        center = (s0 + s1) / 2.0
        cstart, cend = center - half_ctx, center + half_ctx
        if cstart < 0:
            cstart, cend = 0.0, context_sec
        if cend > duration_sec:
            cstart, cend = max(0.0, duration_sec - context_sec), float(duration_sec)
        features = _interpolate_context(times, rr, amp, cstart, cend, args.target_length,
                                        args.min_beats, normalize=win_normalize)
        if features is None:
            skipped_low_beats += 1
            continue
        xs.append(features)
        ys.append(int(y))
        seg_idx.append(int(k))

    if xs:
        features = np.stack(xs).astype(np.float32)
        labels = np.asarray(ys, dtype=np.int64)
        seg_indices = np.asarray(seg_idx, dtype=np.int32)
    else:
        features = np.empty((0, args.target_length, 2), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)
        seg_indices = np.empty((0,), dtype=np.int32)

    stats = {
        "record": record_id,
        "channel": int(channel),
        "duration_sec": int(duration_sec),
        "samples": int(len(labels)),
        "positive": int(labels.sum()) if len(labels) else 0,
        "normal": int((labels == 0).sum()) if len(labels) else 0,
        "rpeaks": int(len(rpeaks)),
        "mean_hr_bpm": float(len(rpeaks) / max(duration_sec, 1) * 60.0),
        "detector_used": detector_used,
        "skipped_low_beats": int(skipped_low_beats),
        "skipped_ambiguous": int(skipped_ambiguous),
        "label_rule": f"clean segment: +1 if {seg_sec}s seg fully inside event, 0 if zero overlap, drop partial; ctx={context_sec}s",
    }
    return LiteratureFeatureRecord(
        record_id=record_id, channel=channel, features=features, labels=labels,
        minute_indices=seg_indices, signal=signal.astype(np.float32, copy=False),
        rpeaks=rpeaks.astype(np.int64, copy=False), detector_used=detector_used, stats=stats,
    )


def build_record_features(args, record_id, channel):
    data_dir = Path(args.ucddb_dir)
    signal = ucddb_highres_trainer.read_ucddb_signal(data_dir, record_id, channel)
    rpeaks, detector_used = detect_rpeaks(signal, FS, args.detector)
    return build_features_from_signal_rpeaks(args, record_id, channel, signal, rpeaks, detector_used)


def save_record_cache(path, record):
    np.savez_compressed(
        path,
        features=record.features,
        labels=record.labels,
        minute_indices=record.minute_indices,
        signal=record.signal,
        rpeaks=record.rpeaks,
        stats_json=json.dumps(record.stats),
    )


def load_record_features(args, record_id, channel):
    path = cache_path(args, record_id, channel)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.rebuild_cache:
        cached = np.load(path, allow_pickle=False)
        stats = json.loads(str(cached["stats_json"]))
        return LiteratureFeatureRecord(
            record_id=record_id,
            channel=channel,
            features=cached["features"].astype(np.float32, copy=False),
            labels=cached["labels"].astype(np.int64, copy=False),
            minute_indices=cached["minute_indices"].astype(np.int32, copy=False),
            signal=cached["signal"].astype(np.float32, copy=False),
            rpeaks=cached["rpeaks"].astype(np.int64, copy=False),
            detector_used=stats.get("detector_used", "unknown"),
            stats=stats,
        )

    if not args.rebuild_cache:
        alternate_amp = "signed" if args.peak_amplitude == "absolute" else "absolute"
        alternate_path = cache_path_for_amp(args, record_id, channel, alternate_amp)
        if alternate_path.exists():
            cached = np.load(alternate_path, allow_pickle=False)
            alternate_stats = json.loads(str(cached["stats_json"]))
            record = build_features_from_signal_rpeaks(
                args,
                record_id,
                channel,
                cached["signal"].astype(np.float32, copy=False),
                cached["rpeaks"].astype(np.int64, copy=False),
                alternate_stats.get("detector_used", "unknown"),
            )
            save_record_cache(path, record)
            return record

    record = build_record_features(args, record_id, channel)
    save_record_cache(path, record)
    return record


def load_records(args, record_ids):
    records = []
    for record_id in tqdm(record_ids, desc="Loading literature features", disable=args.no_progress):
        for channel in args.channels:
            record = load_record_features(args, record_id, channel)
            if len(record.labels):
                records.append(record)
    if not records:
        raise RuntimeError("No UCDDB literature features were loaded.")
    return records


def record_has_positive(args, record_id):
    data_dir = Path(args.ucddb_dir)
    duration_sec = len(ucddb_highres_trainer.read_ucddb_signal(data_dir, record_id, args.channels[0])) // FS
    events = ucddb_runner.parse_respiratory_events(
        data_dir / f"{record_id}_respevt.txt",
        include_hypopnea=not args.apnea_only,
    )
    labels = minute_labels(duration_sec, events, args.label_overlap_sec)
    return bool(labels.sum() > 0)


def available_records(args):
    record_ids = sorted(args.records) if args.records else ucddb_runner.available_record_ids(Path(args.ucddb_dir))
    if args.exclude_no_positive:
        record_ids = [record_id for record_id in record_ids if record_has_positive(args, record_id)]
    return sorted(record_ids)


def add_feature_args(parser):
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--cache-dir", default="aligned_data/ucddb_literature_features")
    parser.add_argument("--records", nargs="*", default=None)
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--context-minutes", type=int, default=5)
    parser.add_argument("--target-length", type=int, default=900)
    parser.add_argument("--label-overlap-sec", type=float, default=5.0)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--exclude-no-positive", action="store_true")
    parser.add_argument(
        "--detector",
        choices=["auto", "biosppy_hamilton", "wfdb_xqrs", "scipy_hamilton_style"],
        default="auto",
    )
    parser.add_argument("--peak-amplitude", choices=["absolute", "signed"], default="absolute")
    parser.add_argument("--min-beats", type=int, default=20)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")


def main():
    parser = argparse.ArgumentParser(description="Build UCDDB RRI/R-peak-amplitude features used in literature baselines.")
    add_feature_args(parser)
    parser.add_argument("--output-summary", default="outputs/ucddb_literature_feature_summary.json")
    args = parser.parse_args()

    record_ids = available_records(args)
    records = load_records(args, record_ids)
    summary = {
        "settings": vars(args),
        "records": [record.stats for record in records],
        "total_samples": int(sum(len(record.labels) for record in records)),
        "total_positive": int(sum(record.labels.sum() for record in records)),
    }
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved feature summary: {args.output_summary}")
    print(f"Records loaded: {len(records)}")
    print(f"Samples: {summary['total_samples']} | Positive: {summary['total_positive']}")


if __name__ == "__main__":
    main()
