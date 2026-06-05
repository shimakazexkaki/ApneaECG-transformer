import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyedflib
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


APNEA_RELEASE_RECORDS = (
    [f"a{i:02d}" for i in range(1, 21)]
    + [f"b{i:02d}" for i in range(1, 6)]
    + [f"c{i:02d}" for i in range(1, 11)]
)
APNEA_WITHHELD_RECORDS = [f"x{i:02d}" for i in range(1, 36)]
UCDDB_EXCLUDED_NO_SA = {"ucddb008", "ucddb011", "ucddb013", "ucddb018"}


@dataclass
class FeatureRecord:
    record_id: str
    features: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    stats: dict


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_records: List[str]
    val_records: List[str]
    test_records: List[str]
    stats: dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - float(np.mean(x))) / (float(np.std(x)) + 1e-8)).astype(np.float32)


def patch_edf_start_time(edf_path: Path) -> None:
    # UCDDB EDF headers sometimes use HH:MM:SS, while pyedflib expects HH.MM.SS.
    with edf_path.open("rb") as f:
        f.seek(176)
        raw = f.read(8)
    fixed = raw.replace(b":", b".")
    if fixed != raw:
        with edf_path.open("r+b") as f:
            f.seek(176)
            f.write(fixed)


def parse_apnea_withheld_answers(path: Path) -> Dict[str, List[int]]:
    labels: Dict[str, List[int]] = {}
    current: Optional[str] = None
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            current = None
            continue
        if re.fullmatch(r"x\d{2}", line):
            current = line
            labels[current] = []
            continue
        match = re.match(r"^\d+\s+([AN]+)$", line)
        if current and match:
            labels[current].extend(1 if c == "A" else 0 for c in match.group(1))
    return labels


def load_apnea_signal_labels(data_dir: Path, record_id: str) -> Tuple[np.ndarray, int, np.ndarray]:
    signal, fields = wfdb.rdsamp(str(data_dir / record_id))
    fs = int(round(float(fields.get("fs", 100))))
    ecg = signal[:, 0].astype(np.float32)

    if record_id.startswith("x"):
        answers = parse_apnea_withheld_answers(data_dir / "event-2-answers")
        if record_id not in answers:
            raise RuntimeError(f"Missing withheld labels for {record_id}")
        labels = np.asarray(answers[record_id], dtype=np.int64)
    else:
        ann = wfdb.rdann(str(data_dir / record_id), "apn")
        labels = np.asarray([1 if s == "A" else 0 for s in ann.symbol if s in ("A", "N")], dtype=np.int64)

    return ecg, fs, labels


def load_apnea_qrs_rpeaks(data_dir: Path, record_id: str) -> Optional[np.ndarray]:
    qrs_path = data_dir / f"{record_id}.qrs"
    if not qrs_path.exists():
        return None
    try:
        ann = wfdb.rdann(str(data_dir / record_id), "qrs")
    except Exception:
        return None
    return np.asarray(ann.sample, dtype=np.int64)


def parse_ucddb_events(path: Path, include_hypopnea: bool = True) -> List[Tuple[int, int, str]]:
    events = []
    for line in path.read_text(errors="ignore").splitlines():
        match = re.match(r"^(\d\d):(\d\d):(\d\d)\s+(\S+)\s+(.*)$", line)
        if not match:
            continue
        hh, mm, ss, event_type, rest = match.groups()
        is_apnea = event_type.startswith("APNEA")
        is_hypopnea = event_type.startswith("HYP")
        if not is_apnea and not (include_hypopnea and is_hypopnea):
            continue
        duration = None
        for token in rest.split():
            if re.fullmatch(r"\d+", token):
                duration = int(token)
                break
        if duration is None:
            continue
        start = int(hh) * 3600 + int(mm) * 60 + int(ss)
        events.append((start, start + duration, event_type))
    return events


def ucddb_minute_labels(duration_sec: float, events: Iterable[Tuple[int, int, str]], overlap_sec: float) -> np.ndarray:
    n_minutes = int(duration_sec // 60)
    labels = np.zeros(n_minutes, dtype=np.int64)
    for minute in range(n_minutes):
        start = minute * 60
        end = start + 60
        for event_start, event_end, _ in events:
            overlap = min(end, event_end) - max(start, event_start)
            if overlap > overlap_sec:
                labels[minute] = 1
                break
    return labels


def available_ucddb_records(data_dir: Path) -> List[str]:
    records = []
    for edf in sorted(data_dir.glob("ucddb*_lifecard.edf")):
        rid = edf.name.replace("_lifecard.edf", "")
        if (data_dir / f"{rid}_respevt.txt").exists():
            records.append(rid)
    return records


def load_ucddb_signal_labels(
    data_dir: Path,
    record_id: str,
    channel: int,
    include_hypopnea: bool,
    label_overlap_sec: float,
) -> Tuple[np.ndarray, int, np.ndarray]:
    edf_path = data_dir / f"{record_id}_lifecard.edf"
    patch_edf_start_time(edf_path)
    with pyedflib.EdfReader(str(edf_path)) as edf:
        fs = int(round(float(edf.getSampleFrequency(channel))))
        duration_sec = float(edf.file_duration)
        ecg = edf.readSignal(channel).astype(np.float32)
    events = parse_ucddb_events(data_dir / f"{record_id}_respevt.txt", include_hypopnea)
    labels = ucddb_minute_labels(duration_sec, events, label_overlap_sec)
    n_minutes = min(len(labels), len(ecg) // (fs * 60))
    return ecg[: n_minutes * fs * 60], fs, labels[:n_minutes]


def refine_to_local_max(signal: np.ndarray, peaks: np.ndarray, fs: int, radius_sec: float = 0.05) -> np.ndarray:
    radius = max(1, int(round(radius_sec * fs)))
    refined = []
    for peak in peaks:
        start = max(0, int(peak) - radius)
        end = min(len(signal), int(peak) + radius + 1)
        if end <= start:
            continue
        refined.append(start + int(np.argmax(signal[start:end])))
    if not refined:
        return np.empty((0,), dtype=np.int64)

    refined = np.unique(np.asarray(refined, dtype=np.int64))
    min_distance = max(1, int(round(0.25 * fs)))
    kept = [int(refined[0])]
    for peak in refined[1:]:
        if int(peak) - kept[-1] >= min_distance:
            kept.append(int(peak))
        elif signal[int(peak)] > signal[kept[-1]]:
            kept[-1] = int(peak)
    return np.asarray(kept, dtype=np.int64)


def detect_rpeaks_hamilton(signal: np.ndarray, fs: int) -> np.ndarray:
    from biosppy.signals import ecg

    peaks, = ecg.hamilton_segmenter(signal=signal.astype(np.float64), sampling_rate=fs)
    corrected, = ecg.correct_rpeaks(
        signal=signal.astype(np.float64),
        rpeaks=peaks,
        sampling_rate=fs,
        tol=0.05,
    )
    return refine_to_local_max(signal, np.asarray(corrected, dtype=np.int64), fs)


def repair_rr_intervals(rr: np.ndarray) -> np.ndarray:
    rr = np.asarray(rr, dtype=np.float32)
    if len(rr) == 0:
        return rr
    rr_med = median_filter(rr, size=5 if len(rr) >= 5 else 3).astype(np.float32)
    median_rr = float(np.median(rr_med))
    valid = np.isfinite(rr_med) & (rr_med >= 0.30) & (rr_med <= 2.50)
    valid &= np.abs(rr_med - median_rr) <= max(0.50 * median_rr, 0.25)
    if valid.sum() >= 4 and valid.sum() < len(rr_med):
        cs = CubicSpline(np.where(valid)[0], rr_med[valid], extrapolate=True)
        rr_med = cs(np.arange(len(rr_med))).astype(np.float32)
    return np.clip(rr_med, 0.30, 2.50).astype(np.float32)


def make_rr_amp_series(signal: np.ndarray, rpeaks: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rpeaks = np.asarray(rpeaks, dtype=np.int64)
    rpeaks = rpeaks[(rpeaks >= 0) & (rpeaks < len(signal))]
    if len(rpeaks) < 3:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty, empty
    times = rpeaks.astype(np.float32) / float(fs)
    rr = np.diff(times).astype(np.float32)
    rr = repair_rr_intervals(rr)
    amp = signal[rpeaks[1:]].astype(np.float32)
    return times[1:].astype(np.float32), rr.astype(np.float32), amp.astype(np.float32)


def interp_feature(
    times: np.ndarray,
    rr: np.ndarray,
    amp: np.ndarray,
    start_sec: float,
    duration_sec: float,
    target_length: int,
    min_beats: int,
) -> Optional[np.ndarray]:
    end_sec = start_sec + duration_sec
    mask = (times >= start_sec) & (times < end_sec)
    if int(mask.sum()) < min_beats:
        return None

    local_t = (times[mask] - start_sec).astype(np.float32)
    rr_local = rr[mask].astype(np.float32)
    amp_local = amp[mask].astype(np.float32)
    unique_t, unique_idx = np.unique(local_t, return_index=True)
    if len(unique_t) < min_beats:
        return None
    rr_local = rr_local[unique_idx]
    amp_local = amp_local[unique_idx]

    grid = np.linspace(0.0, duration_sec, target_length, endpoint=False, dtype=np.float32)
    if len(unique_t) >= 4:
        rr_interp = CubicSpline(unique_t, rr_local, extrapolate=True)(grid).astype(np.float32)
        amp_interp = CubicSpline(unique_t, amp_local, extrapolate=True)(grid).astype(np.float32)
    else:
        rr_interp = np.interp(grid, unique_t, rr_local).astype(np.float32)
        amp_interp = np.interp(grid, unique_t, amp_local).astype(np.float32)

    rr_interp = zscore(np.clip(rr_interp, 0.30, 2.50))
    amp_interp = zscore(amp_interp)
    return np.stack([rr_interp, amp_interp], axis=1).astype(np.float32)


def feature_cache_path(args, dataset: str, record_id: str) -> Path:
    suffix = f"ctx{args.context_minutes}_len{args.target_length}_minbeats{args.min_beats}"
    suffix += f"_edge{args.edge_policy}"
    if dataset == "ucddb":
        hyp = "hyp" if args.include_hypopnea else "apneaonly"
        overlap = str(args.label_overlap_sec).replace(".", "p")
        suffix += f"_ch{args.ucddb_channel}_{hyp}_overlap{overlap}"
    if dataset == "apnea_ecg":
        suffix += f"_rpeaks{args.apnea_rpeak_source}"
    return Path(args.cache_dir) / f"{dataset}_{record_id}_{suffix}.npz"


def load_existing_ucddb_feature_cache(args, record_id: str) -> Optional[FeatureRecord]:
    if args.edge_policy != "skip":
        return None
    cache_dir = Path(args.ucddb_literature_cache_dir) if args.ucddb_literature_cache_dir else None
    if not cache_dir:
        return None
    hyp = "hyp" if args.include_hypopnea else "apneaonly"
    overlap = str(args.label_overlap_sec).replace(".", "p")
    candidates = [
        cache_dir
        / f"{record_id}_ch{args.ucddb_channel}_{hyp}_ctx{args.context_minutes}_len{args.target_length}_overlap{overlap}_biosppyhamilton_absolute.npz"
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        stats_raw = json.loads(str(data["stats_json"]))
        stats = {
            "record_id": record_id,
            "dataset": "ucddb",
            "fs": 100,
            "minutes_total": int(stats_raw.get("minutes_total", 0)),
            "samples": int(stats_raw.get("samples", len(data["labels"]))),
            "positive": int(stats_raw.get("positive", int(data["labels"].sum()))),
            "normal": int(stats_raw.get("normal", int((data["labels"] == 0).sum()))),
            "rpeaks": int(stats_raw.get("rpeaks", 0)),
            "skipped_low_beats": int(stats_raw.get("skipped_low_beats", 0)),
            "feature_shape": list(data["features"].shape[1:]),
            "source_cache": str(path),
        }
        return FeatureRecord(
            record_id=record_id,
            features=data["features"].astype(np.float32, copy=False),
            labels=data["labels"].astype(np.int64, copy=False),
            centers=data["minute_indices"].astype(np.int32, copy=False),
            stats=stats,
        )
    return None


def relabel_ucddb_from_hyp_cache(args, record_id: str) -> Optional[FeatureRecord]:
    if args.include_hypopnea:
        return None

    overlap = str(args.label_overlap_sec).replace(".", "p")
    source_path = (
        Path(args.cache_dir)
        / f"ucddb_{record_id}_ctx{args.context_minutes}_len{args.target_length}_"
        f"minbeats{args.min_beats}_edge{args.edge_policy}_ch{args.ucddb_channel}_hyp_overlap{overlap}.npz"
    )
    if not source_path.exists():
        return None

    data = np.load(source_path, allow_pickle=False)
    stats = json.loads(str(data["stats_json"]))
    centers = data["centers"].astype(np.int32, copy=False)
    minutes_total = int(stats.get("minutes_total", int(centers.max()) + 1 if len(centers) else 0))
    events = parse_ucddb_events(Path(args.ucddb_dir) / f"{record_id}_respevt.txt", include_hypopnea=False)
    labels_by_minute = ucddb_minute_labels(minutes_total * 60.0, events, args.label_overlap_sec)
    labels = labels_by_minute[centers].astype(np.int64, copy=False)

    stats.update(
        {
            "record_id": record_id,
            "dataset": "ucddb",
            "samples": int(len(labels)),
            "positive": int(labels.sum()),
            "normal": int((labels == 0).sum()),
            "source_cache": str(source_path),
            "label_source": "relabelled apnea-only from hyp cache",
        }
    )
    return FeatureRecord(
        record_id=record_id,
        features=data["features"].astype(np.float32, copy=False),
        labels=labels,
        centers=centers,
        stats=stats,
    )


def save_feature_record(path: Path, record: FeatureRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=record.features,
        labels=record.labels,
        centers=record.centers,
        stats_json=json.dumps(record.stats, ensure_ascii=False),
    )


def load_feature_record(path: Path) -> FeatureRecord:
    data = np.load(path, allow_pickle=False)
    stats = json.loads(str(data["stats_json"]))
    return FeatureRecord(
        record_id=stats["record_id"],
        features=data["features"].astype(np.float32, copy=False),
        labels=data["labels"].astype(np.int64, copy=False),
        centers=data["centers"].astype(np.int32, copy=False),
        stats=stats,
    )


def build_feature_record(args, dataset: str, record_id: str) -> FeatureRecord:
    if dataset == "ucddb":
        cached = load_existing_ucddb_feature_cache(args, record_id)
        if cached is not None:
            return cached

    if dataset == "apnea_ecg":
        signal, fs, labels = load_apnea_signal_labels(Path(args.apnea_dir), record_id)
    elif dataset == "ucddb":
        signal, fs, labels = load_ucddb_signal_labels(
            Path(args.ucddb_dir),
            record_id,
            args.ucddb_channel,
            args.include_hypopnea,
            args.label_overlap_sec,
        )
    else:
        raise ValueError(dataset)

    rpeaks = None
    if dataset == "apnea_ecg" and args.apnea_rpeak_source == "qrs":
        rpeaks = load_apnea_qrs_rpeaks(Path(args.apnea_dir), record_id)
    if rpeaks is None:
        rpeaks = detect_rpeaks_hamilton(signal, fs)
    times, rr, amp = make_rr_amp_series(signal, rpeaks, fs)

    half = args.context_minutes // 2
    duration_sec = args.context_minutes * 60.0
    signal_duration_sec = len(signal) / float(fs)
    max_start_sec = max(0.0, signal_duration_sec - duration_sec)
    features = []
    y = []
    centers = []
    skipped = 0
    if args.edge_policy == "skip":
        centers_iter = range(half, len(labels) - half)
    elif args.edge_policy == "clamp":
        centers_iter = range(0, len(labels))
    else:
        raise ValueError(f"Unknown edge policy: {args.edge_policy}")

    for center in centers_iter:
        desired_start = (center - half) * 60.0
        start = min(max(desired_start, 0.0), max_start_sec)
        feat = interp_feature(times, rr, amp, start, duration_sec, args.target_length, args.min_beats)
        if feat is None:
            skipped += 1
            continue
        features.append(feat)
        y.append(int(labels[center]))
        centers.append(int(center))

    if features:
        x_arr = np.stack(features).astype(np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        centers_arr = np.asarray(centers, dtype=np.int32)
    else:
        x_arr = np.empty((0, args.target_length, 2), dtype=np.float32)
        y_arr = np.empty((0,), dtype=np.int64)
        centers_arr = np.empty((0,), dtype=np.int32)

    stats = {
        "record_id": record_id,
        "dataset": dataset,
        "fs": int(fs),
        "minutes_total": int(len(labels)),
        "samples": int(len(y_arr)),
        "positive": int(y_arr.sum()) if len(y_arr) else 0,
        "normal": int((y_arr == 0).sum()) if len(y_arr) else 0,
        "rpeaks": int(len(rpeaks)),
        "skipped_low_beats": int(skipped),
        "feature_shape": list(x_arr.shape[1:]),
    }
    return FeatureRecord(record_id, x_arr, y_arr, centers_arr, stats)


def get_feature_record(args, dataset: str, record_id: str) -> FeatureRecord:
    path = feature_cache_path(args, dataset, record_id)
    if path.exists() and not args.rebuild_cache:
        return load_feature_record(path)
    if dataset == "ucddb" and not args.rebuild_cache:
        record = relabel_ucddb_from_hyp_cache(args, record_id)
        if record is not None:
            save_feature_record(path, record)
            return record
    record = build_feature_record(args, dataset, record_id)
    save_feature_record(path, record)
    return record


def _feature_worker(args_dict: dict, dataset: str, record_id: str) -> FeatureRecord:
    return get_feature_record(Namespace(**args_dict), dataset, record_id)


def get_feature_records(args, dataset: str, record_ids: List[str]) -> List[FeatureRecord]:
    if args.num_workers <= 1 or len(record_ids) <= 1:
        return [get_feature_record(args, dataset, r) for r in record_ids]

    args_dict = vars(args).copy()
    results: Dict[str, FeatureRecord] = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_feature_worker, args_dict, dataset, rid): rid for rid in record_ids}
        for future in as_completed(futures):
            rid = futures[future]
            results[rid] = future.result()
            stats = results[rid].stats
            print(
                f"cached {dataset}:{rid} samples={stats['samples']} "
                f"pos={stats['positive']} skipped={stats['skipped_low_beats']}",
                flush=True,
            )
    return [results[rid] for rid in record_ids]


def concat_records(records: List[FeatureRecord]) -> Tuple[np.ndarray, np.ndarray]:
    valid = [r for r in records if len(r.labels)]
    if not valid:
        raise RuntimeError("No valid feature records")
    return np.concatenate([r.features for r in valid], axis=0), np.concatenate([r.labels for r in valid], axis=0)


def summarize_labels(y: np.ndarray) -> dict:
    return {
        "total": int(len(y)),
        "positive": int(np.sum(y == 1)),
        "normal": int(np.sum(y == 0)),
        "positive_ratio": float(np.mean(y == 1)) if len(y) else 0.0,
    }


def make_apnea_split(args) -> SplitData:
    release = APNEA_RELEASE_RECORDS[:]
    withheld = APNEA_WITHHELD_RECORDS[:]
    if args.max_records:
        release = release[: args.max_records]
        withheld = withheld[: args.max_records]

    train_records, val_records = train_test_split(
        release,
        test_size=0.20,
        random_state=args.seed,
        shuffle=True,
    )
    all_needed = list(dict.fromkeys(train_records + val_records + withheld))
    records_by_id = {r.record_id: r for r in get_feature_records(args, "apnea_ecg", all_needed)}
    train_recs = [records_by_id[r] for r in train_records]
    val_recs = [records_by_id[r] for r in val_records]
    test_recs = [records_by_id[r] for r in withheld]

    x_train, y_train = concat_records(train_recs)
    x_val, y_val = concat_records(val_recs)
    x_test, y_test = concat_records(test_recs)
    stats = {
        "paper_protocol": "Apnea-ECG release records train/val; withheld x01-x35 test; validation is 20% of release records.",
        "train": summarize_labels(y_train),
        "val": summarize_labels(y_val),
        "test": summarize_labels(y_test),
        "records": {
            "train": [r.stats for r in train_recs],
            "val": [r.stats for r in val_recs],
            "test": [r.stats for r in test_recs],
        },
    }
    return SplitData(x_train, y_train, x_val, y_val, x_test, y_test, train_records, val_records, withheld, stats)


def make_ucddb_split(args) -> SplitData:
    records = [r for r in available_ucddb_records(Path(args.ucddb_dir)) if r not in UCDDB_EXCLUDED_NO_SA]
    if args.max_records:
        records = records[: args.max_records]
    recs = get_feature_records(args, "ucddb", records)
    x_all, y_all = concat_records(recs)
    indices = np.arange(len(y_all))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=args.seed,
        shuffle=True,
        stratify=y_all if len(np.unique(y_all)) == 2 else None,
    )
    y_train_val = y_all[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=1 / 9,
        random_state=args.seed,
        shuffle=True,
        stratify=y_train_val if len(np.unique(y_train_val)) == 2 else None,
    )

    stats = {
        "paper_protocol": "UCDDB excludes ucddb008/011/013/018, then segment-level 8:1:1 train/val/test with minority oversampling on train.",
        "all": summarize_labels(y_all),
        "train": summarize_labels(y_all[train_idx]),
        "val": summarize_labels(y_all[val_idx]),
        "test": summarize_labels(y_all[test_idx]),
        "records": [r.stats for r in recs],
        "excluded_records": sorted(UCDDB_EXCLUDED_NO_SA),
    }
    return SplitData(
        x_all[train_idx],
        y_all[train_idx],
        x_all[val_idx],
        y_all[val_idx],
        x_all[test_idx],
        y_all[test_idx],
        records,
        records,
        records,
        stats,
    )


class FeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class SpatioTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, filters: int = 128, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout(dropout)
        self.bigru = nn.GRU(filters, hidden, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)
        x, _ = self.bigru(x)
        x = self.dropout2(x)
        return x.transpose(1, 2)


class DotProductBiGRUAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        half = x.size(-1) // 2
        forward = x[:, :, :half]
        backward = x[:, :, half:]
        scores = torch.bmm(forward, backward.transpose(1, 2)) / math.sqrt(max(half, 1))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.bmm(weights, backward)
        return attended.reshape(attended.size(0), -1)


class FninsCNNBiGRU(nn.Module):
    def __init__(self, input_length: int = 900, input_channels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.initial = nn.Conv1d(input_channels, 128, kernel_size=3, padding=1)
        self.block1 = SpatioTemporalBlock(128, dropout=dropout)
        self.block2 = SpatioTemporalBlock(256, dropout=dropout)
        self.block3 = SpatioTemporalBlock(256, dropout=dropout)
        self.attention = DotProductBiGRUAttention()
        reduced_length = input_length
        for _ in range(3):
            reduced_length = reduced_length // 3
        dense_in = reduced_length * 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.relu(self.initial(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.attention(x)
        return self.classifier(x)


class CNNTransformer(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_ch = input_channels
        for _ in range(3):
            layers.extend(
                [
                    nn.Conv1d(in_ch, d_model, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=3),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = d_model
        self.cnn = nn.Sequential(*layers)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.cnn(x).transpose(1, 2)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def build_model(model_type: str, input_length: int) -> nn.Module:
    if model_type == "cnn_bigru":
        return FninsCNNBiGRU(input_length=input_length)
    if model_type == "cnn_transformer":
        return CNNTransformer()
    raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, oversample: bool, samples_per_epoch: Optional[int]):
    dataset = FeatureDataset(x, y)
    if oversample:
        counts = np.bincount(y, minlength=2)
        sample_weights = len(y) / (2.0 * np.maximum(counts, 1))
        weights = sample_weights[y]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(samples_per_epoch or len(weights)),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels = []
    preds = []
    scores = []
    for batch_x, batch_y in loader:
        logits = model(batch_x.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels.extend(batch_y.numpy().tolist())
        preds.extend((prob >= 0.5).astype(np.int64).tolist())
        scores.extend(prob.tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(preds, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch_x.size(0)
        total += batch_x.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += float(loss.item()) * batch_x.size(0)
        total += batch_x.size(0)
    return total_loss / max(total, 1)


def load_pretrained(model: nn.Module, path: Path, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


def run_training(args) -> dict:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if args.dataset == "apnea_ecg":
        split = make_apnea_split(args)
        oversample = False
    elif args.dataset == "ucddb":
        split = make_ucddb_split(args)
        oversample = True
    else:
        raise ValueError(args.dataset)

    train_loader = make_loader(
        split.x_train,
        split.y_train,
        args.batch_size,
        shuffle=True,
        oversample=oversample,
        samples_per_epoch=args.samples_per_epoch,
    )
    val_loader = make_loader(split.x_val, split.y_val, args.batch_size, shuffle=False, oversample=False, samples_per_epoch=None)
    test_loader = make_loader(split.x_test, split.y_test, args.batch_size, shuffle=False, oversample=False, samples_per_epoch=None)

    model = build_model(args.model_type, args.target_length).to(device)
    if args.pretrained_path:
        load_pretrained(model, Path(args.pretrained_path), device)
        print(f"Loaded pretrained weights: {args.pretrained_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt_path = output_dir / f"{args.experiment_name}.pth"

    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Dataset: {args.dataset} | Model: {args.model_type} | Device: {device}")
    print(f"Params: {count_parameters(model):,} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"Train: {summarize_labels(split.y_train)}")
    print(f"Val:   {summarize_labels(split.y_val)}")
    print(f"Test:  {summarize_labels(split.y_test)}")
    print("=" * 80)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_y, val_pred, val_score = predict(model, val_loader, device)
        val_metrics = compute_metrics(val_y, val_pred, val_score)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "settings": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )
        if epoch <= 3 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} val_rec={val_metrics['recall']:.4f} "
                f"val_spe={val_metrics['specificity']:.4f} val_f1={val_metrics['f1']:.4f} "
                f"val_auc={val_metrics['auc']:.4f}"
            )

    elapsed = time.time() - start_time
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_y, test_pred, test_score = predict(model, test_loader, device)
    test_metrics = compute_metrics(test_y, test_pred, test_score)

    result = {
        "experiment_name": args.experiment_name,
        "dataset": args.dataset,
        "model_type": args.model_type,
        "paper": "Chen et al. 2022 Frontiers in Neuroscience 16:972581",
        "settings": vars(args),
        "n_parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "training_time_sec": float(elapsed),
        "split_stats": split.stats,
        "history": history,
        "test_metrics": test_metrics,
        "model_path": str(ckpt_path),
    }
    result_path = output_dir / f"{args.experiment_name}_results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print(f"Best epoch: {best_epoch} | time: {elapsed:.1f}s")
    print(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"Saved: {result_path}")
    print("=" * 80)
    return result


def inspect_dataset(args) -> None:
    if args.dataset == "apnea_ecg":
        split = make_apnea_split(args)
    elif args.dataset == "ucddb":
        split = make_ucddb_split(args)
    else:
        raise ValueError(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{args.dataset}_inspect.json"
    path.write_text(json.dumps(split.stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(split.stats, indent=2, ensure_ascii=False))
    print(f"Saved inspect report: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FNINS 16:972581 baseline and CNN+Transformer experiments.")
    parser.add_argument("--dataset", choices=["apnea_ecg", "ucddb"], required=True)
    parser.add_argument("--model-type", choices=["cnn_bigru", "cnn_transformer"], default="cnn_bigru")
    parser.add_argument("--mode", choices=["train", "inspect"], default="train")
    parser.add_argument("--apnea-dir", default="apnea-ecg")
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--ucddb-channel", type=int, default=0)
    parser.add_argument("--ucddb-literature-cache-dir", default="aligned_data/ucddb_literature_features")
    parser.add_argument("--apnea-rpeak-source", choices=["hamilton", "qrs"], default="hamilton")
    parser.add_argument("--include-hypopnea", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label-overlap-sec", type=float, default=5.0)
    parser.add_argument("--context-minutes", type=int, default=5)
    parser.add_argument("--target-length", type=int, default=900)
    parser.add_argument("--min-beats", type=int, default=4)
    parser.add_argument("--edge-policy", choices=["clamp", "skip"], default="clamp")
    parser.add_argument("--cache-dir", default="work_fnins_baseline/cache")
    parser.add_argument("--output-dir", default="work_fnins_baseline/outputs")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.context_minutes % 2 != 1:
        raise ValueError("--context-minutes must be odd so the center minute is well-defined")
    if args.experiment_name is None:
        args.experiment_name = f"fnins_{args.model_type}_{args.dataset}"

    if args.mode == "inspect":
        inspect_dataset(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
