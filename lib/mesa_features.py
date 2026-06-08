"""MESA(NSRR)讀取器 —— 產出與 UCDDB 完全同形狀的特徵,供跨資料集訓練。

只換兩塊:
  1. read_mesa_ecg : 從 MESA EDF 抽 EKG channel(256Hz)→ 100Hz → bandpass 0.5–45。
                     (對齊 ucddb_highres_trainer.read_ucddb_signal)
  2. parse_mesa_events : 解析 NSRR XML 的 ScoredEvent,取 apnea/hypopnea → (start, end, concept)。
                     (對齊 ucddb_runner.parse_respiratory_events 的回傳格式)

其餘特徵化(Hamilton R-peak、5 分鐘 context、900 點 RRI/振幅內插、逐窗 z-score、逐分鐘標籤)
全部沿用 ucddb_literature_features,確保 train(MESA)/ test(UCDDB)以相同方式建構特徵。
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyedflib
from scipy.signal import resample_poly
from tqdm import tqdm

import apnea_trainer
import ucddb_literature_features as litfeat

FS = litfeat.FS  # 100
_ECG_LABELS = ("ekg", "ecg", "ecg1", "ecgl", "ekg1")


def find_ecg_channel(edf):
    labels = [str(lab).strip().lower() for lab in edf.getSignalLabels()]
    for i, lab in enumerate(labels):
        if lab in _ECG_LABELS:
            return i
    for i, lab in enumerate(labels):  # 容錯:含 ekg/ecg 但非 *_off 旁路
        if ("ekg" in lab or "ecg" in lab) and not lab.endswith("_off"):
            return i
    raise RuntimeError(f"找不到 ECG/EKG channel,labels={edf.getSignalLabels()}")


def read_mesa_ecg(edf_path):
    with pyedflib.EdfReader(str(edf_path)) as edf:
        ch = find_ecg_channel(edf)
        fs = float(edf.getSampleFrequency(ch))
        duration_sec = float(edf.file_duration)
        signal = edf.readSignal(ch).astype(np.float32)
    if int(round(fs)) != FS:
        signal = resample_poly(signal, FS, int(round(fs))).astype(np.float32)
    usable = int(duration_sec) * FS
    signal = signal[:usable]
    signal = apnea_trainer.butter_bandpass_filter(signal, 0.5, 45.0, FS, order=3).astype(np.float32)
    return signal


def parse_mesa_events(xml_path, include_hypopnea=True):
    events = []
    root = ET.parse(str(xml_path)).getroot()
    for se in root.findall(".//ScoredEvent"):
        concept = (se.findtext("EventConcept") or "").lower()
        is_apnea = "apnea" in concept
        is_hyp = "hypopnea" in concept
        if not is_apnea and not (include_hypopnea and is_hyp):
            continue
        try:
            start = float(se.findtext("Start"))
            dur = float(se.findtext("Duration"))
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        events.append((start, start + dur, concept))
    return events


def locate_files(mesa_dir, record_id):
    mesa_dir = Path(mesa_dir)
    edf = next(iter(mesa_dir.rglob(f"{record_id}.edf")), None)
    xml = next(iter(mesa_dir.rglob(f"{record_id}-nsrr.xml")), None)
    return edf, xml


def available_mesa_records(mesa_dir):
    mesa_dir = Path(mesa_dir)
    ids = []
    for edf in sorted(mesa_dir.rglob("mesa-sleep-*.edf")):
        rid = edf.stem
        if next(iter(mesa_dir.rglob(f"{rid}-nsrr.xml")), None) is not None:
            ids.append(rid)
    return ids


def _cache_path(args, record_id):
    cache_dir = Path(args.mesa_cache_dir)
    hyp = "apneaonly" if args.apnea_only else "hyp"
    overlap = str(args.label_overlap_sec).replace(".", "p")
    return cache_dir / (
        f"{record_id}_ekg_{hyp}_ctx{args.context_minutes}_len{args.target_length}_overlap{overlap}.npz"
    )


def build_mesa_record(args, record_id, mesa_dir):
    edf_path, xml_path = locate_files(mesa_dir, record_id)
    if edf_path is None or xml_path is None:
        raise FileNotFoundError(f"{record_id}: 缺 edf 或 xml")
    signal = read_mesa_ecg(edf_path)
    duration_sec = len(signal) // FS
    rpeaks, detector_used = litfeat.detect_rpeaks(signal, FS, "biosppy_hamilton")
    events = parse_mesa_events(xml_path, include_hypopnea=not args.apnea_only)
    return litfeat.build_features_from_events(
        args, record_id, channel=0, signal=signal, rpeaks=rpeaks,
        events=events, duration_sec=duration_sec, detector_used=detector_used,
    )


def load_mesa_record(args, record_id, mesa_dir):
    path = _cache_path(args, record_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not getattr(args, "rebuild_cache", False):
        c = np.load(path, allow_pickle=False)
        stats = json.loads(str(c["stats_json"]))
        return litfeat.LiteratureFeatureRecord(
            record_id=record_id, channel=0,
            features=c["features"].astype(np.float32, copy=False),
            labels=c["labels"].astype(np.int64, copy=False),
            minute_indices=c["minute_indices"].astype(np.int32, copy=False),
            signal=c["signal"].astype(np.float32, copy=False),
            rpeaks=c["rpeaks"].astype(np.int64, copy=False),
            detector_used=stats.get("detector_used", "biosppy_hamilton"), stats=stats,
        )
    record = build_mesa_record(args, record_id, mesa_dir)
    litfeat.save_record_cache(path, record)
    return record


def _load_cached_signal_rpeaks(args, record_id):
    """從既有 per-minute 快取撈 signal+rpeaks(避免重跑 Hamilton)。"""
    cache_dir = Path(args.mesa_cache_dir)
    for p in sorted(cache_dir.glob(f"{record_id}_ekg_*.npz")):
        if "_seg" in p.name:
            continue
        c = np.load(p, allow_pickle=False)
        if "signal" in c and "rpeaks" in c and c["signal"].size > 1:
            return c["signal"].astype(np.float32), c["rpeaks"].astype(np.int64)
    return None, None


def _seg_cache_path(args, record_id):
    hyp = "apneaonly" if args.apnea_only else "hyp"
    nm = getattr(args, "norm_mode", "window")
    return Path(args.mesa_cache_dir) / (
        f"{record_id}_ekg_seg{args.segment_sec}_str{args.segment_stride_sec}_ctx{args.context_minutes}_{hyp}_norm{nm}.npz"
    )


def build_mesa_segment_record(args, record_id, mesa_dir):
    edf_path, xml_path = locate_files(mesa_dir, record_id)
    if xml_path is None:
        raise FileNotFoundError(f"{record_id}: 缺 xml")
    signal, rpeaks = _load_cached_signal_rpeaks(args, record_id)
    if signal is None:
        if edf_path is None:
            raise FileNotFoundError(f"{record_id}: 缺 edf 且無快取")
        signal = read_mesa_ecg(edf_path)
        rpeaks, _ = litfeat.detect_rpeaks(signal, FS, "biosppy_hamilton")
    events = parse_mesa_events(xml_path, include_hypopnea=not args.apnea_only)
    duration_sec = len(signal) // FS
    return litfeat.build_segment_features_from_events(
        args, record_id, channel=0, signal=signal, rpeaks=rpeaks, events=events,
        duration_sec=duration_sec, detector_used="biosppy_hamilton",
        seg_sec=args.segment_sec, stride_sec=args.segment_stride_sec,
        context_sec=args.context_minutes * 60, norm_mode=getattr(args, "norm_mode", "window"),
    )


def load_mesa_segment_record(args, record_id, mesa_dir):
    path = _seg_cache_path(args, record_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not getattr(args, "rebuild_cache", False):
        c = np.load(path, allow_pickle=False)
        stats = json.loads(str(c["stats_json"]))
        return litfeat.LiteratureFeatureRecord(
            record_id=record_id, channel=0,
            features=c["features"].astype(np.float32, copy=False),
            labels=c["labels"].astype(np.int64, copy=False),
            minute_indices=c["minute_indices"].astype(np.int32, copy=False),
            signal=np.zeros(1, dtype=np.float32), rpeaks=np.zeros(1, dtype=np.int64),
            detector_used="biosppy_hamilton", stats=stats,
        )
    rec = build_mesa_segment_record(args, record_id, mesa_dir)
    np.savez_compressed(
        path, features=rec.features, labels=rec.labels, minute_indices=rec.minute_indices,
        signal=np.zeros(1, dtype=np.float32), rpeaks=np.zeros(1, dtype=np.int64),
        stats_json=json.dumps(rec.stats),
    )
    return rec


def load_mesa_segment_records(args, mesa_dir, record_ids=None):
    if record_ids is None:
        record_ids = available_mesa_records(mesa_dir)
    if getattr(args, "mesa_limit", 0):
        record_ids = record_ids[: args.mesa_limit]
    records = []
    for rid in tqdm(record_ids, desc="Loading MESA seg", disable=getattr(args, "no_progress", False)):
        r = load_mesa_segment_record(args, rid, mesa_dir)
        if len(r.labels):
            records.append(r)
    if not records:
        raise RuntimeError("No MESA segment records loaded.")
    return records


def load_mesa_records(args, mesa_dir, record_ids=None):
    if record_ids is None:
        record_ids = available_mesa_records(mesa_dir)
    if getattr(args, "mesa_limit", 0):
        record_ids = record_ids[: args.mesa_limit]
    records = []
    for rid in tqdm(record_ids, desc="Loading MESA", disable=getattr(args, "no_progress", False)):
        rec = load_mesa_record(args, rid, mesa_dir)
        if len(rec.labels):
            records.append(rec)
    if not records:
        raise RuntimeError("No MESA records loaded.")
    return records
