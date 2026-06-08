"""
資料預處理模組

負責 UCDDB 和 Apnea-ECG 資料集的載入、R peak 偵測、RRI 特徵擷取。

兩種輸入模式:
  1. Raw ECG: 帶通濾波 + Z-score 正規化 → (N, 6000) 每分鐘
  2. RRI Features: R peak 偵測 → RRI + R amplitude → 重取樣至固定長度 → (N, 2, L)

參考論文 Section 3.2:
  - Hamilton 演算法偵測 R 波
  - RRI 計算 + 中位數濾波
  - 三次插值修正假 R peak
  - FIR 帶通濾波 + Z-score 正規化
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyedflib
import wfdb
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks, resample, resample_poly

# ============================================================
# 常數
# ============================================================
FS_APNEA_ECG = 100    # Apnea-ECG 採樣率
FS_UCDDB = 128        # UCDDB Lifecard ECG 採樣率
FS_TARGET = 100        # 統一目標採樣率
MINUTE_SAMPLES = FS_TARGET * 60  # 每分鐘取樣點數 = 6000
RRI_RESAMPLE_LEN = 120  # RRI 特徵重取樣長度


# ============================================================
# 帶通濾波器
# ============================================================
def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                           fs: int, order: int = 3) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data).astype(np.float32)


def normalize_zscore(x: np.ndarray) -> np.ndarray:
    """Z-score 正規化。"""
    return ((x - np.mean(x)) / (np.std(x) + 1e-8)).astype(np.float32)


# ============================================================
# R Peak 偵測 (Pan-Tompkins 風格)
# ============================================================
def detect_r_peaks(ecg: np.ndarray, fs: int = 100, min_distance_sec: float = 0.3) -> np.ndarray:
    """
    偵測 ECG 信號中的 R 波位置。

    使用帶通濾波 (5-15Hz) + 微分 + 平方 + 移動平均的 Pan-Tompkins 風格方法。

    Args:
        ecg: 原始 ECG 信號
        fs: 採樣率
        min_distance_sec: R 波最小間距（秒）

    Returns:
        R 波位置索引陣列
    """
    # 帶通濾波 5-15Hz (隔離 QRS 複合波)
    filtered = butter_bandpass_filter(ecg, 5.0, 15.0, fs, order=2)

    # 微分
    diff = np.diff(filtered)

    # 平方
    squared = diff ** 2

    # 移動平均 (窗口 ~150ms)
    window_size = max(1, int(0.15 * fs))
    kernel = np.ones(window_size) / window_size
    integrated = np.convolve(squared, kernel, mode="same")

    # 尋找峰值
    min_distance = max(1, int(min_distance_sec * fs))
    threshold = np.mean(integrated) + 0.3 * np.std(integrated)

    peaks, properties = find_peaks(
        integrated,
        distance=min_distance,
        height=threshold * 0.5,
    )

    # 在原始信號中精確定位 R 波（在每個偵測到的峰值附近搜索）
    refined_peaks = []
    search_window = int(0.05 * fs)  # ±50ms
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(ecg), peak + search_window + 1)
        local_peak = start + np.argmax(np.abs(ecg[start:end]))
        refined_peaks.append(local_peak)

    return np.array(refined_peaks, dtype=np.int64)


# ============================================================
# RRI 特徵擷取 (論文 Section 3.2)
# ============================================================
def extract_rri_features(
    ecg: np.ndarray,
    fs: int = 100,
    target_len: int = RRI_RESAMPLE_LEN,
    median_kernel: int = 5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    從 ECG 信號擷取 RRI 和 R-peak amplitude 特徵。

    步驟 (對應論文 Section 3.2):
      1. 偵測 R 波
      2. 計算 RRI (R-R Interval)
      3. 中位數濾波去除異常值
      4. 擷取 R 波幅值
      5. 重取樣至固定長度

    Args:
        ecg: ECG 信號
        fs: 採樣率
        target_len: 輸出特徵長度
        median_kernel: 中位數濾波器核大小

    Returns:
        (rri_features, r_amp_features) 各 shape (target_len,)
        若 R 波數量不足則返回 (None, None)
    """
    # 偵測 R 波
    r_peaks = detect_r_peaks(ecg, fs)

    if len(r_peaks) < 4:
        return None, None

    # 計算 RRI (以秒為單位)
    rri = np.diff(r_peaks).astype(np.float32) / fs

    # 中位數濾波去除異常值 (論文建議)
    if len(rri) >= median_kernel:
        rri_filtered = median_filter(rri, size=median_kernel).astype(np.float32)
    else:
        rri_filtered = rri.copy()

    # 三次插值修正 — 偵測並修正假 R peak
    # 偵測異常 RRI (與中位值偏差過大)
    median_rri = np.median(rri_filtered)
    anomaly_mask = np.abs(rri_filtered - median_rri) > 0.5 * median_rri
    if np.sum(~anomaly_mask) >= 2:
        valid_indices = np.where(~anomaly_mask)[0]
        valid_rri = rri_filtered[valid_indices]
        cs = CubicSpline(valid_indices, valid_rri)
        rri_corrected = cs(np.arange(len(rri_filtered))).astype(np.float32)
    else:
        rri_corrected = rri_filtered

    # R 波幅值
    r_amplitudes = ecg[r_peaks].astype(np.float32)
    # 移除第一個 (與 RRI 對齊)
    r_amplitudes = r_amplitudes[1:]
    # 確保長度一致
    min_len = min(len(rri_corrected), len(r_amplitudes))
    rri_corrected = rri_corrected[:min_len]
    r_amplitudes = r_amplitudes[:min_len]

    if min_len < 2:
        return None, None

    # 重取樣至固定長度
    rri_resampled = resample(rri_corrected, target_len).astype(np.float32)
    amp_resampled = resample(r_amplitudes, target_len).astype(np.float32)

    # Z-score 正規化
    rri_resampled = normalize_zscore(rri_resampled)
    amp_resampled = normalize_zscore(amp_resampled)

    return rri_resampled, amp_resampled


# ============================================================
# UCDDB 資料載入
# ============================================================
def patch_edf_start_time(edf_path: Path) -> None:
    """UCDDB Lifecard EDF 使用 HH:MM:SS; pyedflib 需要 HH.MM.SS。"""
    with edf_path.open("rb") as f:
        f.seek(176)
        raw = f.read(8)
    fixed = raw.replace(b":", b".")
    if fixed == raw:
        return
    with edf_path.open("r+b") as f:
        f.seek(176)
        f.write(fixed)


def parse_respiratory_events(path: Path, include_hypopnea: bool = True):
    """解析 UCDDB 呼吸事件標註檔。"""
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


def labels_for_minutes(duration_sec: float, events, min_overlap_sec: float = 1.0):
    """為每分鐘 segment 產生標籤。"""
    n_minutes = int(duration_sec // 60)
    y = np.zeros(n_minutes, dtype=np.int64)
    for minute in range(n_minutes):
        start = minute * 60
        end = start + 60
        for event_start, event_end, _ in events:
            overlap = min(end, event_end) - max(start, event_start)
            if overlap >= min_overlap_sec:
                y[minute] = 1
                break
    return y


def available_ucddb_records(data_dir: Path) -> List[str]:
    """列出可用的 UCDDB 紀錄 ID。"""
    ids = []
    for edf in sorted(data_dir.glob("ucddb*_lifecard.edf")):
        record_id = edf.name.replace("_lifecard.edf", "")
        if (data_dir / f"{record_id}_respevt.txt").exists():
            ids.append(record_id)
    return ids


def load_ucddb_raw(
    data_dir: Path,
    record_id: str,
    channel: int = 0,
    include_hypopnea: bool = True,
    min_overlap_sec: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    載入 UCDDB 紀錄的原始 ECG segments。

    Returns:
        x: (n_minutes, 6000) 原始 ECG segments
        y: (n_minutes,) 標籤 (0=normal, 1=apnea)
        record_id: 紀錄 ID
    """
    edf_path = data_dir / f"{record_id}_lifecard.edf"
    event_path = data_dir / f"{record_id}_respevt.txt"

    patch_edf_start_time(edf_path)

    with pyedflib.EdfReader(str(edf_path)) as edf:
        fs = float(edf.getSampleFrequency(channel))
        duration_sec = float(edf.file_duration)
        signal = edf.readSignal(channel)

    # 重取樣至 100Hz
    if fs != FS_TARGET:
        signal = resample_poly(signal, FS_TARGET, int(fs)).astype(np.float32)
    else:
        signal = signal.astype(np.float32)

    # 解析事件標註
    events = parse_respiratory_events(event_path, include_hypopnea)
    y = labels_for_minutes(duration_sec, events, min_overlap_sec)

    # 切分為每分鐘 segments
    usable_minutes = min(len(y), len(signal) // MINUTE_SAMPLES)
    if usable_minutes == 0:
        return np.empty((0, MINUTE_SAMPLES), dtype=np.float32), np.empty((0,), dtype=np.int64), record_id

    signal = signal[: usable_minutes * MINUTE_SAMPLES]
    y = y[:usable_minutes]
    x = signal.reshape(usable_minutes, MINUTE_SAMPLES)

    # 帶通濾波 + Z-score
    processed = []
    for segment in x:
        segment = butter_bandpass_filter(segment, 0.5, 45.0, FS_TARGET, order=3)
        segment = normalize_zscore(segment)
        processed.append(segment)

    return np.stack(processed), y, record_id


def load_ucddb_rri(
    data_dir: Path,
    record_id: str,
    channel: int = 0,
    include_hypopnea: bool = True,
    min_overlap_sec: float = 1.0,
    rri_len: int = RRI_RESAMPLE_LEN,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    載入 UCDDB 紀錄的 RRI 特徵 segments。

    Returns:
        x: (n_valid, 2, rri_len) — 通道 0: RRI, 通道 1: R amplitude
        y: (n_valid,) 標籤
        record_id: 紀錄 ID
    """
    edf_path = data_dir / f"{record_id}_lifecard.edf"
    event_path = data_dir / f"{record_id}_respevt.txt"

    patch_edf_start_time(edf_path)

    with pyedflib.EdfReader(str(edf_path)) as edf:
        fs = float(edf.getSampleFrequency(channel))
        duration_sec = float(edf.file_duration)
        signal = edf.readSignal(channel)

    if fs != FS_TARGET:
        signal = resample_poly(signal, FS_TARGET, int(fs)).astype(np.float32)
    else:
        signal = signal.astype(np.float32)

    events = parse_respiratory_events(event_path, include_hypopnea)
    y_all = labels_for_minutes(duration_sec, events, min_overlap_sec)

    usable_minutes = min(len(y_all), len(signal) // MINUTE_SAMPLES)
    if usable_minutes == 0:
        return (
            np.empty((0, 2, rri_len), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            record_id,
        )

    xs = []
    ys = []
    for i in range(usable_minutes):
        segment = signal[i * MINUTE_SAMPLES : (i + 1) * MINUTE_SAMPLES]
        segment_filtered = butter_bandpass_filter(segment, 0.5, 45.0, FS_TARGET, order=3)
        rri, amp = extract_rri_features(segment_filtered, FS_TARGET, rri_len)
        if rri is not None:
            feature = np.stack([rri, amp], axis=0)  # (2, rri_len)
            xs.append(feature)
            ys.append(y_all[i])

    if not xs:
        return (
            np.empty((0, 2, rri_len), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            record_id,
        )

    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64), record_id


# ============================================================
# Apnea-ECG 資料載入
# ============================================================
def available_apnea_ecg_records(data_dir: Path) -> List[str]:
    """列出有標註的 Apnea-ECG 紀錄。"""
    list_path = data_dir / "list"
    records = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    return [r for r in records if (data_dir / f"{r}.apn").exists()]


def load_apnea_ecg_raw(
    data_dir: Path,
    record_id: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    載入 Apnea-ECG 紀錄的原始 ECG segments。

    Returns:
        x: (n_segments, 6000)
        y: (n_segments,) 標籤
        record_id: 紀錄 ID
    """
    record_path = str(data_dir / record_id)
    signal, fields = wfdb.rdsamp(record_path)
    annotation = wfdb.rdann(record_path, "apn")

    ecg = signal[:, 0].astype(np.float32)
    xs = []
    ys = []
    for sample, label in zip(annotation.sample, annotation.symbol):
        if label not in ("A", "N"):
            continue
        start = int(sample)
        end = start + MINUTE_SAMPLES
        if start < 0 or end > len(ecg):
            continue
        segment = ecg[start:end]
        segment = butter_bandpass_filter(segment, 0.5, 45.0, FS_APNEA_ECG, order=3)
        segment = normalize_zscore(segment)
        xs.append(segment)
        ys.append(1 if label == "A" else 0)

    if not xs:
        return np.empty((0, MINUTE_SAMPLES), dtype=np.float32), np.empty((0,), dtype=np.int64), record_id

    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64), record_id


def load_apnea_ecg_rri(
    data_dir: Path,
    record_id: str,
    rri_len: int = RRI_RESAMPLE_LEN,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    載入 Apnea-ECG 紀錄的 RRI 特徵 segments。
    """
    record_path = str(data_dir / record_id)
    signal, fields = wfdb.rdsamp(record_path)
    annotation = wfdb.rdann(record_path, "apn")

    ecg = signal[:, 0].astype(np.float32)
    xs = []
    ys = []
    for sample, label in zip(annotation.sample, annotation.symbol):
        if label not in ("A", "N"):
            continue
        start = int(sample)
        end = start + MINUTE_SAMPLES
        if start < 0 or end > len(ecg):
            continue
        segment = ecg[start:end]
        segment = butter_bandpass_filter(segment, 0.5, 45.0, FS_APNEA_ECG, order=3)
        rri, amp = extract_rri_features(segment, FS_APNEA_ECG, rri_len)
        if rri is not None:
            feature = np.stack([rri, amp], axis=0)
            xs.append(feature)
            ys.append(1 if label == "A" else 0)

    if not xs:
        return (
            np.empty((0, 2, rri_len), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            record_id,
        )

    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64), record_id


# ============================================================
# 批量載入
# ============================================================
def load_dataset(
    data_dir: Path,
    record_ids: List[str],
    dataset_type: str = "ucddb",
    feature_type: str = "raw",
    channel: int = 0,
    include_hypopnea: bool = True,
    min_overlap_sec: float = 1.0,
    rri_len: int = RRI_RESAMPLE_LEN,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    批量載入資料集。

    Args:
        data_dir: 資料目錄
        record_ids: 紀錄 ID 列表
        dataset_type: 'ucddb' 或 'apnea_ecg'
        feature_type: 'raw' 或 'rri'
        channel: ECG 通道 (僅 UCDDB)
        include_hypopnea: 是否包含低通氣事件
        min_overlap_sec: 最小重疊秒數
        rri_len: RRI 特徵長度
        verbose: 是否顯示進度

    Returns:
        x: (total_samples, ...) 特徵
        y: (total_samples,) 標籤
        record_stats: 每筆紀錄的統計資訊
    """
    all_x = []
    all_y = []
    record_stats = []

    for rid in record_ids:
        try:
            if dataset_type == "ucddb":
                if feature_type == "raw":
                    x, y, _ = load_ucddb_raw(data_dir, rid, channel, include_hypopnea, min_overlap_sec)
                else:
                    x, y, _ = load_ucddb_rri(data_dir, rid, channel, include_hypopnea, min_overlap_sec, rri_len)
            else:
                if feature_type == "raw":
                    x, y, _ = load_apnea_ecg_raw(data_dir, rid)
                else:
                    x, y, _ = load_apnea_ecg_rri(data_dir, rid, rri_len)

            if len(y) == 0:
                if verbose:
                    print(f"  ⚠️ {rid}: 無可用 segments")
                continue

            all_x.append(x)
            all_y.append(y)
            record_stats.append({
                "record_id": rid,
                "total_segments": int(len(y)),
                "apnea_segments": int(np.sum(y == 1)),
                "normal_segments": int(np.sum(y == 0)),
                "apnea_ratio": float(np.mean(y == 1)),
            })
            if verbose:
                stats = record_stats[-1]
                print(f"  ✓ {rid}: {stats['total_segments']} segs "
                      f"(A={stats['apnea_segments']}, N={stats['normal_segments']})")

        except Exception as e:
            if verbose:
                print(f"  ✗ {rid}: 錯誤 — {e}")

    if not all_x:
        raise RuntimeError(f"No valid samples loaded from {dataset_type} dataset.")

    return np.concatenate(all_x), np.concatenate(all_y), record_stats


if __name__ == "__main__":
    # 快速測試 R peak 偵測
    print("=" * 60)
    print("R Peak 偵測測試")
    print("=" * 60)

    # 產生模擬 ECG 信號
    fs = 100
    t = np.arange(0, 10, 1 / fs)
    ecg_sim = np.sin(2 * np.pi * 1.2 * t)  # 模擬 ~72bpm
    ecg_sim += 0.5 * np.random.randn(len(t))

    peaks = detect_r_peaks(ecg_sim, fs)
    print(f"偵測到 {len(peaks)} 個 R 波 (預期 ~12)")

    # 測試 RRI 特徵擷取
    rri, amp = extract_rri_features(ecg_sim, fs, target_len=120)
    if rri is not None:
        print(f"RRI features shape: {rri.shape}, R-amp shape: {amp.shape}")
    else:
        print("R 波數量不足，無法擷取 RRI 特徵")

    print("\n✅ 預處理模組測試完成！")
