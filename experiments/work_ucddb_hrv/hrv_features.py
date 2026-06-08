"""HRV + CVHR + EDR scalar feature extraction for UCDDB minute-level apnea detection.

Design goals
------------
The existing UCDDB pipelines feed an interpolated (900, 2) RRI/amplitude *sequence*
into a deep model and let it learn everything. Across subjects that plateaus at
AUC ~0.55 because the small UCDDB cohort (~25 subjects) cannot teach a network the
apnea-specific cyclical-variation-of-heart-rate (CVHR) oscillation on its own.

This module instead computes explicit, domain-knowledge scalar features per target
minute (with a multi-minute ECG context):

- Time-domain HRV   : meanRR, SDNN, RMSSD, SDSD, pNN50, CVRR, HR stats, RR spread.
- Poincare nonlinear: SD1, SD2, SD1/SD2, ellipse area.
- Frequency HRV     : VLF / LF / HF / total power, LF/HF, normalized units.
- CVHR / apnea band : power in 0.01-0.04 Hz, its ratio to total, peak freq/power,
                      and an autocorrelation-based CVHR periodicity score in the
                      20-70 s apnea-cycle lag range. This is the single most
                      discriminative apnea signature.
- EDR (respiration) : R-peak amplitude variability and amplitude-spectrum power in
                      the respiration band (0.1-0.4 Hz), which apnea suppresses.

Reuses cached `signal` + `rpeaks` from the literature-feature cache, so no EDF read
or R-peak re-detection is needed.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import welch

# numpy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Allow importing the project-root helper modules when run from anywhere.
_ROOT = Path(__file__).resolve().parents[2]  # apnea project root
for _p in (_ROOT / "lib", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ucddb_runner  # noqa: E402

FS = 100  # cached UCDDB signals are resampled to 100 Hz
RR_FS = 4.0  # resample tachogram to 4 Hz for spectral analysis
LIT_CACHE = _ROOT / "aligned_data" / "ucddb_literature_features"

# Feature column order is fixed so downstream arrays stay aligned.
FEATURE_NAMES = [
    # time-domain HRV
    "mean_rr", "sdnn", "rmssd", "sdsd", "pnn50", "pnn20", "cvrr",
    "mean_hr", "std_hr", "rr_min", "rr_max", "rr_range", "rr_iqr", "mad_rr",
    # Poincare
    "sd1", "sd2", "sd_ratio", "ellipse_area",
    # frequency-domain HRV
    "vlf_power", "lf_power", "hf_power", "total_power",
    "lf_hf_ratio", "lf_nu", "hf_nu",
    # CVHR / apnea band
    "apnea_band_power", "apnea_band_ratio", "apnea_peak_freq", "apnea_peak_power",
    "cvhr_acf_peak", "cvhr_acf_lag", "cvhr_acf_band_max",
    # EDR / respiration from R-peak amplitude
    "amp_std", "amp_cv", "amp_range", "amp_iqr",
    "edr_resp_power", "edr_resp_ratio", "edr_apnea_power",
    # quality / count
    "n_beats", "beat_density",
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class RecordFeatures:
    record_id: str
    channel: int
    features: np.ndarray       # (n_minutes, N_FEATURES) float32
    labels: np.ndarray         # (n_minutes,) int64
    minute_indices: np.ndarray  # (n_minutes,) int32
    mean_hr_bpm: float
    n_minutes_total: int


def _find_cache(record_id, channel):
    """Locate a cached npz holding signal + rpeaks for this record/channel."""
    if not LIT_CACHE.exists():
        return None
    # Prefer the 900-length Hamilton absolute variant; fall back to any match.
    preferred = LIT_CACHE / (
        f"{record_id}_ch{channel}_hyp_ctx5_len900_overlap5p0_biosppyhamilton_absolute.npz"
    )
    if preferred.exists():
        return preferred
    matches = sorted(LIT_CACHE.glob(f"{record_id}_ch{channel}_*.npz"))
    return matches[0] if matches else None


def load_signal_rpeaks(record_id, channel, ucddb_dir="ucddb"):
    """Return (signal, rpeaks) using the cache when available, else compute."""
    path = _find_cache(record_id, channel)
    if path is not None:
        cached = np.load(path, allow_pickle=False)
        return (
            cached["signal"].astype(np.float32, copy=False),
            cached["rpeaks"].astype(np.int64, copy=False),
        )
    # Fallback: read EDF and detect R-peaks (slower, rarely needed here).
    import ucddb_highres_trainer
    import ucddb_literature_features as litfeat

    signal = ucddb_highres_trainer.read_ucddb_signal(Path(ucddb_dir), record_id, channel)
    rpeaks, _ = litfeat.detect_rpeaks(signal, FS, "auto")
    return signal.astype(np.float32), rpeaks.astype(np.int64)


def minute_labels(duration_sec, events, min_overlap_sec=5.0):
    n_minutes = int(duration_sec // 60)
    labels = np.zeros(n_minutes, dtype=np.int64)
    for minute in range(n_minutes):
        start, end = minute * 60, minute * 60 + 60
        for event_start, event_end, _ in events:
            if min(end, event_end) - max(start, event_start) > min_overlap_sec:
                labels[minute] = 1
                break
    return labels


def _safe(x, default=0.0):
    x = float(x)
    return x if np.isfinite(x) else default


def _resample_uniform(times, values, start, end, fs=RR_FS):
    """Linear-interpolate an irregular (times, values) series onto a uniform grid."""
    n = max(8, int(round((end - start) * fs)))
    grid = np.linspace(start, end, n, endpoint=False)
    return grid, np.interp(grid, times, values)


def _band_power(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(_trapz(psd[mask], freqs[mask]))


def compute_minute_features(rr_times, rr, amp, start_sec, end_sec, min_beats=20):
    """Compute the scalar feature vector for one context window. None if too sparse."""
    mask = (rr_times >= start_sec) & (rr_times < end_sec)
    if int(mask.sum()) < min_beats:
        return None

    t = rr_times[mask]
    r = rr[mask].astype(np.float64)
    a = amp[mask].astype(np.float64)
    if len(r) < min_beats or np.std(r) < 1e-6:
        return None

    diff = np.diff(r)
    feat = {}

    # --- time-domain HRV ---
    feat["mean_rr"] = np.mean(r)
    feat["sdnn"] = np.std(r)
    feat["rmssd"] = np.sqrt(np.mean(diff ** 2)) if len(diff) else 0.0
    feat["sdsd"] = np.std(diff) if len(diff) else 0.0
    feat["pnn50"] = np.mean(np.abs(diff) > 0.05) if len(diff) else 0.0
    feat["pnn20"] = np.mean(np.abs(diff) > 0.02) if len(diff) else 0.0
    feat["cvrr"] = feat["sdnn"] / feat["mean_rr"] if feat["mean_rr"] > 0 else 0.0
    hr = 60.0 / np.clip(r, 0.3, 2.5)
    feat["mean_hr"] = np.mean(hr)
    feat["std_hr"] = np.std(hr)
    feat["rr_min"] = np.min(r)
    feat["rr_max"] = np.max(r)
    feat["rr_range"] = feat["rr_max"] - feat["rr_min"]
    feat["rr_iqr"] = np.subtract(*np.percentile(r, [75, 25]))
    feat["mad_rr"] = np.median(np.abs(r - np.median(r)))

    # --- Poincare nonlinear ---
    if len(diff):
        sd1 = np.sqrt(0.5) * np.std(diff)
        sd2 = np.sqrt(max(2.0 * feat["sdnn"] ** 2 - 0.5 * np.std(diff) ** 2, 0.0))
    else:
        sd1 = sd2 = 0.0
    feat["sd1"] = sd1
    feat["sd2"] = sd2
    feat["sd_ratio"] = sd1 / sd2 if sd2 > 1e-6 else 0.0
    feat["ellipse_area"] = np.pi * sd1 * sd2

    # --- frequency-domain HRV on the resampled tachogram ---
    grid, rr_u = _resample_uniform(t, r, start_sec, end_sec, RR_FS)
    rr_u = rr_u - np.mean(rr_u)
    nperseg = min(256, len(rr_u))
    freqs, psd = welch(rr_u, fs=RR_FS, nperseg=nperseg, detrend="linear")
    vlf = _band_power(freqs, psd, 0.0033, 0.04)
    lf = _band_power(freqs, psd, 0.04, 0.15)
    hf = _band_power(freqs, psd, 0.15, 0.40)
    total = _band_power(freqs, psd, 0.0033, 0.40)
    feat["vlf_power"] = vlf
    feat["lf_power"] = lf
    feat["hf_power"] = hf
    feat["total_power"] = total
    feat["lf_hf_ratio"] = lf / hf if hf > 1e-8 else 0.0
    lf_hf = lf + hf
    feat["lf_nu"] = lf / lf_hf if lf_hf > 1e-8 else 0.0
    feat["hf_nu"] = hf / lf_hf if lf_hf > 1e-8 else 0.0

    # --- CVHR / apnea band (the key apnea signature) ---
    apnea = _band_power(freqs, psd, 0.01, 0.04)
    feat["apnea_band_power"] = apnea
    feat["apnea_band_ratio"] = apnea / total if total > 1e-8 else 0.0
    band = (freqs >= 0.008) & (freqs <= 0.05)
    if np.any(band):
        bf, bp = freqs[band], psd[band]
        k = int(np.argmax(bp))
        feat["apnea_peak_freq"] = float(bf[k])
        feat["apnea_peak_power"] = float(bp[k])
    else:
        feat["apnea_peak_freq"] = 0.0
        feat["apnea_peak_power"] = 0.0

    # autocorrelation-based CVHR periodicity in the 20-70 s apnea-cycle lag range
    x = rr_u / (np.std(rr_u) + 1e-8)
    acf = np.correlate(x, x, mode="full")[len(x) - 1:]
    acf = acf / (acf[0] + 1e-8)
    lo_lag, hi_lag = int(20 * RR_FS), int(min(70, (end_sec - start_sec) / 2) * RR_FS)
    if hi_lag > lo_lag + 1 and hi_lag < len(acf):
        seg = acf[lo_lag:hi_lag]
        k = int(np.argmax(seg))
        feat["cvhr_acf_peak"] = float(seg[k])
        feat["cvhr_acf_lag"] = float((lo_lag + k) / RR_FS)
        feat["cvhr_acf_band_max"] = float(np.max(seg))
    else:
        feat["cvhr_acf_peak"] = 0.0
        feat["cvhr_acf_lag"] = 0.0
        feat["cvhr_acf_band_max"] = 0.0

    # --- EDR / respiration from R-peak amplitude ---
    feat["amp_std"] = np.std(a)
    feat["amp_cv"] = np.std(a) / (np.abs(np.mean(a)) + 1e-8)
    feat["amp_range"] = np.max(a) - np.min(a)
    feat["amp_iqr"] = np.subtract(*np.percentile(a, [75, 25]))
    _, amp_u = _resample_uniform(t, a, start_sec, end_sec, RR_FS)
    amp_u = amp_u - np.mean(amp_u)
    fa, pa = welch(amp_u, fs=RR_FS, nperseg=min(256, len(amp_u)), detrend="linear")
    resp = _band_power(fa, pa, 0.10, 0.40)
    amp_total = _band_power(fa, pa, 0.0033, 0.40)
    feat["edr_resp_power"] = resp
    feat["edr_resp_ratio"] = resp / amp_total if amp_total > 1e-8 else 0.0
    feat["edr_apnea_power"] = _band_power(fa, pa, 0.01, 0.04)

    # --- quality / count ---
    feat["n_beats"] = float(len(r))
    feat["beat_density"] = float(len(r) / (end_sec - start_sec))

    return np.array([_safe(feat[name]) for name in FEATURE_NAMES], dtype=np.float32)


def extract_record(
    record_id,
    channel=0,
    apnea_only=False,
    context_minutes=5,
    min_overlap_sec=5.0,
    min_beats=20,
    ucddb_dir="ucddb",
):
    """Compute per-minute scalar features for one UCDDB record/channel."""
    signal, rpeaks = load_signal_rpeaks(record_id, channel, ucddb_dir)
    duration_sec = len(signal) // FS

    events = ucddb_runner.parse_respiratory_events(
        Path(ucddb_dir) / f"{record_id}_respevt.txt",
        include_hypopnea=not apnea_only,
    )
    labels_by_minute = minute_labels(duration_sec, events, min_overlap_sec)

    if len(rpeaks) < 2:
        empty = np.empty((0, N_FEATURES), dtype=np.float32)
        return RecordFeatures(record_id, channel, empty,
                              np.empty(0, np.int64), np.empty(0, np.int32),
                              0.0, len(labels_by_minute))

    rr_times = rpeaks.astype(np.float64) / FS
    rr = np.diff(rr_times, prepend=rr_times[0])
    rr[0] = rr[1] if len(rr) > 1 else 1.0
    rr = np.clip(rr, 0.30, 2.50)
    amp = signal[rpeaks].astype(np.float64)
    amp = np.abs(amp)

    half = context_minutes // 2
    ctx = context_minutes * 60
    feats, ys, mins = [], [], []
    for minute in range(half, len(labels_by_minute) - half):
        start = (minute - half) * 60
        vec = compute_minute_features(rr_times, rr, amp, start, start + ctx, min_beats)
        if vec is None:
            continue
        feats.append(vec)
        ys.append(int(labels_by_minute[minute]))
        mins.append(minute)

    if feats:
        features = np.stack(feats).astype(np.float32)
        labels = np.asarray(ys, dtype=np.int64)
        minute_indices = np.asarray(mins, dtype=np.int32)
    else:
        features = np.empty((0, N_FEATURES), dtype=np.float32)
        labels = np.empty(0, np.int64)
        minute_indices = np.empty(0, np.int32)

    return RecordFeatures(
        record_id=record_id,
        channel=channel,
        features=features,
        labels=labels,
        minute_indices=minute_indices,
        mean_hr_bpm=float(len(rpeaks) / max(duration_sec, 1) * 60.0),
        n_minutes_total=len(labels_by_minute),
    )


if __name__ == "__main__":
    # Quick self-check on a couple of records.
    for rid in ["ucddb002", "ucddb003"]:
        rec = extract_record(rid, channel=0, apnea_only=False)
        print(
            f"{rid}: feats={rec.features.shape} pos={int(rec.labels.sum())}/{len(rec.labels)} "
            f"HR={rec.mean_hr_bpm:.1f} nan={int(np.isnan(rec.features).sum())}"
        )
