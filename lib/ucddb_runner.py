import argparse
import os
import re
from pathlib import Path

import numpy as np
import pyedflib
import torch
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import apnea_trainer


FS_TARGET = 100
MINUTE_SAMPLES = FS_TARGET * 60


def patch_edf_start_time(edf_path: Path) -> None:
    """UCDDB Lifecard EDF files use HH:MM:SS; pyedflib expects HH.MM.SS."""
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


def labels_for_minutes(duration_sec: float, events, min_overlap_sec: float):
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


def load_ucddb_segments(edf_path: Path, event_path: Path, channel: int, include_hypopnea: bool, min_overlap_sec: float):
    patch_edf_start_time(edf_path)

    with pyedflib.EdfReader(str(edf_path)) as edf:
        fs = float(edf.getSampleFrequency(channel))
        duration_sec = float(edf.file_duration)
        signal = edf.readSignal(channel)

    if fs != FS_TARGET:
        # UCDDB Lifecard ECG is 128 Hz. 25/32 converts 128 Hz to 100 Hz exactly.
        signal = resample_poly(signal, FS_TARGET, int(fs)).astype(np.float32)
    else:
        signal = signal.astype(np.float32)

    events = parse_respiratory_events(event_path, include_hypopnea=include_hypopnea)
    y = labels_for_minutes(duration_sec, events, min_overlap_sec=min_overlap_sec)

    usable_minutes = min(len(y), len(signal) // MINUTE_SAMPLES)
    if usable_minutes == 0:
        return np.empty((0, MINUTE_SAMPLES), dtype=np.float32), np.empty((0,), dtype=np.int64)

    signal = signal[: usable_minutes * MINUTE_SAMPLES]
    y = y[:usable_minutes]
    x = signal.reshape(usable_minutes, MINUTE_SAMPLES)

    processed = []
    for segment in x:
        segment = apnea_trainer.butter_bandpass_filter(segment, 0.5, 45.0, FS_TARGET, order=3)
        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
        processed.append(segment.astype(np.float32))

    return np.stack(processed), y


def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return acc, prec, rec, spec, f1, cm


def available_record_ids(data_dir: Path):
    ids = []
    for edf in sorted(data_dir.glob("ucddb*_lifecard.edf")):
        record_id = edf.name.replace("_lifecard.edf", "")
        if (data_dir / f"{record_id}_respevt.txt").exists():
            ids.append(record_id)
    return ids


def run(args):
    data_dir = Path(args.data_dir)
    record_ids = args.records or available_record_ids(data_dir)
    if not record_ids:
        raise SystemExit(f"No downloaded UCDDB lifecard/respevt pairs found in {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = apnea_trainer.ParallelCNNTransformer().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    all_true = []
    all_pred = []

    print(f"Using device: {device}")
    print(f"Records: {len(record_ids)} | channel: {args.channel} | include_hypopnea: {not args.apnea_only}")

    for record_id in record_ids:
        edf_path = data_dir / f"{record_id}_lifecard.edf"
        event_path = data_dir / f"{record_id}_respevt.txt"
        if not edf_path.exists() or not event_path.exists():
            print(f"Skipping {record_id}: missing EDF or respiratory event file")
            continue

        x, y = load_ucddb_segments(
            edf_path,
            event_path,
            channel=args.channel,
            include_hypopnea=not args.apnea_only,
            min_overlap_sec=args.min_overlap_sec,
        )
        if len(x) == 0:
            print(f"Skipping {record_id}: no usable 60-second segments")
            continue

        preds = []
        with torch.no_grad():
            for start in range(0, len(x), args.batch_size):
                batch = torch.from_numpy(x[start : start + args.batch_size]).unsqueeze(1).to(device)
                outputs = model(batch)
                preds.extend(outputs.argmax(dim=1).cpu().numpy())

        preds = np.asarray(preds, dtype=np.int64)
        acc, prec, rec, spec, f1, _ = evaluate_metrics(y, preds)
        print(
            f"{record_id}: n={len(y)} A={int(y.sum())} N={int((y == 0).sum())} "
            f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} Spec={spec:.4f} F1={f1:.4f}"
        )

        all_true.extend(y.tolist())
        all_pred.extend(preds.tolist())

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    acc, prec, rec, spec, f1, cm = evaluate_metrics(y_true, y_pred)

    print("\nUCDDB External Test Metrics")
    print(f"  Samples:     {len(y_true)}")
    print(f"  Apnea/Hyp:   {int(y_true.sum())}")
    print(f"  Normal:      {int((y_true == 0).sum())}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print("  Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Apnea-ECG model on UCDDB Lifecard ECG.")
    parser.add_argument("--data-dir", default="ucddb")
    parser.add_argument("--model", default="apnea_parallel_cnn_transformer.pth")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--records", nargs="*")
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)
    parser.add_argument("--apnea-only", action="store_true", help="Label only APNEA-* events as positive.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
