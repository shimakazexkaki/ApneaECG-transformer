import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import mixed_trainer
import ucddb_runner


FS = 100
RANDOM_STATE = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def detect_r_peaks(segment):
    x = mixed_trainer.normalize_segment(segment)
    candidates = []
    for polarity, signal in ((1.0, x), (-1.0, -x)):
        peaks, _ = find_peaks(signal, distance=int(0.32 * FS), prominence=0.55)
        if len(peaks) < 3:
            continue
        rr = np.diff(peaks) / FS
        rr = rr[(rr >= 0.32) & (rr <= 2.20)]
        if len(rr) < 3:
            continue
        hr = 60.0 / rr
        plausible_hr = np.mean((hr >= 35.0) & (hr <= 190.0))
        regularity = 1.0 / (np.std(rr) / (np.mean(rr) + 1e-8) + 1e-3)
        score = plausible_hr * len(rr) + 0.1 * regularity
        candidates.append((score, polarity, peaks))

    if not candidates:
        return np.empty((0,), dtype=np.int64)

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][2].astype(np.int64)


def safe_stats(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return [0.0] * 9
    q10, q25, q50, q75, q90 = np.percentile(values, [10, 25, 50, 75, 90])
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(q10),
        float(q25),
        float(q50),
        float(q75),
        float(q90),
    ]


def slope(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 2:
        return 0.0
    t = np.arange(len(values), dtype=np.float32)
    t = t - t.mean()
    denom = float(np.dot(t, t))
    if denom == 0.0:
        return 0.0
    return float(np.dot(t, values - values.mean()) / denom)


def extract_features(segment):
    x = mixed_trainer.normalize_segment(segment)
    duration_sec = len(x) / FS
    peaks = detect_r_peaks(x)

    signal_features = []
    signal_features.extend(safe_stats(x))
    signal_features.extend(safe_stats(np.abs(x)))
    signal_features.append(float(np.mean(np.diff(np.signbit(x)) != 0)))
    signal_features.append(float(np.mean(x**2)))

    if len(peaks) < 4:
        rr_features = [0.0] * 31
    else:
        rr = np.diff(peaks) / FS
        rr = rr[(rr >= 0.32) & (rr <= 2.20)]
        if len(rr) < 3:
            rr_features = [0.0] * 31
        else:
            hr = 60.0 / rr
            diff_rr = np.diff(rr)
            rmssd = float(np.sqrt(np.mean(diff_rr**2))) if len(diff_rr) else 0.0
            pnn20 = float(np.mean(np.abs(diff_rr) > 0.02)) if len(diff_rr) else 0.0
            pnn50 = float(np.mean(np.abs(diff_rr) > 0.05)) if len(diff_rr) else 0.0
            rr_features = [
                float(len(peaks) / (duration_sec / 60.0)),
                float(len(rr)),
                float(rmssd),
                float(pnn20),
                float(pnn50),
                float(slope(rr)),
                float(slope(hr)),
            ]
            rr_features.extend(safe_stats(rr))
            rr_features.extend(safe_stats(hr))
            rr_features.extend(safe_stats(diff_rr))

    return np.asarray(signal_features + rr_features, dtype=np.float32)


def extract_feature_matrix(x, label):
    features = []
    total = len(x)
    print(f"Extracting HRV features: {label} ({total} samples)")
    for idx, segment in enumerate(x, start=1):
        features.append(extract_features(segment))
        if idx % 2000 == 0 or idx == total:
            print(f"  {label}: {idx}/{total}")
    return np.vstack(features).astype(np.float32)


def make_sample_weights(y, source, ucddb_weight, apnea_weight):
    y = np.asarray(y, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    class_counts = np.bincount(y, minlength=2)
    class_weights = len(y) / (2.0 * np.maximum(class_counts, 1))
    domain_weights = np.where(source == 0, ucddb_weight, apnea_weight)
    return class_weights[y] * domain_weights


def score_model(model, x, y, threshold):
    scores = model.predict_proba(x)[:, 1]
    metrics = mixed_trainer.evaluate_from_scores(y, scores, threshold=threshold)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, scores))
    except ValueError:
        metrics["roc_auc"] = 0.0
    return metrics, scores


def train(args):
    set_seed(args.seed)
    mixed_trainer.SHOW_PROGRESS = False
    if args.context_minutes < 1 or args.context_minutes % 2 == 0:
        raise ValueError("--context-minutes must be a positive odd number.")

    ucddb_data_dir = Path(args.ucddb_dir)
    apnea_data_dir = Path(args.apnea_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ucddb_records = ucddb_runner.available_record_ids(ucddb_data_dir)
    apnea_records = mixed_trainer.available_apnea_records(apnea_data_dir)
    ucddb_train_records, ucddb_val_records, ucddb_test_records = mixed_trainer.split_records(
        ucddb_records, args.val_size, args.test_size, args.seed
    )
    apnea_train_records, apnea_val_records, apnea_test_records = mixed_trainer.split_records(
        apnea_records, args.val_size, args.test_size, args.seed
    )

    x_u_train, y_u_train, u_train_stats = mixed_trainer.load_ucddb_records(
        ucddb_data_dir,
        Path(args.ucddb_cache_dir),
        ucddb_train_records,
        args.channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
        context_minutes=args.context_minutes,
    )
    x_u_val, y_u_val, u_val_stats = mixed_trainer.load_ucddb_records(
        ucddb_data_dir,
        Path(args.ucddb_cache_dir),
        ucddb_val_records,
        args.channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
        context_minutes=args.context_minutes,
    )
    x_u_test, y_u_test, u_test_stats = mixed_trainer.load_ucddb_records(
        ucddb_data_dir,
        Path(args.ucddb_cache_dir),
        ucddb_test_records,
        args.channels,
        include_hypopnea=not args.apnea_only,
        min_overlap_sec=args.min_overlap_sec,
        context_minutes=args.context_minutes,
    )
    x_a_train, y_a_train, a_train_stats = mixed_trainer.load_apnea_records(
        apnea_data_dir, Path(args.apnea_cache_dir), apnea_train_records, args.context_minutes
    )
    x_a_val, y_a_val, a_val_stats = mixed_trainer.load_apnea_records(
        apnea_data_dir, Path(args.apnea_cache_dir), apnea_val_records, args.context_minutes
    )
    x_a_test, y_a_test, a_test_stats = mixed_trainer.load_apnea_records(
        apnea_data_dir, Path(args.apnea_cache_dir), apnea_test_records, args.context_minutes
    )

    f_u_train = extract_feature_matrix(x_u_train, "ucddb_train")
    f_u_val = extract_feature_matrix(x_u_val, "ucddb_val")
    f_u_test = extract_feature_matrix(x_u_test, "ucddb_test")
    f_a_train = extract_feature_matrix(x_a_train, "apnea_train")
    f_a_val = extract_feature_matrix(x_a_val, "apnea_val")
    f_a_test = extract_feature_matrix(x_a_test, "apnea_test")

    x_train = np.concatenate([f_u_train, f_a_train], axis=0)
    y_train = np.concatenate([y_u_train, y_a_train], axis=0)
    source = mixed_trainer.source_labels(len(y_u_train), len(y_a_train))
    sample_weights = make_sample_weights(y_train, source, args.ucddb_sample_weight, args.apnea_sample_weight)

    if args.model_type == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            random_state=args.seed,
        )

    print(f"Training HRV model: {args.model_type}")
    model.fit(x_train, y_train, sample_weight=sample_weights)

    val_y = np.concatenate([y_u_val, y_a_val])
    val_scores = np.concatenate([
        model.predict_proba(f_u_val)[:, 1],
        model.predict_proba(f_a_val)[:, 1],
    ])
    threshold, val_metrics, constrained = mixed_trainer.best_threshold_balanced(
        val_y, val_scores, args.min_specificity
    )
    ucddb_threshold, ucddb_val_metrics, _ = mixed_trainer.best_threshold_balanced(
        y_u_val, model.predict_proba(f_u_val)[:, 1], args.min_specificity
    )

    result = {
        "settings": vars(args),
        "records": {
            "ucddb_train": ucddb_train_records,
            "ucddb_val": ucddb_val_records,
            "ucddb_test": ucddb_test_records,
            "apnea_train": apnea_train_records,
            "apnea_val": apnea_val_records,
            "apnea_test": apnea_test_records,
        },
        "record_stats": {
            "ucddb_train": u_train_stats,
            "ucddb_val": u_val_stats,
            "ucddb_test": u_test_stats,
            "apnea_train": a_train_stats,
            "apnea_val": a_val_stats,
            "apnea_test": a_test_stats,
        },
        "sample_summaries": {
            "ucddb_train": mixed_trainer.dataset_summary("ucddb_train", y_u_train),
            "ucddb_val": mixed_trainer.dataset_summary("ucddb_val", y_u_val),
            "ucddb_test": mixed_trainer.dataset_summary("ucddb_test", y_u_test),
            "apnea_train": mixed_trainer.dataset_summary("apnea_train", y_a_train),
            "apnea_val": mixed_trainer.dataset_summary("apnea_val", y_a_val),
            "apnea_test": mixed_trainer.dataset_summary("apnea_test", y_a_test),
        },
        "best_threshold_combined_val": threshold,
        "best_threshold_ucddb_val": ucddb_threshold,
        "combined_val_threshold_metrics": val_metrics,
        "ucddb_val_threshold_metrics": ucddb_val_metrics,
        "ucddb_test": {
            "threshold_0_5": score_model(model, f_u_test, y_u_test, 0.5)[0],
            "threshold_combined_val": score_model(model, f_u_test, y_u_test, threshold)[0],
            "threshold_ucddb_val": score_model(model, f_u_test, y_u_test, ucddb_threshold)[0],
        },
        "apnea_test": {
            "threshold_0_5": score_model(model, f_a_test, y_a_test, 0.5)[0],
            "threshold_combined_val": score_model(model, f_a_test, y_a_test, threshold)[0],
            "threshold_ucddb_val": score_model(model, f_a_test, y_a_test, ucddb_threshold)[0],
        },
    }

    model_path = output_dir / args.model_name
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    result["model_path"] = str(model_path)

    result_path = output_dir / args.result_name
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved results: {result_path}")
    print("\nFinal holdout test metrics")
    for domain in ("ucddb_test", "apnea_test"):
        print(f"{domain}:")
        for threshold_name, metrics in result[domain].items():
            print(
                f"  {threshold_name}: Acc={metrics['accuracy']:.4f} "
                f"BAcc={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f} "
                f"Rec={metrics['recall']:.4f} Spec={metrics['specificity']:.4f} "
                f"AUC={metrics['roc_auc']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Train HRV/RR feature models for wearable ECG apnea detection.")
    parser.add_argument("--ucddb-dir", default="ucddb")
    parser.add_argument("--apnea-dir", default="apnea-ecg")
    parser.add_argument("--ucddb-cache-dir", default="aligned_data/ucddb")
    parser.add_argument("--apnea-cache-dir", default="aligned_data/apnea_ecg")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-name", default="hrv_ucddb_focused_model.pkl")
    parser.add_argument("--result-name", default="hrv_ucddb_focused_results.json")
    parser.add_argument("--channels", nargs="+", type=int, default=[0])
    parser.add_argument("--context-minutes", type=int, default=5)
    parser.add_argument("--min-overlap-sec", type=float, default=1.0)
    parser.add_argument("--apnea-only", action="store_true")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ucddb-sample-weight", type=float, default=2.0)
    parser.add_argument("--apnea-sample-weight", type=float, default=1.0)
    parser.add_argument("--min-specificity", type=float, default=0.65)
    parser.add_argument("--model-type", choices=["extra_trees", "hist_gradient_boosting"], default="extra_trees")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--min-samples-leaf", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
