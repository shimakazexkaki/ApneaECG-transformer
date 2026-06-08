"""
評估與視覺化腳本

功能:
  - 載入已訓練模型，評估 per-segment 和 per-recording 指標
  - 繪製 Confusion Matrix
  - 繪製 ROC 曲線
  - 繪製訓練 loss/accuracy 曲線
  - 外部驗證 (Apnea-ECG)

用法:
  python evaluate.py --model results/cnn_transformer_lstm_ucddb_raw_holdout.pth \
                     --results results/cnn_transformer_lstm_ucddb_raw_holdout_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_curve,
    auc,
)

from models import build_model, count_parameters
from data_preprocessing import (
    available_apnea_ecg_records,
    available_ucddb_records,
    load_dataset,
)
from train import ECGDataset, predict, compute_metrics, compute_per_recording_metrics

plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 150


# ============================================================
# 視覺化函數
# ============================================================
def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """繪製並儲存 Confusion Matrix。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Normal", "Apnea"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion Matrix 已儲存: {save_path}")


def plot_roc_curve(y_true, y_scores, title, save_path):
    """繪製並儲存 ROC 曲線。"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", alpha=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC 曲線已儲存: {save_path}")


def plot_training_curves(history, title_prefix, save_path):
    """
    繪製訓練過程的 loss 和 accuracy 曲線 (對應論文 Fig. 7, 8)。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(f"{title_prefix} — Loss", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["val_acc"], "g-", label="Validation Accuracy", linewidth=1.5)
    if "val_f1" in history:
        axes[1].plot(epochs, history["val_f1"], "m-", label="Validation F1-score", linewidth=1.5)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title(f"{title_prefix} — Accuracy & F1", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  訓練曲線已儲存: {save_path}")


def plot_cv_summary(cv_results, save_path):
    """繪製 CV 結果的摘要長條圖。"""
    metrics_keys = ["accuracy", "sensitivity", "specificity", "f1_score", "auc", "kappa"]
    summary = cv_results["summary"]

    means = [summary[k]["mean"] for k in metrics_keys]
    stds = [summary[k]["std"] for k in metrics_keys]
    labels = ["Acc", "Sen", "Spe", "F1", "AUC", "κ"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=[
        "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"
    ], edgecolor="white", linewidth=0.5, alpha=0.85)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                f"{mean:.3f}±{std:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{cv_results['settings']['folds']}-Fold CV Results — "
                 f"{cv_results['settings']['model_type'].upper()}", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  CV 摘要圖已儲存: {save_path}")


# ============================================================
# 外部驗證
# ============================================================
def external_validation(args):
    """
    載入 UCDDB 訓練的模型，在 Apnea-ECG 上進行外部驗證。
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print("=" * 70)
    print("外部驗證: Apnea-ECG Dataset")
    print("=" * 70)

    # 載入模型
    in_channels = 1 if args.feature_type == "raw" else 2
    pool_size = 4 if args.feature_type == "raw" else 2

    model = build_model(
        args.model_type,
        in_channels=in_channels,
        cnn_filters=(64, 128, 128),
        pool_size=pool_size,
    ).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    print(f"模型已載入: {args.model}")

    # 載入 Apnea-ECG
    apnea_dir = Path(args.apnea_dir)
    all_records = available_apnea_ecg_records(apnea_dir)
    print(f"Apnea-ECG 紀錄: {len(all_records)}")

    print(f"\n載入資料 ({args.feature_type})...")
    x, y, stats = load_dataset(
        apnea_dir, all_records, "apnea_ecg", args.feature_type,
    )

    segment_rids = []
    for stat in stats:
        segment_rids.extend([stat["record_id"]] * stat["total_segments"])

    loader = torch.utils.data.DataLoader(
        ECGDataset(x, y), batch_size=args.batch_size, shuffle=False
    )

    labels, preds, scores = predict(model, loader, device)

    # Per-segment
    seg_m = compute_metrics(labels, preds, scores)
    print(f"\nPer-Segment 指標:")
    for k, v in seg_m.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")

    # Per-recording
    rec_m = compute_per_recording_metrics(all_records, segment_rids, labels, preds, scores)
    print(f"\nPer-Recording 指標:")
    for k, v in rec_m.items():
        if k not in ("per_recording_details", "confusion_matrix"):
            print(f"  {k}: {v}")

    # 儲存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext_results = {
        "external_validation": "apnea_ecg",
        "model_path": args.model,
        "per_segment": seg_m,
        "per_recording": rec_m,
    }
    result_path = output_dir / f"{Path(args.model).stem}_external_apnea_ecg.json"
    result_path.write_text(json.dumps(ext_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n外部驗證結果已儲存: {result_path}")

    # 圖表
    plot_confusion_matrix(labels, preds,
                          "External Validation: Apnea-ECG (Per-Segment)",
                          output_dir / f"{Path(args.model).stem}_ext_confusion_matrix.png")
    plot_roc_curve(labels, scores,
                   "External Validation: Apnea-ECG ROC",
                   output_dir / f"{Path(args.model).stem}_ext_roc.png")


# ============================================================
# 從已有結果生成視覺化
# ============================================================
def visualize_results(args):
    """從 JSON 結果檔生成視覺化圖表。"""
    result_path = Path(args.results)
    if not result_path.exists():
        raise FileNotFoundError(f"結果檔不存在: {result_path}")

    results = json.loads(result_path.read_text(encoding="utf-8"))
    output_dir = result_path.parent
    stem = result_path.stem

    # 訓練曲線
    if "history" in results:
        plot_training_curves(
            results["history"],
            results.get("experiment_name", "Training"),
            output_dir / f"{stem}_training_curves.png",
        )

    # 測試集 confusion matrix
    if "test_per_segment" in results:
        cm = np.array(results["test_per_segment"]["confusion_matrix"])
        y_true = []
        y_pred = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                y_true.extend([i] * cm[i, j])
                y_pred.extend([j] * cm[i, j])
        plot_confusion_matrix(
            y_true, y_pred,
            "Test Set: Per-Segment Confusion Matrix",
            output_dir / f"{stem}_confusion_matrix.png",
        )

    # CV 結果
    if "summary" in results and "fold_results" in results:
        plot_cv_summary(results, output_dir / f"{stem}_cv_summary.png")

    print("視覺化完成！")


# ============================================================
# 結果摘要表格 (對應論文 Table 5)
# ============================================================
def print_results_table(results_path: str):
    """以表格形式印出結果，對齊論文 Table 5 的格式。"""
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))

    print("\n" + "=" * 90)
    print("  結果摘要 (對應論文 Table 5 格式)")
    print("=" * 90)

    if "test_per_segment" in results:
        m = results["test_per_segment"]
        print(f"\n  Per-Segment Classification:")
        print(f"  {'Metric':<15} {'Value':>10}")
        print(f"  {'-'*25}")
        print(f"  {'Accuracy':<15} {m['accuracy']*100:>9.1f}%")
        print(f"  {'Sensitivity':<15} {m['sensitivity']*100:>9.1f}%")
        print(f"  {'Specificity':<15} {m['specificity']*100:>9.1f}%")
        print(f"  {'F1-score':<15} {m['f1_score']:>10.3f}")
        print(f"  {'AUC':<15} {m['auc']:>10.3f}")
        print(f"  {'Cohen κ':<15} {m['kappa']:>10.3f}")

    if "test_per_recording" in results:
        m = results["test_per_recording"]
        if "accuracy" in m:
            print(f"\n  Per-Recording Classification:")
            print(f"  {'Metric':<15} {'Value':>10}")
            print(f"  {'-'*25}")
            print(f"  {'Accuracy':<15} {m['accuracy']*100:>9.1f}%")
            print(f"  {'Sensitivity':<15} {m.get('sensitivity', 0)*100:>9.1f}%")
            print(f"  {'Specificity':<15} {m.get('specificity', 0)*100:>9.1f}%")
            print(f"  {'AUC':<15} {m.get('auc', 0):>10.3f}")
            print(f"  {'Pearson Corr':<15} {m.get('pearson_corr', 0):>10.3f}")

    if "summary" in results:
        print(f"\n  {results['settings']['folds']}-Fold CV Summary:")
        print(f"  {'Metric':<15} {'Mean ± Std':>20}")
        print(f"  {'-'*35}")
        for key in ["accuracy", "sensitivity", "specificity", "f1_score", "auc", "kappa"]:
            s = results["summary"][key]
            print(f"  {key:<15} {s['mean']*100:>8.1f}% ± {s['std']*100:.1f}%")

    print("=" * 90)


# ============================================================
# 主程式
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CNN-Transformer-LSTM 評估與視覺化")

    subparsers = parser.add_subparsers(dest="command")

    # 視覺化子命令
    vis_parser = subparsers.add_parser("visualize", help="從結果檔生成視覺化")
    vis_parser.add_argument("--results", required=True, help="結果 JSON 檔案路徑")

    # 表格子命令
    table_parser = subparsers.add_parser("table", help="印出結果表格")
    table_parser.add_argument("--results", required=True, help="結果 JSON 檔案路徑")

    # 外部驗證子命令
    ext_parser = subparsers.add_parser("external", help="外部驗證 (Apnea-ECG)")
    ext_parser.add_argument("--model", required=True, help="模型檔案路徑")
    ext_parser.add_argument("--model-type", default="cnn_transformer_lstm")
    ext_parser.add_argument("--feature-type", choices=["raw", "rri"], default="raw")
    ext_parser.add_argument("--apnea-dir", default=None)
    ext_parser.add_argument("--output-dir", default="results")
    ext_parser.add_argument("--batch-size", type=int, default=64)
    ext_parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    if args.command == "visualize":
        visualize_results(args)
    elif args.command == "table":
        print_results_table(args.results)
    elif args.command == "external":
        if args.apnea_dir is None:
            args.apnea_dir = str(Path(__file__).parent.parent / "apnea-ecg")
        external_validation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
