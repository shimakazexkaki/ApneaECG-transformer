"""
一鍵實驗腳本

自動執行:
  1. 模型維度驗證
  2. UCDDB Hold-out 訓練 + 評估 (Raw ECG)
  3. UCDDB Hold-out 訓練 + 評估 (RRI Features)
  4. UCDDB 5-Fold CV (Raw ECG)
  5. Apnea-ECG 外部驗證
  6. 結果彙總

用法:
  # 完整實驗
  python run_experiment.py --all

  # 僅 hold-out (Raw ECG)
  python run_experiment.py --holdout-raw

  # 僅 5-fold CV
  python run_experiment.py --cv

  # 僅外部驗證
  python run_experiment.py --external --model results/xxx.pth
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

# 取得 Python 執行檔路徑
PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


def run_cmd(cmd: list, description: str):
    """執行子程序並即時顯示輸出。"""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(
        cmd, cwd=str(SCRIPT_DIR),
        text=True, encoding="utf-8", errors="replace",
    )
    elapsed = time.time() - start
    status = "✅ 成功" if result.returncode == 0 else "❌ 失敗"
    print(f"\n{status} ({elapsed:.1f}s)")
    return result.returncode == 0


def verify_models():
    """步驟 1: 驗證模型維度。"""
    return run_cmd(
        [PYTHON, str(SCRIPT_DIR / "models.py")],
        "Step 1: 模型維度驗證",
    )


def train_holdout_raw(args):
    """步驟 2: UCDDB Hold-out (Raw ECG)。"""
    cmd = [
        PYTHON, str(SCRIPT_DIR / "train.py"),
        "--mode", "holdout",
        "--dataset", "ucddb",
        "--feature-type", "raw",
        "--data-dir", str(Path(SCRIPT_DIR).parent / "ucddb"),
        "--output-dir", str(RESULTS_DIR),
        "--experiment-name", "cnn_transformer_lstm_ucddb_raw_holdout",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
    ]
    if args.cpu:
        cmd.append("--cpu")
    return run_cmd(cmd, "Step 2: UCDDB Hold-out Training (Raw ECG)")


def train_holdout_rri(args):
    """步驟 3: UCDDB Hold-out (RRI Features)。"""
    cmd = [
        PYTHON, str(SCRIPT_DIR / "train.py"),
        "--mode", "holdout",
        "--dataset", "ucddb",
        "--feature-type", "rri",
        "--data-dir", str(Path(SCRIPT_DIR).parent / "ucddb"),
        "--output-dir", str(RESULTS_DIR),
        "--experiment-name", "cnn_transformer_lstm_ucddb_rri_holdout",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
    ]
    if args.cpu:
        cmd.append("--cpu")
    return run_cmd(cmd, "Step 3: UCDDB Hold-out Training (RRI Features)")


def train_cv_raw(args):
    """步驟 4: UCDDB 5-Fold CV (Raw ECG)。"""
    cmd = [
        PYTHON, str(SCRIPT_DIR / "train.py"),
        "--mode", "cv",
        "--folds", str(args.folds),
        "--dataset", "ucddb",
        "--feature-type", "raw",
        "--data-dir", str(Path(SCRIPT_DIR).parent / "ucddb"),
        "--output-dir", str(RESULTS_DIR),
        "--experiment-name", "cnn_transformer_lstm_ucddb_raw_cv",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
    ]
    if args.cpu:
        cmd.append("--cpu")
    return run_cmd(cmd, "Step 4: UCDDB 5-Fold Cross-Validation (Raw ECG)")


def train_cv_rri(args):
    """步驟 4b: UCDDB 5-Fold CV (RRI Features)。"""
    cmd = [
        PYTHON, str(SCRIPT_DIR / "train.py"),
        "--mode", "cv",
        "--folds", str(args.folds),
        "--dataset", "ucddb",
        "--feature-type", "rri",
        "--data-dir", str(Path(SCRIPT_DIR).parent / "ucddb"),
        "--output-dir", str(RESULTS_DIR),
        "--experiment-name", "cnn_transformer_lstm_ucddb_rri_cv",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
    ]
    if args.cpu:
        cmd.append("--cpu")
    return run_cmd(cmd, "Step 4b: UCDDB 5-Fold Cross-Validation (RRI Features)")


def external_validation(args, model_path: str):
    """步驟 5: Apnea-ECG 外部驗證。"""
    cmd = [
        PYTHON, str(SCRIPT_DIR / "evaluate.py"),
        "external",
        "--model", model_path,
        "--feature-type", "raw",
        "--apnea-dir", str(Path(SCRIPT_DIR).parent / "apnea-ecg"),
        "--output-dir", str(RESULTS_DIR),
        "--batch-size", str(args.batch_size),
    ]
    if args.cpu:
        cmd.append("--cpu")
    return run_cmd(cmd, "Step 5: External Validation on Apnea-ECG")


def visualize_results(result_path: str):
    """步驟 6: 生成視覺化圖表。"""
    return run_cmd(
        [PYTHON, str(SCRIPT_DIR / "evaluate.py"), "visualize", "--results", result_path],
        f"Step 6: 視覺化 ({Path(result_path).name})",
    )


def generate_summary():
    """步驟 7: 生成結果彙總 Markdown。"""
    print(f"\n{'='*70}")
    print("▶ Step 7: 生成結果彙總")
    print(f"{'='*70}\n")

    summary_lines = [
        "# CNN-Transformer-LSTM 實驗結果彙總",
        "",
        "## 模型配置 (M11 — 論文最佳)",
        "- CNN Filters: (64, 128, 128), Kernel=7, MaxPool=4",
        "- Transformer: d_model=128, nhead=8, dim_ff=256",
        "- LSTM: hidden=128",
        "- Optimizer: Adam, lr=0.001",
        "- Loss: CrossEntropyLoss (weighted)",
        "- Early Stopping: patience=30",
        "",
    ]

    # 讀取 hold-out 結果
    for feature_type in ["raw", "rri"]:
        result_file = RESULTS_DIR / f"cnn_transformer_lstm_ucddb_{feature_type}_holdout_results.json"
        if result_file.exists():
            results = json.loads(result_file.read_text(encoding="utf-8"))
            m = results["test_per_segment"]
            summary_lines.extend([
                f"## Hold-out 結果 ({feature_type.upper()} Features)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Accuracy | {m['accuracy']*100:.1f}% |",
                f"| Sensitivity | {m['sensitivity']*100:.1f}% |",
                f"| Specificity | {m['specificity']*100:.1f}% |",
                f"| F1-score | {m['f1_score']:.3f} |",
                f"| AUC | {m['auc']:.3f} |",
                f"| Cohen's κ | {m['kappa']:.3f} |",
                "",
            ])

            if "test_per_recording" in results and "accuracy" in results["test_per_recording"]:
                rm = results["test_per_recording"]
                summary_lines.extend([
                    f"### Per-Recording ({feature_type.upper()})",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Accuracy | {rm['accuracy']*100:.1f}% |",
                    f"| Pearson Corr | {rm.get('pearson_corr', 0):.3f} |",
                    "",
                ])

    # 讀取 CV 結果
    for feature_type in ["raw", "rri"]:
        cv_file = RESULTS_DIR / f"cnn_transformer_lstm_ucddb_{feature_type}_cv_cv_results.json"
        if cv_file.exists():
            results = json.loads(cv_file.read_text(encoding="utf-8"))
            s = results["summary"]
            summary_lines.extend([
                f"## 5-Fold CV 結果 ({feature_type.upper()} Features)",
                "",
                "| Metric | Mean ± Std |",
                "|--------|-----------|",
                f"| Accuracy | {s['accuracy']['mean']*100:.1f}% ± {s['accuracy']['std']*100:.1f}% |",
                f"| Sensitivity | {s['sensitivity']['mean']*100:.1f}% ± {s['sensitivity']['std']*100:.1f}% |",
                f"| Specificity | {s['specificity']['mean']*100:.1f}% ± {s['specificity']['std']*100:.1f}% |",
                f"| F1-score | {s['f1_score']['mean']:.3f} ± {s['f1_score']['std']:.3f} |",
                f"| AUC | {s['auc']['mean']:.3f} ± {s['auc']['std']:.3f} |",
                f"| Cohen's κ | {s['kappa']['mean']:.3f} ± {s['kappa']['std']:.3f} |",
                "",
            ])

    # 讀取外部驗證結果
    ext_files = list(RESULTS_DIR.glob("*_external_apnea_ecg.json"))
    for ext_file in ext_files:
        results = json.loads(ext_file.read_text(encoding="utf-8"))
        m = results["per_segment"]
        summary_lines.extend([
            f"## Apnea-ECG 外部驗證 ({ext_file.stem})",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {m['accuracy']*100:.1f}% |",
            f"| Sensitivity | {m['sensitivity']*100:.1f}% |",
            f"| Specificity | {m['specificity']*100:.1f}% |",
            f"| F1-score | {m['f1_score']:.3f} |",
            f"| AUC | {m['auc']:.3f} |",
            "",
        ])

    summary_path = RESULTS_DIR / "experiment_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"結果彙總已儲存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CNN-Transformer-LSTM 一鍵實驗腳本"
    )

    # 實驗選擇
    parser.add_argument("--all", action="store_true", help="執行所有實驗")
    parser.add_argument("--verify", action="store_true", help="僅驗證模型維度")
    parser.add_argument("--holdout-raw", action="store_true", help="Hold-out (Raw ECG)")
    parser.add_argument("--holdout-rri", action="store_true", help="Hold-out (RRI Features)")
    parser.add_argument("--cv", action="store_true", help="5-Fold CV (Raw ECG)")
    parser.add_argument("--cv-rri", action="store_true", help="5-Fold CV (RRI Features)")
    parser.add_argument("--external", action="store_true", help="外部驗證 (Apnea-ECG)")
    parser.add_argument("--model", default=None, help="外部驗證時使用的模型路徑")
    parser.add_argument("--visualize", action="store_true", help="生成視覺化圖表")
    parser.add_argument("--summary", action="store_true", help="生成結果彙總")

    # 訓練參數
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    success_count = 0
    fail_count = 0

    if args.all or args.verify:
        if verify_models():
            success_count += 1
        else:
            fail_count += 1

    if args.all or args.holdout_raw:
        if train_holdout_raw(args):
            success_count += 1
            # 視覺化
            result_file = str(RESULTS_DIR / "cnn_transformer_lstm_ucddb_raw_holdout_results.json")
            visualize_results(result_file)
        else:
            fail_count += 1

    if args.all or args.holdout_rri:
        if train_holdout_rri(args):
            success_count += 1
            result_file = str(RESULTS_DIR / "cnn_transformer_lstm_ucddb_rri_holdout_results.json")
            visualize_results(result_file)
        else:
            fail_count += 1

    if args.all or args.cv:
        if train_cv_raw(args):
            success_count += 1
            result_file = str(RESULTS_DIR / "cnn_transformer_lstm_ucddb_raw_cv_cv_results.json")
            visualize_results(result_file)
        else:
            fail_count += 1

    if args.all or args.cv_rri:
        if train_cv_rri(args):
            success_count += 1
            result_file = str(RESULTS_DIR / "cnn_transformer_lstm_ucddb_rri_cv_cv_results.json")
            visualize_results(result_file)
        else:
            fail_count += 1

    if args.all or args.external:
        model_path = args.model
        if model_path is None:
            # 自動尋找 holdout raw 模型
            default_model = RESULTS_DIR / "cnn_transformer_lstm_ucddb_raw_holdout.pth"
            if default_model.exists():
                model_path = str(default_model)
            else:
                print("⚠️ 未指定模型，且找不到預設模型，跳過外部驗證。")
        if model_path:
            if external_validation(args, model_path):
                success_count += 1
            else:
                fail_count += 1

    if args.all or args.summary:
        generate_summary()

    if args.visualize and not args.all:
        for result_file in RESULTS_DIR.glob("*_results.json"):
            visualize_results(str(result_file))

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"🏁 實驗完成！成功: {success_count} | 失敗: {fail_count} | 總耗時: {total_elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
