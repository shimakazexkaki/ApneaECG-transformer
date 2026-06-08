"""UCDDB-only segment split(M11)兩個消融,合在一起跑:

實驗 A（標註規則）：掃 --label-overlap-sec ∈ {5,15,30}（一分鐘需重疊多少秒才算 apnea）。
實驗 B（更平衡的操作點）：每一版都自動報「F1-最佳閾值」vs 預設(balanced-acc)閾值的 Acc/Rec/Prec/F1。

全部 segment split(--protocol literature, 8:1:1),序列執行避免搶資源。
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable
TRAINER = HERE / "paper_cnn_transformer_lstm_trainer.py"
OVERLAPS = [5, 15, 30]
EPOCHS = 40


def run(overlap):
    name = f"m11_seg_ov{overlap}"
    cmd = [PY, str(TRAINER), "--protocol", "literature", "--label-overlap-sec", str(overlap),
           "--experiment-name", name, "--epochs", str(EPOCHS), "--no-progress"]
    print(f"\n>>> overlap={overlap}s  {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT))
    p = HERE / "outputs" / name / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    r = d["results"][0]
    sc, test = r["sample_counts"], r["test"]
    prevalence = sc["test_positive"] / max(sc["test"], 1)
    return {
        "overlap": overlap,
        "test_windows": sc["test"], "test_positive": sc["test_positive"],
        "prevalence": prevalence,
        "auc": test["threshold_val"]["roc_auc"],
        "default": test["threshold_val"],          # balanced-acc 閾值
        "f1opt": test["threshold_f1_val"],         # F1-最佳閾值(val 上挑)
        "f1_threshold": test.get("f1_threshold_val"),
    }


def fmt(m):
    return f"Acc={m['accuracy']:.3f} Rec={m['recall']:.3f} Prec={m['precision']:.3f} F1={m['f1']:.3f} Spec={m['specificity']:.3f}"


def main():
    rows = []
    for ov in OVERLAPS:
        res = run(ov)
        if res:
            rows.append(res)
    (HERE / "outputs" / "seg_label_experiments.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    print("\n========== UCDDB segment-split：overlap 標註門檻 × 操作點 ==========", flush=True)
    for r in rows:
        print(f"\n--- overlap {r['overlap']}s | 陽性率={r['prevalence']:.3f} ({r['test_positive']}/{r['test_windows']}) | AUC={r['auc']:.4f} ---")
        print(f"  預設(bacc)閾值 : {fmt(r['default'])}")
        print(f"  F1-最佳閾值({r['f1_threshold']:.3f}): {fmt(r['f1opt'])}")
    print("\nSaved: outputs/seg_label_experiments.json", flush=True)


if __name__ == "__main__":
    main()
