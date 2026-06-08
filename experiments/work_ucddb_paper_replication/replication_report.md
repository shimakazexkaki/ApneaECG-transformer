# UCDDB CNN-Transformer-LSTM Replication Report

Date: 2026-06-01

Environment:
- Python: `C:\Users\a2003\miniconda3\envs\apnea\python.exe`
- GPU detected: NVIDIA GeForce RTX 3070 Laptop GPU
- Work directory: `work_ucddb_paper_replication`

## What Was Built

I added `paper_cnn_transformer_lstm_trainer.py`, a clean local replication wrapper for the Pham and Moucek 2025 paper's M11-style architecture:

`RRI + R-peak amplitude -> CNN -> Transformer -> LSTM -> classifier`

The script reuses the existing UCDDB feature pipeline:
- Hamilton R-peak detection via BioSPPy.
- 5-minute context windows.
- 900 interpolated RRI points and 900 R-peak-amplitude points.
- UCDDB channel 0 + channel 2 by default.
- Outputs saved under `work_ucddb_paper_replication/outputs`.

The implementation is intentionally separate from the existing project files.

## Results

| Run | Protocol | Epochs | BAcc | AUC | F1 | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| `smoke_paper_m11_ch0_ch2` | 4-record segment split | 1 | 0.4874 | 0.4715 | 0.3237 | 0.4948 | 0.4801 |
| `paper_m11_ch0_ch2_lit_5ep` | full UCDDB segment split | 5 | 0.5909 | 0.6605 | 0.3552 | 0.3654 | 0.8163 |
| `paper_m11_ch0_ch2_lit_20ep` | full UCDDB segment split | 20 | 0.6389 | 0.6982 | 0.4275 | 0.5223 | 0.7555 |
| `paper_m11_ch0_ch2_cv_3ep` | full UCDDB grouped 5-fold CV | 3 | 0.5318 +/- 0.0139 | 0.5732 +/- 0.0202 | 0.2846 +/- 0.1082 | 0.5371 +/- 0.3509 | 0.5266 +/- 0.3403 |

Metrics above use validation-selected thresholds.

## Comparison With Existing Local Runs

Existing useful baselines already in `outputs/`:

| Existing run | Protocol | BAcc | AUC | Note |
|---|---|---:|---:|---|
| `hybrid_transformer_ch0_ch2_hamilton_lit` | segment split | 0.7302 | 0.8063 | Best existing relaxed split among the RRI/raw hybrid runs. |
| `hybrid_transformer_ch0_ch2_hamilton_cv` | grouped CV | 0.5123 +/- 0.0275 | 0.5010 +/- 0.0668 | Honest cross-subject result for 5-minute hybrid model. |
| `ucddb_highres_ch0_ch2_overlap5_cnn_transformer_smooth_5fold` | grouped CV, minute level | 0.5355 +/- 0.0355 | 0.5522 +/- 0.0474 | Best direction for wearable-style event/minute output. |

## Interpretation

The paper-style M11 model can learn useful RRI/R-amplitude signal under a relaxed segment-level split. The 20 epoch run reached AUC 0.6982, so the replication path is functional.

For the actual wearable-device goal, the grouped-CV result is more important. M11 grouped CV reached AUC 0.5732 and BAcc 0.5318 after only 3 epochs. This is close to the existing high-resolution UCDDB grouped-CV range, but it does not yet solve cross-subject generalization.

This supports the current project direction:
- Use UCDDB as the primary benchmark.
- Report grouped record-level CV for final claims.
- Treat paper-like segment splits as debugging/literature comparison only.
- Keep the high-resolution 11-second UCDDB pipeline as the main wearable-oriented path because it matches UCDDB event annotations better than 5-minute labels.

## Recommended Next Work

1. Run `paper_m11_ch0_ch2_cv` for longer training, such as 20-40 epochs, using the existing feature cache.
2. Add an M11 variant to the existing high-resolution 11-second UCDDB pipeline, because the PDF explicitly uses 11-second windows with 10-second overlap for UCDDB.
3. Compare full 25-patient and reduced 21-patient UCDDB settings, but present the 25-patient grouped-CV result as the main wearable claim.
4. Report both minute-level classification and AHI-like apnea burden per recording.
5. Use Apnea-ECG only as auxiliary/pretraining evidence, not as the primary result.

## Reproducible Commands

The exact command templates are stored in `commands.md`.
