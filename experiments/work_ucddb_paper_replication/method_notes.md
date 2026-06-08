# UCDDB Paper Replication Notes

Goal from the project proposal:
- Build an ECG-based sleep apnea screening model.
- Prefer a wearable-friendly single-lead ECG pipeline.
- Use UCDDB as the main dataset because it has PSG respiratory event annotations.

Reference paper:
- Pham and Moucek, 2025, *Efficient sleep apnea detection using single-lead ECG: A CNN-Transformer-LSTM approach*.

Useful method details extracted from the PDF:
- Input physiology: RRI and R-peak amplitude derived from ECG.
- R peaks: Hamilton detector.
- Segment context: 5 minutes with 900 RRI points and 900 R-amplitude points.
- UCDDB-specific event-window note: the paper also describes 11-second windows with 10-second overlap for second-level UCDDB detection.
- M11 model: CNN-Transformer-LSTM with CNN filters 64, 128, 128; kernel size 7; max-pool size 4; Transformer dimension 128; LSTM hidden size 128.
- Training: Adam, initial learning rate 0.001, ReduceLROnPlateau, early stopping.
- UCDDB holdout in paper: 8:1:1 train/val/test; paper reports both full 25-patient and reduced 21-patient datasets.

Local adaptation:
- I kept UCDDB as the primary dataset.
- I used the existing local UCDDB literature feature pipeline for Hamilton R peaks, 5-minute contexts, 900 RRI + 900 R-amplitude points.
- I added a clean M11-style `CNN -> Transformer -> LSTM` trainer in this work directory.
- For wearable-device claims, grouped record-level CV is the honest protocol. Segment-level random splits are only literature-comparable debugging because windows from the same subject can leak subject-specific patterns.

Important caution:
- The paper's UCDDB accuracy is from holdout-style validation and very likely not directly comparable with grouped record-level CV.
- Existing local grouped-CV results show that UCDDB generalization is much harder than the paper's headline numbers suggest.
- For the final project, report grouped-CV UCDDB results and AHI-like burden estimates, then present Apnea-ECG or segment-level splits only as auxiliary evidence.
