# Paper Protocol Audit: Why Their UCDDB Scores Are So High

Date: 2026-06-01

Paper: Pham and Moucek, 2025, *Efficient sleep apnea detection using single-lead ECG: A CNN-Transformer-LSTM approach*.

## Main Finding

Their UCDDB evaluation is not a subject-held-out test.

The paper says UCDDB is split into train, validation, and test in an 8:1:1 ratio. Their GitHub code then loads `*_ecg_train.mat`, `*_ecg_valid.mat`, and `*_ecg_test.mat` for each UCDDB record and appends all records into global train/valid/test arrays.

That means the same patient/record contributes windows to train, validation, and test.

For wearable-device generalization, this is much easier than our grouped CV, where entire subjects are held out.

## Evidence From Paper

UCDDB setup:
- 25 PSG records.
- ECG sampled at 128 Hz.
- Full dataset: 25/25 patients.
- Reduced dataset: excludes `ucddb008`, `ucddb011`, `ucddb013`, `ucddb018`.
- Training and validation are balanced by oversampling SA events.

UCDDB preprocessing:
- 11-second windows.
- 10-second overlap, so stride is 1 second.
- Label is based on the state of the 2nd second in the window.
- Detection resolution is reported as 1 second.

Training:
- UCDDB uses hold-out validation.
- Split ratio is 8:1:1.
- The paper does not explicitly state that the UCDDB split is patient-level.

Reported UCDDB results:
- CNN-LSTM full 25/25: Acc 98.69%, AUC 0.999.
- CNN-LSTM reduced 21/25: Acc 99.61%, AUC 0.999.
- CNN-Transformer-LSTM full 25/25: Acc 98.34%, AUC 0.995.
- CNN-Transformer-LSTM reduced 21/25: Acc 99.37%, AUC 0.999.

## Evidence From Their GitHub

Their UCDDB model code:
- Defines a `list_string` of records.
- For every record, loads:
  - `{record}_ecg_valid.mat`
  - `{record}_valid_labels.mat`
  - `{record}_ecg_test.mat`
  - `{record}_test_labels.mat`
  - `{record}_ecg_train.mat`
  - `{record}_train_labels.mat`
- Appends all records' train/valid/test windows into global arrays.

Their preprocessing README says:
- `splitting_datasets` normalizes the signal record, windows the signals, and splits them into training, validation, and test set.
- Minority upsampling is carried out on training and validation.

This strongly indicates a per-record/window split, not a subject-level split.

## Why This Inflates UCDDB Performance

1. Same-subject leakage:
- The model sees ECG morphology, noise, lead characteristics, heart-rate range, and event distribution from the same patient during training.
- The test set is then mostly asking whether it can classify another window from a known patient.

2. Overlapping-window similarity:
- UCDDB windows are 11 seconds long with 10 seconds overlap.
- Adjacent windows share 10/11 of their raw ECG content.
- If split after windowing, train/valid/test can contain nearly identical neighboring windows.

3. Class balancing:
- They oversample minority apnea windows in train and validation.
- This helps learning and validation selection, but it does not solve true cross-subject generalization.

4. Reduced dataset:
- Removing four no-apnea records makes the task less imbalanced and more homogeneous.
- Their best CNN-Transformer-LSTM result is on the reduced 21-subject set.

## Comparison With Our Protocol

Our grouped CV:
- Entire UCDDB records/subjects are held out.
- No test subject appears in training.
- This is much closer to a real wearable device being used by a new person.

Therefore, our lower metrics are not just poor modeling. They reflect a harder and more honest evaluation.

## Practical Conclusion

For the final project:
- We can cite their method as inspiration.
- We should not compare our grouped-CV result directly against their 98-99% UCDDB accuracy.
- We can include a sentence like:

  "The reference paper reports high UCDDB accuracy under an 8:1:1 hold-out split of windowed data. In this project, we use record-level grouped cross-validation to better estimate performance on unseen wearable-device users, which is a stricter protocol."

