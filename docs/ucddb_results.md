# UCDDB External Test Results

Model: `apnea_parallel_cnn_transformer.pth`

Runner: `ucddb_runner.py`

Label rule:
- Positive = UCDDB respiratory event overlaps a 60-second ECG segment.
- Main result includes both `APNEA-*` and `HYP-*` events, matching the common apnea/hypopnea minute definition.
- Lifecard ECG is resampled from 128 Hz to 100 Hz, then filtered with the same 0.5-45 Hz bandpass and z-score normalization used by `apnea_trainer.py`.

Downloaded data:
- 25 Lifecard EDF files
- 25 respiratory event text files
- Directory: `ucddb/`

## Main Result: APNEA + HYP, Channel 0

Samples: 12,206

Positive minutes: 2,677

Normal minutes: 9,529

| Metric | Value |
|---|---:|
| Accuracy | 0.6275 |
| Precision | 0.2813 |
| Recall | 0.4494 |
| Specificity | 0.6775 |
| F1-score | 0.3460 |

Confusion matrix `[[TN, FP], [FN, TP]]`:

```text
[[6456 3073]
 [1474 1203]]
```

## Channel Sensitivity

| Channel | Accuracy | Precision | Recall | Specificity | F1-score |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6275 | 0.2813 | 0.4494 | 0.6775 | 0.3460 |
| 1 | 0.6091 | 0.2532 | 0.4012 | 0.6675 | 0.3104 |
| 2 | 0.7752 | 0.1474 | 0.0052 | 0.9915 | 0.0101 |

## APNEA-only Check, Channel 0

Samples: 12,206

Positive minutes: 630

Normal minutes: 11,576

| Metric | Value |
|---|---:|
| Accuracy | 0.6513 |
| Precision | 0.0760 |
| Recall | 0.5159 |
| Specificity | 0.6587 |
| F1-score | 0.1325 |

Confusion matrix `[[TN, FP], [FN, TP]]`:

```text
[[7625 3951]
 [ 305  325]]
```

## Interpretation

The Apnea-ECG-trained model does not transfer well to UCDDB without retraining or domain adaptation. The best channel-level F1-score is 0.3460 on channel 0 when hypopnea events are included. This is useful as an external validation result, but it should not be presented as a successful cross-dataset deployment result.
