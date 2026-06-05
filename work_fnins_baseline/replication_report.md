# FNINS 16:972581 Replication Report

## Baseline Being Replicated

Paper: Chen et al. (2022), *A spatio-temporal learning-based model for sleep apnea detection using single-lead ECG signals*, Frontiers in Neuroscience 16:972581.

Implemented in: `work_fnins_baseline/fnins_experiment.py`

Main preprocessing choices:

- Input is not raw ECG. It is 5-minute context around the target minute.
- R peaks are detected by Hamilton, then refined to the local maximum.
- RR intervals and R-peak amplitudes are extracted.
- RR intervals are median-filtered and repaired for anomalous values.
- RR intervals and amplitudes are interpolated to 900 points over the 5-minute context.
- Final model input shape is `(900, 2)`: RR interval plus R-peak amplitude.
- Edge minutes use `edge-policy=clamp` so first/last minutes still produce a 5-minute context. This matches the paper's Apnea-ECG segment count much better than dropping the first/last two minutes.

Dataset protocol:

- Apnea-ECG: release records `a01-a20`, `b01-b05`, `c01-c10`; withheld records `x01-x35` for test. 20% of release records are used for validation.
- UCDDB: exclude `ucddb008`, `ucddb011`, `ucddb013`, `ucddb018`; split remaining segments 8:1:1 into train/validation/test; oversample minority class in training; initialize from the Apnea-ECG model before fine-tuning.

Model protocol:

- FNINS baseline: CNN-BiGRU with 3 spatio-temporal blocks, dot-product attention, dense classifier.
- Our comparison: CNN + Transformer using the same FNINS preprocessing and the same dataset splits.
- Optimizer: Adam, LR 0.001.
- Epochs: 40.
- Batch size: 128.
- Loss: two-class cross entropy, equivalent to softmax binary classification for this implementation.

## Formal Hamilton Results

These are the main results to cite.

| Dataset | Model | Accuracy | Recall | Specificity | Precision | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Apnea-ECG | FNINS CNN-BiGRU baseline | 0.8873 | 0.7823 | 0.9513 | 0.9073 | 0.8401 | 0.9456 |
| UCDDB | FNINS CNN-BiGRU baseline, fine-tuned from Apnea | 0.7688 | 0.5625 | 0.8329 | 0.5114 | 0.5357 | 0.7764 |
| Apnea-ECG | CNN + Transformer | 0.8555 | 0.8169 | 0.8789 | 0.8044 | 0.8106 | 0.9229 |
| UCDDB | CNN + Transformer, fine-tuned from Apnea | 0.7372 | 0.4750 | 0.8187 | 0.4488 | 0.4615 | 0.7358 |

## UCDDB Apnea-Only Relabel Results

This run only labels `APNEA-*` events as positive and excludes `HYP-*` events from the positive class. This appears much closer to the UCDDB prevalence implied by the paper's reported metrics.

| Dataset | Model | Accuracy | Recall | Specificity | Precision | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| UCDDB apnea-only | FNINS CNN-BiGRU baseline, fine-tuned from Apnea | 0.9101 | 0.6271 | 0.9276 | 0.3491 | 0.4485 | 0.8583 |
| UCDDB apnea-only | CNN + Transformer, fine-tuned from Apnea | 0.9259 | 0.3051 | 0.9643 | 0.3462 | 0.3243 | 0.7940 |

Paper comparison:

- Paper UCDDB: Accuracy 0.923, Recall 0.705, Specificity 0.939, Precision 0.467, F1 0.760, AUC about 0.890.
- Our apnea-only CNN-BiGRU: Accuracy 0.910, Recall 0.627, Specificity 0.928, Precision 0.349, F1 0.448, AUC 0.858.
- The accuracy/specificity gap is now small, so the earlier large mismatch was mainly caused by label definition.
- The paper's reported Precision 0.467 and Recall 0.705 imply an F1 of about 0.562 by the standard formula, not 0.760, so the F1 entry in the paper table is likely inconsistent or computed differently.

Apnea-only segment counts:

- Total UCDDB samples after four-record exclusion: 10,117.
- Positive samples: 586.
- Positive ratio: 5.79%.
- Train/validation/test: 8,093 / 1,012 / 1,012 samples.

Segment counts:

- Apnea-ECG Hamilton release train+val: 17,044 segments, close to the paper's 17,045.
- Apnea-ECG Hamilton withheld test: 17,213 usable segments from the local `event-2-answers` labels after low-beat windows are skipped.
- UCDDB Hamilton/clamp after excluding four records: train 8,093, validation 1,012, test 1,012.

Interpretation:

- The replicated CNN-BiGRU baseline currently beats the CNN+Transformer idea on both datasets.
- On Apnea-ECG, CNN+Transformer is close but still lower: F1 0.8106 vs 0.8401.
- On UCDDB, the gap is larger: F1 0.4615 vs 0.5357.
- This suggests the FNINS recurrent block is doing useful work on RRI/R-peak temporal dynamics; our Transformer needs more tuning before it is a credible improvement.

## Fast QRS Results

These runs used local `.qrs` annotations for Apnea-ECG to accelerate early iteration. They are useful as a debugging comparison, but the Hamilton table above is the formal baseline.

| Dataset | Model | Accuracy | Recall | Specificity | Precision | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Apnea-ECG | CNN-BiGRU, QRS accelerated | 0.8472 | 0.7712 | 0.8937 | 0.8160 | 0.7930 | 0.9154 |
| UCDDB | CNN-BiGRU from QRS Apnea model | 0.7490 | 0.4417 | 0.8446 | 0.4690 | 0.4549 | 0.7420 |
| Apnea-ECG | CNN + Transformer, QRS accelerated | 0.8331 | 0.6719 | 0.9316 | 0.8572 | 0.7533 | 0.9109 |
| UCDDB | CNN + Transformer from QRS Apnea model | 0.7648 | 0.4375 | 0.8666 | 0.5048 | 0.4688 | 0.7298 |

## Key Artifacts

- Main script: `work_fnins_baseline/fnins_experiment.py`
- Extracted PDF text: `work_fnins_baseline/fnins-16-972581_text_clean.txt`
- Formal Hamilton outputs: `work_fnins_baseline/outputs/hamilton_clamp_m4/`
- QRS accelerated outputs: `work_fnins_baseline/outputs/qrs_clamp_m4/`
- Hamilton Apnea cache: `work_fnins_baseline/cache_hamilton_clamp_m4/`
- UCDDB Hamilton/clamp cache: `work_fnins_baseline/cache_ucddb_clamp_m4/`

## Next Moves

The baseline is now established. The next improvement attempts should focus on the Transformer side:

- Add positional encoding to CNN+Transformer.
- Try class-weighted loss or focal loss for UCDDB.
- Tune the decision threshold on validation F1 instead of fixed 0.5.
- Try 1-3 Transformer encoder layers and wider feed-forward sizes.
- Repeat the random split several times, because UCDDB is small and unstable.
