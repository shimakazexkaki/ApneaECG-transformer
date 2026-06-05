# Sleep Apnea Detection Experiments

This repository contains the final project code and selected artifacts for ECG-based sleep apnea detection.

## Current Mainline

The final usable pipeline is in:

- `work_fnins_baseline/fnins_experiment.py`
- `work_fnins_baseline/replication_report.md`
- `work_fnins_baseline/commands.md`

The implemented baseline follows Chen et al. (2022), Frontiers in Neuroscience 16:972581:

- 5-minute ECG context around each target minute
- Hamilton R-peak detection
- RR interval and R-peak amplitude features
- Cubic interpolation to `(900, 2)`
- CNN-BiGRU baseline with attention
- CNN + Transformer comparison

## Data

The extracted datasets are committed for project teammates:

- `apnea-ecg/`
- `ucddb/`

The original archive `apnea-ecg.zip` is not committed because it is larger than GitHub's normal 100MB single-file limit. Large generated folders such as `aligned_data/`, `outputs/`, and feature caches are ignored.

## Main Results

See:

- `work_fnins_baseline/replication_report.md`

Selected model checkpoints and result JSON files are committed under:

- `work_fnins_baseline/outputs/hamilton_clamp_m4/`
- `work_fnins_baseline/outputs/ucddb_apneaonly/`

## Reproduce

Use the project conda environment:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe'
```

Full commands are listed in:

- `work_fnins_baseline/commands.md`

## Notes

The UCDDB results differ substantially depending on whether hypopnea events are treated as positive. The paper's reported UCDDB metrics are much closer to an apnea-only labeling setup, so both the original hypopnea-inclusive run and the apnea-only relabel run are documented in the report.
