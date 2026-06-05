\# GitHub Cleanup Plan

## Recommended Minimal GitHub Push

This is the cleanest set for the current final-project direction: FNINS baseline replication plus our CNN+Transformer comparison.

- `.gitignore`
- `GITHUB_CLEANUP_PLAN.md`
- `work_fnins_baseline/fnins_experiment.py`
- `work_fnins_baseline/replication_report.md`
- `work_fnins_baseline/commands.md`

These three `work_fnins_baseline` files are enough to reproduce the current main results, assuming the user downloads Apnea-ECG and UCDDB locally.

Suggested add command:

```powershell
git add .gitignore GITHUB_CLEANUP_PLAN.md work_fnins_baseline/fnins_experiment.py work_fnins_baseline/replication_report.md work_fnins_baseline/commands.md
```

## Optional To Push

Push these only if you want to preserve the earlier research path and paper audit.

- `work_ucddb_paper_replication/paper_protocol_audit.md`
- `work_ucddb_paper_replication/replication_report.md`
- `work_ucddb_paper_replication/method_notes.md`
- `work_ucddb_paper_replication/project_update_for_ppt.md`
- `work_ucddb_paper_replication/commands.md`
- `work_ucddb_paper_replication/paper_cnn_transformer_lstm_trainer.py`

These explain why the other paper's very high UCDDB scores are likely caused by a different split/protocol.

## Optional Research Scripts

These are useful if the repo should show the whole experiment history, but they are not required for the current FNINS baseline deliverable.

- `ucddb_literature_features.py`
- `ucddb_literature_train_common.py`
- `literature_bigru_trainer.py`
- `transformer_rri_trainer.py`
- `hybrid_transformer_trainer.py`
- `ucddb_highres_trainer.py`
- `ucddb_highres_grouped_cv.py`
- `summarize_literature_experiments.py`
- `combine_cv_summaries.py`
- `diagnose_ucddb_signal_quality.py`
- `diagnose_ucddb_cv_results.py`
- `diagnose_highres_burden.py`

These support the previous UCDDB grouped-CV, high-resolution, and literature-feature experiments.

## Do Not Push

These are local-only files and are now covered by `.gitignore`.

- `apnea-ecg/`
- `ucddb/`
- `apnea-ecg.zip`
- `aligned_data/`
- `outputs/`
- `work_fnins_baseline/cache*/`
- `work_fnins_baseline/outputs/`
- `work_ucddb_paper_replication/cache/`
- `work_ucddb_paper_replication/outputs/`
- `cnn_transformer_lstm/results/`
- `*.pth`, `*.pt`
- `*.npz`
- `*.pdf`
- `*.pptx`
- `pptx_content.txt`
- `__pycache__/`
- `.vscode/`

## Safe To Delete If You Want Space

These are generated or obsolete local artifacts. Deleting them will not remove source code, but some experiments would need to recompute caches or retrain models.

Definitely safe:

- `__pycache__/`
- `work_fnins_baseline/__pycache__/`
- `cnn_transformer_lstm/__pycache__/`
- `work_ucddb_paper_replication/__pycache__/`
- `work_fnins_baseline/cache_smoke/`
- `work_fnins_baseline/cache_smoke_reuse/`
- `work_fnins_baseline/cache_qrs/`
- `work_fnins_baseline/cache_qrs_clamp/`
- `cnn_transformer_lstm/results/`
- root old checkpoints: `apnea_1dcnn.pth`, `apnea_cnn_transformer.pth`, `apnea_parallel_cnn_transformer.pth`, `apnea_resnet_transformer.pth`

Can delete after you are done rerunning:

- `work_fnins_baseline/cache_qrs_clamp_m4/`
- `work_fnins_baseline/cache_hamilton_clamp_m4/`
- `work_fnins_baseline/cache_ucddb_clamp_m4/`
- `work_fnins_baseline/outputs/`
- `work_ucddb_paper_replication/cache/`
- `work_ucddb_paper_replication/outputs/`
- `aligned_data/`
- `outputs/`

Keep locally if you still need to rerun without waiting:

- `apnea-ecg/`
- `ucddb/`
- `work_fnins_baseline/cache_hamilton_clamp_m4/`
- `work_fnins_baseline/cache_ucddb_clamp_m4/`

## Current Size Hotspots

- `aligned_data/`: about 2.9 GB
- `work_fnins_baseline/`: about 853 MB, mostly caches and model outputs
- `apnea-ecg/`: about 581 MB
- `ucddb/`: about 537 MB
- `outputs/`: about 193 MB
- `work_ucddb_paper_replication/`: about 131 MB
- `apnea-ecg.zip`: about 313 MB

The source code itself is tiny. The disk usage is almost entirely datasets, caches, and checkpoints.
