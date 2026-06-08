# Commands

All commands use the requested conda environment:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_ucddb_paper_replication\paper_cnn_transformer_lstm_trainer.py --help
```

Smoke run:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_ucddb_paper_replication\paper_cnn_transformer_lstm_trainer.py `
  --experiment-name smoke_paper_m11_ch0_ch2 `
  --protocol literature `
  --records ucddb002 ucddb003 ucddb005 ucddb006 `
  --epochs 1 `
  --samples-per-epoch 256 `
  --batch-size 64 `
  --channels 0 2 `
  --no-progress
```

Honest grouped-CV first fold:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_ucddb_paper_replication\paper_cnn_transformer_lstm_trainer.py `
  --experiment-name paper_m11_ch0_ch2_cv_fold0 `
  --protocol cv `
  --folds 0 `
  --epochs 10 `
  --samples-per-epoch 4096 `
  --batch-size 128 `
  --channels 0 2 `
  --no-progress
```

Runs completed for this report:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_ucddb_paper_replication\paper_cnn_transformer_lstm_trainer.py `
  --experiment-name paper_m11_ch0_ch2_lit_20ep `
  --protocol literature `
  --epochs 20 `
  --samples-per-epoch 4096 `
  --batch-size 128 `
  --channels 0 2 `
  --cache-dir aligned_data\ucddb_literature_features `
  --no-progress
```

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_ucddb_paper_replication\paper_cnn_transformer_lstm_trainer.py `
  --experiment-name paper_m11_ch0_ch2_cv_3ep `
  --protocol cv `
  --epochs 3 `
  --samples-per-epoch 4096 `
  --batch-size 128 `
  --channels 0 2 `
  --cache-dir aligned_data\ucddb_literature_features `
  --no-progress
```

Full grouped CV:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_ucddb_paper_replication\paper_cnn_transformer_lstm_trainer.py `
  --experiment-name paper_m11_ch0_ch2_cv `
  --protocol cv `
  --epochs 40 `
  --samples-per-epoch 0 `
  --batch-size 128 `
  --channels 0 2 `
  --no-progress
```
