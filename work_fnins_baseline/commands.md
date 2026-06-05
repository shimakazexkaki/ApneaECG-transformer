# Commands

All commands use:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe'
```

## Inspect / Build Caches

Formal Hamilton Apnea-ECG cache:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_fnins_baseline\fnins_experiment.py --dataset apnea_ecg --mode inspect --apnea-rpeak-source hamilton --cache-dir work_fnins_baseline\cache_hamilton_clamp_m4 --output-dir work_fnins_baseline\outputs\hamilton_clamp_m4 --num-workers 6
```

UCDDB Hamilton/clamp cache:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_fnins_baseline\fnins_experiment.py --dataset ucddb --mode inspect --cache-dir work_fnins_baseline\cache_ucddb_clamp_m4 --output-dir work_fnins_baseline\outputs\hamilton_clamp_m4 --num-workers 4
```

## Formal Hamilton Baseline

Apnea-ECG CNN-BiGRU:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_fnins_baseline\fnins_experiment.py --dataset apnea_ecg --model-type cnn_bigru --epochs 40 --batch-size 128 --apnea-rpeak-source hamilton --cache-dir work_fnins_baseline\cache_hamilton_clamp_m4 --output-dir work_fnins_baseline\outputs\hamilton_clamp_m4 --experiment-name fnins_cnn_bigru_apnea_hamilton_clamp --log-every 5
```

UCDDB CNN-BiGRU fine-tune:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_fnins_baseline\fnins_experiment.py --dataset ucddb --model-type cnn_bigru --epochs 40 --batch-size 128 --cache-dir work_fnins_baseline\cache_ucddb_clamp_m4 --output-dir work_fnins_baseline\outputs\hamilton_clamp_m4 --experiment-name fnins_cnn_bigru_ucddb_from_apnea_hamilton_clamp --pretrained-path work_fnins_baseline\outputs\hamilton_clamp_m4\fnins_cnn_bigru_apnea_hamilton_clamp.pth --num-workers 4 --log-every 5
```

## CNN + Transformer

Apnea-ECG:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_fnins_baseline\fnins_experiment.py --dataset apnea_ecg --model-type cnn_transformer --epochs 40 --batch-size 128 --apnea-rpeak-source hamilton --cache-dir work_fnins_baseline\cache_hamilton_clamp_m4 --output-dir work_fnins_baseline\outputs\hamilton_clamp_m4 --experiment-name fnins_cnn_transformer_apnea_hamilton_clamp --log-every 5
```

UCDDB fine-tune:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_fnins_baseline\fnins_experiment.py --dataset ucddb --model-type cnn_transformer --epochs 40 --batch-size 128 --cache-dir work_fnins_baseline\cache_ucddb_clamp_m4 --output-dir work_fnins_baseline\outputs\hamilton_clamp_m4 --experiment-name fnins_cnn_transformer_ucddb_from_apnea_hamilton_clamp --pretrained-path work_fnins_baseline\outputs\hamilton_clamp_m4\fnins_cnn_transformer_apnea_hamilton_clamp.pth --num-workers 4 --log-every 5
```

## UCDDB Apnea-Only Runs

Inspect / relabel from existing UCDDB feature cache:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_fnins_baseline\fnins_experiment.py --dataset ucddb --mode inspect --no-include-hypopnea --cache-dir work_fnins_baseline\cache_ucddb_clamp_m4 --output-dir work_fnins_baseline\outputs\ucddb_apneaonly --num-workers 4
```

CNN-BiGRU apnea-only fine-tune:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_fnins_baseline\fnins_experiment.py --dataset ucddb --model-type cnn_bigru --epochs 40 --batch-size 128 --no-include-hypopnea --cache-dir work_fnins_baseline\cache_ucddb_clamp_m4 --output-dir work_fnins_baseline\outputs\ucddb_apneaonly --experiment-name fnins_cnn_bigru_ucddb_apneaonly_from_apnea_hamilton_clamp --pretrained-path work_fnins_baseline\outputs\hamilton_clamp_m4\fnins_cnn_bigru_apnea_hamilton_clamp.pth --num-workers 4 --log-every 5
```

CNN + Transformer apnea-only fine-tune:

```powershell
& 'C:\Users\a2003\miniconda3\envs\apnea\python.exe' work_fnins_baseline\fnins_experiment.py --dataset ucddb --model-type cnn_transformer --epochs 40 --batch-size 128 --no-include-hypopnea --cache-dir work_fnins_baseline\cache_ucddb_clamp_m4 --output-dir work_fnins_baseline\outputs\ucddb_apneaonly --experiment-name fnins_cnn_transformer_ucddb_apneaonly_from_apnea_hamilton_clamp --pretrained-path work_fnins_baseline\outputs\hamilton_clamp_m4\fnins_cnn_transformer_apnea_hamilton_clamp.pth --num-workers 4 --log-every 5
```
