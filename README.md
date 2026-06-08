# 基於單導程 ECG 的睡眠呼吸中止症(SA)篩檢

資料探勘專題。用單導程心電圖(ECG)→ 心率變異(HRV)/RRI 特徵 → 深度學習 / 古典 ML,做睡眠呼吸中止的逐分鐘偵測與整晚 AHI 篩檢。
主資料集:**Apnea-ECG(乾淨 benchmark,與 Polar H10 胸帶波形相似)**、**UCDDB**、**MESA(大世代對照)**。

> 📊 **完整結果先看 [`RESULTS.md`](RESULTS.md)** —— 每個模型的架構 × 流程 × 完整指標(AUC/F1/Acc/Precision/Recall)。

---

## 0. 環境設定

```powershell
# 用 conda 建環境(Python 3.11)
conda create -n apnea python=3.11
conda activate apnea
pip install torch numpy scipy scikit-learn wfdb pyedflib biosppy tqdm
# (MESA 下載才需要)pip install sleepecg pip-system-certs
```

本專案開發用的直譯器:`C:\Users\<user>\miniconda3\envs\apnea\python.exe`。下面指令以 `python` 代表此直譯器。

- numpy 2.x:用 `np.trapezoid`(非 `np.trapz`)。
- GPU 選用(CUDA);沒有 GPU 加 `--cpu`。

---

## 1. 資料集(放哪 / 怎麼來)

| 資料集 | 位置 | 來源 | 說明 |
|---|---|---|---|
| **Apnea-ECG** | `apnea-ecg/`(repo 內) | [PhysioNet apnea-ecg](https://physionet.org/content/apnea-ecg/) | 70 筆單導 ECG(release a/b/c + withheld x01–x35),逐分鐘 apnea 標註 |
| **UCDDB** | `ucddb/`(repo 內) | [PhysioNet ucddb](https://physionet.org/content/ucddb/) | 25 位整夜 PSG(Lifecard EDF + 呼吸事件 `*_respevt.txt`) |
| **MESA** | `D:/mesa/`(repo 外) | [NSRR mesa](https://sleepdata.org/datasets/mesa) | ~2000 位整夜 PSG;需 NSRR 帳號 + DAUA,用 `download_mesa.py` 下載 |

> Apnea-ECG / UCDDB 已隨 repo 提供。MESA 因授權與體積放在 repo 外(`D:/mesa`),需自行下載(見 §5)。

---

## 2. 共用訓練流程(所有模型的前處理 → 特徵 → 訓練 → 評估)

這是整個專案的核心 pipeline,實作在 `lib/`,被各 experiment 重用:

**(1) 前處理(原始 ECG → 心跳序列)** — `lib/ucddb_highres_trainer.read_ucddb_signal` / `lib/mesa_features.read_mesa_ecg`
1. 讀 EDF 的 ECG channel(UCDDB ch0/2;Apnea-ECG lead V2;MESA `EKG`)。
2. **重採樣 100 Hz**(`scipy.resample_poly`)。
3. **Butterworth 帶通 0.5–45 Hz**(`lib/apnea_trainer.butter_bandpass_filter`,去基線漂移/工頻)。

**(2) R-peak 偵測** — `lib/ucddb_literature_features.detect_rpeaks`
- BioSPPy **Hamilton segmenter** + `correct_rpeaks`(備援:wfdb xqrs / scipy)。

**(3) 特徵(序列,非標量)** — `lib/ucddb_literature_features`
- 由 R-peak 算 **RRI(clip 0.30–2.50 s)** 與 **R-peak 振幅**。
- 每個目標分鐘取 **5 分鐘 context**,RRI 與振幅各內插成 **900 點** → 輸入張量 **(900, 2)**。
- **正規化**:`_interpolate_context` 內逐窗 z-score(`norm_mode=window`);MESA 篩檢用整夜 z-score(`norm_mode=recording`,保留個體水準)。

**(4) 標籤** — `minute_labels` / `build_segment_features_from_events`
- 逐分鐘:該分鐘與呼吸事件(apnea+hypopnea)重疊 **>5 s** 為陽性。
- (MESA 進階)10 秒段「完整落在事件內=陽性、零重疊=陰性、部分丟棄」→ 較乾淨但稀疏。

**(5) 模型 + 訓練** — `lib/ucddb_literature_train_common`
- 模型:`CNNTransformer` / `CNNBiGRUAttention` / M11 `PaperCNNTransformerLSTM`(架構見 `RESULTS.md §0`)。
- AdamW + AMP、loss∈{CE / weighted-CE / focal}、`WeightedRandomSampler` 過採樣、ReduceLROnPlateau、early stopping、grad-clip 1.0。

**(6) 評估** — `train_one_split` / `best_threshold`
- 協定:**segment split**(8:1:1,⚠️洩漏、樂觀)或 **record-level grouped CV**(整人 held out,✅誠實)。
- 在 val 挑閾值,test 報 Acc/Recall/Precision/F1/Spec/AUC;整晚聚合報 per-recording AHI 篩檢。

---

## 3. 怎麼跑各個模型(執行指令)

### A. Apnea-ECG(主結果 — CNN+Transformer + per-recording AHI 篩檢)
```powershell
# 訓練:release 訓練 → withheld x01-x35 測試(標準協定)
python experiments/work_fnins_baseline/fnins_experiment.py --dataset apnea_ecg `
  --model-type cnn_transformer --apnea-dir apnea-ecg `
  --cache-dir experiments/work_fnins_baseline/cache `
  --output-dir experiments/work_fnins_baseline/outputs `
  --experiment-name apnea_cnntf --epochs 40
# per-recording AHI 篩檢(用上面訓練好的權重)
python experiments/work_fnins_baseline/apnea_screening.py
```
→ per-minute AUC 0.931 / F1 0.820;**per-recording 篩檢 97.1% / AHI MAE 5.37**。

### B. UCDDB —— HRV+CVHR ensemble(誠實基準最佳)與深度模型
```powershell
python models/hrv_ensemble.py        # HRV+CVHR ensemble(grouped CV AUC 0.570,最佳且可解釋)
python models/hrv_mlp.py             # HRV 深度 MLP 對照
python models/cnn_transformer.py     # 提案模型 CNN+Transformer(UCDDB 從頭 + focal)
python models/cnn_bigru.py           # FNINS baseline CNN-BiGRU
python models/screening.py           # 受試者層級 AHI 篩檢
```
(`models/` 是一鍵啟動器,免打參數;細節見 [`models/README.md`](models/README.md))

### C. M11 論文複現(Pham & Moucek 2025)+ 洩漏示範
```powershell
# 5 分鐘 RRI M11:segment split(洩漏,0.79)vs grouped CV(誠實,0.54)
python experiments/work_ucddb_paper_replication/paper_cnn_transformer_lstm_trainer.py --protocol literature --experiment-name paper_m11_seg --epochs 40
python experiments/work_ucddb_paper_replication/paper_cnn_transformer_lstm_trainer.py --protocol cv --experiment-name paper_m11_cv --epochs 40
```
→ 重現論文 98% 的「視窗洩漏」機制,見 `m11_protocol_results.md`。

### D. MESA(大世代對照,需先下載 — 見 §5)
```powershell
python experiments/work_mesa_transfer/mesa_to_ucddb_trainer.py --mesa-limit 74 --experiment-name mesa2ucddb   # 跨資料集 train-MESA→test-UCDDB
python experiments/work_mesa_transfer/mesa_internal_trainer.py --label-mode segment --norm-mode recording --use-all-available --mesa-limit 149 --experiment-name mesa149  # MESA-internal grouped CV
python experiments/work_mesa_transfer/mesa_screening.py --experiment-name mesa149 --label-mode segment --norm-mode recording --use-all-available --mesa-limit 149  # per-subject AHI 篩檢
```

---

## 4. 目錄與每個檔案的用途

### `lib/` — 共用核心庫(被 import,**不要直接執行**)
| 檔案 | 用途 |
|---|---|
| `apnea_trainer.py` | Apnea-ECG 訓練工具 + `butter_bandpass_filter`(訊號帶通) |
| `mixed_trainer.py` | `normalize_segment`/`normalize_matrix`(逐窗 z-score)等共用工具 |
| `ucddb_runner.py` | 讀 UCDDB EDF、`parse_respiratory_events`(解析呼吸事件)、`available_record_ids` |
| `ucddb_trainer.py` | UCDDB 一般訓練工具 |
| `ucddb_highres_trainer.py` | 11 秒高解析視窗版;`read_ucddb_signal`(EDF→100Hz→帶通) |
| `ucddb_literature_features.py` | **核心特徵管線**:`detect_rpeaks`、RRI/振幅、5分鐘→(900,2)、`minute_labels`、`build_segment_features_from_events`(乾淨標籤) |
| `ucddb_literature_train_common.py` | **核心訓練庫**:模型定義(CNNTransformer/CNNBiGRU 等)、`train_one_split`、`records_to_arrays`、`best_threshold`、grouped CV folds、指標 |
| `mesa_features.py` | **MESA reader**:`read_mesa_ecg`(EDF EKG→100Hz)、`parse_mesa_events`(NSRR XML)、段特徵建構 |

### `models/` — 一鍵啟動器(同學從這裡開始)
| 檔案 | 用途 |
|---|---|
| `cnn_transformer.py` / `cnn_bigru.py` / `cnn_transformer_segment.py` | 提案 CNN+Transformer / CNN-BiGRU / 段版,免參數直跑 |
| `hrv_ensemble.py` / `hrv_mlp.py` | HRV+CVHR ensemble(最佳)/ HRV MLP |
| `screening.py` | 受試者層級 AHI 篩檢 |
| `README.md` | 各啟動器說明 |

### `experiments/work_fnins_baseline/` — Apnea-ECG + UCDDB(CNN+Transformer / BiGRU)
| 檔案 | 用途 |
|---|---|
| `fnins_experiment.py` | **主程式**:Apnea-ECG(release/withheld)+ UCDDB 訓練,CNN-BiGRU/CNN+Transformer,支援 `--loss`、SQI gating |
| `apnea_screening.py` | **Apnea-ECG per-recording AHI 篩檢**(對標 Liu 2023) |
| `cnn_transformer_grouped_cv.py` | UCDDB record-level grouped CV harness |
| `cnn_transformer_tune.py` | CNN+Transformer 超參數搜尋 |
| `*_report.md` / `commands.md` | 複現報告與指令 |

### `experiments/work_ucddb_hrv/` — HRV 特徵法(誠實基準)
| 檔案 | 用途 |
|---|---|
| `hrv_features.py` | **41 個手工 HRV/CVHR/EDR 標量特徵**(5分鐘 context) |
| `hrv_grouped_cv.py` | grouped CV + 指標 + `screening_report`(被多處重用) |
| `hrv_mlp_cv.py` | HRV 深度 MLP |
| `hrv_imbalance_test.py` | 不平衡處理對照(SMOTE/欠採樣/EasyEnsemble…) |
| `diagnose_signal.py` | 訊號可分性診斷 |
| **`model_comparison.md`** ⭐ | **各協定×各模型完整對照表** |
| **`hrv_vs_deeplearning.md`** ⭐ | **HRV vs 深度學習:何時用哪個、為何 DL 失敗時 HRV 較好** |
| `results_report.md` | HRV 結果報告 |

### `experiments/work_ucddb_paper_replication/` — M11 複現 + 洩漏稽核
| 檔案 | 用途 |
|---|---|
| `paper_cnn_transformer_lstm_trainer.py` | **M11 CNN-Transformer-LSTM**(5分鐘 RRI);`--protocol literature/cv` |
| `paper_m11_highres.py` | M11 on 11 秒原始 ECG 視窗(密度→洩漏示範) |
| `paper_m11_tune.py` | M11 nested grouped-CV 調參 |
| `run_seg_label_experiments.py` | overlap 標註門檻 × 操作點消融 |
| **`m11_protocol_results.md`** ⭐ | **重現論文 98% 的「視窗洩漏」機制** |
| `method_notes.md` / `paper_protocol_audit.md` / `replication_report.md` | 方法與協定稽核筆記 |

### `experiments/work_mesa_transfer/` — MESA 大世代
| 檔案 | 用途 |
|---|---|
| `download_mesa.py` | 從 NSRR 下載 MESA 子集(EDF + 事件 XML) |
| `mesa_to_ucddb_trainer.py` | **跨資料集**:train MESA → test 全 UCDDB(零洩漏) |
| `mesa_internal_trainer.py` | **MESA-internal grouped CV**(段/分鐘標籤、window/recording 正規化、層級消融) |
| `mesa_screening.py` | per-subject AHI 篩檢(Olsen-style) |
| `run_pipeline.py` / `run_mesa_scale.py` / `run_ablation.py` | 自動編排:下載+訓練 / 149人規模 / 層級消融 |
| **`mesa_transfer_results.md`** ⭐ | **MESA 結果(含 Olsen 對照、規模測試)** |

### 其他
| 路徑 | 用途 |
|---|---|
| `experiments/cnn_transformer_lstm/` | 早期自包式 M11 實作(train/evaluate/models),已被 lib 版取代 |
| `archive/` | 探索階段舊腳本(不影響上面流程) |
| `docs/` | 零散說明、企畫書 |
| `RESULTS.md` ⭐ | **主結果總表** |

---

## 5. 下載 MESA(選用)

```powershell
# 1. NSRR 帳號 + MESA DAUA 核准後,從 https://sleepdata.org/token 取 token
$env:NSRR_TOKEN = "你的token"
# 2. 下載子集(EDF + 事件 XML)到 D:/mesa
python experiments/work_mesa_transfer/download_mesa.py --pattern "mesa-sleep-00[0-9]*" --data-dir D:/mesa
```
注意:VPN 若做 TLS 攔截,需 `pip install pip-system-certs`(讓 Python 信任 Windows 憑證庫)。

---

## 6. 成果文件(看這幾份就懂全部)

| 文件 | 內容 |
|---|---|
| **[`RESULTS.md`](RESULTS.md)** | 主結果總表:架構×流程×完整指標 |
| [`experiments/work_ucddb_hrv/model_comparison.md`](experiments/work_ucddb_hrv/model_comparison.md) | 各協定×各模型對照 |
| [`experiments/work_ucddb_hrv/hrv_vs_deeplearning.md`](experiments/work_ucddb_hrv/hrv_vs_deeplearning.md) | HRV vs 深度學習決策準則 |
| [`experiments/work_mesa_transfer/mesa_transfer_results.md`](experiments/work_mesa_transfer/mesa_transfer_results.md) | MESA 結果 + Olsen 對照 |
| [`experiments/work_ucddb_paper_replication/m11_protocol_results.md`](experiments/work_ucddb_paper_replication/m11_protocol_results.md) | 論文 98% 洩漏重現 |

## 7. 一句話結論

- **乾淨資料(Apnea-ECG / Polar H10 相似)→ CNN+Transformer 勝**:per-minute AUC 0.931,**per-recording 篩檢 97.1%**。
- **小/雜/跨人(UCDDB 誠實 grouped CV)→ 深度優勢消失,HRV+CVHR ensemble AUC 0.570 追平且可解釋**。
- **評估協定是關鍵**:segment split 會洩漏虛高(論文 98%);grouped CV 才誠實。
