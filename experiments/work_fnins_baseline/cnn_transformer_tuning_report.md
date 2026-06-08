# CNN+Transformer 調參報告（UCDDB，誠實 record-level grouped CV）

評估：5-fold record-level grouped CV（整位受試者 held out），與 HRV ensemble（0.570）同協定可直接比較。
搜尋方式：座標式（每次只改一維，與 baseline 比），各組跑完整 5-fold，搜尋階段用 15 epoch。

## 誠實基準（30 epoch，預設超參）
- Pooled out-of-fold AUC **0.552**、per-fold AUC **0.572**、val-threshold BAcc **0.556**。
- Fold 3（ucddb012/019/021/022/027）是穩定的失敗 fold（AUC ≈ 0.49）。

## 調參結果（15 epoch 搜尋；依 pooled AUC）

| 設定 | pooled AUC | 設定 | pooled AUC |
|---|---:|---|---:|
| **Baseline（ctx5, ch0+2, d96/l2/h4/drop0.3/lr5e-4/focal γ2）** | **0.554** | layers=1 | 0.530 |
| dropout=0.2 | 0.551 | context=3 | 0.526 |
| weight_decay=1e-3 | 0.551 | dropout=0.4 | 0.515 |
| focal_gamma=1.0 | 0.547 | lr=1e-3 | 0.514 |
| d_model=64 | 0.544 | layers=3 | 0.510 |
| d_model=128 | 0.535 | lr=3e-4 | 0.508 |
| nhead=8 | 0.532 | channels=[0]（單導程） | 0.504 |
| loss=wce | 0.531 | | |

## 結論
- **預設超參已是最佳（~0.554）；沒有任何一組調整能贏過它。**
- 單導程（ch0 only）與較短 context（3 分鐘）都更差 → 雙通道 + 5 分鐘 context 是對的。
- focal > wce；focal γ2 > γ1；d_model 96、layers 2、nhead 4、dropout 0.3、lr 5e-4 為最佳區。
- **調參無法突破 ~0.55 的跨受試者天花板** → 瓶頸是輸入特徵/表徵品質，不是模型超參數。
  下一步改攻會改變「輸入品質」的 P3（R-peak 偵測 + 訊號品質 gating）。

## 最佳組態（後續 P3/P1 沿用）
`channels=[0,2]  context=5  d_model=96  layers=2  nhead=4  dropout=0.3  lr=5e-4  weight_decay=1e-4  loss=focal  focal_gamma=2.0`
