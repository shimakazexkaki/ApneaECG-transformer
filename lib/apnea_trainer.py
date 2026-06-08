"""
復現論文: "Leveraging Transformer to Detect Obstructive Sleep Apnea from Single-Lead ECG Signals"
(Biswas & Abu Yousuf, ICCA '24, ACM, DOI: 10.1145/3723178.3723252)

架構: 三平行卷積路徑 (Parallel CNN) + Transformer
資料切分: Segment-level 80/20 split (與論文一致)
"""

import os
import wfdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, lfilter

# ==========================================
# 1. 參數設定 (與論文一致)
# ==========================================
DATA_DIR = 'apnea-ecg'
FS = 100  # 採樣率 100 Hz
MINUTE_SAMPLES = FS * 60  # 每分鐘 6000 個取樣點
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
PATIENCE = 10  # Early Stopping patience
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 資料集準備與濾波
# ==========================================
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

class ApneaECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # shape: (N, 1, 6000)
        self.y = torch.LongTensor(y)                 # shape: (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    """讀取 Apnea-ECG 資料, 濾波 0.5-45Hz (與論文一致), Z-score 正規化"""
    print("正在載入 Apnea-ECG 資料集 (Butterworth 0.5-45Hz)...")
    X_all, y_all = [], []
    
    list_path = os.path.join(DATA_DIR, 'list')
    with open(list_path, 'r') as f:
        records = [line.strip() for line in f.readlines() if line.strip()]

    annotated_records = [r for r in records if os.path.exists(os.path.join(DATA_DIR, f"{r}.apn"))]
    print(f"找到 {len(annotated_records)} 筆具有標記的紀錄。")

    for record in tqdm(annotated_records, desc="資料處理進度"):
        record_path = os.path.join(DATA_DIR, record)
        try:
            signal, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'apn')
            
            ecg_signal = signal[:, 0]
            sample_pts = annotation.sample
            labels = annotation.symbol
            
            for pt, label in zip(sample_pts, labels):
                if label not in ['A', 'N']: continue
                if pt < 0 or pt + MINUTE_SAMPLES > len(ecg_signal): continue
                
                segment = ecg_signal[pt : pt + MINUTE_SAMPLES]
                # 帶通濾波 0.5-45Hz (與論文一致)
                segment = butter_bandpass_filter(segment, 0.5, 45.0, FS, order=3)
                # Z-score 正規化
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                
                X_all.append(segment)
                y_all.append(1 if label == 'A' else 0)
                
        except Exception as e:
            print(f"處理 {record} 時發生錯誤: {e}")

    return np.array(X_all), np.array(y_all)

# ==========================================
# 3. 論文模型架構: Parallel CNN + Transformer
# ==========================================

class ParallelCNNBlock(nn.Module):
    """三條平行卷積路徑，分別擷取不同尺度的 ECG 特徵"""
    def __init__(self):
        super(ParallelCNNBlock, self).__init__()
        # 路徑 1: 大 kernel (捕捉長期心率趨勢, ~3秒)
        self.path1 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=300, stride=1, padding=150),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 路徑 2: 中 kernel (捕捉中期特徵, ~0.3秒)
        self.path2 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=30, stride=1, padding=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 路徑 3: 小 kernel (捕捉 R 波細節, ~0.1秒)
        self.path3 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=10, stride=1, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        # x: [batch, 1, 6000]
        p1 = self.path1(x)  # [batch, 48, 3000]
        p2 = self.path2(x)  # [batch, 48, 3000]
        p3 = self.path3(x)  # [batch, 48, 3000]
        
        # 取最小長度對齊後串接
        min_len = min(p1.size(2), p2.size(2), p3.size(2))
        p1 = p1[:, :, :min_len]
        p2 = p2[:, :, :min_len]
        p3 = p3[:, :, :min_len]
        
        out = torch.cat([p1, p2, p3], dim=1)  # [batch, 144, min_len]
        return out

class TransformerBlock(nn.Module):
    """論文中的 Transformer Block"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-Head Attention + Residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-Forward Network + Residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class ParallelCNNTransformer(nn.Module):
    """
    完整論文架構:
    Parallel CNN (3 paths) → Concatenation → MaxPool → 
    Addition (Skip Connection) → Dense → GAP → 
    Transformer → Softmax
    """
    def __init__(self):
        super(ParallelCNNTransformer, self).__init__()
        
        # 1. 三平行卷積路徑
        self.parallel_cnn = ParallelCNNBlock()
        
        # 2. 融合後的池化
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 3. 降維卷積 (將 144 channels 降至 d_model)
        self.channel_reduce = nn.Sequential(
            nn.Conv1d(144, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 4. Dense Layer + Dropout (論文: 64 units, LeakyReLU)
        self.dense_block = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        
        # 5. Global Average Pooling → 固定長度
        self.gap = nn.AdaptiveAvgPool1d(64)
        
        # 6. Transformer Block
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, 64) * 0.02)
        self.transformer = TransformerBlock(d_model=64, nhead=8, dim_feedforward=256, dropout=0.1)
        
        # 7. 分類器 (Softmax 輸出 2 類)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # Parallel CNN
        x = self.parallel_cnn(x)        # [batch, 144, ~3000]
        x = self.pool(x)                 # [batch, 144, ~1500]
        
        # 降維
        x = self.channel_reduce(x)       # [batch, 64, ~750]
        
        # Dense + Skip Connection (Addition Layer)
        identity = x
        x = self.dense_block(x)
        # 確保維度匹配後相加
        if x.shape == identity.shape:
            x = x + identity
        
        # GAP
        x = self.gap(x)                  # [batch, 64, 64]
        
        # Transformer
        x = x.permute(0, 2, 1)           # [batch, 64, 64]
        x = x + self.pos_embedding
        x = self.transformer(x)          # [batch, 64, 64]
        x = x.mean(dim=1)                # [batch, 64]
        
        # Classification
        x = self.classifier(x)           # [batch, 2]
        return x

# ==========================================
# 4. 訓練與評估流程
# ==========================================
def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spec = 0.0
        
    return acc, prec, rec, spec, f1

def train_model():
    print(f"使用裝置: {DEVICE}")
    
    # 1. 載入資料
    X, y = load_data()
    print(f"總共 {len(X)} 段 | Apnea: {sum(y == 1)}, Normal: {sum(y == 0)}")
    
    # 2. Segment-level 80/20 split (與論文一致)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"訓練集: {len(X_train)}, 測試集: {len(X_test)}")
    
    train_loader = DataLoader(ApneaECGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ApneaECGDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. 初始化模型
    model = ParallelCNNTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # 論文使用 Softmax + CE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    
    print("開始訓練【論文復現版】Parallel CNN + Transformer 模型...")
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)  # [batch, 2]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step(train_loss)
        
        # 評估
        model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X.to(DEVICE))
                preds = outputs.argmax(dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_trues.extend(batch_y.numpy())
                
        acc, prec, rec, spec, f1 = evaluate_metrics(np.array(test_trues), np.array(test_preds))
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Spec: {spec:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), 'apnea_parallel_cnn_transformer.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early Stopping at Epoch {epoch}!")
                break

    print(f"訓練完成！最佳 F1: {best_f1:.4f}, 模型已儲存至 'apnea_parallel_cnn_transformer.pth'")
    
if __name__ == '__main__':
    train_model()
