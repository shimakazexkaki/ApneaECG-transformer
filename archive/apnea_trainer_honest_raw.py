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
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.signal import butter, lfilter

# ==========================================
# 1. 參數設定
# ==========================================
DATA_DIR = 'apnea-ecg'
FS = 100  # 採樣率 100 Hz
MINUTE_SAMPLES = FS * 60  # 每分鐘 6000 個取樣點
BATCH_SIZE = 64
EPOCHS = 40  # 提升 Epoch 以便模型收斂到更高點
LEARNING_RATE = 5e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 資料集準備與濾波 (Dataset & Preprocessing)
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
        self.X = torch.FloatTensor(X).unsqueeze(1) # shape: (N, 1, 18000)
        self.y = torch.FloatTensor(y)              # shape: (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def extract_features_for_record(record_path):
    """處理單一病患的訊號，確保時間連續性 (修正時間邊界問題)"""
    try:
        signal, fields = wfdb.rdsamp(record_path)
        annotation = wfdb.rdann(record_path, 'apn')
    except Exception as e:
        print(f"處理 {record_path} 時發生錯誤: {e}")
        return [], []
        
    ecg_signal = signal[:, 0]
    sample_pts = annotation.sample
    labels = annotation.symbol
    
    valid_segments = []
    valid_labels = []
    valid_pts = []
    
    for pt, label in zip(sample_pts, labels):
        if label not in ['A', 'N']: continue
        if pt < 0 or pt + MINUTE_SAMPLES > len(ecg_signal): continue
        
        segment = ecg_signal[pt : pt + MINUTE_SAMPLES]
        # 帶通濾波 (0.5Hz - 40Hz) 消除基線漂移與高頻雜訊
        segment = butter_bandpass_filter(segment, 0.5, 40.0, FS, order=3)
        # Z-score 正規化
        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
        
        valid_segments.append(segment)
        valid_labels.append(1 if label == 'A' else 0)
        valid_pts.append(pt)
        
    X_record, y_record = [], []
    # 建立 3 分鐘的時間序列脈絡 (Context Window = 3)
    for i in range(1, len(valid_segments) - 1):
        # 檢查前後分鐘是否真的「連續」(差距剛好是 1 分鐘的 sample 數)
        if valid_pts[i] - valid_pts[i-1] == MINUTE_SAMPLES and valid_pts[i+1] - valid_pts[i] == MINUTE_SAMPLES:
            context_segment = np.concatenate((valid_segments[i-1], valid_segments[i], valid_segments[i+1]))
            X_record.append(context_segment)
            y_record.append(valid_labels[i])
            
    return X_record, y_record

def load_data_record_split():
    """解決資料洩漏問題：先切分病患，再擷取特徵"""
    list_path = os.path.join(DATA_DIR, 'list')
    with open(list_path, 'r') as f:
        records = [line.strip() for line in f.readlines() if line.strip()]

    annotated_records = [r for r in records if os.path.exists(os.path.join(DATA_DIR, f"{r}.apn"))]
    print(f"找到 {len(annotated_records)} 筆具有標記的紀錄。")
    
    # 1. 解決 Data Leakage: 依照 record 切分 train/test
    train_recs, test_recs = train_test_split(annotated_records, test_size=0.2, random_state=42)
    print(f"訓練集病患數: {len(train_recs)}, 測試集病患數: {len(test_recs)}")
    
    def build_dataset(recs, desc):
        X, y = [], []
        for r in tqdm(recs, desc=desc):
            X_rec, y_rec = extract_features_for_record(os.path.join(DATA_DIR, r))
            X.extend(X_rec)
            y.extend(y_rec)
        return np.array(X), np.array(y)
        
    X_train, y_train = build_dataset(train_recs, "處理訓練集")
    X_test, y_test = build_dataset(test_recs, "處理測試集")
    
    return X_train, y_train, X_test, y_test

# ==========================================
# 3. 定義 ResNet1D + Transformer 模型架構
# ==========================================
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.gelu(out)
        return out

class ApneaResNetTransformer(nn.Module):
    def __init__(self):
        super(ApneaResNetTransformer, self).__init__()
        # 使用 ResNet Block 讓特徵萃取更深、更好
        self.conv_blocks = nn.Sequential(
            ResBlock1D(1, 16, kernel_size=7, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=2),
            ResBlock1D(16, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool1d(kernel_size=2),
            ResBlock1D(32, 64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool1d(kernel_size=2),
            ResBlock1D(64, 128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool1d(kernel_size=2)
        )
        self.adaptive_pool = nn.AdaptiveMaxPool1d(64) # 因時間延長為3倍，將長度拉升至 64
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, 128) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.3, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_blocks(x)          # [batch, 128, length]
        x = self.adaptive_pool(x)        # [batch, 128, 64]
        x = x.permute(0, 2, 1)           # [batch, 64, 128]
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)  # [batch, 64, 128]
        x = x.mean(dim=1)                # Global Average Pooling -> [batch, 128]
        x = self.classifier(x)
        return x

# ==========================================
# 4. 訓練與評估流程
# ==========================================
def evaluate_metrics(y_true, y_pred_logits):
    y_pred = (y_pred_logits > 0.0).astype(int)
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
    
    # 使用正確的 Record-level Data Split
    X_train, y_train, X_test, y_test = load_data_record_split()
    print(f"訓練集樣本數: {len(X_train)}, 測試集樣本數: {len(X_test)}")
    print(f"訓練集 Apnea(1): {sum(y_train == 1)}, Normal(0): {sum(y_train == 0)}")
    
    train_loader = DataLoader(ApneaECGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ApneaECGDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 計算正樣本權重
    num_neg = sum(y_train == 0)
    num_pos = sum(y_train == 1)
    smooth_weight = np.sqrt(num_neg / num_pos) if num_pos > 0 else 1.0
    pos_weight = torch.tensor([smooth_weight], dtype=torch.float32).to(DEVICE)
    
    model = ApneaResNetTransformer().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print("開始訓練超級進化版 ResNet + Transformer 模型 (無 Data Leakage)...")
    best_f1 = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1) # 修正 squeeze 導致 batch=1 崩潰的 bug
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X.to(DEVICE)).squeeze(-1)
                test_preds.extend(outputs.cpu().numpy())
                test_trues.extend(batch_y.numpy())
                
        acc, prec, rec, spec, f1 = evaluate_metrics(np.array(test_trues), np.array(test_preds))
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Spec: {spec:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'apnea_resnet_transformer.pth')

    print(f"訓練完成！最佳模型已儲存至 'apnea_resnet_transformer.pth', 最高 F1: {best_f1:.4f}")
    
if __name__ == '__main__':
    train_model()
