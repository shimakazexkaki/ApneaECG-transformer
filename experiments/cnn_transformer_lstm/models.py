"""
CNN-Transformer-LSTM Model for Sleep Apnea Detection

復現論文: Pham & Mouček (2025)
"Sleep apnea detection from single-lead ECG: A CNN-Transformer-LSTM approach"
Computers in Biology and Medicine, 196, 110655.

模型配置 M11 (論文最佳):
  CNN filters: (64, 128, 128), kernel=7, pool=4
  Transformer: d_model=128, nhead=8, dim_ff=256
  LSTM: hidden=128
"""

import math

import torch
import torch.nn as nn


# ============================================================
# 位置編碼 (Positional Encoding)
# ============================================================
class PositionalEncoding(nn.Module):
    """
    標準正弦/餘弦位置編碼，讓 Transformer 感知序列順序。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ============================================================
# CNN 模組 (論文 Section 3.3.1)
# ============================================================
class CNNBlock(nn.Module):
    """
    三層 1D CNN，擷取 ECG 訊號的空間與通道特徵。

    每層結構: Conv1d → BatchNorm → ReLU → MaxPool1d
    最後加 Dropout(0.5)。

    論文設定:
      - Filter 數: (64, 128, 128) [M11] 或 (64, 128, 256) [M12]
      - Kernel Size: 7, Stride: 1
      - MaxPool Size: 4
    """

    def __init__(
        self,
        in_channels: int = 1,
        filters: tuple = (64, 128, 128),
        kernel_size: int = 7,
        pool_size: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        layers = []
        ch_in = in_channels
        for f in filters:
            layers.extend(
                [
                    nn.Conv1d(ch_in, f, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                    nn.BatchNorm1d(f),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=pool_size),
                ]
            )
            ch_in = f
        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.out_channels = filters[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Transformer Encoder 模組 (論文 Section 3.3.2)
# ============================================================
class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder，捕捉長期依賴與時序動態。

    結構:
      1. Multi-Head Self-Attention + Residual + LayerNorm
      2. Feed-Forward Network + Residual + LayerNorm
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Attention + Residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-Forward Network + Residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ============================================================
# 完整 CNN-Transformer-LSTM 模型 (論文 Fig. 3)
# ============================================================
class CNNTransformerLSTM(nn.Module):
    """
    CNN-Transformer-LSTM 模型 (M11 配置)

    架構流程:
      Input ECG → CNN(64,128,128) → Transformer Encoder → LSTM → Dense → Softmax

    支援兩種輸入模式:
      1. 原始 ECG 信號: (batch, 1, 6000) — 1分鐘 100Hz
      2. RRI 特徵:      (batch, 2, N)    — 2通道 (RRI + R-peak amplitude)
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_filters: tuple = (64, 128, 128),
        kernel_size: int = 7,
        pool_size: int = 4,
        cnn_dropout: float = 0.5,
        d_model: int = None,  # 預設等於 cnn_filters[-1]
        nhead: int = 8,
        dim_feedforward: int = 256,
        transformer_dropout: float = 0.1,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__()

        if d_model is None:
            d_model = cnn_filters[-1]

        self.d_model = d_model

        # 1. CNN
        self.cnn = CNNBlock(
            in_channels=in_channels,
            filters=cnn_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            dropout=cnn_dropout,
        )

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, transformer_dropout)

        # 3. Transformer Encoder
        self.transformer = TransformerEncoderBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
        )

        # 4. LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # 5. 分類器
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        # CNN 特徵擷取
        x = self.cnn(x)  # (batch, d_model, reduced_len)

        # 轉置為 (batch, seq_len, d_model) 給 Transformer
        x = x.permute(0, 2, 1)

        # 加入位置編碼
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer(x)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最後一層隱藏狀態
        x = h_n[-1]  # (batch, lstm_hidden)

        # 分類
        x = self.classifier(x)  # (batch, num_classes)
        return x


# ============================================================
# 消融模型 (論文 Table 4): CNN-only, CNN-Transformer, CNN-LSTM
# ============================================================
class CNNOnly(nn.Module):
    """M1/M2: 僅 CNN + Global Average Pooling + Dense"""

    def __init__(self, in_channels=1, cnn_filters=(64, 128, 128), kernel_size=7,
                 pool_size=4, cnn_dropout=0.5, num_classes=2):
        super().__init__()
        self.cnn = CNNBlock(in_channels, cnn_filters, kernel_size, pool_size, cnn_dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(cnn_filters[-1], num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        return x


class CNNTransformer(nn.Module):
    """M7/M8: CNN + Transformer (無 LSTM)"""

    def __init__(self, in_channels=1, cnn_filters=(64, 128, 128), kernel_size=7,
                 pool_size=4, cnn_dropout=0.5, d_model=None, nhead=8,
                 dim_feedforward=256, transformer_dropout=0.1, num_classes=2):
        super().__init__()
        if d_model is None:
            d_model = cnn_filters[-1]
        self.cnn = CNNBlock(in_channels, cnn_filters, kernel_size, pool_size, cnn_dropout)
        self.pos_encoder = PositionalEncoding(d_model, transformer_dropout)
        self.transformer = TransformerEncoderBlock(d_model, nhead, dim_feedforward, transformer_dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        return x


class CNNLSTM(nn.Module):
    """M5/M6: CNN + LSTM (無 Transformer)"""

    def __init__(self, in_channels=1, cnn_filters=(64, 128, 128), kernel_size=7,
                 pool_size=4, cnn_dropout=0.5, lstm_hidden=128, lstm_layers=1,
                 num_classes=2):
        super().__init__()
        self.cnn = CNNBlock(in_channels, cnn_filters, kernel_size, pool_size, cnn_dropout)
        self.lstm = nn.LSTM(cnn_filters[-1], lstm_hidden, lstm_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = self.classifier(x)
        return x


# ============================================================
# 工具函數
# ============================================================
def build_model(model_type: str = "cnn_transformer_lstm", **kwargs) -> nn.Module:
    """
    根據模型類型建立模型。

    Args:
        model_type: 'cnn_transformer_lstm' (M11), 'cnn_only' (M1),
                     'cnn_transformer' (M7), 'cnn_lstm' (M5)
    """
    builders = {
        "cnn_transformer_lstm": CNNTransformerLSTM,
        "cnn_only": CNNOnly,
        "cnn_transformer": CNNTransformer,
        "cnn_lstm": CNNLSTM,
    }
    if model_type not in builders:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(builders.keys())}")
    return builders[model_type](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """計算可訓練參數數量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 維度驗證
    print("=" * 60)
    print("CNN-Transformer-LSTM 模型維度驗證")
    print("=" * 60)

    # 測試 1: Raw ECG 輸入 (1分鐘, 100Hz)
    model_raw = CNNTransformerLSTM(in_channels=1, pool_size=4)
    x_raw = torch.randn(4, 1, 6000)
    out_raw = model_raw(x_raw)
    print(f"\n[Raw ECG] Input: {x_raw.shape} → Output: {out_raw.shape}")
    print(f"  參數量: {count_parameters(model_raw):,}")

    # 測試 2: RRI 特徵輸入 (2通道)
    model_rri = CNNTransformerLSTM(in_channels=2, pool_size=2)
    x_rri = torch.randn(4, 2, 120)
    out_rri = model_rri(x_rri)
    print(f"\n[RRI Features] Input: {x_rri.shape} → Output: {out_rri.shape}")
    print(f"  參數量: {count_parameters(model_rri):,}")

    # 測試 3: 消融模型
    for name in ["cnn_only", "cnn_transformer", "cnn_lstm", "cnn_transformer_lstm"]:
        m = build_model(name, in_channels=1)
        o = m(x_raw)
        print(f"\n[{name}] Output: {o.shape}, Params: {count_parameters(m):,}")

    print("\n[OK] All model dimension checks passed!")
