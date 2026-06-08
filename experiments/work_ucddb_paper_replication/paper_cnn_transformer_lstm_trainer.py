"""UCDDB replication trainer for Pham & Moucek (2025) CNN-Transformer-LSTM.

This script keeps the original project files untouched. It reuses the local
UCDDB literature feature pipeline:

- Hamilton R-peak detection via ``ucddb_literature_features.py``
- 5-minute contexts
- 900 interpolated RRI points + 900 R-peak-amplitude points
- record-grouped CV / holdout / segment-level literature split

The added model is the paper's M11-style stack:

    Conv1D(64, 128, 128; kernel=7) -> MaxPool(4) -> Transformer -> LSTM -> Dense

Run with the requested environment, for example:

    C:\\Users\\a2003\\miniconda3\\envs\\apnea\\python.exe ^
      work_ucddb_paper_replication\\paper_cnn_transformer_lstm_trainer.py ^
      --protocol cv --folds 0 --epochs 3 --channels 0 2 --no-progress
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn


WORK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WORK_DIR.parents[1]  # apnea project root
for _p in (PROJECT_DIR / "lib", PROJECT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ucddb_literature_train_common as common  # noqa: E402


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class PaperCNNTransformerLSTM(nn.Module):
    """M11-style CNN-Transformer-LSTM over RRI/R-peak-amplitude sequences."""

    def __init__(
        self,
        input_dim: int = 2,
        filters: tuple[int, int, int] = (64, 128, 128),
        kernel_size: int = 7,
        pool_size: int = 4,
        cnn_dropout: float = 0.5,
        nhead: int = 8,
        dim_feedforward: int = 256,
        transformer_dropout: float = 0.1,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        classifier_dropout: float = 0.3,
        use_transformer: bool = True,
        use_lstm: bool = True,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_lstm = use_lstm
        layers: list[nn.Module] = []
        in_channels = input_dim
        for out_channels in filters:
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(pool_size),
                ]
            )
            in_channels = out_channels
        layers.append(nn.Dropout(cnn_dropout))
        self.cnn = nn.Sequential(*layers)

        d_model = filters[-1]
        if use_transformer:
            self.positional_encoding = SinusoidalPositionalEncoding(d_model, transformer_dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=transformer_dropout,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=0.0 if lstm_layers == 1 else transformer_dropout,
            )
        classifier_in = lstm_hidden if use_lstm else d_model
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_in, 2),
        )

    def forward(self, features: torch.Tensor, raw: torch.Tensor | None = None) -> torch.Tensor:
        del raw
        x = features.permute(0, 2, 1)
        x = self.cnn(x).permute(0, 2, 1)
        if self.use_transformer:
            x = self.positional_encoding(x)
            x = self.transformer(x)
        if self.use_lstm:
            _, (hidden, _) = self.lstm(x)
            feat = hidden[-1]
        else:
            feat = x.mean(dim=1)  # global average pool over time
        return self.classifier(feat)


def patch_model_factory() -> None:
    original_make_model = common.make_model

    def make_model(args, model_kind):
        if model_kind == "paper_cnn_transformer_lstm":
            return PaperCNNTransformerLSTM(
                input_dim=getattr(args, "input_dim", 2),
                nhead=args.nhead,
                dim_feedforward=args.dim_feedforward,
                transformer_dropout=args.transformer_dropout,
                cnn_dropout=args.cnn_dropout,
                lstm_hidden=args.lstm_hidden,
                lstm_layers=args.lstm_layers,
                classifier_dropout=args.classifier_dropout,
                use_transformer=getattr(args, "use_transformer", True),
                use_lstm=getattr(args, "use_lstm", True),
            )
        return original_make_model(args, model_kind)

    common.make_model = make_model


def relax_required(parser: argparse.ArgumentParser, dest: str, default) -> None:
    for action in parser._actions:
        if action.dest == dest:
            action.required = False
            action.default = default
            return
    raise ValueError(f"Parser action not found: {dest}")


def main() -> None:
    patch_model_factory()
    parser = argparse.ArgumentParser(
        description="Pham & Moucek 2025 M11-style CNN-Transformer-LSTM replication on UCDDB."
    )
    common.add_train_args(parser)
    relax_required(parser, "experiment_name", "paper_m11_ch0_ch2_hamilton")
    parser.set_defaults(
        output_dir=str(WORK_DIR / "outputs"),
        cache_dir=str(WORK_DIR / "cache" / "ucddb_literature_features"),
        channels=[0, 2],
        detector="biosppy_hamilton",
        lr=1e-3,
        patience=30,
        batch_size=128,
        nhead=8,
        epochs=40,
    )
    parser.add_argument("--cnn-dropout", type=float, default=0.5)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--classifier-dropout", type=float, default=0.3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    args = parser.parse_args()
    common.run_training(args, model_kind="paper_cnn_transformer_lstm", include_raw=False)


if __name__ == "__main__":
    main()
