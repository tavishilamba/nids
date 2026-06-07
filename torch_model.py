"""
PyTorch Models for NIDS
- DeepMLP: 4-layer residual MLP (best for tabular NSL-KDD)
- LSTMClassifier: sequence model for temporal traffic patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.relu(x + self.net(x)))


class DeepMLP(nn.Module):
    """
    Residual MLP for tabular intrusion detection.
    Input: (batch, n_features)
    Output: (batch, n_classes) logits
    """
    def __init__(self, n_features, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)


class LSTMClassifier(nn.Module):
    """
    LSTM for sequential traffic windows.
    Input: (batch, seq_len, n_features)
    Output: (batch, n_classes) logits
    """
    def __init__(self, n_features, n_classes, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)           # (batch, seq_len, hidden*2)
        attn = torch.softmax(self.attention(out), dim=1)  # (batch, seq_len, 1)
        ctx = (out * attn).sum(dim=1)   # (batch, hidden*2)
        return self.head(ctx)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced attack classes."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def get_class_weights(y, n_classes, device):
    counts = torch.bincount(torch.tensor(y), minlength=n_classes).float()
    weights = 1.0 / (counts + 1e-6)
    return (weights / weights.sum() * n_classes).to(device)
