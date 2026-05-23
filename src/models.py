import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CO2Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading tabular features and dual targets (classification & regression).
    """
    def __init__(self, X, yc, yr_norm):
        self.X = torch.from_numpy(X)
        self.yc = torch.from_numpy(yc).unsqueeze(1)       # [N, 1] for BCE loss
        self.yr = torch.from_numpy(yr_norm).unsqueeze(1)  # [N, 1] for MSE / Pinball loss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.yc[i], self.yr[i]


class PinballLoss(nn.Module):
    """
    Quantile (pinball) loss for a single quantile level q.
    L_q(y, yhat) = q * (y - yhat) if y >= yhat (underprediction)
                 = (1-q) * (yhat - y) if y < yhat (overprediction)
    """
    def __init__(self, quantile: float):
        super().__init__()
        assert 0 < quantile < 1, "quantile must be in (0, 1)"
        self.q = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target - pred
        loss = torch.max(self.q * errors, (self.q - 1) * errors)
        return loss.mean()


class MainMLP(nn.Module):
    """
    Dual-head MLP (classification logit + regression normalized CO2) used for Gasoline/Diesel vehicles.
    """
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        trunk = []
        prev = in_dim
        for h in hidden:
            trunk += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*trunk)
        
        self.clf_head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()  # output in [0,1] range
        )

    def forward(self, x):
        shared = self.trunk(x)
        return self.clf_head(shared), self.reg_head(shared)

# Alias for backward-compatibility with notebook 4 checkpoint
CO2DualHeadMLP = MainMLP


class CO2QuantileMLP(nn.Module):
    """
    Quantile MLP with a shared trunk, a classification head, and three separate
    quantile regression heads (q10, q50, q90) to fix tail bias and provide uncertainty ranges.
    """
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        trunk = []
        prev = in_dim
        for h in hidden:
            trunk += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*trunk)

        def _head():
            return nn.Sequential(
                nn.Linear(prev, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 1), nn.Sigmoid()
            )

        self.clf_head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)  # BCEWithLogitsLoss expects raw logit
        )
        self.q10_head = _head()
        self.q50_head = _head()
        self.q90_head = _head()

    def forward(self, x):
        shared = self.trunk(x)
        return (self.clf_head(shared),
                self.q10_head(shared),
                self.q50_head(shared),
                self.q90_head(shared))


class E85MLP(nn.Module):
    """
    Compact dual-head MLP tailored specifically for the E85 alternative fuel submodel.
    Uses smaller hidden dimensions to prevent overfitting on smaller sample sizes.
    """
    def __init__(self, in_dim, dropout=0.4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,  32),     nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(dropout)
        )
        self.clf_head = nn.Sequential(nn.Linear(32, 1))
        self.reg_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        shared = self.trunk(x)
        return self.clf_head(shared), self.reg_head(shared)
