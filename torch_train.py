"""Churn model in PyTorch: same data, split and imbalance handling as train.py.

Model: 34 -> 64 -> ReLU -> Dropout -> 1 (logit), trained with
BCEWithLogitsLoss(pos_weight = non-churners / churners in train).
Early stopping on validation PR-AUC; the test set is used once at the end.

Usage: python torch_train.py
"""

import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from churn import TARGET, clean, encode

SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 1e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ChurnMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),  # (B, 34) -> (B, 64)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),           # (B, 64) -> (B, 1) raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits; sigmoid is applied inside the loss


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(X.to(device))          # (N, 1)
    return torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (N,)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = clean(pd.read_csv("Customer-Churn.csv"))
    X = encode(df).to_numpy(dtype=np.float32)
    y = df[TARGET].map({"No": 0, "Yes": 1}).to_numpy(dtype=np.float32)

    # Same test split as train.py, then 10% of train held out for early stopping
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=SEED, stratify=y_tr)

    scaler = StandardScaler().fit(X_tr)  # fitted on training rows only
    X_tr, X_val, X_te = (scaler.transform(a).astype(np.float32) for a in (X_tr, X_val, X_te))

    X_tr_t = torch.from_numpy(X_tr)                # (N_train, 34) float32
    y_tr_t = torch.from_numpy(y_tr).unsqueeze(1)   # (N_train, 1)  float32
    X_val_t, X_te_t = torch.from_numpy(X_val), torch.from_numpy(X_te)
    print(f"device={device} X_train={tuple(X_tr_t.shape)} y_train={tuple(y_tr_t.shape)} "
          f"X_val={tuple(X_val_t.shape)} X_test={tuple(X_te_t.shape)}")

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))

    model = ChurnMLP(X_tr.shape[1]).to(device)
    pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"pos_weight={pos_weight.item():.3f} parameters={n_params}")

    xb0, yb0 = next(iter(loader))
    print(f"one batch: x={tuple(xb0.shape)} y={tuple(yb0.shape)} logits={tuple(model(xb0.to(device)).shape)}")

    best_ap, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        val_ap = average_precision_score(y_val, predict_proba(model, X_val_t, device))
        if val_ap > best_ap:
            best_ap, best_epoch, bad = val_ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    print(f"stopped at epoch {epoch}; best epoch {best_epoch}, val PR-AUC {best_ap:.3f}, "
          f"last train loss {total / len(X_tr):.4f}")

    model.load_state_dict(best_state)
    p = predict_proba(model, X_te_t, device)
    pred = (p >= 0.5).astype(int)
    metrics = {
        "model": "PyTorch MLP 34-64-1, BCEWithLogitsLoss(pos_weight)",
        "test_rows": int(len(y_te)),
        "churn_recall": round(recall_score(y_te, pred), 3),
        "churn_precision": round(precision_score(y_te, pred), 3),
        "churn_f1": round(f1_score(y_te, pred), 3),
        "accuracy": round(accuracy_score(y_te, pred), 3),
        "roc_auc": round(roc_auc_score(y_te, p), 3),
        "pr_auc": round(average_precision_score(y_te, p), 3),
        "best_epoch": best_epoch,
    }
    print(json.dumps(metrics, indent=2))
    with open("torch_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_}, "churn_torch.pt")


if __name__ == "__main__":
    main()
