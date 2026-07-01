"""Improved CNN-LSTM v3 sequence model for solar flare forecasting.

Architecture (input: B x 12 x 3600):
    4-stage Conv1d feature extractor (32->64->128->256) with BatchNorm + GELU
    -> bidirectional 2-layer LSTM (hidden=256) -> temporal attention
    -> 3-layer MLP head with Sigmoid.

Designed for the A100 80GB GPU with bfloat16 mixed-precision training, focal
loss for the ~6% positive class, cosine-annealing warm-restart LR schedule,
gradient clipping, early stopping, and rolling-origin cross-validation.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

_RANDOM_STATE = 42


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with tunable gamma and alpha.

    L = -alpha_t * pos_w * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1 else 1-p, alpha_t = alpha if y=1 else 1-alpha, and
    pos_w = pos_weight if y=1 else 1 (class-weighting on top of the focal term).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        p_t = target * pred + (1 - target) * (1 - pred)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        pos_w = target * self.pos_weight + (1 - target)
        loss = -alpha_t * pos_w * (1 - p_t) ** self.gamma * torch.log(p_t)
        return loss.mean()


class TemporalAttention(nn.Module):
    """Simple additive attention over the time dimension.

    Input: (B, T, D) -> Output: (B, D)
    Learns to weight which time steps are most important.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        scores = self.attention(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context


class CNNLSTMv3(nn.Module):
    """CNN feature extractor + bidirectional LSTM + temporal attention head.

    Input:  (B, n_channels, seq_len)  e.g. (B, 12, 3600)
    Output: (B,) flare probabilities in [0, 1].

    When n_features > 0, the forward pass accepts an additional ``features``
    tensor of shape (B, n_features). The features are projected
    (Linear 128 → LayerNorm → GELU), expanded to (B, T, 128), and
    concatenated with the Conv1D output on the channel dim before BiLSTM.
    """

    def __init__(self, n_channels: int = 12, seq_len: int = 3600, n_features: int = 0):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.n_features = n_features

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
        )

        conv_out_dim = 256
        lstm_in_dim = conv_out_dim
        if n_features > 0:
            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.LayerNorm(128),
                nn.GELU(),
            )
            lstm_in_dim = conv_out_dim + 128

        self.lstm = nn.LSTM(
            lstm_in_dim,
            256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        self.attention = TemporalAttention(512)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, features: Tensor | None = None) -> Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        if features is not None and self.n_features > 0:
            f = self.feature_proj(features)
            f = f.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, f], dim=-1)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.head(x)
        return x.squeeze(-1)


def _tss_at_threshold(
    probs: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, tuple[int, int, int, int]]:
    pred = (probs >= threshold).astype(np.int8)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr - fpr, (tn, fp, fn, tp)


def _best_tss(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[float, float, tuple[int, int, int, int]]:
    best_tss = -1.0
    best_threshold = 0.5
    best_conf = (0, 0, 0, 0)
    thresholds = np.unique(
        np.concatenate([[0.0, 0.25, 0.5, 0.75, 1.0], np.linspace(0.01, 0.99, 99)])
    )
    for t in thresholds:
        tss, conf = _tss_at_threshold(probs, labels, float(t))
        if tss > best_tss:
            best_tss = tss
            best_threshold = float(t)
            best_conf = conf
    return best_tss, best_threshold, best_conf


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a data loader.

    Returns dict with:
    - 'probabilities': (N,) array
    - 'labels': (N,) array
    - 'tss': best TSS over thresholds
    - 'auc_roc', 'auc_pr', 'f1', 'precision', 'recall'
    - 'best_threshold'
    - 'confusion': (tn, fp, fn, tp)
    """
    model.eval()
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                xb, fb, yb = batch
                fb = fb.to(device)
            else:
                xb, yb = batch
                fb = None
            xb = xb.to(device)
            with amp_ctx:
                p = model(xb, features=fb)
            probs_list.append(p.float().cpu().numpy())
            labels_list.append(yb.cpu().numpy())
    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list).astype(np.int8)

    best_tss, best_threshold, (tn, fp, fn, tp) = _best_tss(probs, labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    auc_roc = float("nan")
    auc_pr = float("nan")
    if len(np.unique(labels)) > 1:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            auc_roc = float(roc_auc_score(labels, probs))
            auc_pr = float(average_precision_score(labels, probs))
        except Exception:
            pass

    return {
        "probabilities": probs,
        "labels": labels,
        "tss": best_tss,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "best_threshold": best_threshold,
        "confusion": (tn, fp, fn, tp),
    }


class FlareForecasterCNNLSTMv3:
    """Full training wrapper for CNN-LSTM v3.

    Handles:
    - Mixed precision training (bfloat16)
    - CosineAnnealingWarmRestarts scheduler
    - Early stopping with patience
    - Gradient clipping
    - Class-weighted focal loss
    - Rolling-origin cross-validation support
    - Model checkpointing
    """

    def __init__(
        self,
        n_channels: int = 12,
        seq_len: int = 3600,
        n_features: int = 0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        pos_weight: float = 16.0,
        device: str = "auto",
    ):
        torch.manual_seed(_RANDOM_STATE)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.n_features = n_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.pos_weight = pos_weight

        self.model = CNNLSTMv3(n_channels, seq_len, n_features).to(self.device)
        self.criterion = FocalLoss(
            gamma=focal_gamma, alpha=focal_alpha, pos_weight=pos_weight
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=True) if self.use_amp else None
        )
        self.history: dict | None = None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        checkpoint_path: str | None = None,
    ) -> dict:
        """Train the model with early stopping.

        Returns dict with 'train_losses', 'val_losses', 'val_tss', 'best_epoch',
        'best_tss'.
        """
        self.history = {
            "train_losses": [],
            "val_losses": [],
            "val_tss": [],
            "best_epoch": -1,
            "best_tss": -1.0,
        }
        best_state: dict | None = None
        epochs_no_improve = 0
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_amp
            else nullcontext()
        )

        def _unpack(batch):
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
            return batch[0], None, batch[1]

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_seen = 0
            for batch in train_loader:
                xb, fb, yb = _unpack(batch)
                xb = xb.to(self.device)
                if fb is not None:
                    fb = fb.to(self.device)
                yb = yb.to(self.device).float()
                self.optimizer.zero_grad()
                if self.use_amp:
                    with amp_ctx:
                        pred = self.model(xb, features=fb)
                        loss = self.criterion(pred, yb)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(xb, features=fb)
                    loss = self.criterion(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                running_loss += loss.item() * xb.size(0)
                n_seen += xb.size(0)
            train_loss = running_loss / max(n_seen, 1)
            self.scheduler.step()

            self.model.eval()
            val_loss_sum = 0.0
            val_n = 0
            probs_list: list[np.ndarray] = []
            labels_list: list[np.ndarray] = []
            with torch.no_grad():
                for batch in val_loader:
                    xb, fb, yb = _unpack(batch)
                    xb = xb.to(self.device)
                    if fb is not None:
                        fb = fb.to(self.device)
                    yb = yb.to(self.device).float()
                    with amp_ctx:
                        pred = self.model(xb, features=fb)
                        loss = self.criterion(pred, yb)
                    val_loss_sum += loss.item() * xb.size(0)
                    val_n += xb.size(0)
                    probs_list.append(pred.float().cpu().numpy())
                    labels_list.append(yb.cpu().numpy())
            val_loss = val_loss_sum / max(val_n, 1)
            probs = np.concatenate(probs_list)
            labels = np.concatenate(labels_list).astype(np.int8)
            val_tss, _, _ = _best_tss(probs, labels)

            self.history["train_losses"].append(train_loss)
            self.history["val_losses"].append(val_loss)
            self.history["val_tss"].append(val_tss)

            lr_now = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_tss={val_tss:.4f} | lr={lr_now:.2e}"
            )

            if val_tss > self.history["best_tss"]:
                self.history["best_tss"] = val_tss
                self.history["best_epoch"] = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                if checkpoint_path:
                    self.save(checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)."
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    def predict_proba(self, X: np.ndarray | Tensor) -> np.ndarray:
        """Predict flare probability for a batch of sequences.

        X: (N, 12, 3600) -> returns (N,) array of probabilities.
        """
        self.model.eval()
        if not isinstance(X, Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        preds: list[np.ndarray] = []
        batch = 256
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_amp
            else nullcontext()
        )
        with torch.no_grad():
            for i in range(0, X.size(0), batch):
                xb = X[i : i + batch]
                with amp_ctx:
                    p = self.model(xb)
                preds.append(p.float().cpu().numpy())
        return np.concatenate(preds)

    def predict_proba_dataloader(self, loader: DataLoader) -> np.ndarray:
        """Predict probabilities for all samples in a DataLoader."""
        self.model.eval()
        preds: list[np.ndarray] = []
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_amp
            else nullcontext()
        )
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    xb, fb, _ = batch
                    fb = fb.to(self.device)
                else:
                    xb, _ = batch
                    fb = None
                xb = xb.to(self.device)
                with amp_ctx:
                    p = self.model(xb, features=fb)
                preds.append(p.float().cpu().numpy())
        return np.concatenate(preds)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": {
                    "n_channels": self.n_channels,
                    "seq_len": self.seq_len,
                    "n_features": self.n_features,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "focal_gamma": self.focal_gamma,
                    "focal_alpha": self.focal_alpha,
                    "pos_weight": self.pos_weight,
                },
                "history": self.history,
            },
            path,
        )

    def load(self, path: str) -> "FlareForecasterCNNLSTMv3":
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.history = ckpt.get("history")
        return self


def rolling_origin_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 256,
) -> dict:
    """Rolling-origin cross-validation for time series.

    Splits data chronologically into n_folds train/val/test splits. Trains a
    fresh model for each fold using an expanding training window.

    Returns dict with 'fold_results' (list of per-fold metrics), 'mean_tss',
    'std_tss'.
    """
    n = len(X)
    fold_size = n // (n_folds + 1)
    if fold_size < 1:
        raise ValueError(
            f"Not enough samples ({n}) for {n_folds} folds "
            f"(need at least {n_folds + 1})."
        )

    n_channels = X.shape[1]
    seq_len = X.shape[2]
    fold_results: list[dict] = []

    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min(train_end + fold_size, n)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        val_cut = int(len(X_train) * 0.8)
        X_tr, y_tr = X_train[:val_cut], y_train[:val_cut]
        X_val, y_val = X_train[val_cut:], y_train[val_cut:]

        train_dl = DataLoader(
            TensorDataset(
                torch.as_tensor(X_tr, dtype=torch.float32),
                torch.as_tensor(y_tr, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_dl = DataLoader(
            TensorDataset(
                torch.as_tensor(X_val, dtype=torch.float32),
                torch.as_tensor(y_val, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        test_dl = DataLoader(
            TensorDataset(
                torch.as_tensor(X_test, dtype=torch.float32),
                torch.as_tensor(y_test, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        forecaster = FlareForecasterCNNLSTMv3(n_channels=n_channels, seq_len=seq_len)
        forecaster.fit(train_dl, val_dl, epochs=epochs)

        res = evaluate_model(forecaster.model, test_dl, forecaster.device)
        fold_results.append(
            {
                "fold": i,
                "tss": res["tss"],
                "auc_roc": res["auc_roc"],
                "auc_pr": res["auc_pr"],
                "f1": res["f1"],
                "precision": res["precision"],
                "recall": res["recall"],
                "best_threshold": res["best_threshold"],
                "n_test": int(len(y_test)),
                "n_pos_test": int(np.sum(y_test)),
            }
        )
        print(
            f"Fold {i + 1}/{n_folds}: TSS={res['tss']:.4f} "
            f"AUC-ROC={res['auc_roc']:.4f} F1={res['f1']:.4f}"
        )

    tss_arr = np.array([f["tss"] for f in fold_results])
    return {
        "fold_results": fold_results,
        "mean_tss": float(np.mean(tss_arr)),
        "std_tss": float(np.std(tss_arr)),
    }
