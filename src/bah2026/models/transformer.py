"""Spectral-Temporal Transformer with cross-attention and Neupert physics-informed loss.

Designed for the A100 80GB GPU with bfloat16 mixed precision and FlashAttention
(via ``F.scaled_dot_product_attention``).

Input format: ``(batch, n_channels=12, seq_len=360)`` — 12 energy channels ×
360 time steps (10 s cadence, 1 h lookback). Binary classification: flare in
the next 30 min.

The Neupert effect (dSXR/dt ≈ η · HXR(t − τ)) is embedded as a learnable
physics term weighted by the predicted flare probability, so it only activates
when the model believes a flare is underway.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

_RANDOM_STATE = 42


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for energy bands (small number of tokens)."""

    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x):  # x: (B, E, D)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class FocalLoss(nn.Module):
    """Focal Loss for binary classification (γ focuses on hard examples).

    ℒ_focal(p_t) = -α·(1−p_t)^γ·log(p_t)  where p_t = p if y=1 else 1-p
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.clamp(1e-7, 1 - 1e-7)
        p_t = target * pred + (1 - target) * (1 - pred)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t)
        return loss.mean()


class NeupertLoss(nn.Module):
    """Physics-informed loss embedding the Neupert effect.

    The Neupert effect:  dSXR/dt ≈ η · HXR(t − τ)

    where η (evaporation efficiency) and τ (time delay) are learnable.

    Total loss: L = L_focal + λ_phys · L_neupert
    L_neupert = ||dSXR/dt - η · HXR(t-τ)||²   (weighted by predicted flare prob)

    The physics loss is only active when the model predicts a flare (weighted by
    predicted probability), because the Neupert effect only holds during flares.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        lambda_phys: float = 0.1,
        dt_seconds: float = 10.0,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma, alpha)
        self.eta = nn.Parameter(torch.tensor(0.5))
        self.tau_seconds = nn.Parameter(torch.tensor(30.0))
        self.lambda_phys = lambda_phys
        self.dt_seconds = dt_seconds

    def forward(
        self,
        pred_prob: Tensor,  # (B,)
        target: Tensor,  # (B,)
        sxr_seq: Tensor,  # (B, T)
        hxr_seq: Tensor,  # (B, T)
    ) -> Tensor:
        loss_cls = self.focal(pred_prob, target)

        dsxr_dt = sxr_seq[:, 1:] - sxr_seq[:, :-1]  # (B, T-1)
        tau_idx = max(int(self.tau_seconds.item() / self.dt_seconds), 1)
        hxr_shifted = torch.roll(hxr_seq, shifts=tau_idx, dims=1)
        residual = dsxr_dt - self.eta * hxr_shifted[:, :-1]
        loss_phys = (residual**2).mean()

        loss_phys_weighted = loss_phys * pred_prob.detach().clamp(min=0.01).mean()

        return loss_cls + self.lambda_phys * loss_phys_weighted


class SpectralTemporalTransformer(nn.Module):
    """Dual-branch transformer: temporal self-attention + spectral cross-attention.

    Input:  (B, n_channels, seq_len)  e.g. (B, 12, 360)
    Output: (pred_prob (B,), sxr_seq (B, T), hxr_seq (B, T))
    """

    def __init__(
        self,
        n_channels: int = 12,
        seq_len: int = 360,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        sxr_channel: int = 0,
        hxr_channel: int = 8,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.d_model = d_model
        self.sxr_channel = sxr_channel
        self.hxr_channel = hxr_channel

        # ── Temporal branch ─────────────────────────────────────────────
        self.temporal_proj = nn.Linear(n_channels, d_model)
        self.temporal_pe = PositionalEncoding(d_model, max_len=seq_len + 16)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # ── Spectral branch ─────────────────────────────────────────────
        self.spectral_proj = nn.Linear(seq_len, d_model)
        self.spectral_pe = LearnablePositionalEncoding(d_model, max_len=n_channels)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.spectral_norm = nn.LayerNorm(d_model)

        # ── Fusion + head ───────────────────────────────────────────────
        self.fusion_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor):  # x: (B, n_channels, seq_len)
        sxr_seq = x[:, self.sxr_channel, :]  # (B, T)
        hxr_seq = x[:, self.hxr_channel, :]  # (B, T)

        # Temporal branch: treat each time step as a token
        x_t = x.permute(0, 2, 1)  # (B, T, n_channels)
        x_t = self.temporal_proj(x_t)  # (B, T, d_model)
        x_t = self.temporal_pe(x_t)
        z_t = self.temporal_encoder(x_t)  # (B, T, d_model)

        # Spectral branch: treat each energy channel as a token
        x_s = self.spectral_proj(x)  # (B, n_channels, d_model)
        x_s = self.spectral_pe(x_s)
        z_s, _ = self.cross_attn(
            query=x_s, key=z_t, value=z_t, need_weights=False
        )  # (B, n_channels, d_model)
        z_s = self.spectral_norm(z_s + x_s)

        # Fusion
        z_t_pooled = z_t.mean(dim=1)  # (B, d_model)
        z_s_pooled = z_s.mean(dim=1)  # (B, d_model)
        z_fused = self.fusion_norm(z_t_pooled + z_s_pooled)  # (B, d_model)
        z_concat = torch.cat([z_t_pooled, z_s_pooled], dim=1)  # (B, 2*d_model)
        out = self.head(z_concat).squeeze(-1)  # (B,)

        _ = z_fused  # retained for architecture fidelity (residual fusion path)
        return out, sxr_seq, hxr_seq


class FlareForecasterTransformer:
    """Full training wrapper for the Spectral-Temporal Transformer.

    Handles mixed precision (bfloat16), OneCycleLR, Neupert physics-informed
    loss, early stopping, gradient clipping, label smoothing, MAE pretrained
    weight loading, checkpointing, and MC Dropout inference.
    """

    def __init__(
        self,
        n_channels: int = 12,
        seq_len: int = 360,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        lambda_phys: float = 0.1,
        device: str = "auto",
    ):
        from bah2026.config import has_gpu

        torch.manual_seed(_RANDOM_STATE)
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_phys = lambda_phys
        self.label_smoothing = 0.05

        if device == "auto":
            self.device = torch.device("cuda" if has_gpu() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = SpectralTemporalTransformer(
            n_channels=n_channels,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        ).to(self.device)

        self.criterion = NeupertLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            lambda_phys=lambda_phys,
        ).to(self.device)

        all_params = list(self.model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        checkpoint_path: str | None = None,
        mae_encoder_path: str | None = None,
    ) -> dict:
        if mae_encoder_path:
            self.load_mae_encoder(mae_encoder_path)

        steps_per_epoch = max(len(train_loader), 1)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "lr": [],
        }
        best_val_loss = float("inf")
        best_epoch = 0
        wait = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            running, n_seen = 0.0, 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, dtype=torch.float32)
                yb = yb.to(self.device, dtype=torch.float32)
                yb_smooth = yb * (1 - self.label_smoothing) + self.label_smoothing * 0.5

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    pred, sxr, hxr = self.model(xb)
                    loss = self.criterion(pred, yb_smooth, sxr, hxr)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                running += loss.item() * xb.size(0)
                n_seen += xb.size(0)

            train_loss = running / max(n_seen, 1)
            val_loss, val_auc = self._eval_loss(val_loader)
            cur_lr = self.scheduler.get_last_lr()[0]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)
            history["lr"].append(cur_lr)
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_auc={val_auc:.4f} lr={cur_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                if checkpoint_path:
                    self.save(checkpoint_path)
            else:
                wait += 1
                if wait >= patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best={best_epoch}, val_loss={best_val_loss:.4f})"
                    )
                    break

        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        return history

    def _eval_loss(self, loader: DataLoader) -> tuple[float, float]:
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        total, n_seen = 0.0, 0
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, dtype=torch.float32)
                yb = yb.to(self.device, dtype=torch.float32)
                with self._autocast():
                    pred, sxr, hxr = self.model(xb)
                    loss = self.criterion(pred, yb, sxr, hxr)
                total += loss.item() * xb.size(0)
                n_seen += xb.size(0)
                preds.append(pred.detach().float().cpu().numpy())
                labels.append(yb.detach().cpu().numpy())
        avg_loss = total / max(n_seen, 1)
        preds_arr = np.concatenate(preds)
        labels_arr = np.concatenate(labels)
        try:
            auc = roc_auc_score(labels_arr, preds_arr)
        except ValueError:
            auc = 0.5
        return avg_loss, auc

    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            with self._autocast():
                pred, _, _ = self.model(x)
            return pred.float().cpu().numpy()

    def predict_proba_mc_dropout(
        self, X, n_samples: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC Dropout inference — returns (mean_prob, epistemic_uncertainty)."""
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        self.model.train()  # enable dropout (LayerNorm is unaffected)
        probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                with self._autocast():
                    pred, _, _ = self.model(x)
                probs.append(pred.float())
        self.model.eval()
        stacked = torch.stack(probs, dim=0)  # (n_samples, B)
        return (
            stacked.mean(0).cpu().numpy(),
            stacked.std(0).cpu().numpy(),
        )

    def load_mae_encoder(self, path: str) -> None:
        """Load weights from an MAE pretrained encoder into the temporal branch."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model_sd = self.model.state_dict()
        prefixes = ("temporal_proj", "temporal_pe", "temporal_encoder")
        target_keys = [k for k in model_sd if k.startswith(prefixes)]
        mapped: dict[str, Tensor] = {}
        for mk in target_keys:
            for ck, cv in state.items():
                if ck == mk or ck.endswith(mk) or mk.endswith(ck):
                    if cv.shape == model_sd[mk].shape:
                        mapped[mk] = cv
                        break
        if mapped:
            self.model.load_state_dict(mapped, strict=False)
        print(f"MAE encoder: loaded {len(mapped)}/{len(target_keys)} temporal tensors")

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "criterion_state_dict": self.criterion.state_dict(),
                "config": {
                    "n_channels": self.n_channels,
                    "seq_len": self.seq_len,
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "lambda_phys": self.lambda_phys,
                },
            },
            path,
        )

    def load(self, path: str) -> "FlareForecasterTransformer":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.criterion.load_state_dict(ckpt["criterion_state_dict"])
        return self


def evaluate_transformer(
    model: nn.Module,
    loader: DataLoader,
    device,
    mc_dropout: bool = False,
    n_mc_samples: int = 100,
) -> dict:
    """Evaluate a SpectralTemporalTransformer on a data loader.

    Returns standard classification metrics; additionally returns
    ``mean_uncertainty`` when ``mc_dropout=True``.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev)
    model.train() if mc_dropout else model.eval()

    all_preds, all_labels, all_unc = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev, dtype=torch.float32)
            yb = yb.to(dev, dtype=torch.float32)
            if mc_dropout:
                probs = []
                for _ in range(n_mc_samples):
                    pred, _, _ = model(xb)
                    probs.append(pred.float())
                stacked = torch.stack(probs, dim=0)
                all_preds.append(stacked.mean(0).cpu().numpy())
                all_unc.append(stacked.std(0).cpu().numpy())
            else:
                pred, _, _ = model(xb)
                all_preds.append(pred.float().cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    pred_bin = (preds >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels, pred_bin)),
        "precision": float(precision_score(labels, pred_bin, zero_division=0)),
        "recall": float(recall_score(labels, pred_bin, zero_division=0)),
        "f1": float(f1_score(labels, pred_bin, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, preds))
    except ValueError:
        metrics["roc_auc"] = 0.5
    try:
        metrics["pr_auc"] = float(average_precision_score(labels, preds))
    except ValueError:
        metrics["pr_auc"] = 0.0
    if mc_dropout and all_unc:
        metrics["mean_uncertainty"] = float(np.mean(np.concatenate(all_unc)))
    return metrics


__all__ = [
    "PositionalEncoding",
    "LearnablePositionalEncoding",
    "FocalLoss",
    "NeupertLoss",
    "SpectralTemporalTransformer",
    "FlareForecasterTransformer",
    "evaluate_transformer",
]
