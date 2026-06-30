"""Masked Autoencoder (MAE) for self-supervised pretraining on solar X-ray time series.

Learns representations of solar X-ray variability from unlabeled SoLEXS + HEL1OS
combined light curves. The pretrained encoder transfers into the Spectral-Temporal
Transformer's temporal branch (Linear(12->256) + sinusoidal PE + TransformerEncoder).

Input: (B, 12, 360) -- 12 energy channels x 360 time steps (10s cadence, 1h lookback).
Designed for A100 80GB: bf16 autocast, FlashAttention via nn.TransformerEncoderLayer.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (SDPA via nn.TransformerEncoderLayer)
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class MaskingStrategy:
    """Random masking of time steps for MAE pretraining.

    Masks a fraction of time steps (default 50%) across ALL channels.
    The masked positions are the same across all channels for a given sample.
    """

    @staticmethod
    def create_mask(
        batch_size: int,
        seq_len: int,
        mask_ratio: float = 0.5,
        device: str = "cpu",
    ) -> Tensor:
        """Create boolean mask: True = masked, False = visible.

        Returns: (B, seq_len) boolean tensor.
        """
        num_masked = int(seq_len * mask_ratio)
        masks = []
        for _ in range(batch_size):
            perm = torch.randperm(seq_len, device=device)
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            mask[perm[:num_masked]] = True
            masks.append(mask)
        return torch.stack(masks)


def _sinusoidal_pe(max_len: int, d_model: int) -> Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _visible_sort_idx(mask: Tensor) -> Tensor:
    """Return (B, T) permutation that brings visible (unmasked) tokens to the front,
    preserving original temporal order. Visible tokens come first (mask=False sorts
    before mask=True), so [:n_visible] yields ascending original indices."""
    B, T = mask.shape
    sort_key = mask.long() * T + torch.arange(T, device=mask.device).unsqueeze(0)
    return sort_key.argsort(dim=1, stable=True)


class MAEEncoder(nn.Module):
    """Transformer encoder for MAE -- processes only VISIBLE (unmasked) tokens.

    Architecture:
    - Linear projection: (12 channels -> d_model)
    - Sinusoidal positional encoding (added to visible tokens, by original position)
    - TransformerEncoder (6 layers, 8 heads, d_model=256, d_ff=1024, pre-LN, GELU)
    - Final LayerNorm

    Input: (B, n_visible, 12) -- only visible time steps.
    Output: (B, n_visible, d_model) -- encoded visible tokens.

    The linear projection + transformer layers are structurally compatible with the
    Spectral-Temporal Transformer's temporal branch (same d_model=256, nhead=8,
    norm_first layers) so state_dict keys transfer directly.
    """

    pos_encoding: Tensor

    def __init__(
        self,
        n_channels: int = 12,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 360,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.register_buffer(
            "pos_encoding", _sinusoidal_pe(max_seq_len, d_model), persistent=False
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x_visible: Tensor) -> Tensor:
        B, N, _ = x_visible.shape
        x = self.input_proj(x_visible)
        x = x + self.pos_encoding[:N].unsqueeze(0).to(device=x.device, dtype=x.dtype)
        x = self.transformer(x)
        x = self.norm(x)
        return x


class MAEDecoder(nn.Module):
    """Lightweight transformer decoder for MAE -- reconstructs masked tokens.

    Architecture:
    - Linear: (d_model -> d_decoder)
    - Add mask tokens at masked positions, visible tokens at visible positions
    - Add sinusoidal positional encoding for ALL positions
    - TransformerEncoder (4 layers, 4 heads, d_decoder=128, pre-LN, GELU)
    - Final LayerNorm
    - Linear head: (d_decoder -> n_channels) -- reconstruct original channels

    Input:
    - encoded_visible: (B, n_visible, d_model) -- from encoder
    - mask: (B, seq_len) -- boolean, True = masked

    Output: (B, seq_len, n_channels) -- reconstructed full sequence.
    """

    pos_encoding: Tensor

    def __init__(
        self,
        d_model: int = 256,
        d_decoder: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        n_channels: int = 12,
        max_seq_len: int = 360,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(d_model, d_decoder)
        self.mask_token = nn.Parameter(torch.zeros(d_decoder))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer(
            "pos_encoding", _sinusoidal_pe(max_seq_len, d_decoder), persistent=False
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_decoder,
            nhead=nhead,
            dim_feedforward=d_decoder * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_decoder)
        self.head = nn.Linear(d_decoder, n_channels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, encoded_visible: Tensor, mask: Tensor) -> Tensor:
        B, T = mask.shape
        N = encoded_visible.shape[1]
        x = self.proj(encoded_visible)
        d_out = x.shape[-1]
        sorted_idx = _visible_sort_idx(mask)
        vis_idx = sorted_idx[:, :N]
        full = (
            self.mask_token.view(1, 1, -1)
            .expand(B, T, -1)
            .clone()
            .to(dtype=x.dtype, device=x.device)
        )
        full.scatter_(1, vis_idx.unsqueeze(-1).expand(-1, -1, d_out), x)
        full = full + self.pos_encoding[:T].unsqueeze(0).to(
            device=full.device, dtype=full.dtype
        )
        full = self.transformer(full)
        full = self.norm(full)
        return self.head(full)


class MaskedAutoencoder(nn.Module):
    """Full MAE: encoder + decoder + masking.

    Pretraining:
    1. Mask 50% of time steps (same positions across all channels per sample).
    2. Encode only visible tokens (fast -- 50% of sequence).
    3. Decode full sequence (mask tokens at masked positions).
    4. Loss: MSE on masked positions only.

    Fine-tuning:
    - Load encoder weights (input_proj + transformer + norm) into the downstream
      Spectral-Temporal Transformer's temporal branch.
    """

    def __init__(
        self,
        n_channels: int = 12,
        seq_len: int = 360,
        d_model: int = 256,
        mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.encoder = MAEEncoder(
            n_channels=n_channels,
            d_model=d_model,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            max_seq_len=seq_len,
        )
        self.decoder = MAEDecoder(
            d_model=d_model,
            d_decoder=128,
            nhead=4,
            num_layers=4,
            n_channels=n_channels,
            max_seq_len=seq_len,
            dropout=0.1,
        )
        self.mask_ratio = mask_ratio
        self.seq_len = seq_len

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass: mask -> encode visible -> decode full.

        x: (B, n_channels, seq_len).
        Returns (reconstruction (B, C, T), mask (B, T)).
        """
        B, C, T = x.shape
        mask = MaskingStrategy.create_mask(B, T, self.mask_ratio, device=str(x.device))
        x_perm = x.permute(0, 2, 1)  # (B, T, C)
        n_visible = T - int(T * self.mask_ratio)
        sorted_idx = _visible_sort_idx(mask)
        vis_idx = sorted_idx[:, :n_visible]  # (B, n_visible) ascending
        x_visible = torch.gather(
            x_perm, 1, vis_idx.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, n_visible, C)
        encoded = self.encoder(x_visible)  # (B, n_visible, d_model)
        reconstruction = self.decoder(encoded, mask)  # (B, T, C)
        reconstruction = reconstruction.permute(0, 2, 1)  # (B, C, T)
        return reconstruction, mask

    def compute_loss(self, x: Tensor, reconstruction: Tensor, mask: Tensor) -> Tensor:
        """MSE loss on masked positions only."""
        mask_expanded = mask.unsqueeze(1).float()  # (B, 1, T)
        loss = (
            (reconstruction - x) ** 2 * mask_expanded
        ).sum() / mask_expanded.sum().clamp(min=1)
        return loss


class MAEPretrainer:
    """Self-supervised MAE pretraining on unlabeled solar X-ray data.

    Trains on ALL windows (flare and non-flare) to learn the "language" of solar
    X-ray variability. Uses bf16 autocast (no GradScaler needed for bf16), AdamW
    with high weight decay, and cosine LR schedule.
    """

    def __init__(
        self,
        n_channels: int = 12,
        seq_len: int = 360,
        d_model: int = 256,
        mask_ratio: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = MaskedAutoencoder(
            n_channels=n_channels,
            seq_len=seq_len,
            d_model=d_model,
            mask_ratio=mask_ratio,
        ).to(self.device)
        self.lr = lr
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        self._use_cuda = self.device.type == "cuda"

    def _amp_ctx(self):
        if self._use_cuda:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _step(self, x: Tensor, train: bool) -> float:
        grad_ctx = torch.enable_grad() if train else torch.no_grad()
        with grad_ctx:
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with self._amp_ctx():
                reconstruction, mask = self.model(x)
                loss = self.model.compute_loss(x, reconstruction, mask)
            if train:
                loss.backward()
                self.optimizer.step()
        return loss.item()

    def pretrain(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        checkpoint_path: str | None = None,
    ) -> dict:
        """Pretrain the MAE.

        Returns dict with 'train_losses', 'val_losses', 'best_epoch'.
        Saves encoder weights to checkpoint_path on improvement.
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr * 0.01
        )
        train_losses: list[float] = []
        val_losses: list[float] = []
        best = float("inf")
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            running, nb = 0.0, 0
            for batch in train_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device, non_blocking=True)
                running += self._step(x, train=True)
                nb += 1
            scheduler.step()
            train_loss = running / max(nb, 1)
            train_losses.append(train_loss)

            val_loss = float("nan")
            if val_loader is not None:
                self.model.eval()
                v_run, v_nb = 0.0, 0
                for batch in val_loader:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    x = x.to(self.device, non_blocking=True)
                    v_run += self._step(x, train=False)
                    v_nb += 1
                val_loss = v_run / max(v_nb, 1)
                val_losses.append(val_loss)
                metric = val_loss
            else:
                metric = train_loss

            improved = metric < best
            if improved:
                best = metric
                best_epoch = epoch
                if checkpoint_path:
                    self.save_encoder(checkpoint_path)

            cur_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | lr={cur_lr:.2e}"
                + (" | *saved*" if improved and checkpoint_path else "")
            )

        if checkpoint_path and not Path(checkpoint_path).exists():
            self.save_encoder(checkpoint_path)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
        }

    def get_encoder_state_dict(self) -> dict:
        """Return encoder weights for loading into downstream model."""
        return self.model.encoder.state_dict()

    def save_encoder(self, path: str) -> None:
        """Save only the encoder weights (input_proj + transformer + norm)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.encoder.state_dict(), path)

    def save(self, path: str) -> None:
        """Save full MAE (encoder + decoder)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> "MAEPretrainer":
        """Load full MAE."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        return self


def prepare_pretraining_data(
    x_seq_path: str,
    y_path: str | None = None,
    batch_size: int = 512,
    downsample_factor: int = 10,
) -> DataLoader:
    """Create a DataLoader for MAE pretraining (uses ALL data, ignores labels).

    Loads sequences from a .npy file of shape (N, n_channels, T_raw). If
    downsample_factor > 1, averages consecutive samples: (N, C, T_raw) ->
    (N, C, T_raw // factor). Labels (if provided) are loaded but ignored by the
    pretrainer; they remain in the dataset so the same loader can be shared.
    """
    x = np.load(x_seq_path, mmap_mode="r")
    if x.ndim == 2:
        x = x[:, None, :]
    if downsample_factor > 1 and x.shape[-1] >= downsample_factor:
        N, C, T = x.shape
        T_new = T // downsample_factor
        x = (
            x[:, :, : T_new * downsample_factor]
            .reshape(N, C, T_new, downsample_factor)
            .mean(axis=-1)
        )
    x = np.ascontiguousarray(x, dtype=np.float32)
    x_tensor = torch.from_numpy(x)
    if y_path is not None:
        y = np.load(y_path)
        y_tensor = torch.from_numpy(y.astype(np.float32))
        ds = TensorDataset(x_tensor, y_tensor)
    else:
        ds = TensorDataset(x_tensor)
    try:
        from bah2026.config import N_WORKERS

        nw = min(N_WORKERS, 8, os.cpu_count() or 1)
    except Exception:
        nw = min(4, os.cpu_count() or 1)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )
