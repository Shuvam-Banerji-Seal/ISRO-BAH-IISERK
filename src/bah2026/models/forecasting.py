"""Flare forecasting models: LightGBM, XGBoost, CatBoost, CNN-LSTM.

All models now:
  - Use explicit random_state for reproducibility
  - Support early stopping on validation
  - Use focal loss for CNN-LSTM (addresses class imbalance)
"""

from __future__ import annotations

import numpy as np

from bah2026.config import (
    LGBM_N_ESTIMATORS,
    LGBM_LEARNING_RATE,
    LGBM_MAX_DEPTH,
    LGBM_VERBOSE,
    XGB_N_ESTIMATORS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    CATBOOST_ITERATIONS,
    CATBOOST_LEARNING_RATE,
    CATBOOST_DEPTH,
)

_RANDOM_STATE = 42


class FlareForecasterLightGBM:
    """LightGBM-based flare forecaster with early stopping."""

    def __init__(
        self,
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        scale_pos_weight: float = 1.0,
    ):
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators or LGBM_N_ESTIMATORS,
            learning_rate=learning_rate or LGBM_LEARNING_RATE,
            max_depth=max_depth or LGBM_MAX_DEPTH,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            verbose=LGBM_VERBOSE,
            random_state=_RANDOM_STATE,
            n_jobs=-1,  # use all CPU cores
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import lightgbm as lgb

        if X_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self):
        return self.model.feature_importances_


class FlareForecasterXGBoost:
    """XGBoost-based flare forecaster."""

    def __init__(
        self,
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        scale_pos_weight: float = 1.0,
    ):
        import xgboost as xgb

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators or XGB_N_ESTIMATORS,
            learning_rate=learning_rate or XGB_LEARNING_RATE,
            max_depth=max_depth or XGB_MAX_DEPTH,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            verbosity=0,
            random_state=_RANDOM_STATE,
            n_jobs=-1,  # use all CPU cores
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self):
        return self.model.feature_importances_


class FlareForecasterCatBoost:
    """CatBoost-based flare forecaster."""

    def __init__(
        self,
        iterations: int | None = None,
        learning_rate: float | None = None,
        depth: int | None = None,
        auto_class_weights: str = "Balanced",
    ):
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier(
            iterations=iterations or CATBOOST_ITERATIONS,
            learning_rate=learning_rate or CATBOOST_LEARNING_RATE,
            depth=depth or CATBOOST_DEPTH,
            auto_class_weights=auto_class_weights,
            verbose=0,
            eval_metric="Logloss",
            random_seed=_RANDOM_STATE,
            thread_count=-1,  # CPU fallback
            task_type="GPU",  # use A100 80GB when available
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                verbose=0,
                use_best_model=True,
            )
        else:
            self.model.fit(X_train, y_train, verbose=0)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self):
        return self.model.feature_importances_


class FocalLoss:
    """Focal Loss for binary classification (γ focuses on hard examples).

    ℒ_focal(p_t) = -α·(1−p_t)^γ·log(p_t)
    where p_t = p if y=1 else 1-p
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, pred, target):
        import torch

        pred = pred.clamp(1e-7, 1 - 1e-7)
        p_t = target * pred + (1 - target) * (1 - pred)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t)
        return loss.mean()


class FlareForecasterCNNLSTM:
    """CNN-LSTM hybrid model for flare forecasting (PyTorch) with focal loss.

    Input shape: (batch, n_channels, seq_len) for Conv1d.
    Expects sequence-formatted data (not flat feature vectors).
    """

    def __init__(
        self,
        input_len: int | None = None,
        n_channels: int | None = None,
        lr: float | None = None,
    ):
        import torch
        import torch.nn as nn

        from bah2026.config import CNNLSTM_INPUT_LEN, CNNLSTM_N_CHANNELS, CNNLSTM_LR

        self.input_len = input_len or CNNLSTM_INPUT_LEN
        self.n_channels = n_channels or CNNLSTM_N_CHANNELS
        self.lr = lr or CNNLSTM_LR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(_RANDOM_STATE)

        class CNNLSTMNet(nn.Module):
            def __init__(self, n_ch, seq_len):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv1d(n_ch, 32, kernel_size=7, padding=3),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(32),
                )
                self.lstm = nn.LSTM(
                    128, 64, num_layers=2, batch_first=True, dropout=0.3
                )
                self.head = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                c = self.cnn(x).permute(0, 2, 1)
                _, (h, _) = self.lstm(c)
                return self.head(h[-1]).squeeze(-1)

        self.net = CNNLSTMNet(self.n_channels, self.input_len).to(self.device)
        self.criterion = FocalLoss(gamma=2.0, alpha=0.25)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int | None = None,
        batch_size: int | None = None,
    ):
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        from bah2026.config import CNNLSTM_EPOCHS, CNNLSTM_BATCH_SIZE

        epochs = epochs or CNNLSTM_EPOCHS
        batch_size = batch_size or CNNLSTM_BATCH_SIZE

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.net.train()
        for _ in range(epochs):
            for xb, yb in dl:
                optimizer.zero_grad()
                loss = self.criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        import torch

        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.net(X_t).cpu().numpy()
