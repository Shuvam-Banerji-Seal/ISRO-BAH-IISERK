# Forecasting Pipeline Plan

## Objective
Train a predictive model that forecasts solar flares N minutes before they occur, using precursor patterns in combined SoLEXS + HEL1OS light curves.

---

## Pipeline Architecture

```
Historical Data → Event Labeling → Feature Extraction → Sliding Windows → Model Training → Forecasting
```

---

## Step 1: Label Creation (from Nowcast Catalog)

```python
import pandas as pd
import numpy as np
from pathlib import Path

def create_flare_labels(nowcast_catalog, time_grid):
    """
    Create binary + class labels from nowcast catalog.
    
    Parameters
    ----------
    nowcast_catalog : DataFrame
        Must have: time_peak, flare_class, start, end
    time_grid : array
        1-second time grid for the full observation period
    
    Returns
    -------
    labels : DataFrame with columns:
        - time: time_grid
        - is_flare: bool (1 if within any flare interval)
        - flare_class: str (A/B/C/M/X or "quiet")
        - time_to_peak: float (seconds until next flare peak)
    """
    labels = pd.DataFrame({"time": time_grid})
    labels["is_flare"] = False
    labels["flare_class"] = "quiet"
    labels["time_to_peak"] = np.inf
    
    for _, row in nowcast_catalog.iterrows():
        mask = (labels["time"] >= row["start"]) & (labels["time"] <= row["end"])
        labels.loc[mask, "is_flare"] = True
        labels.loc[mask, "flare_class"] = row["flare_class"]
        
        # Time from each point to this flare's peak
        peak_time = row["time_peak"]
        dt = peak_time - labels["time"]
        labels["time_to_peak"] = np.minimum(labels["time_to_peak"], dt)
    
    # For quiet periods, set time_to_peak to infinity
    labels.loc[labels["time_to_peak"] < 0, "time_to_peak"] = np.inf
    
    return labels
```

## Step 2: Sliding Window Features

```python
def create_sliding_windows(counts, labels, lookback=3600, lookahead=1800):
    """
    Create training examples from light curve + labels.
    
    Parameters
    ----------
    counts : array
        Full-day count rate (86400 seconds)
    labels : DataFrame
        Labels with time_to_peak
    lookback : int
        Input window (seconds, e.g., 1 hour)
    lookahead : int
        Prediction window (seconds, e.g., 30 min)
    
    Returns
    -------
    X : array, shape (N, lookback)
    y : array, shape (N,)  - time to next flare (seconds)
    y_class : array, shape (N,) - binary: flare within lookahead?
    """
    X, y_reg, y_bin = [], [], []
    
    for i in range(lookback, len(counts) - lookahead):
        # Input: lookback window
        window = counts[i-lookback:i]
        
        # Regression target: seconds to next flare
        ttp = labels["time_to_peak"].iloc[i]
        reg_target = min(ttp, lookahead)  # cap at lookahead
        bin_target = 1 if ttp <= lookahead else 0
        
        X.append(window)
        y_reg.append(reg_target)
        y_bin.append(bin_target)
    
    return np.array(X), np.array(y_reg), np.array(y_bin)
```

## Step 3: Model Training

### Approach A: LightGBM (Fast, Interpretable)

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_fscore_support

def train_lgbm_forecaster(X, y_reg, y_bin):
    """
    Train dual-output LightGBM model:
    1. Binary: will a flare occur within next 30 min?
    2. Regression: how many seconds until next flare?
    """
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Binary classification
    lgb_binary = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Regression
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31
    )
    
    # Cross-validation
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train_b, y_val_b = y_bin[train_idx], y_bin[val_idx]
        y_train_r, y_val_r = y_reg[train_idx], y_reg[val_idx]
        
        lgb_binary.fit(X_train, y_train_b)
        lgb_reg.fit(X_train, y_train_r)
    
    return lgb_binary, lgb_reg
```

### Approach B: 1D-CNN + LSTM (Deep Learning)

```python
import torch
import torch.nn as nn

class FlareForecasterCNNLSTM(nn.Module):
    """
    CNN + LSTM model for solar flare forecasting.
    
    Architecture:
        Input (3600 × 2) → Conv1D → MaxPool → Conv1D → MaxPool
        → LSTM → FC → Output (2)
    
    The two channels are:
        Channel 0: SoLEXS count rate (normalized)
        Channel 1: HEL1OS count rate (normalized)
    """
    
    def __init__(self, input_len=3600, n_channels=2):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
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
            nn.MaxPool1d(4),
        )
        
        # LSTM temporal model
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Dual heads
        self.binary_head = nn.Linear(64, 1)      # Will flare?
        self.regression_head = nn.Linear(64, 1)   # Time to flare
    
    def forward(self, x):
        # x: (batch, 2, time_len)
        x = self.cnn(x)          # (batch, 128, time_len/16)
        x = x.permute(0, 2, 1)   # (batch, time_len/16, 128)
        _, (h_n, _) = self.lstm(x)
        
        last_hidden = h_n[-1]    # (batch, 64)
        
        binary_logit = self.binary_head(last_hidden)   # (batch, 1)
        time_to_flare = self.regression_head(last_hidden)  # (batch, 1)
        
        return binary_logit, time_to_flare.squeeze(-1)
```

## Step 4: Evaluation Metrics

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

def evaluate_forecast(y_true_bin, y_pred_prob, y_true_reg, y_pred_reg):
    """
    Comprehensive evaluation of forecasting model.
    
    Metrics:
        1. AUC-ROC: Overall classification ability
        2. AUC-PR: Performance on imbalanced classes (flares are rare)
        3. F1-score: Precision-recall balance
        4. Lead Time: Average prediction lead time (seconds)
        5. Mean Absolute Error: Regression accuracy (seconds)
    """
    # Classification metrics
    y_pred_bin = (y_pred_prob > 0.5).astype(int)
    auc_roc = roc_auc_score(y_true_bin, y_pred_prob)
    auc_pr = average_precision_score(y_true_bin, y_pred_prob)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary"
    )
    
    # Regression metrics (for true flare events)
    flare_mask = y_true_bin == 1
    if flare_mask.sum() > 0:
        mae = np.mean(np.abs(y_true_reg[flare_mask] - y_pred_reg[flare_mask]))
    else:
        mae = np.nan
    
    return {
        "AUC_ROC": auc_roc,
        "AUC_PR": auc_pr,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MAE_seconds": mae
    }
```

## Step 5: Prediction Interface

```python
class FlareForecaster:
    """Production forecasting interface."""
    
    def __init__(self, binary_model, reg_model, lookback=3600):
        self.binary_model = binary_model
        self.reg_model = reg_model
        self.lookback = lookback
    
    def predict(self, recent_counts_solexs, recent_counts_hel1os):
        """
        Predict flare probability for next N minutes.
        
        Parameters
        ----------
        recent_counts_solexs : array, shape (3600,)
            Last hour of SoLEXS count rates
        recent_counts_hel1os : array, shape (3600,)
            Last hour of HEL1OS count rates (interpolated if needed)
        
        Returns
        -------
        dict with:
            - probability: float (0-1), flare within next 30 min
            - lead_time: float (seconds until predicted peak)
            - confidence: str ("HIGH"/"MEDIUM"/"LOW")
        """
        # Normalize and stack
        solexs_norm = recent_counts_solexs / np.median(recent_counts_solexs)
        hel1os_norm = recent_counts_hel1os / np.median(recent_counts_hel1os)
        
        X = np.stack([solexs_norm, hel1os_norm], axis=0)  # (2, 3600)
        X = X.reshape(1, 2, self.lookback)  # (1, 2, 3600)
        
        prob = self.binary_model.predict_proba(X)[:, 1][0]
        lead = self.reg_model.predict(X)[0]
        
        confidence = "LOW"
        if prob > 0.8 and lead < 900:
            confidence = "HIGH"
        elif prob > 0.5:
            confidence = "MEDIUM"
        
        return {
            "probability": float(prob),
            "lead_time": max(0, float(lead)),
            "confidence": confidence
        }
```

## Expected Output: Forecast Results

```
Column           Type        Description
-----------      --------    -----------
time_forecast    float64     Time of prediction (from day start)
probability      float64     P(flare within 30 min), range [0, 1]
predicted_peak   float64     Predicted time of flare peak
lead_time        float64     Seconds before predicted peak
confidence       str         HIGH/MEDIUM/LOW
model_name       str         lgbm/cnn_lstm
version          str         Model version string
```

## Target Performance

| Metric | Target | Description |
|--------|--------|-------------|
| AUC-ROC | > 0.95 | Overall classification |
| AUC-PR  | > 0.85 | Imbalanced class detection |
| F1      | > 0.80 | Precision-recall balance |
| MAE     | < 300s | Regression accuracy |
| Lead    | > 10 min | Average prediction lead time |
