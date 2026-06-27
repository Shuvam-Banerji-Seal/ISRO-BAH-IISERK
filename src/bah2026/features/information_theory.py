"""Information-theoretic features for flare precursor detection.

Functions:
  - transfer_entropy(signal_x, signal_y, k=1, bins=16):
        Quantifies causal flow HXR → SXR (precursor index rises pre-flare)
  - sample_entropy(signal, m=2, r_factor=0.2):
        Measures signal complexity (drops during flare onset)
  - mutual_information(signal_x, signal_y, bins=16):
        Shared information between energy bands
  - lagged_cross_correlation(x, y, max_lag=300):
        Finds optimal time lag between SXR and HXR (Neupert delay estimate)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate
from scipy.stats import entropy


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    k: int = 1,
    bins: int = 16,
) -> float:
    """Compute transfer entropy from source to target.

    TE(src → tgt) = Σ p(tgt_{n+1}, tgt_n, src_n) ·
                        log(p(tgt_{n+1} | tgt_n, src_n) / p(tgt_{n+1} | tgt_n))

    A rise in TE(HXR → SXR) is expected 1-15 minutes before flare onset
    due to the Neupert effect (energetic electrons precede chromospheric
    evaporation).

    Parameters
    ----------
    source : ndarray
        Source signal (e.g., HEL1OS HXR count rate).
    target : ndarray
        Target signal (e.g., SoLEXS dSXR/dt).
    k : int
        History length (default 1).
    bins : int
        Number of bins for histogram discretization (default 16).

    Returns
    -------
    te : float
        Transfer entropy value. Higher values indicate stronger causality.
    """
    if len(source) != len(target):
        min_len = min(len(source), len(target))
        source = source[:min_len]
        target = target[:min_len]

    n = len(target) - k
    if n < 10:
        return 0.0

    # Discretize signals
    s_bins = np.linspace(np.min(source), np.max(source), bins + 1)
    t_bins = np.linspace(np.min(target), np.max(target), bins + 1)
    s_disc = np.digitize(source, s_bins) - 1
    t_disc = np.digitize(target, t_bins) - 1

    # Build transition tables
    s_hist = source[:n]
    t_hist = target[:n]
    t_next = target[k:].copy()

    s_bins_a = (
        np.linspace(np.min(s_hist), np.max(s_hist), bins + 1)
        if np.max(s_hist) > np.min(s_hist)
        else np.arange(bins + 1)
    )
    t_bins_a = (
        np.linspace(np.min(t_hist), np.max(t_hist), bins + 1)
        if np.max(t_hist) > np.min(t_hist)
        else np.arange(bins + 1)
    )
    t_next_bins = (
        np.linspace(np.min(t_next), np.max(t_next), bins + 1)
        if np.max(t_next) > np.min(t_next)
        else np.arange(bins + 1)
    )

    s_disc = np.digitize(s_hist, s_bins_a) - 1
    t_disc = np.digitize(t_hist, t_bins_a) - 1
    t_next_disc = np.digitize(t_next, t_next_bins) - 1

    # Clip to valid bin indices
    s_disc = np.clip(s_disc, 0, bins - 1)
    t_disc = np.clip(t_disc, 0, bins - 1)
    t_next_disc = np.clip(t_next_disc, 0, bins - 1)

    # Compute joint and conditional probabilities
    p_joint = np.zeros((bins, bins, bins))
    for i in range(len(s_disc)):
        p_joint[s_disc[i], t_disc[i], t_next_disc[i]] += 1.0
    p_joint /= max(p_joint.sum(), 1)

    te = 0.0
    for s in range(bins):
        for t in range(bins):
            p_st = p_joint[s, t, :].sum()  # p(src, tgt)
            if p_st <= 0:
                continue
            for tn in range(bins):
                p_stn = p_joint[s, t, tn]
                if p_stn <= 0:
                    continue

                # p(tgt_n+1 | tgt_n) — marginal over src
                p_t_given_t = p_joint[:, t, tn].sum() / max(
                    p_joint[:, t, :].sum(), 1e-10
                )

                # p(tgt_n+1 | tgt_n, src_n)
                p_t_given_st = p_stn / p_st

                if p_t_given_st > 0 and p_t_given_t > 0:
                    te += p_stn * np.log2(p_t_given_st / p_t_given_t)

    return float(te)


def sample_entropy(
    signal: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
) -> float:
    """Compute sample entropy of a signal.

    Measures complexity/regularity. Lower values during flares indicate
    a more ordered (less complex) emission pattern.

    Parameters
    ----------
    signal : ndarray
        Input time series.
    m : int
        Embedding dimension (default 2).
    r_factor : float
        Tolerance factor × std (default 0.2).

    Returns
    -------
    sampen : float
        Sample entropy. Higher = more complex/noisy.
    """
    n = len(signal)
    if n < m + 5:
        return 0.0

    r = r_factor * np.std(signal)
    if r < 1e-10:
        return 0.0

    def _count_matches(template_length: int) -> int:
        count = 0
        for i in range(n - template_length):
            for j in range(i + 1, n - template_length):
                if (
                    np.max(
                        np.abs(
                            signal[i : i + template_length]
                            - signal[j : j + template_length]
                        )
                    )
                    < r
                ):
                    count += 1
        return count

    b = _count_matches(m)
    a = _count_matches(m + 1)

    if b == 0 or a == 0:
        return 0.0

    return float(-np.log(a / b))


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 16,
) -> float:
    """Compute mutual information I(X; Y) between two signals.

    Parameters
    ----------
    x : ndarray
        First signal.
    y : ndarray
        Second signal.
    bins : int
        Number of bins for the 2D histogram (default 16).

    Returns
    -------
    mi : float
        Mutual information in bits.
    """
    if len(x) != len(y):
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
    if len(x) < 10:
        return 0.0

    # Discretize
    x_bins = (
        np.linspace(np.min(x), np.max(x), bins + 1)
        if np.max(x) > np.min(x)
        else np.arange(bins + 1)
    )
    y_bins = (
        np.linspace(np.min(y), np.max(y), bins + 1)
        if np.max(y) > np.min(y)
        else np.arange(bins + 1)
    )
    x_disc = np.clip(np.digitize(x, x_bins) - 1, 0, bins - 1)
    y_disc = np.clip(np.digitize(y, y_bins) - 1, 0, bins - 1)

    # 2D histogram → joint probability
    joint = np.zeros((bins, bins))
    for i in range(len(x_disc)):
        joint[x_disc[i], y_disc[i]] += 1.0
    joint /= max(joint.sum(), 1)

    # Marginal probabilities
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    # MI = ΣΣ p(x,y) log(p(x,y) / (p(x)·p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))

    return float(mi)


def lagged_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 300,
) -> tuple[float, int]:
    """Compute lagged cross-correlation between two signals.

    Finds the optimal time lag that maximizes correlation. A positive
    lag means x leads y (useful for detecting HXR leading SXR).

    Parameters
    ----------
    x : ndarray
        First signal (e.g., HEL1OS HXR).
    y : ndarray
        Second signal (e.g., SoLEXS dSXR/dt).
    max_lag : int
        Maximum lag in samples (default 300 s).

    Returns
    -------
    max_corr : float
        Maximum cross-correlation value.
    best_lag : int
        Lag at which max correlation occurs (positive = x leads y).
    """
    if len(x) != len(y):
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
    if len(x) < max_lag * 2:
        return 0.0, 0

    # Normalize
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)

    # Cross-correlation
    corr = correlate(x, y, mode="full", method="auto")
    lags = np.arange(-len(x) + 1, len(x))

    # Restrict to max_lag
    center = len(x) - 1
    lag_range = slice(center - max_lag, center + max_lag + 1)
    corr_restricted = corr[lag_range]
    lags_restricted = lags[lag_range]

    # Normalize
    n = len(x)
    corr_normalized = corr_restricted / n

    best_idx = np.argmax(np.abs(corr_normalized))
    return float(corr_normalized[best_idx]), int(lags_restricted[best_idx])
