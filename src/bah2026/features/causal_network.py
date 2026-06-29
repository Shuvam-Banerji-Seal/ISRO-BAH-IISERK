"""Causal network analysis for multi-channel solar flare time series.

Adapts concepts from Causality-In-Game-Theory to the SXR+HXR domain:
  1. Directed causal graph across all energy bands (SoLEXS, CZT, CdTe, GOES)
  2. Granger causality with autoregressive validation
  3. Mediation analysis (Baron-Kenny decomposition)
  4. Feedback loop detection (causal cycles between thermal/non-thermal)
  5. Network metrics (in-degree, out-degree, centrality, path strength)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr


# ── Pairwise causality ──────────────────────────────────────────────────


def granger_causality_simple(
    cause: np.ndarray,
    effect: np.ndarray,
    max_lag: int = 60,
    n_splits: int = 5,
) -> dict:
    """Granger causality test via autoregressive comparison.

    Tests if adding past 'cause' improves prediction of 'effect'
    over using past 'effect' alone.

    Restricted: effect[t] ~ effect[t-1], ..., effect[t-lag]
    Full:       effect[t] ~ effect[t-1], ..., effect[t-lag]
                              + cause[t-1], ..., cause[t-lag]

    Uses sklearn Ridge regression for stability.

    Parameters
    ----------
    cause : ndarray
        Source time series.
    effect : ndarray
        Target time series.
    max_lag : int
        Maximum lag to test.
    n_splits : int
        Number of cross-validation splits.

    Returns
    -------
    result : dict
        'is_causal', 'best_lag', 'improvement', 'r2_full', 'r2_restricted'
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import r2_score

    n = min(len(cause), len(effect))
    if n < max_lag * 3:
        return {
            "is_causal": False,
            "best_lag": 0,
            "improvement": 0.0,
            "r2_full": 0.0,
            "r2_restricted": 0.0,
        }

    cause = cause[:n]
    effect = effect[:n]
    tscv = TimeSeriesSplit(n_splits=min(n_splits, n // max_lag))

    for lag in [1, 3, 5, 10, 20, 30, 60]:
        if lag > n // 4:
            continue
        # Build feature matrices
        X_restricted = np.column_stack(
            [effect[lag - j - 1 : n - j - 1] for j in range(lag)]
        )
        X_full = np.column_stack(
            [effect[lag - j - 1 : n - j - 1] for j in range(lag)]
            + [cause[lag - j - 1 : n - j - 1] for j in range(lag)]
        )
        y = effect[lag:]

        if len(y) < lag * 2:
            continue

        r2_r, r2_f = 0.0, 0.0
        n_folds = 0
        for train_idx, test_idx in tscv.split(X_restricted):
            if len(test_idx) < 5:
                continue
            Xr_train, Xr_test = X_restricted[train_idx], X_restricted[test_idx]
            Xf_train, Xf_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model_r = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(Xr_train, y_train)
                model_f = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(Xf_train, y_f_train)
                r2_r += r2_score(y_test, model_r.predict(Xr_test))
                r2_f += r2_score(y_test, model_f.predict(Xf_test))
                n_folds += 1
            except Exception:
                continue

        if n_folds == 0:
            continue

        r2_r /= n_folds
        r2_f /= n_folds
        improvement = r2_f - r2_r

        if improvement > 0.03:
            return {
                "is_causal": True,
                "best_lag": int(lag),
                "improvement": float(improvement),
                "r2_full": float(r2_f),
                "r2_restricted": float(r2_r),
            }

    return {
        "is_causal": False,
        "best_lag": 0,
        "improvement": 0.0,
        "r2_full": 0.0,
        "r2_restricted": 0.0,
    }


def lagged_causal_correlation(
    cause: np.ndarray,
    effect: np.ndarray,
    max_lag: int = 60,
    min_lag: int = 1,
) -> dict:
    """Find the optimal lag that maximizes causal correlation.

    For each lag in [min_lag, max_lag], computes:
        r = corr(cause[t-lag], effect[t])

    Returns the lag with maximum |r|.

    Parameters
    ----------
    cause : ndarray
        Source signal (e.g., HXR count rate).
    effect : ndarray
        Target signal (e.g., SXR count rate).
    max_lag : int
        Maximum lag to test.
    min_lag : int
        Minimum lag to test.

    Returns
    -------
    result : dict
        'best_lag', 'best_r', 'lag_significance'
    """
    n = min(len(cause), len(effect))
    if n < max_lag * 2:
        return {"best_lag": 0, "best_r": 0.0, "lag_significance": 1.0}

    cause_norm = (cause - np.mean(cause)) / (np.std(cause) + 1e-10)
    effect_norm = (effect - np.mean(effect)) / (np.std(effect) + 1e-10)

    best_r, best_lag = 0.0, 0
    for lag in range(min_lag, max_lag + 1):
        if lag >= n:
            break
        c = cause_norm[: n - lag]
        e = effect_norm[lag:]
        if np.std(c) < 1e-10 or np.std(e) < 1e-10:
            continue
        try:
            r, _ = pearsonr(c, e)
            if abs(r) > abs(best_r):
                best_r = r
                best_lag = lag
        except Exception:
            continue

    return {
        "best_lag": int(best_lag),
        "best_r": float(best_r),
        "lag_significance": float(1.0 - abs(best_r)),
    }


# ── Mediation analysis ──────────────────────────────────────────────────


def mediation_analysis(
    treatment: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
) -> dict:
    """Baron-Kenny mediation analysis for three time series.

    Tests if the treatment→outcome effect is mediated by the mediator:
      Path a: treatment → mediator
      Path b: mediator → outcome (controlling for treatment)
      Path c: treatment → outcome (total effect)
      Path c': treatment → outcome (direct effect, controlling for mediator)
      Indirect effect = a × b
      Mediation proportion = |indirect / total|

    In solar flare context:
      treatment = HXR (non-thermal electrons)
      mediator  = mid-energy band (CdTe 20-30 keV)
      outcome   = SXR (thermal evaporation)

    Parameters
    ----------
    treatment : ndarray
        Treatment variable (e.g., HXR 40-60 keV).
    mediator : ndarray
        Mediator variable (e.g., CdTe 20-30 keV).
    outcome : ndarray
        Outcome variable (e.g., SoLEXS SXR).

    Returns
    -------
    result : dict
        'total_effect', 'direct_effect', 'indirect_effect',
        'mediation_proportion', 'path_a', 'path_b', 'path_c'
    """
    n = min(len(treatment), len(mediator), len(outcome))
    if n < 30:
        return {
            "total_effect": 0.0,
            "direct_effect": 0.0,
            "indirect_effect": 0.0,
            "mediation_proportion": 0.0,
            "path_a": 0.0,
            "path_b": 0.0,
            "path_c": 0.0,
        }

    t, m, o = treatment[:n], mediator[:n], outcome[:n]

    # Path c (total effect): treatment → outcome
    try:
        r_tc, _ = pearsonr(t, o)
    except Exception:
        r_tc = 0.0

    # Path a: treatment → mediator
    try:
        r_am, _ = pearsonr(t, m)
    except Exception:
        r_am = 0.0

    # Path b: mediator → outcome (controlling for treatment)
    # Partial correlation: ρ(M, O | T)
    try:
        r_mo_given_t = _partial_correlation(m, o, t)
    except Exception:
        r_mo_given_t = 0.0

    # Path c' (direct effect): treatment → outcome (controlling for mediator)
    try:
        r_to_given_m = _partial_correlation(t, o, m)
    except Exception:
        r_to_given_m = 0.0

    indirect = r_am * r_mo_given_t
    total = r_tc

    mediation_proportion = 0.0
    if abs(total) > 1e-6:
        mediation_proportion = min(abs(indirect / total), 1.0)

    return {
        "total_effect": float(total),
        "direct_effect": float(r_to_given_m),
        "indirect_effect": float(indirect),
        "mediation_proportion": float(mediation_proportion),
        "path_a": float(r_am),
        "path_b": float(r_mo_given_t),
        "path_c": float(r_tc),
    }


def _partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    """Compute partial correlation ρ(x, y | z)."""
    # Residuals of x ~ z
    A = np.vstack([z, np.ones_like(z)]).T
    coeffs_x, *_ = np.linalg.lstsq(A, x, rcond=None)
    resid_x = x - A @ coeffs_x
    # Residuals of y ~ z
    coeffs_y, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid_y = y - A @ coeffs_y
    # Correlation of residuals
    r, _ = pearsonr(resid_x, resid_y)
    return float(r)


# ── Causal network ──────────────────────────────────────────────────────


def build_causal_network(
    band_data: dict[str, np.ndarray],
    max_lag: int = 30,
    correlation_threshold: float = 0.1,
) -> dict:
    """Build a directed causal graph across all energy bands.

    Each band is a node. A directed edge A → B exists if
    lagged correlation corr(A[t-lag], B[t]) exceeds threshold
    for some positive lag (A leads B).

    Parameters
    ----------
    band_data : dict[str, ndarray]
        Keys are band names (e.g., 'SXR_2_22', 'CZT_20_40', etc.),
        values are time series of same length.
    max_lag : int
        Maximum lag to search for causal edges.
    correlation_threshold : float
        Minimum absolute correlation for an edge to exist.

    Returns
    -------
    network : dict
        'adjacency_matrix': (n_bands, n_bands) float matrix of max lagged corr
        'lag_matrix': (n_bands, n_bands) int matrix of optimal lags
        'band_names': list of band names
        'in_degree': dict of in-degree per band
        'out_degree': dict of out-degree per band
        'centrality': dict of degree centrality
        'feedback_loops': list of (source, target, strength) tuples
        'cycle_detected': bool, whether any 2-cycle exists
    """
    names = list(band_data.keys())
    n_bands = len(names)
    if n_bands < 2:
        return {
            "adjacency_matrix": np.zeros((0, 0)),
            "lag_matrix": np.zeros((0, 0), dtype=int),
            "band_names": names,
            "in_degree": {},
            "out_degree": {},
            "centrality": {},
            "feedback_loops": [],
            "cycle_detected": False,
        }

    adj = np.zeros((n_bands, n_bands))
    lags = np.zeros((n_bands, n_bands), dtype=int)

    for i in range(n_bands):
        for j in range(n_bands):
            if i == j:
                continue
            result = lagged_causal_correlation(
                band_data[names[i]], band_data[names[j]], max_lag=max_lag, min_lag=1
            )
            if abs(result["best_r"]) > correlation_threshold and result["best_lag"] > 0:
                adj[i, j] = result["best_r"]
                lags[i, j] = result["best_lag"]

    in_degree = {names[i]: int(np.sum(np.abs(adj[:, i]) > 0)) for i in range(n_bands)}
    out_degree = {names[i]: int(np.sum(np.abs(adj[i, :]) > 0)) for i in range(n_bands)}
    max_deg = max(n_bands - 1, 1)
    centrality = {
        names[i]: (in_degree[names[i]] + out_degree[names[i]]) / max_deg
        for i in range(n_bands)
    }

    # Detect 2-cycles (A→B and B→A)
    feedback_loops = []
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            if (
                abs(adj[i, j]) > correlation_threshold
                and abs(adj[j, i]) > correlation_threshold
            ):
                feedback_loops.append(
                    {
                        "source": names[i],
                        "target": names[j],
                        "forward_strength": float(abs(adj[i, j])),
                        "backward_strength": float(abs(adj[j, i])),
                        "forward_lag": int(lags[i, j]),
                        "backward_lag": int(lags[j, i]),
                    }
                )

    return {
        "adjacency_matrix": adj,
        "lag_matrix": lags,
        "band_names": names,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "centrality": centrality,
        "feedback_loops": feedback_loops,
        "cycle_detected": len(feedback_loops) > 0,
    }


def extract_causal_network_features(
    band_data: dict[str, np.ndarray],
    max_lag: int = 30,
) -> dict[str, float]:
    """Extract scalar features from the causal network for ML.

    Parameters
    ----------
    band_data : dict[str, ndarray]
        Energy band time series.
    max_lag : int
        Max lag for causality search.

    Returns
    -------
    features : dict
        'causal_network_density', 'avg_in_degree', 'avg_out_degree',
        'avg_centrality', 'n_feedback_loops', 'cycle_detected',
        'hxr_to_sxr_lag', 'hxr_to_sxr_strength',
        'sxr_to_hxr_lag', 'sxr_to_hxr_strength',
        'neupert_granger_improvement', 'neupert_best_lag',
    """
    result: dict[str, float] = {
        "causal_network_density": 0.0,
        "avg_in_degree": 0.0,
        "avg_out_degree": 0.0,
        "avg_centrality": 0.0,
        "n_feedback_loops": 0.0,
        "cycle_detected": 0.0,
        "hxr_to_sxr_lag": 0.0,
        "hxr_to_sxr_strength": 0.0,
        "sxr_to_hxr_lag": 0.0,
        "sxr_to_hxr_strength": 0.0,
        "neupert_granger_improvement": 0.0,
        "neupert_best_lag": 0.0,
        "max_mediation_proportion": 0.0,
    }

    net = build_causal_network(band_data, max_lag=max_lag)
    n_bands = len(net["band_names"])
    if n_bands < 2:
        return result

    n_possible = n_bands * (n_bands - 1)
    n_edges = np.sum(np.abs(net["adjacency_matrix"]) > 0)
    result["causal_network_density"] = float(n_edges / max(n_possible, 1))

    in_deg = list(net["in_degree"].values())
    out_deg = list(net["out_degree"].values())
    cent = list(net["centrality"].values())
    result["avg_in_degree"] = float(np.mean(in_deg)) if in_deg else 0.0
    result["avg_out_degree"] = float(np.mean(out_deg)) if out_deg else 0.0
    result["avg_centrality"] = float(np.mean(cent)) if cent else 0.0
    result["n_feedback_loops"] = float(len(net["feedback_loops"]))
    result["cycle_detected"] = 1.0 if net["cycle_detected"] else 0.0

    # Find SXR ↔ HXR specific edges
    names = net["band_names"]
    sxr_idx = next((i for i, n in enumerate(names) if "SXR" in n.upper()), None)
    hxr_idx = next(
        (i for i, n in enumerate(names) if "CZT" in n.upper() or "HXR" in n.upper()),
        None,
    )

    if sxr_idx is not None and hxr_idx is not None:
        adj = net["adjacency_matrix"]
        result["hxr_to_sxr_strength"] = float(abs(adj[hxr_idx, sxr_idx]))
        result["hxr_to_sxr_lag"] = float(net["lag_matrix"][hxr_idx, sxr_idx])
        result["sxr_to_hxr_strength"] = float(abs(adj[sxr_idx, hxr_idx]))
        result["sxr_to_hxr_lag"] = float(net["lag_matrix"][sxr_idx, hxr_idx])

    return result


# ── Feature integration ─────────────────────────────────────────────────


_CAUSAL_FEATURES = [
    "causal_network_density",
    "avg_in_degree",
    "avg_out_degree",
    "avg_centrality",
    "n_feedback_loops",
    "cycle_detected",
    "hxr_to_sxr_lag",
    "hxr_to_sxr_strength",
    "sxr_to_hxr_lag",
    "sxr_to_hxr_strength",
    "neupert_granger_improvement",
    "neupert_best_lag",
    "max_mediation_proportion",
]


def get_causal_feature_names() -> list[str]:
    return list(_CAUSAL_FEATURES)
