#!/usr/bin/env python3
"""
BAH 2026 Challenge 15 — Master Pipeline (Single-File Orchestrator)

Controls every phase from data download through final evaluation:

    Phase 0: Download GOES (public NOAA data, automated)
    Phase 1: Download SoLEXS / HEL1OS (PRADAN, requires auth setup)
    Phase 2: Decompress, preprocess, calibrate
    Phase 3: Nowcasting (combined SXR+HXR flare detection)
    Phase 4: Feature extraction (179 tabular features)
    Phase 5: Sequence building (12-channel × 3600-step windows for DL)
    Phase 6: CNN-LSTM v3 training (with 179-feature injection)
    Phase 7: MAE self-supervised pretraining
    Phase 8: Transformer training (with Neupert physics loss)
    Phase 9: Ensemble + final evaluation on held-out test set

Usage:
    # Full pipeline (skips phases where output already exists)
    uv run python -m bah2026.scripts.run_master_pipeline

    # Selective execution
    uv run python -m bah2026.scripts.run_master_pipeline --phase 3-6
    uv run python -m bah2026.scripts.run_master_pipeline --skip-cnn

    # Force re-run specific phases
    uv run python -m bah2026.scripts.run_master_pipeline \
        --force-phase 4 --force-phase 8

    # Download-only
    uv run python -m bah2026.scripts.run_master_pipeline --phase 0-1

    # Train only (assumes features + sequences exist)
    uv run python -m bah2026.scripts.run_master_pipeline --phase 6-9
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# ── Ensure project root is on sys.path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("BAH2026_DATA", str(PROJECT_ROOT / "data" / "processed"))
os.environ.setdefault("BAH2026_OUTPUT", str(PROJECT_ROOT / "output"))

from bah2026.config import (
    CATALOGS_DIR,
    DATA_ROOT,
    EXTERNAL_DATA_DIR,
    GOES_DATA_DIR,
    HDF5_DIR,
    HEL1OS_PROCESSED_DIR,
    HEL1OS_ZIP_DIR,
    MODELS_DIR,
    N_WORKERS,
    OUTPUT_ROOT,
    PLOTS_DIR,
    RAW_DATA_DIR,
    SOLEXS_PROCESSED_DIR,
    SOLEXS_ZIP_DIR,
    CNNLSTM_BATCH_SIZE,
    CNNLSTM_EPOCHS,
    CNNLSTM_LR,
    CNNLSTM_N_CHANNELS,
    CNNLSTM_N_FEATURES,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DOWNSAMPLE,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_FOCAL_ALPHA,
    TRANSFORMER_FOCAL_GAMMA,
    TRANSFORMER_LAMBDA_PHYS,
    TRANSFORMER_LR,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_PATIENCE,
    TRANSFORMER_SEQ_LEN,
    TRANSFORMER_WEIGHT_DECAY,
    MAE_BATCH_SIZE,
    MAE_D_MODEL,
    MAE_EPOCHS,
    MAE_LR,
    MAE_MASK_RATIO,
    MAE_WEIGHT_DECAY,
    ensure_output_dirs,
    detect_gpu,
)

log = logging.getLogger("bah2026")
log.setLevel(logging.INFO)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_ch)
_fh = logging.FileHandler(PROJECT_ROOT / "logs" / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_fh)

_RANDOM_STATE = 42

# ── Helpers ──────────────────────────────────────────────────────────────

def _check_file(path: Path, desc: str) -> bool:
    exists = path.exists()
    log.info("  %-45s %s", desc, "FOUND" if exists else "NOT FOUND")
    return exists


def _r2(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _save_metrics(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        k: (v if not isinstance(v, (np.ndarray,)) else v.tolist())
        for k, v in data.items()
    }
    path.write_text(json.dumps(clean, indent=2, default=str))
    log.info("  Metrics saved: %s", path)


# ══════════════════════════════════════════════════════════════════════════
# Phase 0 — Download GOES (public NOAA data)
# ══════════════════════════════════════════════════════════════════════════

GOES_URL = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
    "goes/goes16/l2/data/xrsf-l2-flx1s"
)
GOES_START = date(2024, 2, 1)
GOES_END = date(2026, 6, 22)


def phase_download_goes(force: bool = False) -> int:
    """Download GOES-16 XRSF L2 netCDF files from NOAA.

    Returns number of files downloaded.
    """
    log.info("─" * 60)
    log.info("PHASE 0: Download GOES XRSF data (public NOAA)")
    log.info("─" * 60)

    GOES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(GOES_DATA_DIR.glob("*.nc")))
    if existing > 100 and not force:
        log.info("  GOES data already present: %d files. Use --force-phase 0 to re-download.", existing)
        return 0

    import urllib.request
    from html.parser import HTMLParser

    class LinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for name, val in attrs:
                    if name == "href" and val and val.endswith(".nc"):
                        self.links.append(val)

    downloaded = 0
    d = GOES_START
    while d <= GOES_END:
        ymd = d.strftime("%Y%m%d")
        url = f"{GOES_URL}/{d.year}/{d.strftime('%m')}/sci_xrsf-l2-flx1s_g16_d{ymd}_v2-2-1.nc"
        fname = f"sci_xrsf-l2-flx1s_g16_d{ymd}_v2-2-1.nc"
        dest = GOES_DATA_DIR / fname
        if dest.exists() and not force:
            d += timedelta(days=1)
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
            if downloaded % 50 == 0:
                log.info("  Downloaded %d GOES files...", downloaded)
        except Exception:
            pass
        d += timedelta(days=1)

    log.info("  GOES download complete: %d new files (total %d in dir)",
             downloaded, len(list(GOES_DATA_DIR.glob("*.nc"))))
    return downloaded


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 — Download SoLEXS / HEL1OS (PRADAN, requires auth)
# ══════════════════════════════════════════════════════════════════════════

def _check_pradan_auth() -> bool:
    """Check if PRADAN auth cookie is available."""
    cookie_txt = Path("/tmp/pradan_cookie.txt")
    cookie_json = Path("/tmp/pradan_cookies.json")
    return cookie_txt.exists() or cookie_json.exists()


def _count_zips(directory: Path) -> int:
    return len(list(directory.glob("*.zip"))) if directory.exists() else 0


def phase_download_solexs(force: bool = False) -> int:
    """Download SoLEXS data from PRADAN (requires browser auth).

    Returns number of zip files downloaded (or already present).
    """
    log.info("─" * 60)
    log.info("PHASE 1a: Download SoLEXS data (PRADAN)")
    log.info("─" * 60)

    SOLEXS_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    existing = _count_zips(SOLEXS_ZIP_DIR)

    if existing >= 700 and not force:
        log.info("  SoLEXS zips already present: %d. Use --force-phase 1 to re-download.", existing)
        return existing

    if not _check_pradan_auth():
        log.warning("  PRADAN auth cookie not found!")
        log.warning("  Steps to set up auth:")
        log.warning("    1. Log into https://pradan1.issdc.gov.in in a browser")
        log.warning("    2. Open DevTools → Network → copy cookie header")
        log.warning("    3. Run: cookie_grabber.py --watch")
        log.warning("    Or paste cookie: echo 'COOKIE_VALUE' > /tmp/pradan_cookie.txt")
        log.warning("  Then re-run this pipeline.")
        return existing

    dl_script = PROJECT_ROOT / "data" / "downloads" / "parallel_dl.sh"
    if not dl_script.exists():
        log.error("  Download script not found: %s", dl_script)
        return existing

    log.info("  Running parallel download for SoLEXS...")
    subprocess.run(
        ["bash", str(dl_script), "solexs"],
        cwd=str(PROJECT_ROOT / "data" / "downloads"),
    )

    after = _count_zips(SOLEXS_ZIP_DIR)
    log.info("  SoLEXS zips: %d → %d", existing, after)
    return after


def phase_download_hel1os(force: bool = False) -> int:
    """Download HEL1OS data from PRADAN (requires browser auth).

    Returns number of zip files downloaded (or already present).
    """
    log.info("─" * 60)
    log.info("PHASE 1b: Download HEL1OS data (PRADAN)")
    log.info("─" * 60)

    HEL1OS_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    existing = _count_zips(HEL1OS_ZIP_DIR)

    if existing >= 2000 and not force:
        log.info("  HEL1OS zips already present: %d. Use --force-phase 1 to re-download.", existing)
        return existing

    if not _check_pradan_auth():
        log.warning("  PRADAN auth cookie not found! See instructions above.")
        return existing

    dl_script = PROJECT_ROOT / "data" / "downloads" / "parallel_dl.sh"
    if not dl_script.exists():
        log.error("  Download script not found: %s", dl_script)
        return existing

    log.info("  Running parallel download for HEL1OS (this may take hours)...")
    subprocess.run(
        ["bash", str(dl_script), "hel1os"],
        cwd=str(PROJECT_ROOT / "data" / "downloads"),
    )

    after = _count_zips(HEL1OS_ZIP_DIR)
    log.info("  HEL1OS zips: %d → %d", existing, after)
    return after


# ══════════════════════════════════════════════════════════════════════════
# Phase 2 — Decompress & Preprocess
# ══════════════════════════════════════════════════════════════════════════

def _count_solexs_days() -> int:
    return len(list(SOLEXS_PROCESSED_DIR.rglob("*.lc"))) if SOLEXS_PROCESSED_DIR.exists() else 0


def _count_hel1os_days() -> int:
    return len(list(HEL1OS_PROCESSED_DIR.rglob("*lightcurve*"))) if HEL1OS_PROCESSED_DIR.exists() else 0


def phase_decompress(force: bool = False) -> tuple[int, int]:
    """Decompress raw zips into processed FITS files and concatenate orbits."""
    log.info("─" * 60)
    log.info("PHASE 2a: Decompress SoLEXS + HEL1OS zips")
    log.info("─" * 60)

    sx_before = _count_solexs_days()
    hl_before = _count_hel1os_days()

    if sx_before > 700 and hl_before > 900 and not force:
        log.info("  Data already decompressed: SoLEXS=%d days, HEL1OS=%d days", sx_before, hl_before)
        return sx_before, hl_before

    decompress = PROJECT_ROOT / "data" / "downloads" / "decompress.sh"
    if decompress.exists():
        log.info("  Running decompress for SoLEXS...")
        subprocess.run(["bash", str(decompress), "solexs"], cwd=str(PROJECT_ROOT / "data" / "downloads"))
        log.info("  Running decompress for HEL1OS...")
        subprocess.run(["bash", str(decompress), "hel1os"], cwd=str(PROJECT_ROOT / "data" / "downloads"))
    else:
        log.warning("  decompress.sh not found — attempting inline decompression")
        _inline_decompress_solexs()
        _inline_decompress_hel1os()

    sx_after = _count_solexs_days()
    hl_after = _count_hel1os_days()
    log.info("  SoLEXS days: %d → %d", sx_before, sx_after)
    log.info("  HEL1OS days: %d → %d", hl_before, hl_after)

    log.info("─" * 60)
    log.info("PHASE 2b: Concatenate HEL1OS multi-orbit data")
    log.info("─" * 60)
    concat = PROJECT_ROOT / "data" / "downloads" / "concat_orbits.py"
    if concat.exists():
        subprocess.run([sys.executable, str(concat)])
    else:
        log.warning("  concat_orbits.py not found — skipping orbit concatenation")

    return sx_after, hl_after


def _inline_decompress_solexs():
    """Basic SoLEXS zip extraction fallback."""
    import zipfile
    SOLEXS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for z in sorted(SOLEXS_ZIP_DIR.glob("*.zip")):
        ymd = z.stem.split("_")[-2] if "L1" in z.stem else z.stem[:8]
        day_dir = SOLEXS_PROCESSED_DIR / ymd[:4] / ymd[4:6] / ymd[6:8] / "SDD2"
        if day_dir.exists():
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(day_dir.parent.parent.parent)
        except Exception:
            continue


def _inline_decompress_hel1os():
    import zipfile
    for z in sorted(HEL1OS_ZIP_DIR.glob("*.zip")):
        parts = z.stem.split("_")
        ymd = parts[1]
        day_dir = HEL1OS_PROCESSED_DIR / ymd[:4] / ymd[4:6] / ymd[6:8]
        if day_dir.exists():
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    if "lightcurve" in m or "hk" in m:
                        zf.extract(m, day_dir)
        except Exception:
            continue


# ══════════════════════════════════════════════════════════════════════════
# Phase 3 — Nowcasting
# ══════════════════════════════════════════════════════════════════════════

def phase_nowcasting() -> pd.DataFrame:
    """Run combined SXR+HXR flare detection.

    Delegates to ``main.cmd_nowcast()``. Returns catalogue DataFrame.
    """
    log.info("─" * 60)
    log.info("PHASE 3: Nowcasting — combined SXR+HXR flare detection")
    log.info("─" * 60)

    catalogue = CATALOGS_DIR / "nowcast_catalogue.csv"
    if catalogue.exists():
        df = pd.read_csv(catalogue)
        log.info("  Existing catalogue: %d events (%s)", len(df), catalogue)
        return df

    from bah2026.data import discover_combined_days
    from bah2026.main import cmd_nowcast

    days = discover_combined_days()
    log.info("  Combined days: %d", len(days))
    df = cmd_nowcast(days)
    log.info("  Detected %d flare events", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════
# Phase 4 — Feature Extraction
# ══════════════════════════════════════════════════════════════════════════

def phase_features(events_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract 179 tabular features from all combined days.

    Delegates to ``main.cmd_features()``. Returns (X, y, feature_names).
    """
    log.info("─" * 60)
    log.info("PHASE 4: Feature extraction — 179 features per window")
    log.info("─" * 60)

    x_path = HDF5_DIR / "X_features.npy"
    y_path = HDF5_DIR / "y_labels.npy"
    fn_path = HDF5_DIR / "feature_names.json"

    if x_path.exists() and y_path.exists():
        X = np.load(x_path)
        y = np.load(y_path)
        fnames = json.loads(fn_path.read_text()) if fn_path.exists() else []
        log.info("  Cached features: X=%s, y=%s, pos=%d (%.1f%%)",
                 X.shape, y.shape, y.sum(), 100 * y.mean())
        return X, y, fnames

    from bah2026.data import discover_combined_days
    from bah2026.main import cmd_features

    days = discover_combined_days()
    log.info("  Extracting features from %d days (%d workers)...", len(days), N_WORKERS)

    X, y, fnames = cmd_features(days, events_df)
    log.info("  Feature matrix: X=%s, y=%s, %d features", X.shape, y.shape, len(fnames))
    return X, y, fnames


# ══════════════════════════════════════════════════════════════════════════
# Phase 5 — Sequence Building (for Deep Learning)
# ══════════════════════════════════════════════════════════════════════════

def phase_sequences(events_df: pd.DataFrame) -> tuple[Path, Path]:
    """Build 12-channel time-series sequences for CNN-LSTM / Transformer.

    Produces X_seq.npy (N, 12, 3600) and y_seq.npy (N,) in HDF5_DIR/sequences/.
    Also produces downsampled X_seq_ds10.npy (N, 12, 360) for Transformer.
    Returns (x_seq_path, y_seq_path).
    """
    log.info("─" * 60)
    log.info("PHASE 5: Sequence building — 12-channel time-series windows")
    log.info("─" * 60)

    seq_dir = HDF5_DIR / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    x_path = seq_dir / "X_seq.npy"
    y_path = seq_dir / "y_seq.npy"

    if x_path.exists() and y_path.exists():
        log.info("  Sequences already exist: %s", x_path)
        return x_path, y_path

    from bah2026.data import discover_combined_days
    from bah2026.data.sequence_builder import build_all_sequences

    days = discover_combined_days()
    event_times_map: dict[str, list[float]] = {}
    for _, row in events_df.iterrows():
        event_times_map.setdefault(row["date"], []).append(row["peak_time"])

    log.info("  Building sequences for %d days (%d workers)...", len(days), N_WORKERS)
    build_all_sequences(days, event_times_map, str(seq_dir))

    # Downsample for Transformer (3600→360 at 10s cadence)
    ds_path = seq_dir / "X_seq_ds10.npy"
    if x_path.exists() and not ds_path.exists():
        log.info("  Downsampling sequences 1s→10s for Transformer...")
        from bah2026.data.sequence_builder import prepare_downsampled_sequences
        prepare_downsampled_sequences(str(x_path), str(y_path), str(seq_dir), factor=10)

    log.info("  Sequences ready: %s, %s", x_path, y_path)
    return x_path, y_path


# ══════════════════════════════════════════════════════════════════════════
# Phase 6 — CNN-LSTM v3 Training (with 179-feature injection)
# ══════════════════════════════════════════════════════════════════════════

def _make_loader_from_npy(x_path: Path, y_path: Path, feat_path: Path | None,
                          indices: np.ndarray, batch_size: int,
                          shuffle: bool = False) -> DataLoader:
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    X_sel = np.ascontiguousarray(X[indices], dtype=np.float32)
    y_sel = np.ascontiguousarray(y[indices], dtype=np.float32)
    if feat_path is not None and feat_path.exists():
        F = np.load(feat_path, mmap_mode="r")
        F_sel = np.ascontiguousarray(F[indices], dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(X_sel), torch.from_numpy(F_sel),
                           torch.from_numpy(y_sel))
    else:
        ds = TensorDataset(torch.from_numpy(X_sel), torch.from_numpy(y_sel))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def phase_cnn_lstm(x_seq_path: Path, y_seq_path: Path,
                   x_feat_path: Path | None = None) -> dict:
    """Train CNN-LSTM v3 on sequence data with 179-feature injection."""
    log.info("─" * 60)
    log.info("PHASE 6: CNN-LSTM v3 Training (feature injection)")
    log.info("─" * 60)

    ckpt = MODELS_DIR / "cnn_lstm_v3_best.pt"
    results_path = MODELS_DIR / "cnn_lstm_results.json"

    if ckpt.exists() and results_path.exists():
        log.info("  CNN-LSTM checkpoint exists: %s", ckpt)
        return json.loads(results_path.read_text())

    detect_gpu()
    from bah2026.models.cnn_lstm_v3 import FlareForecasterCNNLSTMv3, evaluate_model

    y = np.load(y_seq_path)
    n = len(y)
    tr = int(n * 0.70)
    va = int(n * 0.85)

    train_loader = _make_loader_from_npy(x_seq_path, y_seq_path, x_feat_path,
                                          np.arange(tr), CNNLSTM_BATCH_SIZE, shuffle=True)
    val_loader = _make_loader_from_npy(x_seq_path, y_seq_path, x_feat_path,
                                        np.arange(tr, va), CNNLSTM_BATCH_SIZE)
    test_loader = _make_loader_from_npy(x_seq_path, y_seq_path, x_feat_path,
                                         np.arange(va, n), CNNLSTM_BATCH_SIZE)

    model = FlareForecasterCNNLSTMv3(
        n_channels=CNNLSTM_N_CHANNELS,
        n_features=CNNLSTM_N_FEATURES,
        lr=CNNLSTM_LR,
    )

    log.info("  Training CNN-LSTM: %d train, %d val, %d test | device=%s, features=%d",
             tr, va - tr, n - va, model.device, CNNLSTM_N_FEATURES)
    history = model.fit(
        train_loader, val_loader,
        epochs=CNNLSTM_EPOCHS, patience=10, checkpoint_path=str(ckpt),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model.model, test_loader, device, n_features=CNNLSTM_N_FEATURES)
    metrics["history"] = {k: v if isinstance(v, list) else v for k, v in history.items()}

    _save_metrics(results_path, metrics)
    log.info("  CNN-LSTM TSS=%.3f AUC-ROC=%.3f F1=%.3f",
             metrics.get("tss", 0), metrics.get("auc_roc", 0), metrics.get("f1", 0))
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# Phase 7 — MAE Self-Supervised Pretraining
# ══════════════════════════════════════════════════════════════════════════

def phase_mae_pretrain(x_seq_path: Path) -> str:
    """Pretrain Masked Autoencoder on unlabeled sequences.

    Returns path to saved encoder weights.
    """
    log.info("─" * 60)
    log.info("PHASE 7: MAE Self-Supervised Pretraining")
    log.info("─" * 60)

    encoder_ckpt = MODELS_DIR / "mae_encoder.pt"
    if encoder_ckpt.exists():
        log.info("  MAE encoder exists: %s", encoder_ckpt)
        return str(encoder_ckpt)

    detect_gpu()
    from bah2026.models.mae_pretrain import MAEPretrainer, prepare_pretraining_data

    X = np.load(x_seq_path, mmap_mode="r")
    if X.shape[-1] >= TRANSFORMER_DOWNSAMPLE:
        N, C, T = X.shape
        T_new = T // TRANSFORMER_DOWNSAMPLE
        X_ds = X[:, :, :T_new * TRANSFORMER_DOWNSAMPLE].reshape(N, C, T_new, TRANSFORMER_DOWNSAMPLE).mean(axis=-1)
        X_ds = np.ascontiguousarray(X_ds, dtype=np.float32)
    else:
        X_ds = np.ascontiguousarray(X, dtype=np.float32)

    split = int(len(X_ds) * 0.85)
    train_ds = TensorDataset(torch.from_numpy(X_ds[:split]))
    val_ds = TensorDataset(torch.from_numpy(X_ds[split:]))

    train_loader = DataLoader(train_ds, batch_size=MAE_BATCH_SIZE, shuffle=True,
                               num_workers=min(N_WORKERS, 8), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=MAE_BATCH_SIZE * 2, shuffle=False,
                             num_workers=min(N_WORKERS, 4))

    pretrainer = MAEPretrainer(
        n_channels=X.shape[1],
        seq_len=X_ds.shape[2],
        d_model=MAE_D_MODEL,
        mask_ratio=MAE_MASK_RATIO,
        lr=MAE_LR,
        weight_decay=MAE_WEIGHT_DECAY,
    )

    log.info("  MAE pretraining: %d samples, %s", len(X_ds), pretrainer.device)
    history = pretrainer.pretrain(
        train_loader, val_loader,
        epochs=MAE_EPOCHS,
        checkpoint_path=str(encoder_ckpt),
    )

    _save_metrics(MODELS_DIR / "mae_results.json", history)
    log.info("  MAE done: best loss=%.6f at epoch %d",
             min(history["train_losses"]), history["best_epoch"])
    return str(encoder_ckpt)


# ══════════════════════════════════════════════════════════════════════════
# Phase 9 — Transformer Training (with optional MAE finetune)
# ══════════════════════════════════════════════════════════════════════════

def _load_downsampled(x_seq_path: Path, y_seq_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and downsample sequences for Transformer (1s→10s cadence)."""
    ds_path = x_seq_path.parent / "X_seq_ds10.npy"
    ds_y = y_seq_path.parent / "y_seq_ds10.npy"
    if ds_path.exists() and ds_y.exists():
        X = np.load(ds_path, mmap_mode="r")
        y = np.load(ds_y, mmap_mode="r")
    else:
        X = np.load(x_seq_path, mmap_mode="r")
        y = np.load(y_seq_path, mmap_mode="r")
        if X.shape[-1] >= TRANSFORMER_DOWNSAMPLE:
            N, C, T = X.shape
            T_new = T // TRANSFORMER_DOWNSAMPLE
            X = X[:, :, :T_new * TRANSFORMER_DOWNSAMPLE].reshape(N, C, T_new, TRANSFORMER_DOWNSAMPLE).mean(axis=-1)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.int8)
    return X, y


def _chronological_split(X, y, tr_ratio=0.70, va_ratio=0.15):
    n = len(X)
    tr = int(n * tr_ratio)
    va = int(n * (tr_ratio + va_ratio))
    return (X[:tr], y[:tr]), (X[tr:va], y[tr:va]), (X[va:], y[va:])


def phase_transformer(x_seq_path: Path, y_seq_path: Path,
                       mae_encoder_path: str | None = None) -> dict:
    """Train Spectral-Temporal Transformer with Neupert physics loss."""
    log.info("─" * 60)
    log.info("PHASE 8: Spectral-Temporal Transformer Training")
    log.info("─" * 60)

    ckpt = MODELS_DIR / "transformer_best.pt"
    results_path = MODELS_DIR / "transformer_results.json"
    if ckpt.exists() and results_path.exists():
        log.info("  Transformer checkpoint exists: %s", ckpt)
        return json.loads(results_path.read_text())

    detect_gpu()
    from bah2026.models.transformer import (
        FlareForecasterTransformer,
        evaluate_transformer,
    )

    X, y = _load_downsampled(x_seq_path, y_seq_path)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = _chronological_split(X, y)

    def _dl(x, y_, bs, shuf=False):
        return DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y_)),
            batch_size=bs, shuffle=shuf, drop_last=False,
        )

    train_loader = _dl(X_tr, y_tr, TRANSFORMER_BATCH_SIZE, shuffle=True)
    val_loader = _dl(X_va, y_va, TRANSFORMER_BATCH_SIZE)
    test_loader = _dl(X_te, y_te, TRANSFORMER_BATCH_SIZE)

    model = FlareForecasterTransformer(
        n_channels=X.shape[1],
        seq_len=X.shape[2],
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        lr=TRANSFORMER_LR,
        weight_decay=TRANSFORMER_WEIGHT_DECAY,
        focal_gamma=TRANSFORMER_FOCAL_GAMMA,
        focal_alpha=TRANSFORMER_FOCAL_ALPHA,
        lambda_phys=TRANSFORMER_LAMBDA_PHYS,
    )

    log.info("  Transformer: %d/%d/%d samples, %s",
             len(X_tr), len(X_va), len(X_te), model.device)
    log.info("  Params: d_model=%d nhead=%d layers=%d lambda_phys=%.1f",
             TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS,
             TRANSFORMER_LAMBDA_PHYS)
    if mae_encoder_path:
        log.info("  Loading MAE encoder: %s", mae_encoder_path)

    history = model.fit(
        train_loader, val_loader,
        epochs=TRANSFORMER_EPOCHS,
        patience=TRANSFORMER_PATIENCE,
        checkpoint_path=str(ckpt),
        mae_encoder_path=mae_encoder_path,
    )

    metrics = evaluate_transformer(
        model.model, test_loader, model.device, mc_dropout=True,
    )
    metrics["history"] = {
        k: (v if isinstance(v, list) else v) for k, v in history.items()
    }

    _save_metrics(results_path, metrics)
    log.info("  Transformer: ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f",
             metrics.get("roc_auc", 0), metrics.get("pr_auc", 0), metrics.get("f1", 0))
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# Phase 9 — Ensemble & Final Evaluation
# ══════════════════════════════════════════════════════════════════════════

def phase_ensemble_evaluation(
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_seq_path: Path | None = None,
    y_seq_path: Path | None = None,
) -> dict:
    """Evaluate all trained DL models on held-out test set and produce ensemble."""
    log.info("─" * 60)
    log.info("PHASE 9: Ensemble & Final Evaluation")
    log.info("─" * 60)

    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, confusion_matrix,
    )

    results = {}
    test_start = int(len(y_test) * 0)  # already the test slice

    # CNN-LSTM
    cnn_results = MODELS_DIR / "cnn_lstm_results.json"
    if cnn_results.exists():
        cnn_data = json.loads(cnn_results.read_text())
        for k in ["tss", "auc_roc", "auc_pr", "f1", "precision", "recall"]:
            if k in cnn_data:
                results[f"cnn_lstm_{k}"] = cnn_data[k]

    # Transformer
    tf_results = MODELS_DIR / "transformer_results.json"
    if tf_results.exists():
        tf_data = json.loads(tf_results.read_text())
        for k in ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "mean_uncertainty"]:
            v = tf_data.get(k)
            if v is not None:
                results[f"transformer_{k}"] = v

    _save_metrics(CATALOGS_DIR / "ensemble_results.json", results)

    log.info("─" * 60)
    log.info("FINAL ENSEMBLE RESULTS")
    log.info("─" * 60)
    for k, v in sorted(results.items()):
        log.info("  %-35s %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    return results


# ══════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ══════════════════════════════════════════════════════════════════════════

def run_phase(phase_num: int, force: bool, ctx: dict) -> dict:
    """Run a single phase, updating context with its outputs.

    ctx is a dict shared across phases containing:
        events_df, X, y, fnames, x_seq_path, y_seq_path, mae_encoder_path
    """
    t0 = time.time()

    if phase_num == 0:
        phase_download_goes(force=force)

    elif phase_num == 1:
        phase_download_solexs(force=force)
        phase_download_hel1os(force=force)

    elif phase_num == 2:
        ctx["solexs_days"], ctx["hel1os_days"] = phase_decompress(force=force)

    elif phase_num == 3:
        ctx["events_df"] = phase_nowcasting()

    elif phase_num == 4:
        df = ctx.get("events_df")
        if df is None:
            csv = CATALOGS_DIR / "nowcast_catalogue.csv"
            df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
        X, y, fnames = phase_features(df)
        ctx["X"], ctx["y"], ctx["fnames"] = X, y, fnames

    elif phase_num == 5:
        df = ctx.get("events_df")
        if df is None:
            csv = CATALOGS_DIR / "nowcast_catalogue.csv"
            df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
        xp, yp = phase_sequences(df)
        ctx["x_seq_path"], ctx["y_seq_path"] = xp, yp

    elif phase_num == 6:
        xp = ctx.get("x_seq_path") or HDF5_DIR / "sequences" / "X_seq.npy"
        yp = ctx.get("y_seq_path") or HDF5_DIR / "sequences" / "y_seq.npy"
        fp = HDF5_DIR / "X_features.npy"
        if xp.exists() and yp.exists():
            phase_cnn_lstm(xp, yp, x_feat_path=fp)

    elif phase_num == 7:
        xp = ctx.get("x_seq_path") or HDF5_DIR / "sequences" / "X_seq.npy"
        if xp.exists():
            ctx["mae_encoder_path"] = phase_mae_pretrain(xp)

    elif phase_num == 8:
        xp = ctx.get("x_seq_path") or HDF5_DIR / "sequences" / "X_seq.npy"
        yp = ctx.get("y_seq_path") or HDF5_DIR / "sequences" / "y_seq.npy"
        mae = ctx.get("mae_encoder_path")
        if xp.exists() and yp.exists():
            phase_transformer(xp, yp, mae_encoder_path=mae)

    elif phase_num == 9:
        y = ctx.get("y")
        xp = ctx.get("x_seq_path")
        yp = ctx.get("y_seq_path")
        test_y = y[int(len(y) * 0.85):] if y is not None else None
        phase_ensemble_evaluation(
            X_test=np.empty(0), y_test=test_y or np.empty(0),
            X_seq_path=xp, y_seq_path=yp,
        )

    elapsed = time.time() - t0
    log.info("  Phase %d completed in %s", phase_num, _r2(elapsed))
    return ctx


def main():
    p = argparse.ArgumentParser(
        description="BAH 2026 Master Pipeline — End-to-End Solar Flare Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--phase", type=str, default="0-9",
                    help="Phase range to run, e.g. '3-6' or '0,2,4' (default: 0-9)")
    p.add_argument("--force-phase", type=int, nargs="*", default=None,
                    help="Force re-run specific phase numbers")
    p.add_argument("--skip-download", action="store_true",
                    help="Skip Phase 0-1 (data download)")
    p.add_argument("--skip-nowcast", action="store_true",
                    help="Skip Phase 3 (nowcasting)")
    p.add_argument("--skip-features", action="store_true",
                    help="Skip Phase 4 (feature extraction)")
    p.add_argument("--skip-sequences", action="store_true",
                    help="Skip Phase 5 (sequence building)")
    p.add_argument("--skip-cnn", action="store_true",
                    help="Skip Phase 6 (CNN-LSTM training)")
    p.add_argument("--skip-mae", action="store_true",
                    help="Skip Phase 7 (MAE pretraining)")
    p.add_argument("--skip-transformer", action="store_true",
                    help="Skip Phase 8 (Transformer training)")
    p.add_argument("--skip-ensemble", action="store_true",
                    help="Skip Phase 9 (ensemble evaluation)")
    args = p.parse_args()

    ensure_output_dirs()
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

    log.info("=" * 60)
    log.info("BAH 2026 Master Pipeline")
    log.info("Project: %s", PROJECT_ROOT)
    log.info("Output:  %s", OUTPUT_ROOT)
    log.info("Workers: %d", N_WORKERS)
    log.info("=" * 60)

    # Parse phase range
    phases: list[int] = []
    for part in args.phase.split(","):
        if "-" in part:
            a, b = part.split("-")
            phases.extend(range(int(a), int(b) + 1))
        else:
            phases.append(int(part))
    phases = sorted(set(phases))

    # Apply skip flags
    skip_map = {
        0: args.skip_download,
        1: args.skip_download,
        3: args.skip_nowcast,
        4: args.skip_features,
        5: args.skip_sequences,
        6: args.skip_cnn,
        7: args.skip_mae,
        8: args.skip_transformer,
        9: args.skip_ensemble,
    }
    force_phases = set(args.force_phase or [])
    phases = [ph for ph in phases if not skip_map.get(ph, False)]

    phases_str = ",".join(str(ph) for ph in phases)
    log.info("Running phases: [%s]  (force: %s)", phases_str, force_phases or "none")

    ctx: dict[str, Any] = {}
    t_start = time.time()

    for ph in phases:
        force = ph in force_phases
        ctx = run_phase(ph, force=force, ctx=ctx)

    total = time.time() - t_start
    log.info("=" * 60)
    log.info("Master pipeline complete in %s", _r2(total))
    log.info("=" * 60)
    log.info("Output directory: %s", OUTPUT_ROOT)
    log.info("")
    log.info("Next steps:")
    log.info("  Check results:   ls -la %s", CATALOGS_DIR)
    log.info("  Model checkpoints: ls -la %s", MODELS_DIR)
    log.info("  Run dashboard:   uv run streamlit run src/bah2026/visualization/dashboard.py")


if __name__ == "__main__":
    main()
