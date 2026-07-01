"""
Stage 2 — Phase 9: Feature matrix assembly.
Merge all phase NPZs into one unified feature matrix.
"""
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

STAGE1 = Path("data/processed/stage1_20260623.npz")
PHASES = [
    ("phase1_direct", Path("dist/features/phase1_direct.npz")),
    ("phase2_perflare", Path("dist/features/phase2_perflare.npz")),
    ("phase3_tem_goes", Path("dist/features/phase3_tem_goes.npz")),
    ("phase4_tem_solexs", Path("dist/features/phase4_tem_solexs.npz")),
    ("phase5_wavelet", Path("dist/features/phase5_wavelet.npz")),
    ("phase6_spectral", Path("dist/features/phase6_spectral_index.npz")),
    ("phase7_nonlinear", Path("dist/features/phase7_nonlinear.npz")),
    ("phase8_event", Path("dist/features/phase8_event_aux.npz")),
    ("phase10_extended", Path("dist/features/phase10_extended.npz")),
]
OUT_DIR = Path("dist")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "stage2_feature_matrix_20260623.npz"
OUT_CSV = OUT_DIR / "stage2_feature_matrix.csv"
OUT_PLOT = OUT_DIR / "plots/feature_nan_heatmap.png"


def assemble():
    print("Phase 9: Feature matrix assembly")
    print("=" * 60)

    ds1 = np.load(STAGE1, allow_pickle=True)
    master_flag = ds1["master_flag"]
    t = ds1["time"].astype(np.float64)

    all_features = {}
    phase_map = {}
    all_nan_frac = {}
    n = 86400

    for phase_name, path in PHASES:
        if not path.exists():
            print(f"  [{phase_name}] MISSING: {path}")
            continue
        ds = np.load(path, allow_pickle=True)
        if "__metadata__" in ds:
            meta = ds["__metadata__"].item()
        else:
            meta = {}

        n_feat = 0
        for k in ds.files:
            if k == "__metadata__":
                continue
            v = ds[k]
            if not hasattr(v, "shape") or v.shape != (n,):
                continue
            if v.dtype.kind not in ("f", "i", "u", "b"):
                continue

            # Rename to avoid collisions (add phase prefix if name collision)
            out_name = k
            if out_name in all_features:
                out_name = f"{phase_name}_{k}"

            all_features[out_name] = v.astype(np.float32) if v.dtype.kind == "f" else v
            phase_map[out_name] = phase_name
            nfeat = int(np.isnan(v).sum()) if v.dtype.kind == "f" else 0
            all_nan_frac[out_name] = nfeat / n
            n_feat += 1

        print(f"  [{phase_name}] loaded {n_feat} features from {path.name}")

    print(f"\n  Total features: {len(all_features)}")

    # ── Apply master_flag mask ─────────────────────────────────
    # For non-GOOD bins (master_flag != 0), set float features to NaN
    bad = master_flag != 0
    n_bad = int(bad.sum())
    for k, v in list(all_features.items()):
        if v.dtype.kind == "f":
            all_features[k] = np.where(bad, np.nan, v)
        elif v.dtype.kind == "i" or v.dtype.kind == "u":
            # For int features, keep but track status
            pass

    print(f"  Masked {n_bad} non-GOOD bins to NaN")

    # ── Feature statistics ─────────────────────────────────────
    n_float = sum(1 for v in all_features.values() if v.dtype.kind == "f")
    print(f"  Float features: {n_float}, Int/bool: {len(all_features) - n_float}")

    # ── Save as NPZ ───────────────────────────────────────────
    metadata = {
        "date_created": datetime.now(timezone.utc).isoformat(),
        "date_obs": "20260623",
        "n_samples": n,
        "n_features": len(all_features),
        "n_good_bins": n - n_bad,
        "phases_loaded": [p[0] for p in PHASES],
        "phase_map": phase_map,
        "nan_fraction_per_feature": all_nan_frac,
        "features_with_high_nan": sorted(
            [k for k, v in all_nan_frac.items() if v > 0.5],
            key=lambda x: all_nan_frac[x], reverse=True,
        ),
        "notes": "Non-GOOD bins (master_flag != 0) have NaN for float features",
    }

    np.savez_compressed(OUT_PATH, **all_features, __metadata__=metadata)
    print(f"\n  Saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")

    # ── CSV summary ───────────────────────────────────────────
    print(f"\n  Top-15 features by NaN fraction:")
    sorted_feats = sorted(all_nan_frac.items(), key=lambda x: -x[1])
    for name, frac in sorted_feats[:15]:
        print(f"    {name:40s} NaN={frac*100:5.1f}%")

    # ── Quick plot: NaN heatmap ───────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_feats_plot = min(60, len(all_features))
        feature_names = [f[0] for f in sorted_feats[:n_feats_plot]]
        nan_fracs = [f[1] for f in sorted_feats[:n_feats_plot]]

        fig, ax = plt.subplots(figsize=(12, max(6, n_feats_plot * 0.35)))
        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#1a1a2e")

        colors = ["#00cc66" if f < 0.1 else "#ffcc00" if f < 0.5 else "#ff4444" for f in nan_fracs]
        ax.barh(range(len(feature_names)), nan_fracs, color=colors, height=0.7)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=6, color="#ccc")
        ax.set_xlabel("NaN fraction", color="white", fontsize=9)
        ax.set_title("Stage 2 Feature Matrix — NaN fractions per feature", color="white", fontsize=11)
        ax.set_xlim(0, 1)
        ax.tick_params(colors="#aaa", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#444")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#00cc66", label="< 10% NaN"),
            Patch(facecolor="#ffcc00", label="10-50% NaN"),
            Patch(facecolor="#ff4444", label="> 50% NaN"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

        plt.tight_layout()
        OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUT_PLOT, dpi=120, facecolor="#0d0d1a")
        plt.close()
        print(f"  Plot saved: {OUT_PLOT}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    # ── Quick CSV export ──────────────────────────────────────
    try:
        import pandas as pd
        df = pd.DataFrame({k: v.astype(np.float32) if v.dtype.kind == "f" else v
                           for k, v in all_features.items()})
        df.to_csv(OUT_CSV, index=False)
        print(f"  CSV saved: {OUT_CSV}")
    except Exception as e:
        print(f"  CSV export: {e}")

    return all_features


if __name__ == "__main__":
    assemble()
