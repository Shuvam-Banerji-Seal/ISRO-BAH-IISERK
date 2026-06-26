"""Build HDF5 database from processed FITS files."""

from __future__ import annotations

import h5py
import numpy as np
from pathlib import Path
from datetime import date
from tqdm import tqdm

from bah2026.config import HDF5_DIR
from bah2026.data.reader import (
    load_solexs_lc, load_solexs_pi, load_solexs_gti,
    load_hel1os_lc, discover_solexs_days, discover_hel1os_days,
)


def build_hdf5(output_path: Path | None = None) -> Path:
    """Build the master HDF5 database from all processed FITS files.

    Structure:
        /solexs/lc/{YYYYMMDD}       — TIME(86400), COUNTS(86400)
        /solexs/gti/{YYYYMMDD}      — START(M), STOP(M)
        /hel1os/czt1/{YYYYMMDD}     — MJD(N), CTR(N,5), STAT_ERR(N,5)
        /hel1os/czt2/{YYYYMMDD}     — same
        /hel1os/cdte1/{YYYYMMDD}    — same
        /hel1os/cdte2/{YYYYMMDD}    — same
        /metadata/solexs_days       — list of date strings
        /metadata/hel1os_days       — list of date strings
        /metadata/combined_days     — list of date strings
    """
    if output_path is None:
        output_path = HDF5_DIR / "flare_data.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    solexs_days = discover_solexs_days()
    hel1os_days = discover_hel1os_days()
    combined = sorted(set(solexs_days) & set(hel1os_days))

    print(f"Building HDF5: {len(solexs_days)} SoLEXS, {len(hel1os_days)} HEL1OS, {len(combined)} combined days")

    with h5py.File(output_path, "w") as f:
        f.attrs["description"] = "Aditya-L1 Solar Flare Data (SoLEXS + HEL1OS)"
        f.attrs["solexs_count"] = len(solexs_days)
        f.attrs["hel1os_count"] = len(hel1os_days)
        f.attrs["combined_count"] = len(combined)

        meta = f.create_group("metadata")
        meta.create_dataset("solexs_days", data=np.array([str(d) for d in solexs_days], dtype="S10"))
        meta.create_dataset("hel1os_days", data=np.array([str(d) for d in hel1os_days], dtype="S10"))
        meta.create_dataset("combined_days", data=np.array([str(d) for d in combined], dtype="S10"))

        solexs_grp = f.create_group("solexs")
        hel1os_grp = f.create_group("hel1os")

        for d in tqdm(solexs_days, desc="SoLEXS"):
            key = d.strftime("%Y%m%d")
            try:
                lc = load_solexs_lc(d)
                g = solexs_grp.create_group(f"lc/{key}")
                g.create_dataset("time", data=lc["time"], dtype="f8", compression="gzip", compression_opts=4)
                counts_clean = np.where(np.isfinite(lc["counts"]), lc["counts"], 0.0)
                g.create_dataset("counts", data=counts_clean, dtype="f8", compression="gzip", compression_opts=4)
                g.attrs["tstart"] = lc["tstart"]
                g.attrs["mjdrefi"] = lc["mjdrefi"]
                g.attrs["mjdreff"] = lc["mjdreff"]
                g.attrs["date_obs"] = lc["date_obs"]

                gti = load_solexs_gti(d)
                gg = solexs_grp.create_group(f"gti/{key}")
                gg.create_dataset("start", data=gti[:, 0], dtype="f8")
                gg.create_dataset("stop", data=gti[:, 1], dtype="f8")
            except Exception as e:
                print(f"  SoLEXS {key}: {e}")

        for det_name in ["czt1", "czt2", "cdte1", "cdte2"]:
            det = det_name[:3]
            num = int(det_name[-1])
            for d in tqdm(hel1os_days, desc=f"HEL1OS {det_name}"):
                key = d.strftime("%Y%m%d")
                try:
                    lc = load_hel1os_lc(d, detector=det, num=num)
                    if lc["ctr"].size == 0:
                        continue
                    g = hel1os_grp.create_group(f"{det_name}/{key}")
                    g.create_dataset("mjd", data=lc["mjd"], dtype="f8", compression="gzip", compression_opts=4)
                    g.create_dataset("ctr", data=lc["ctr"], dtype="f8", compression="gzip", compression_opts=4)
                    g.create_dataset("stat_err", data=lc["stat_err"], dtype="f8", compression="gzip", compression_opts=4)
                    g.attrs["band_names"] = [b.encode() for b in lc["band_names"]]
                    g.attrs["nrows"] = len(lc["mjd"])
                except Exception:
                    pass

    print(f"HDF5 saved: {output_path} ({output_path.stat().st_size / 1e9:.1f} GB)")
    return output_path


if __name__ == "__main__":
    build_hdf5()
