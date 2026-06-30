import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

GOES_DATA_DIR = Path(__file__).parent.parent / "data/raw/goes"

def _norm_date(d):
    if isinstance(d, int):
        return f"202606{d:02d}" if d < 100 else str(d)
    s = str(d).replace("-", "")
    if len(s) == 8:
        return s
    raise ValueError(f"Can't parse date: {d}")

def _find_file(date, resolution="avg1m"):
    dstr = _norm_date(date)
    pat = f"*{resolution}*{dstr}*.nc"
    files = sorted(GOES_DATA_DIR.glob(pat))
    if not files:
        raise FileNotFoundError(f"No GOES file for {dstr} {resolution} in {GOES_DATA_DIR}")
    return files[0]

def _cls_str(f):
    if np.isnan(f):
        return "---"
    if f >= 1e-4:
        return f"X{f/1e-4:.1f}"
    if f >= 1e-5:
        return f"M{f/1e-5:.1f}"
    if f >= 1e-6:
        return f"C{f/1e-6:.1f}"
    if f >= 1e-7:
        return f"B{f/1e-7:.1f}"
    return f"A{f/1e-8:.2f}"

class GOESData:
    def __init__(self, date, resolution="avg1m"):
        self._date_str = _norm_date(date)
        path = _find_file(date, resolution)
        self._ds = xr.open_dataset(path)
        self.resolution = resolution
        self.satellite = "GOES-18"

        dt64 = self._ds.time.values
        self.time = np.array([t.astype("datetime64[s]").astype(float) for t in dt64])
        self.xrsa = self._ds.xrsa_flux.values.astype(float)
        self.xrsb = self._ds.xrsb_flux.values.astype(float)

        self.tstart = self.time[0]
        self.tstop = self.time[-1]
        self.date = f"{self._date_str[:4]}-{self._date_str[4:6]}-{self._date_str[6:]}"

    def __repr__(self):
        n = len(self.time)
        c = self.class_str(self.xrsb[np.nanargmax(self.xrsb)])
        return (f"GOESData({self.date}, {self.resolution}): "
                f"{n} samples, peak {c}")

    @staticmethod
    def class_str(flux):
        return _cls_str(flux)

    def slice(self, t_start, t_stop):
        mask = (self.time >= t_start) & (self.time <= t_stop)
        return self.time[mask], self.xrsa[mask], self.xrsb[mask]

    def max_in_window(self, t_center, half_width=300):
        t0 = t_center - half_width
        t1 = t_center + half_width
        t, a, b = self.slice(t0, t1)
        if len(t) == 0:
            return None, None, None
        i = np.nanargmax(b)
        return t[i], a[i], b[i]

    def peak_near(self, unix_time, window=600):
        return self.max_in_window(unix_time, window // 2)

    def close(self):
        self._ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def load_goes(date, resolution="avg1m"):
    return GOESData(date, resolution)

def demo():
    for day in [16, 21, 23]:
        for res in ["avg1m", "flx1s"]:
            try:
                g = GOESData(day, res)
                pi = np.nanargmax(g.xrsb)
                print(f"  {g}")
                g.close()
            except FileNotFoundError:
                print(f"  No GOES {res} for day {day}")

if __name__ == "__main__":
    demo()
