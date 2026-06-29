"""Tests for new HEL1OS readers (HK, GTI).

These tests require actual data on disk. They are skipped if data is missing.
"""

import numpy as np
import pytest
from datetime import date

from bah2026.data.reader import (
    load_hel1os_hk,
    load_hel1os_gti,
    load_hel1os_all_gti,
)


# Use a known data day for integration tests
TEST_DAY = date(2024, 5, 5)


def _has_data(d: date) -> bool:
    """Check if HEL1OS data exists for the given date."""
    from bah2026.config import DATA_ROOT
    from pathlib import Path

    return (
        DATA_ROOT / "hel1os" / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"
    ).exists()


skip_if_no_data = pytest.mark.skipif(
    not _has_data(TEST_DAY), reason=f"No HEL1OS data for {TEST_DAY}"
)


# ── HK reader ───────────────────────────────────────────────────────────


@skip_if_no_data
class TestLoadHel1osHk:
    def test_returns_dict(self):
        """HK reader returns a dict with expected columns."""
        hk = load_hel1os_hk(TEST_DAY)
        assert isinstance(hk, dict)
        assert "columns" in hk

    def test_has_62_columns(self):
        """HK has 62 columns."""
        hk = load_hel1os_hk(TEST_DAY)
        assert len(hk["columns"]) == 62

    def test_detector_temps(self):
        """Detector temperatures are in expected ranges."""
        hk = load_hel1os_hk(TEST_DAY)
        assert "czt1temp" in hk
        assert "czt2temp" in hk
        assert "cdte1temp" in hk
        assert "cdte2temp" in hk
        # CZT: 15-25°C
        assert 10 < np.mean(hk["czt1temp"]) < 30
        # CdTe: -50 to -20°C
        assert -60 < np.mean(hk["cdte1temp"]) < -10

    def test_hv_monitors(self):
        """HV monitors are present and positive."""
        hk = load_hel1os_hk(TEST_DAY)
        assert "czthvmon" in hk
        assert "cdtehvmon" in hk
        assert np.mean(hk["czthvmon"]) > 0
        assert np.mean(hk["cdtehvmon"]) > 0

    def test_pileup_counters(self):
        """Pile-up and saturation counters are present."""
        hk = load_hel1os_hk(TEST_DAY)
        assert "cdte1pilectr" in hk
        assert "czt1satctr1" in hk

    def test_mjd_present(self):
        """MJD time array is present."""
        hk = load_hel1os_hk(TEST_DAY)
        assert "mjd" in hk
        assert len(hk["mjd"]) > 0


# ── GTI reader ──────────────────────────────────────────────────────────


@skip_if_no_data
class TestLoadHel1osGti:
    def test_single_detector(self):
        """GTI for a single detector returns (N, 2) array."""
        gti = load_hel1os_gti(TEST_DAY, "czt", 1)
        assert gti.shape[1] == 2
        assert len(gti) >= 1
        # tstart < tstop
        assert np.all(gti[:, 0] < gti[:, 1])

    def test_all_detectors(self):
        """All 4 detectors have GTI."""
        gti_all = load_hel1os_all_gti(TEST_DAY)
        assert "czt1" in gti_all
        assert "czt2" in gti_all
        assert "cdte1" in gti_all
        assert "cdte2" in gti_all
        for key, gti in gti_all.items():
            assert gti.shape[1] == 2


# ── Missing data handling ───────────────────────────────────────────────


class TestMissingData:
    def test_hk_missing(self):
        """Missing HK file raises FileNotFoundError."""
        from bah2026.config import DATA_ROOT

        far_future = date(2099, 1, 1)
        # Only test if data definitely doesn't exist
        if not (DATA_ROOT / "hel1os" / "2099" / "01" / "01").exists():
            with pytest.raises(FileNotFoundError):
                load_hel1os_hk(far_future)

    def test_gti_missing(self):
        """Missing GTI file returns empty array."""
        from bah2026.config import DATA_ROOT

        far_future = date(2099, 1, 1)
        if not (DATA_ROOT / "hel1os" / "2099" / "01" / "01").exists():
            gti = load_hel1os_gti(far_future, "czt", 1)
            assert gti.shape == (0, 2)
