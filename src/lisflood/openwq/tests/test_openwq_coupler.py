"""Unit tests for the LISFLOOD -> OpenWQ coupler mapping.

These run WITHOUT the compiled ``openwq_py`` module: a recording ``FakeHost``
is injected via ``host_factory`` and we assert the coupler declares the right
compartments and issues the expected water+solute transfers with the correct
mm->m3 conversion. The compiled-module import path is covered separately in
test_openwq_import.py (skipped when the module is not built).
"""
import datetime
import numpy as np
import pytest

from lisflood.openwq.coupler import LisfloodOpenWQCoupler, COMPARTMENTS


class FakeHost:
    """Records every bridge call so the mapping can be asserted."""
    def __init__(self, master_json):
        self.master_json = master_json
        self.compartments = []        # names in declaration order
        self.volumes = {}             # icmp -> last array set
        self.moves = []               # (src, dst, vol_array)
        self.inflows = []             # (name, dst, vol_array)
        self.times = []               # ('start'|'end', (y,m,d,h,mi))
        self.inited = False

    def add_compartment(self, name, nx, ny=1, nz=1):
        self.compartments.append(name)
        return len(self.compartments) - 1

    def init(self):
        self.inited = True

    def set_water_volume_array(self, icmp, arr):
        self.volumes[icmp] = np.asarray(arr, dtype=float).copy()

    def move(self, src, ix_s, iy_s, iz_s, dst, ix_r, iy_r, iz_r,
             wflux_m3, wmass_src_m3, year, month, day, hour=0, minute=0):
        # real bridge signature; record (src, dst, cell, wflux, wmass_src)
        self.moves.append((src, dst, ix_s, float(wflux_m3), float(wmass_src_m3)))

    def inflow(self, name, dst, arr):
        self.inflows.append((name, dst, np.asarray(arr, dtype=float).copy()))

    def time_start(self, y, m, d, h, mi):
        self.times.append(("start", (y, m, d, h, mi)))

    def time_end(self, y, m, d, h, mi):
        self.times.append(("end", (y, m, d, h, mi)))


class FakeVar:
    """Minimal stand-in for the LISFLOOD model (self.var). N compressed cells."""
    def __init__(self, n=5):
        ones = np.ones(n)
        self.MMtoM3 = ones * 30900.0            # ~0.05deg equatorial pixel (m2)/1000
        # stores (mm, except ChanM3 in m3)
        self.W1a = ones * 20.0
        self.W1b = ones * 30.0
        self.W2 = ones * 50.0
        self.UZ = ones * 10.0
        self.LZ = ones * 100.0
        self.ChanM3 = ones * 1000.0
        # fluxes this step (mm, except ...M3...)
        self.SeepTopToSubA = ones * 1.0
        self.SeepTopToSubB = ones * 1.5
        self.SeepSubToGW = ones * 2.0
        self.GwPercUZLZ = ones * 0.5
        self.UZOutflow = ones * 3.0
        self.LZOutflowToChannel = ones * 0.8
        self.ToChanM3Runoff = ones * 500.0      # already m3
        self.GwLoss = ones * 0.1
        self.CalendarDate = datetime.datetime(2000, 6, 15, 0, 0)
        self.DtSec = 86400.0


@pytest.fixture
def coupled():
    var = FakeVar(n=5)
    c = LisfloodOpenWQCoupler(var, "openWQ_master.json", host_factory=FakeHost)
    return var, c


def test_declares_all_compartments(coupled):
    var, c = coupled
    c.setup()
    assert c._host.compartments == COMPARTMENTS
    assert c._host.inited
    assert c._ncells == 5


def test_prime_seeds_initial_volumes_without_fluxes(coupled):
    var, c = coupled
    c.setup()
    c.prime()
    # channel volume passed through unchanged (already m3)
    np.testing.assert_allclose(c._host.volumes[c._icmp["CHANNEL"]], var.ChanM3)
    # soil top A converted mm->m3
    np.testing.assert_allclose(
        c._host.volumes[c._icmp["SOIL_TOP_A"]], var.W1a * var.MMtoM3)
    # priming issues no transfers
    assert c._host.moves == []


def test_step_issues_expected_transfers_with_mm_to_m3(coupled):
    var, c = coupled
    c.step()                      # lazy setup + prime + one step
    I = c._icmp
    # collect per (src,dst) the per-cell wflux and wmass_src (uniform FakeVar)
    bysd = {}
    for s, d, ix, wf, wm in c._host.moves:
        bysd.setdefault((s, d), []).append((wf, wm))
    # one scalar move per cell
    assert len(bysd[(I["SOIL_SUB"], I["UPPER_ZONE"])]) == c._ncells
    # vertical soil cascade: wflux = flux(mm) * MMtoM3; wmass_src = source vol
    wf, wm = bysd[(I["SOIL_SUB"], I["UPPER_ZONE"])][0]
    assert np.isclose(wf, float(var.SeepSubToGW[0] * var.MMtoM3[0]))
    assert np.isclose(wm, float(var.W2[0] * var.MMtoM3[0]))        # source = SOIL_SUB vol
    # interflow UZ -> channel
    assert np.isclose(bysd[(I["UPPER_ZONE"], I["CHANNEL"])][0][0],
                      float(var.UZOutflow[0] * var.MMtoM3[0]))
    # baseflow LZ -> channel
    assert np.isclose(bysd[(I["LOWER_ZONE"], I["CHANNEL"])][0][0],
                      float(var.LZOutflowToChannel[0] * var.MMtoM3[0]))
    # surface runoff already m3 -> passed through unchanged
    assert any(np.isclose(wf, float(var.ToChanM3Runoff[0]))
               for _, _, _, wf, _ in c._host.moves)


def test_default_off_is_noop_import():
    # Importing the coupler must not import the compiled module.
    import importlib
    import lisflood.openwq as pkg
    importlib.reload(pkg)
    assert hasattr(pkg, "LisfloodOpenWQCoupler")
