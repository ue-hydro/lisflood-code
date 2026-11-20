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

    def add_ext_flux(self, name, nx, ny=1, nz=1):
        self.ext_fluxes = getattr(self, "ext_fluxes", []) + [name]
        return len(self.ext_fluxes) - 1

    def add_depend(self, name, nx, ny=1, nz=1):
        self.depends = getattr(self, "depends", []) + [name]
        return len(self.depends) - 1

    def set_depend_array(self, idep, arr):
        self.depend_values = getattr(self, "depend_values", {})
        self.depend_values[idep] = np.asarray(arr, dtype=float).copy()

    def init(self):
        self.inited = True

    def set_water_volume_array(self, icmp, arr):
        self.volumes[icmp] = np.asarray(arr, dtype=float).copy()

    def move_array(self, src, dst, wflux_m3, wmass_src_m3,
                   year, month, day, hour=0, minute=0):
        # vectorised bridge signature; record (src, dst, flux[], srcvol[])
        self.moves.append((src, dst,
                           np.asarray(wflux_m3, dtype=float).copy(),
                           np.asarray(wmass_src_m3, dtype=float).copy()))

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
        self.Tavg = ones * 15.0                 # daily mean air temperature (deg C)
        self.CalendarDate = datetime.datetime(2000, 6, 15, 0, 0)
        self.DtSec = 86400.0


@pytest.fixture
def coupled(tmp_path):
    var = FakeVar(n=5)
    c = LisfloodOpenWQCoupler(var, tmp_path / "openWQ_master.json",
                              host_factory=FakeHost, ext_flux_names=["LOAD"])
    return var, c


def test_declares_all_compartments(coupled):
    var, c = coupled
    c.setup()
    assert c._host.compartments == COMPARTMENTS
    assert c._host.inited
    assert c._ncells == 5


def test_declares_ext_flux_and_temperature_dependency(coupled):
    var, c = coupled
    c.step()
    assert c._host.ext_fluxes == ["LOAD"]
    assert c._host.depends == ["T"]                      # name used by BGC kinetics
    np.testing.assert_allclose(c._host.depend_values[c._idep_T], var.Tavg)


def test_openwq_calls_run_in_case_folder_and_cwd_restored(tmp_path):
    seen = []

    class CwdHost(FakeHost):
        def time_end(self, *a):
            seen.append(__import__("os").getcwd())
            super().time_end(*a)

    import os
    case = tmp_path / "case"
    case.mkdir()
    before = os.getcwd()
    c = LisfloodOpenWQCoupler(FakeVar(n=2), case / "openWQ_master.json",
                              host_factory=CwdHost, ext_flux_names=[])
    c.step()
    assert seen and all(os.path.realpath(p) == os.path.realpath(case) for p in seen)
    assert os.getcwd() == before                         # LISFLOOD's CWD untouched


def test_ext_flux_names_read_from_generated_master(tmp_path):
    # openWQ generator style: full-line // comments + relative EWF path
    from lisflood.openwq.coupler import ewf_names_from_master
    (tmp_path / "openwq_in").mkdir()
    (tmp_path / "openWQ_master.json").write_text(
        "\n// generated using OpenWQ's supporting scripts\n{\n"
        ' "OPENWQ_INPUT": {"EXTERNAL_WATER_FLUXES": {"1": {"LABEL": "x",'
        ' "FILEPATH": "openwq_in/openWQ_EWF_fixed_value.json"}}}\n}\n')
    (tmp_path / "openwq_in" / "openWQ_EWF_fixed_value.json").write_text(
        '// header\n{"METADATA": {"Comment": "c"}, "1": {"DATA_FORMAT": "JSON",'
        ' "EXTERNAL_INPUTFLUX_NAME": "LOAD", "CHEMICAL_NAME": "NO3-N"}}\n')
    assert ewf_names_from_master(tmp_path / "openWQ_master.json") == ["LOAD"]


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
    # one move_array call per (src,dst) transfer; arrays over all cells
    bysd = {(s, d): (wf, wm) for s, d, wf, wm in c._host.moves}
    # vertical soil cascade: wflux = flux(mm) * MMtoM3; wmass_src = source vol
    np.testing.assert_allclose(
        bysd[(I["SOIL_SUB"], I["UPPER_ZONE"])][0], var.SeepSubToGW * var.MMtoM3)
    np.testing.assert_allclose(
        bysd[(I["SOIL_SUB"], I["UPPER_ZONE"])][1], var.W2 * var.MMtoM3)  # src = SOIL_SUB vol
    # interflow UZ -> channel
    np.testing.assert_allclose(
        bysd[(I["UPPER_ZONE"], I["CHANNEL"])][0], var.UZOutflow * var.MMtoM3)
    # baseflow LZ -> channel
    np.testing.assert_allclose(
        bysd[(I["LOWER_ZONE"], I["CHANNEL"])][0], var.LZOutflowToChannel * var.MMtoM3)
    # surface runoff already m3 -> passed through unchanged
    assert any(np.allclose(wf, var.ToChanM3Runoff) for _, _, wf, _ in c._host.moves)


def test_default_off_is_noop_import():
    # Importing the coupler must not import the compiled module.
    import importlib
    import lisflood.openwq as pkg
    importlib.reload(pkg)
    assert hasattr(pkg, "LisfloodOpenWQCoupler")
