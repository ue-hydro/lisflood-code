"""Unit tests for the LISFLOOD -> OpenWQ coupler mapping.

These run WITHOUT the compiled ``openwq_py`` module: a recording ``FakeHost`` is
injected via ``host_factory``. ``FakeVar`` mimics LISFLOOD v5.0.0's layout:
soil and upper-zone stores/fluxes carry a 'vegetation' dimension (3 prescribed
fractions x pixels) and are reduced with ``deffraction`` (SoilFraction-weighted
sum), the ``*Pixel`` fluxes are those pixel averages, LZ/ChanM3 are per pixel,
``downstruct`` is the downstream pixel with pits = N.
"""
import datetime
import os

import numpy as np
import pytest

from lisflood.openwq.coupler import (LisfloodOpenWQCoupler, LisfloodOpenWQRecorder,
                                     COMPARTMENTS, TRANSFERS, store_volumes, step_fluxes,
                                     downstream_index, ewf_names_from_master)


class VegArray(np.ndarray):
    """(vegetation, pixel) array with LISFLOOD's dims label."""
    def __new__(cls, a):
        obj = np.asarray(a, dtype=float).view(cls)
        return obj

    @property
    def dims(self):
        return ["vegetation", "pixel"]

    @property
    def values(self):
        return self


class FakeHost:
    """Records every bridge call so the mapping can be asserted."""
    def __init__(self, master_json):
        self.master_json = master_json
        self.compartments, self.ext_fluxes, self.depends = [], [], []
        self.calls = []                      # ordered log of runtime calls
        self.inited = False

    def add_compartment(self, name, nx, ny=1, nz=1):
        self.compartments.append(name); return len(self.compartments) - 1

    def add_ext_flux(self, name, nx, ny=1, nz=1):
        self.ext_fluxes.append(name); return len(self.ext_fluxes) - 1

    def add_depend(self, name, nx, ny=1, nz=1):
        self.depends.append(name); return len(self.depends) - 1

    def init(self):
        self.inited = True

    def set_water_volume_array(self, icmp, arr):
        self.calls.append(("vol", icmp, np.array(arr, dtype=float)))

    def set_depend_array(self, idep, arr):
        self.calls.append(("dep", idep, np.array(arr, dtype=float)))

    def time_start(self, *t):
        self.calls.append(("start", tuple(t)))

    def time_end(self, *t):
        self.calls.append(("end", tuple(t)))

    def move_array(self, src, dst, wflux, wsrc, *t):
        self.calls.append(("move", src, dst, np.array(wflux), np.array(wsrc), tuple(t)))

    def move_array_to(self, src, dst, dst_ix, wflux, wsrc, *t):
        self.calls.append(("moveto", src, dst, np.array(dst_ix), np.array(wflux),
                           np.array(wsrc), tuple(t)))

    def inflow_array(self, name, dst, wflux, *t):
        self.calls.append(("inflow", name, dst, np.array(wflux), tuple(t)))


class FakeVar:
    """Minimal stand-in for LISFLOOD's self.var (v5.0.0 layout), n pixels."""
    def __init__(self, n=5):
        self.n = n
        self.vegetation = ["Rainfed_prescribed", "Forest_prescribed", "Irrigated_prescribed"]
        self.VEGETATION_LANDUSE = {"Rainfed_prescribed": "Rainfed",
                                   "Forest_prescribed": "Forest",
                                   "Irrigated_prescribed": "Irrigated"}
        sf = np.tile(np.array([[0.5], [0.3], [0.1]]), (1, n))     # 0.1 = water/sealed
        self.SoilFraction = VegArray(sf)
        self.PixelArea = np.full(n, 3.09e7)
        self.MMtoM3 = 0.001 * self.PixelArea
        veg = lambda a, b, c: VegArray(np.vstack([np.full(n, a), np.full(n, b), np.full(n, c)]))
        self.W1a, self.W1b, self.W2 = veg(10, 12, 14), veg(30, 32, 34), veg(50, 55, 60)
        self.UZ = veg(5, 6, 7)
        self.LZ = np.full(n, 100.0)
        self.ChanM3 = np.linspace(1e4, 5e4, n)
        self.SeepSubToGW = veg(2.0, 2.0, 4.0)
        self.DrainedFraction = 0.25
        self.InfiltrationPixel = np.full(n, 6.0)
        self.SeepTopToSubPixelA = np.full(n, 1.0)
        self.SeepTopToSubPixelB = np.full(n, 1.5)
        self.SeepSubToGWPixel = self.deffraction(self.SeepSubToGW)
        self.GwPercUZLZPixel = np.full(n, 0.5)
        drained = 0.1 * 0.25 * 4.0                                 # irrigated fraction only
        self.UZOutflowPixel = np.full(n, 3.0) + drained
        self.LZOutflowToChannelPixel = np.full(n, 0.8)
        self.GwLossPixel = np.full(n, 0.1)
        self.ChanQAvg = np.linspace(1.0, 2.0, n)
        self.downstruct = np.array([1, 2, 3, 4, n][:n], dtype=np.int32)  # chain, last = pit
        self.Tavg = np.full(n, 15.0)
        self.DtSec = 86400.0
        self.NoRoutSteps = 4
        self.CalendarDate = datetime.datetime(2015, 6, 1, 0, 0)

    def deffraction(self, x):
        return (np.asarray(self.SoilFraction) * np.asarray(x)).sum(0)


def make(tmp_path, var=None, host=FakeHost, names=("LOAD",)):
    var = var or FakeVar()
    return var, LisfloodOpenWQCoupler(var, tmp_path / "openWQ_master.json",
                                      host_factory=host, ext_flux_names=list(names))


def run_one_step(var, c):
    c.begin_step()
    c.step()
    return c._host.calls


def test_declares_compartments_ewf_and_temperature(tmp_path):
    var, c = make(tmp_path)
    run_one_step(var, c)
    h = c._host
    assert h.compartments == COMPARTMENTS and h.inited
    assert h.ext_fluxes == ["LOAD"] and h.depends == ["T"]
    assert c._ncells == var.n and c._n_sub == var.NoRoutSteps


def test_store_volumes_use_lisflood_fraction_weighting():
    var = FakeVar()
    v = store_volumes(var)
    sf = np.asarray(var.SoilFraction)
    np.testing.assert_allclose(v["SOIL_TOP_A"], (sf * np.asarray(var.W1a)).sum(0) * var.MMtoM3)
    np.testing.assert_allclose(v["UPPER_ZONE"], (sf * np.asarray(var.UZ)).sum(0) * var.MMtoM3)
    np.testing.assert_allclose(v["LOWER_ZONE"], var.LZ * var.MMtoM3)
    np.testing.assert_allclose(v["CHANNEL"], var.ChanM3)           # already m3


def test_drained_irrigation_split_and_flux_conversion():
    var = FakeVar()
    f = step_fluxes(var)
    drained_mm = 0.1 * 0.25 * 4.0
    np.testing.assert_allclose(f["SUB_CH"], drained_mm * var.MMtoM3)
    np.testing.assert_allclose(f["SUB_UZ"] + f["SUB_CH"], var.SeepSubToGWPixel * var.MMtoM3)
    np.testing.assert_allclose(f["UZ_CH"], 3.0 * var.MMtoM3)
    np.testing.assert_allclose(f["TOPA_TOPB"], var.SeepTopToSubPixelA * var.MMtoM3)
    np.testing.assert_allclose(f["CH_DOWN"], var.ChanQAvg * var.DtSec)


def test_downstream_index_pits_leave_domain():
    var = FakeVar()
    np.testing.assert_array_equal(downstream_index(var), [1, 2, 3, 4, -1])


def test_substeps_transfers_volumes_and_times(tmp_path):
    var, c = make(tmp_path)
    calls = run_one_step(var, c)
    N = var.NoRoutSteps
    starts = [x[1] for x in calls if x[0] == "start"]
    ends = [x[1] for x in calls if x[0] == "end"]
    # prime at the START of the LISFLOOD step (end - DtSec), then N sub-steps whose
    # end times are evenly spaced and the last equals the step's date stamp
    assert starts[0] == (2015, 5, 31, 0, 0) and len(starts) == N + 1
    assert starts[1:] == [(2015, 5, 31, 6, 0), (2015, 5, 31, 12, 0),
                          (2015, 5, 31, 18, 0), (2015, 6, 1, 0, 0)]
    assert ends == starts
    I = c._icmp
    moves = [x for x in calls if x[0] == "move"]
    assert len(moves) == (len(TRANSFERS) - 1) * N   # all but CH_DOWN go through move_array
    topab = [m for m in moves if m[1] == I["SOIL_TOP_A"] and m[2] == I["SOIL_TOP_B"]]
    np.testing.assert_allclose(topab[0][3], var.SeepTopToSubPixelA * var.MMtoM3 / N)
    loss = [m for m in moves if m[1] == I["LOWER_ZONE"] and m[2] == -1]
    np.testing.assert_allclose(loss[0][3], var.GwLossPixel * var.MMtoM3 / N)
    mt = [x for x in calls if x[0] == "moveto"]
    assert len(mt) == N
    np.testing.assert_array_equal(mt[0][3], [1, 2, 3, 4, -1])
    np.testing.assert_allclose(mt[0][4], var.ChanQAvg * var.DtSec / N)
    inf = [x for x in calls if x[0] == "inflow"]
    assert len(inf) == N and inf[0][1] == "LOAD" and inf[0][2] == I["SOIL_TOP_A"]
    np.testing.assert_allclose(inf[0][3], var.InfiltrationPixel * var.MMtoM3 / N)


def test_volume_interpolation_and_source_volume(tmp_path):
    var, c = make(tmp_path)
    c.begin_step()
    v0 = store_volumes(var)
    var.ChanM3 = var.ChanM3 * 3.0                    # hydrology changed the channel store
    c.step()
    I, N = c._icmp, var.NoRoutSteps
    calls = c._host.calls
    mt = [x for x in calls if x[0] == "moveto"]
    # source volume of sub-step k = start volume + k/N of the step's change
    for k, m in enumerate(mt):
        np.testing.assert_allclose(m[5], v0["CHANNEL"] + k / N * (var.ChanM3 - v0["CHANNEL"]))
    # the volume set right before the last time_end is the end-of-step volume
    last_end = max(i for i, x in enumerate(calls) if x[0] == "end")
    vols = [x for x in calls[:last_end] if x[0] == "vol" and x[1] == I["CHANNEL"]]
    np.testing.assert_allclose(vols[-1][2], var.ChanM3)


def test_record_then_replay_equals_in_process(tmp_path):
    # in-process
    var1, c1 = make(tmp_path / "a")
    os.makedirs(tmp_path / "a", exist_ok=True)
    run_one_step(var1, c1)
    # record (LISFLOOD side) then replay (openWQ side) through advance()
    var2 = FakeVar()
    rec = LisfloodOpenWQRecorder(var2, tmp_path / "rec")
    rec.begin_step(); rec.step()
    st = np.load(tmp_path / "rec" / "static.npz")
    f = dict(np.load(tmp_path / "rec" / "step_00001.npz"))
    c2 = LisfloodOpenWQCoupler(None, tmp_path / "openWQ_master.json",
                               host_factory=FakeHost, ext_flux_names=["LOAD"])
    c2.setup(int(st["ncells"]), st["dst_ix"], float(st["dtsec"]), substeps=int(st["noroutsteps"]))
    c2.advance(f)
    a, b = c1._host.calls, c2._host.calls
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[0] == y[0]
        for u, w in zip(x[1:], y[1:]):
            if isinstance(u, np.ndarray):
                np.testing.assert_array_equal(u, w)
            else:
                assert u == w


def test_openwq_calls_run_in_case_folder_and_cwd_restored(tmp_path):
    seen = []

    class CwdHost(FakeHost):
        def time_end(self, *a):
            seen.append(os.getcwd()); super().time_end(*a)

    case = tmp_path / "case"
    case.mkdir()
    before = os.getcwd()
    var = FakeVar(n=2)
    var.downstruct = np.array([1, 2], dtype=np.int32)
    c = LisfloodOpenWQCoupler(var, case / "openWQ_master.json", host_factory=CwdHost,
                              ext_flux_names=[])
    run_one_step(var, c)
    assert seen and all(os.path.realpath(p) == os.path.realpath(case) for p in seen)
    assert os.getcwd() == before                         # LISFLOOD's CWD untouched


def test_ext_flux_names_read_from_generated_master(tmp_path):
    (tmp_path / "openwq_in").mkdir()
    (tmp_path / "openWQ_master.json").write_text(
        "\n// generated using OpenWQ's supporting scripts\n{\n"
        ' "OPENWQ_INPUT": {"EXTERNAL_WATER_FLUXES": {"1": {"LABEL": "x",'
        ' "FILEPATH": "openwq_in/openWQ_EWF_fixed_value.json"}}}\n}\n')
    (tmp_path / "openwq_in" / "openWQ_EWF_fixed_value.json").write_text(
        '// header\n{"METADATA": {"Comment": "c"}, "1": {"DATA_FORMAT": "JSON",'
        ' "EXTERNAL_INPUTFLUX_NAME": "LOAD", "CHEMICAL_NAME": "NO3-N"}}\n')
    assert ewf_names_from_master(tmp_path / "openWQ_master.json") == ["LOAD"]


def test_default_off_is_noop_import():
    # Importing the coupler must not import the compiled module.
    import importlib
    import lisflood.openwq as pkg
    importlib.reload(pkg)
    assert hasattr(pkg, "LisfloodOpenWQCoupler")
