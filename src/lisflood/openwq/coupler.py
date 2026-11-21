"""OpenWQ coupler for OS-LISFLOOD (the JRC/GloFAS hydrological model).

Forward (one-way) coupling: the LISFLOOD water balance drives the OpenWQ solute
balance. The hydrology is NOT modified -- the hooks only *read* LISFLOOD state
and flux variables (``self.var.*``) and replay the water transfers into OpenWQ,
as the other OpenWQ hosts (SUMMA, CRHM, mizuRoute, FLUXOS, dHBV) do.

Because the coupling is one-way, the SAME transfers can be replayed either
  * in-process  (settings option ``OpenWQ``: LISFLOOD -> openwq_py every step), or
  * offline     (option ``OpenWQRecord``: LISFLOOD writes the coupling fields of
                 every step to ``OpenWQRecordDir``; ``replay.py`` then drives
                 openWQ from them).
Both paths go through ``collect_fields`` (what is read from LISFLOOD) and
``LisfloodOpenWQCoupler.advance`` (what is sent to openWQ), so they are
identical by construction. The offline path keeps LISFLOOD's and openWQ's
memory peaks apart (global GloFAS domain: 6.2 M cells x 6 compartments).

OpenWQ cells = LISFLOOD *compressed* (masked) 1-D pixels; each compartment is
declared with nx = number of pixels, ny = nz = 1.

Compartments and transfers (verified against OS-LISFLOOD v5.0.0 source:
soilloop.py, soil.py, groundwater.py, surface_routing.py, routing.py,
Lisflood_dynamic.py). LISFLOOD keeps soil/UZ stores PER VEGETATION FRACTION
(dims vegetation x pixel); every store and flux is reduced to a pixel average
with LISFLOOD's own weighting (``var.deffraction`` = sum over soil fractions
weighted by SoilFraction, i.e. the ``*Pixel`` variables LISFLOOD itself uses
for its water balance), then converted mm -> m3 with ``var.MMtoM3``.

  OpenWQ compartment  LISFLOOD store            transfers replayed (per step)
  ------------------  ------------------------  ----------------------------------
  SOIL_TOP_A          deffraction(W1a)          in : InfiltrationPixel  (EWF, topsoil)
                                                out: SeepTopToSubPixelA -> SOIL_TOP_B
  SOIL_TOP_B          deffraction(W1b)          out: SeepTopToSubPixelB -> SOIL_SUB
  SOIL_SUB            deffraction(W2)           out: SeepSubToGWPixel   -> UPPER_ZONE
                                                     (drained-irrigation part -> CHANNEL)
  UPPER_ZONE          deffraction(UZ)           out: GwPercUZLZPixel    -> LOWER_ZONE
                                                     UZOutflowPixel     -> CHANNEL
  LOWER_ZONE          LZ                        out: LZOutflowToChannelPixel -> CHANNEL
                                                     GwLossPixel        -> out of domain
                                                     abstraction_GW_actual_M3 (water use):
                                                     return-flow part -> CHANNEL, rest out
  CHANNEL             ChanM3 (m3)               out: withdrawal_CH_actual_M3 -> out (water use)
                                                     TransLossWBM3 -> out (transmission loss)
                                                     ChanQAvg*DtSec -> downstream pixel
                                                     (var.downstruct, LddChan); pits
                                                     -> out of domain (sea / sink)

Water that changes a store without a solute transfer (rain/snow on open water,
overland flow, preferential flow, evapotranspiration, open-water evaporation,
irrigation water applied to soils, the groundwater-body
smoothing of LZ) only changes the volume OpenWQ gets
from LISFLOOD each sub-step (solute is conserved; concentration follows).
Lakes/reservoirs have no OpenWQ compartment in the generator's LISFLOOD set:
solute leaving the pixel upstream of a structure enters the structure pixel's
CHANNEL and continues downstream (pass-through, zero residence time).

Time stepping: OpenWQ advances in ``substeps`` equal parts of the LISFLOOD
step (default: LISFLOOD's own channel-routing sub-steps, ``NoRoutSteps``), so
solute can travel as many pixels per LISFLOOD day as LISFLOOD's routing has
sub-steps. Per sub-step: volumes interpolated linearly between the start and
end of the LISFLOOD step, fluxes = step total / substeps, source volume for
the exported fraction = start-of-sub-step volume (openWQ convention), volume at
time_end = end-of-sub-step volume (so output concentration = mass / volume).
"""
from __future__ import annotations

import contextlib
import datetime
import json
import os

import numpy as np


def read_owq_json(path):
    """Read an openWQ JSON file (openWQ allows full-line ``//`` comments, as
    written by openWQ's supporting-scripts generator)."""
    with open(path) as fh:
        txt = "\n".join(l for l in fh.read().splitlines()
                        if not l.lstrip().startswith("//"))
    return json.loads(txt)


def ewf_names_from_master(master_json):
    """External-water-flux names declared in a generated openWQ master file
    (EXTERNAL_INPUTFLUX_NAME of each OPENWQ_INPUT/EXTERNAL_WATER_FLUXES file).
    Paths in the master are relative to its own folder (openWQ's convention)."""
    base = os.path.dirname(os.path.abspath(master_json))
    master = read_owq_json(master_json)
    names = []
    for entry in master.get("OPENWQ_INPUT", {}).get("EXTERNAL_WATER_FLUXES", {}).values():
        ewf = read_owq_json(os.path.join(base, entry["FILEPATH"]))
        for k, block in ewf.items():
            if k != "METADATA" and isinstance(block, dict):
                n = block.get("EXTERNAL_INPUTFLUX_NAME")
                if n and n not in names:
                    names.append(n)
    return names


# Ordered so the index is stable across declare/set/move calls. Must equal the
# LISFLOOD compartments of openWQ's generator (Gen_Input_Driver.LISFLOOD_COMPARTMENTS,
# hostmodel = "lisflood"); tests/deep_test_nitrogen.py checks the two agree.
COMPARTMENTS = [
    "SOIL_TOP_A",
    "SOIL_TOP_B",
    "SOIL_SUB",
    "UPPER_ZONE",
    "LOWER_ZONE",
    "CHANNEL",
]

# (flux key, source compartment, recipient compartment or None = out of domain)
# in the order they are replayed within a sub-step. CH_DOWN is the per-pixel
# downstream transfer (recipient pixel = downstream pixel).
TRANSFERS = [
    ("TOPA_TOPB", "SOIL_TOP_A", "SOIL_TOP_B"),
    ("TOPB_SUB",  "SOIL_TOP_B", "SOIL_SUB"),
    ("SUB_UZ",    "SOIL_SUB",   "UPPER_ZONE"),
    ("SUB_CH",    "SOIL_SUB",   "CHANNEL"),
    ("UZ_LZ",     "UPPER_ZONE", "LOWER_ZONE"),
    ("UZ_CH",     "UPPER_ZONE", "CHANNEL"),
    ("LZ_CH",     "LOWER_ZONE", "CHANNEL"),
    ("LZ_OUT",    "LOWER_ZONE", None),
    ("LZ_ABS_CH", "LOWER_ZONE", "CHANNEL"),   # groundwater abstraction returned to channel
    ("LZ_ABS_OUT", "LOWER_ZONE", None),       # groundwater abstraction (water use) leaving
    ("CH_ABS_OUT", "CHANNEL",   None),        # channel abstraction (water use) leaving
    ("CH_TRANS_OUT", "CHANNEL", None),        # channel transmission loss (option TransLoss)
    ("CH_DOWN",   "CHANNEL",    "CHANNEL"),
]
INFLOW_KEY = "INFIL_TOPA"          # carrier of the topsoil external water flux
FLUX_KEYS = [INFLOW_KEY] + [t[0] for t in TRANSFERS]


# ---------------------------------------------------------------------------
# What is read from LISFLOOD
# ---------------------------------------------------------------------------
def _pix(var, x):
    """Pixel average (mm over the pixel) of a LISFLOOD store/flux: arrays with a
    'vegetation' dimension are reduced with LISFLOOD's own ``deffraction``
    (sum over soil fractions weighted by SoilFraction); a bare (vegetation,
    pixel) array that lost its dims label gets the identical weighting."""
    dims = getattr(x, "dims", None)
    if dims is not None and "vegetation" in tuple(dims):
        return np.asarray(var.deffraction(x), dtype=float)
    a = np.asarray(x, dtype=float)
    if a.ndim == 2:
        return (np.asarray(var.SoilFraction, dtype=float) * a).sum(0)
    return a


def store_volumes(var):
    """Water volume (m3) of every OpenWQ compartment from the current LISFLOOD
    state (pixel averages, LISFLOOD's own fraction weighting)."""
    mm = np.asarray(var.MMtoM3, dtype=float)
    n = int(np.asarray(var.LZ).shape[-1])
    v = {
        "SOIL_TOP_A": _pix(var, var.W1a) * mm,
        "SOIL_TOP_B": _pix(var, var.W1b) * mm,
        "SOIL_SUB":   _pix(var, var.W2) * mm,
        "UPPER_ZONE": _pix(var, var.UZ) * mm,
        "LOWER_ZONE": np.asarray(var.LZ, dtype=float) * mm,
        "CHANNEL":    np.asarray(var.ChanM3, dtype=float),
    }
    return {k: np.ascontiguousarray(np.broadcast_to(a, (n,)), dtype=float) for k, a in v.items()}


def _drained_seepage_mm(var):
    """Part of SeepSubToGW that LISFLOOD sends straight to the channel through
    drainage systems (option drainedIrrigation: soilloop.py adds
    DrainedFraction*SeepSubToGW to UZOutflow of IRRIGATED, non-paddy vegetation
    fractions). Returned as a pixel average in mm."""
    n = int(np.asarray(var.LZ).shape[-1])
    df = np.asarray(getattr(var, "DrainedFraction", 0.0), dtype=float)
    seep = np.asarray(var.SeepSubToGW, dtype=float)
    if not np.any(df > 0) or seep.ndim != 2:
        return np.zeros(n)
    veg = list(var.vegetation)
    is_irr = np.array([var.VEGETATION_LANDUSE[v] == "Irrigated" for v in veg])
    crop = getattr(var, "crop_module", None)
    paddy = list(getattr(crop, "paddy_crops", [])) if crop is not None else []
    is_irr &= ~np.array([v in paddy for v in veg])
    sf = np.asarray(var.SoilFraction, dtype=float)   # (vegetation, pixel), as deffraction
    drained = np.zeros(n)
    for i, irr in enumerate(is_irr):
        if irr:
            drained += sf[i] * np.where(df > 0, df * seep[i], 0.0)
    return drained


def step_fluxes(var):
    """Water transfers of the LISFLOOD step just computed (m3 per step)."""
    mm = np.asarray(var.MMtoM3, dtype=float)
    dr = _drained_seepage_mm(var)
    # water use (option wateruse, waterabstraction.py): groundwater abstraction
    # (m3/step, removed from LZ) of which a part returns to the channel within
    # the step (returnflow_GwAbs2Channel_M3_routStep per routing sub-step), and
    # channel abstraction (m3/step). Absent when water use is off -> zero.
    gw_abs = np.asarray(getattr(var, "abstraction_GW_actual_M3", 0.0), dtype=float)
    gw_ret = (np.asarray(getattr(var, "returnflow_GwAbs2Channel_M3_routStep", 0.0), dtype=float)
              * float(getattr(var, "NoRoutSteps", 1)))
    f = {
        "INFIL_TOPA": np.asarray(var.InfiltrationPixel, dtype=float) * mm,
        "TOPA_TOPB":  np.asarray(var.SeepTopToSubPixelA, dtype=float) * mm,
        "TOPB_SUB":   np.asarray(var.SeepTopToSubPixelB, dtype=float) * mm,
        "SUB_UZ":     (np.asarray(var.SeepSubToGWPixel, dtype=float) - dr) * mm,
        "SUB_CH":     dr * mm,
        "UZ_LZ":      np.asarray(var.GwPercUZLZPixel, dtype=float) * mm,
        "UZ_CH":      (np.asarray(var.UZOutflowPixel, dtype=float) - dr) * mm,
        "LZ_CH":      np.asarray(var.LZOutflowToChannelPixel, dtype=float) * mm,
        "LZ_OUT":     np.asarray(var.GwLossPixel, dtype=float) * mm,
        "LZ_ABS_CH":  gw_ret,
        "LZ_ABS_OUT": np.maximum(gw_abs - gw_ret, 0.0),
        "CH_ABS_OUT": np.asarray(getattr(var, "withdrawal_CH_actual_M3", 0.0), dtype=float),
        # transmission.py: daily total of the routing sub-steps' losses (m3/step),
        # water that leaves the domain through the river bed (option TransLoss)
        "CH_TRANS_OUT": np.asarray(getattr(var, "TransLossWBM3", 0.0), dtype=float),
        "CH_DOWN":    np.asarray(var.ChanQAvg, dtype=float) * float(var.DtSec),
    }
    n = int(np.asarray(var.LZ).shape[-1])
    return {k: np.ascontiguousarray(np.broadcast_to(a, (n,)), dtype=float) for k, a in f.items()}


def downstream_index(var):
    """Recipient pixel of each pixel's channel outflow: LISFLOOD's
    ``var.downstruct`` (downstream pixel on LddChan; pits are set to N), with
    pits mapped to -1 (= out of the domain: sea or endorheic sink)."""
    ds = np.asarray(var.downstruct).astype(np.int64)
    n = ds.size
    out = np.where((ds < 0) | (ds >= n), -1, ds)
    return np.ascontiguousarray(out, dtype=np.int32)


def collect_fields(var, v_start):
    """Everything the coupler needs from one LISFLOOD step."""
    d = var.CalendarDate
    fields = {"date": np.array([d.year, d.month, d.day, d.hour, d.minute], dtype=np.int64),
              "T": np.ascontiguousarray(np.broadcast_to(
                  np.asarray(var.Tavg, dtype=float), (int(np.asarray(var.LZ).shape[-1]),)))}
    v_end = store_volumes(var)
    for c in COMPARTMENTS:
        fields["V0_" + c] = v_start[c]
        fields["V1_" + c] = v_end[c]
    for k, a in step_fluxes(var).items():
        fields["F_" + k] = a
    return fields


# ---------------------------------------------------------------------------
# What is sent to openWQ
# ---------------------------------------------------------------------------
class LisfloodOpenWQCoupler:
    """One-way LISFLOOD -> OpenWQ coupler (in-process or replay).

    Parameters
    ----------
    var : LISFLOOD variable container (``self.var``), read-only; None in replay.
    master_json : OpenWQ master JSON generated by openWQ's template generator
        (hostmodel = "lisflood").
    host_factory : optional callable(master_basename) -> host (tests inject a fake).
    ext_flux_names : EWFs to declare; default = the ones the generated master lists.
    topsoil_ewf : EWF applied to the infiltration into SOIL_TOP_A; default = the
        first EWF of the generated master (None if it declares none).
    substeps : OpenWQ sub-steps per LISFLOOD step; default = LISFLOOD's
        channel-routing sub-steps (var.NoRoutSteps), 1 if unknown.
    """

    def __init__(self, var, master_json, host_factory=None, ext_flux_names=None,
                 topsoil_ewf="__default__", substeps=None):
        self.var = var
        self.master_json = os.path.abspath(str(master_json))
        # openWQ resolves the generated (relative) paths and writes openwq_out/
        # relative to the CWD: the coupler enters this folder around its openWQ
        # calls only, so LISFLOOD's own file I/O is unaffected.
        self._case_dir = os.path.dirname(self.master_json)
        self.host_factory = host_factory
        self.ext_flux_names = ext_flux_names
        self._topsoil_ewf = topsoil_ewf
        self._substeps = substeps
        self._host = None
        self._icmp = {}
        self._iewf = {}
        self._idep_T = None
        self._primed = False
        self._v_start = None
        self._restore = None     # state file to restore right after priming (chunked runs)
        self.diag = []           # per-LISFLOOD-step diagnostics (dicts)

    @contextlib.contextmanager
    def _in_case(self):
        prev = os.getcwd()
        os.chdir(self._case_dir)
        try:
            yield
        finally:
            os.chdir(prev)

    # -- setup -----------------------------------------------------------
    def setup(self, ncells, dst_ix, dtsec, substeps=None):
        if self.host_factory is not None:
            make_host = self.host_factory
        else:
            import openwq_py  # pybind11 module built from OpenWQ_hydrolink + openwq/
            make_host = openwq_py.Host
        self._ncells = int(ncells)
        self._dst_ix = np.ascontiguousarray(dst_ix, dtype=np.int32)
        self._dtsec = float(dtsec)
        self._n_sub = max(1, int(substeps if substeps is not None else
                                 (self._substeps if self._substeps is not None else 1)))
        names = (self.ext_flux_names if self.ext_flux_names is not None
                 else ewf_names_from_master(self.master_json))
        if self._topsoil_ewf == "__default__":
            self._topsoil_ewf = names[0] if names else None
        with self._in_case():
            self._host = make_host(os.path.basename(self.master_json))
            for name in COMPARTMENTS:
                self._icmp[name] = self._host.add_compartment(name, self._ncells, 1, 1)
            for name in names:
                self._iewf[name] = self._host.add_ext_flux(name, self._ncells, 1, 1)
            # BGC templates may use "T" (deg C) in their kinetics, e.g. 1.08^(T - 20)
            self._idep_T = self._host.add_depend("T", self._ncells, 1, 1)
            self._host.init()

    def _sub_times(self, date_end):
        """End time of each sub-step of the LISFLOOD step that ends at date_end
        (LISFLOOD/GloFAS stamp a daily step with the END of the interval)."""
        t_end = datetime.datetime(*[int(x) for x in date_end])
        dt = datetime.timedelta(seconds=self._dtsec / self._n_sub)
        t0 = t_end - datetime.timedelta(seconds=self._dtsec)
        return t0, [t0 + (k + 1) * dt for k in range(self._n_sub)]

    @staticmethod
    def _tt(t):
        return t.year, t.month, t.day, t.hour, t.minute

    def _set_volumes(self, vols):
        for c, arr in vols.items():
            self._host.set_water_volume_array(self._icmp[c], arr)

    # -- core: one LISFLOOD step ----------------------------------------------
    def advance(self, fields, on_substep=None):
        """Replay one LISFLOOD step (fields from ``collect_fields``) in OpenWQ.
        ``on_substep(k, vk, flux)`` (optional, diagnostics) is called after each
        sub-step's time_start, before its transfers."""
        n_sub = self._n_sub
        t0, tsub = self._sub_times(fields["date"])
        v0 = {c: fields["V0_" + c] for c in COMPARTMENTS}
        v1 = {c: fields["V1_" + c] for c in COMPARTMENTS}
        T = np.ascontiguousarray(fields["T"], dtype=float)
        with self._in_case():
            if not self._primed:
                # OpenWQ ignores fluxes on its first interaction step: seed the
                # initial conditions (IC conc x volume) at the start of the run.
                self._set_volumes(v0)
                self._host.set_depend_array(self._idep_T, T)
                self._host.time_start(*self._tt(t0))
                self._host.time_end(*self._tt(t0))
                self._primed = True
                if self._restore is not None:            # continue a previous run exactly
                    self._apply_state(self._restore)
                    self._restore = None
            I = self._icmp
            flux = {k: np.ascontiguousarray(fields["F_" + k] / n_sub) for k in FLUX_KEYS}
            clamp = 0
            for k, t in enumerate(tsub):
                a0, a1 = k / n_sub, (k + 1) / n_sub
                vk = {c: np.ascontiguousarray(v0[c] + a0 * (v1[c] - v0[c])) for c in COMPARTMENTS}
                vk1 = {c: np.ascontiguousarray(v0[c] + a1 * (v1[c] - v0[c])) for c in COMPARTMENTS}
                tt = self._tt(t)
                self._set_volumes(vk)
                self._host.set_depend_array(self._idep_T, T)
                self._host.time_start(*tt)
                if on_substep is not None:
                    on_substep(k, vk, flux)
                if self._topsoil_ewf is not None:
                    self._host.inflow_array(self._topsoil_ewf, I["SOIL_TOP_A"],
                                            flux[INFLOW_KEY], *tt)
                for key, src, dst in TRANSFERS:
                    if key == "CH_DOWN":
                        self._host.move_array_to(I[src], I[dst], self._dst_ix,
                                                 flux[key], vk[src], *tt)
                        clamp = max(clamp, int(np.count_nonzero(flux[key] > vk[src])))
                    else:
                        self._host.move_array(I[src], -1 if dst is None else I[dst],
                                              flux[key], vk[src], *tt)
                self._set_volumes(vk1)
                self._host.time_end(*tt)
        self.diag.append({"date": "%04d-%02d-%02d" % tuple(fields["date"][:3]),
                          "substeps": n_sub, "channel_cfl_gt1_cells": clamp})

    # -- state carry-over (chunked runs) ----------------------------------
    def save_state(self, path):
        """Solute state between two steps: mass of every species in every cell
        of every compartment + the cells whose initial condition is still
        pending (dry when the IC was seeded)."""
        h, ns, st = self._host, self._host.num_species(), {}
        for c, i in self._icmp.items():
            for k in range(ns):
                st[f"m|{c}|{k}"] = np.asarray(h.mass_array(i, k), dtype=float).ravel()
                st[f"p|{c}|{k}"] = np.asarray(h.pending_ic_array(i, k), dtype=np.int32).ravel()
        np.savez(path, **st)

    def restore_state_after_prime(self, path):
        self._restore = str(path)

    def _apply_state(self, path):
        st = np.load(path)
        for key in st.files:
            kind, c, k = key.split("|")
            if kind == "m":
                self._host.set_mass_array(self._icmp[c], int(k), np.ascontiguousarray(st[key]))
            else:
                self._host.set_pending_ic_array(self._icmp[c], int(k), np.ascontiguousarray(st[key]))

    # -- in-process LISFLOOD hooks ---------------------------------------
    def begin_step(self):
        """Hook at the START of Lisflood_dynamic.dynamic(): start-of-step stores."""
        self._v_start = store_volumes(self.var)

    def step(self):
        """Hook at the END of the hydrology of Lisflood_dynamic.dynamic()."""
        v = self.var
        if self._host is None:
            self.setup(int(np.asarray(v.LZ).shape[-1]), downstream_index(v), v.DtSec,
                       substeps=self._substeps if self._substeps is not None
                       else int(getattr(v, "NoRoutSteps", 1)))
        if self._v_start is None:
            raise RuntimeError("begin_step() was not called at the start of the step")
        self.advance(collect_fields(v, self._v_start))
        self._v_start = None

    def finalize(self):
        pass


class LisfloodOpenWQRecorder:
    """Writes, for every LISFLOOD step, exactly the fields the coupler replays
    (``collect_fields``) to <out_dir>/step_NNNNN.npz, plus static.npz once
    (pixel count, downstream index, DtSec, routing sub-steps, mask, pixel area)."""

    def __init__(self, var, out_dir, dtype="float64"):
        self.var = var
        self.out_dir = os.path.abspath(str(out_dir))
        os.makedirs(self.out_dir, exist_ok=True)
        self._v_start = None
        self._n = 0
        # float32 + compression halves/thirds the disk for long runs (a global
        # day is ~1.3 GB in float64); the replay casts back to float64.
        self.dtype = np.dtype(dtype)

    def _write_static(self):
        v = self.var
        static = dict(ncells=np.int64(np.asarray(v.LZ).shape[-1]),
                      dst_ix=downstream_index(v),
                      dtsec=np.float64(v.DtSec),
                      noroutsteps=np.int64(getattr(v, "NoRoutSteps", 1)),
                      pixel_area_m2=np.asarray(v.PixelArea, dtype=float))
        try:
            from lisflood.global_modules.settings import MaskInfo
            static["mask"] = np.asarray(MaskInfo.instance().info.mask)
        except Exception:
            pass
        np.savez(os.path.join(self.out_dir, "static.npz"), **static)

    def begin_step(self):
        self._v_start = store_volumes(self.var)

    def step(self):
        if self._n == 0:
            self._write_static()
        self._n += 1
        f = collect_fields(self.var, self._v_start)
        if self.dtype != np.float64:
            f = {k: (v.astype(self.dtype) if v.dtype == np.float64 else v) for k, v in f.items()}
        dst = os.path.join(self.out_dir, "step_%05d.npz" % self._n)
        tmp = dst + ".tmp.npz"                 # atomic: readers never see a partial day
        (np.savez if self.dtype == np.float64 else np.savez_compressed)(tmp, **f)
        os.replace(tmp, dst)
        self._v_start = None

    def finalize(self):
        pass
