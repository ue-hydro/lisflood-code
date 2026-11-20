"""OpenWQ coupler for OS-LISFLOOD (the JRC/GloFAS hydrological model).

Forward (one-way) coupling: the LISFLOOD water balance drives the OpenWQ solute
balance. The hydrology is NOT modified -- the hook only *reads* LISFLOOD state
and flux variables (``self.var.*``) at the end of each time step and replays the
water transfers into OpenWQ, exactly as the other OpenWQ hosts (SUMMA, CRHM,
mizuRoute, FLUXOS, the differentiable HBV) do.

Status of this file
-------------------
VERIFIED against OS-LISFLOOD v5.0.0 source: the compartment <-> variable mapping
below uses the real ``self.var`` names (hydrological_modules/{soil,soilloop,
groundwater,surface_routing,routing}.py). The lifecycle (declare -> init ->
prime -> per-step time_start/move/inflow/time_end) follows the generic
``openwq_py`` pybind11 bridge (OpenWQ_hydrolink.{h,cpp}, reused unchanged).

NOT YET DONE (the real implementation / research work, flagged `# TODO`):
  (1) within-step flux ordering (LISFLOOD updates several stores within one
      step; OpenWQ applies transfers from start-of-step state -- the dHBV
      coupler's origin-of-water replay is the reference pattern to port here);
  (2) in-channel downstream advection along the LDD network (needs the
      drainage direction + ChanQ to move solute cell -> downstream cell);
  (3) external loads (diffuse HaNi/HaPi at the soil surface; point sources /
      HydroWASTE at the channel) via ``inflow``;
  (4) scaling: LISFLOOD is ~6.2 M compressed cells; this MUST use the array
      API (``set_water_volume_array``) and OpenWQ's many-cell scaling is
      unproven -- profile before any global run. Prototype on one catchment.

OpenWQ cells = LISFLOOD *compressed* (masked) 1-D cells, so each compartment is
declared with nx = number of masked cells, ny = nz = 1, and the compressed
numpy arrays are passed straight through.

Compartment <-> LISFLOOD store / flux map (all verified in v5.0.0 source):

  OpenWQ compartment   LISFLOOD store (mm)      outgoing water transfers
  ------------------   --------------------     ---------------------------------
  SOIL_TOP_A           var.W1a                  Infiltration(in); SeepTopToSubA
  SOIL_TOP_B           var.W1b                  SeepTopToSubB
  SOIL_SUB             var.W2                   SeepSubToGW -> UZ
  UPPER_ZONE (UZ)      var.UZ                   UZOutflow(interflow->CHANNEL);
                                                GwPercUZLZ -> LZ
  LOWER_ZONE (LZ)      var.LZ                   LZOutflowToChannel(baseflow);
                                                GwLoss (deep, out of domain)
  CHANNEL              var.ChanM3 (already m3)  ChanQ routing (downstream, LDD);
                                                ToChanM3Runoff (surface in)

Storages are mm -> convert with ``var.MMtoM3`` (pixel area / 1000); ``ChanM3``
is already m3. Fluxes ending in ``...M3...`` are already volumetric; the rest
are mm/step and convert the same way.
"""
from __future__ import annotations

import contextlib
import json
import os


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

# LISFLOOD self.var store name backing each compartment (mm, except CHANNEL=m3).
STORE_VAR = {
    "SOIL_TOP_A": "W1a",
    "SOIL_TOP_B": "W1b",
    "SOIL_SUB": "W2",
    "UPPER_ZONE": "UZ",
    "LOWER_ZONE": "LZ",
    "CHANNEL": "ChanM3",
}


class LisfloodOpenWQCoupler:
    """One-way LISFLOOD -> OpenWQ coupler.

    Parameters
    ----------
    var : the LISFLOOD variable container (``self.var``) -- read-only here.
    master_json : path to the OpenWQ master JSON generated by openWQ's template
        generator (model_config_lisflood_*.py, hostmodel = "lisflood").
    host_factory : optional callable(master_basename) -> host (tests inject a fake).
    ext_flux_names : external water fluxes to declare; default = the ones the
        generated master lists (so the coupler follows the generated inputs).
    """

    def __init__(self, var, master_json, host_factory=None, ext_flux_names=None):
        self.var = var
        self.master_json = os.path.abspath(str(master_json))
        # openWQ resolves the generated (relative) paths and writes openwq_out/
        # relative to the CWD: the coupler enters this folder around its openWQ
        # calls only, so LISFLOOD's own file I/O is unaffected.
        self._case_dir = os.path.dirname(self.master_json)
        self.host_factory = host_factory
        self.ext_flux_names = ext_flux_names
        self._host = None
        self._icmp = {}          # compartment name -> OpenWQ index
        self._iewf = {}          # external water flux name -> OpenWQ index
        self._idep_T = None      # temperature dependency "T" (deg C)
        self._primed = False

    @contextlib.contextmanager
    def _in_case(self):
        prev = os.getcwd()
        os.chdir(self._case_dir)
        try:
            yield
        finally:
            os.chdir(prev)

    # -- lifecycle -------------------------------------------------------
    def setup(self):
        """Declare compartments, external fluxes and the temperature dependency
        on LISFLOOD's compressed grid, then init OpenWQ."""
        if self.host_factory is not None:
            make_host = self.host_factory
        else:
            import openwq_py  # pybind11 module built from OpenWQ_hydrolink + openwq/
            make_host = openwq_py.Host

        ncells = int(self.var.W1a.shape[-1])
        self._ncells = ncells
        names = (self.ext_flux_names if self.ext_flux_names is not None
                 else ewf_names_from_master(self.master_json))
        with self._in_case():
            self._host = make_host(os.path.basename(self.master_json))
            for name in COMPARTMENTS:
                self._icmp[name] = self._host.add_compartment(name, ncells, 1, 1)
            for name in names:
                self._iewf[name] = self._host.add_ext_flux(name, ncells, 1, 1)
            # BGC templates use "T" (deg C) in their kinetics, e.g. 1.08^(T - 20)
            self._idep_T = self._host.add_depend("T", ncells, 1, 1)
            self._host.init()
        # TODO(2): register the LDD downstream index per channel cell so solute
        # can be advected cell -> downstream cell with ChanQ each step.

    def _mm_to_m3(self, arr_mm):
        # var.MMtoM3 = PixelArea/1000 (compressed). TODO: confirm name/units.
        return arr_mm * self.var.MMtoM3

    def _push_volumes(self):
        """Set each compartment's water volume (m3) from current LISFLOOD state,
        and cache the arrays -- move() needs the source start-of-step volume."""
        v = self.var
        self._vol = {
            self._icmp["SOIL_TOP_A"]: self._mm_to_m3(v.W1a),
            self._icmp["SOIL_TOP_B"]: self._mm_to_m3(v.W1b),
            self._icmp["SOIL_SUB"]:   self._mm_to_m3(v.W2),
            self._icmp["UPPER_ZONE"]: self._mm_to_m3(v.UZ),
            self._icmp["LOWER_ZONE"]: self._mm_to_m3(v.LZ),
            self._icmp["CHANNEL"]:    v.ChanM3,   # already m3
        }
        for icmp, arr in self._vol.items():
            self._host.set_water_volume_array(icmp, arr)
        # temperature for the BGC kinetics: LISFLOOD daily mean air temperature
        # (var.Tavg, deg C; LISFLOOD has no soil/water temperature yet)
        import numpy as np
        self._host.set_depend_array(self._idep_T, np.ascontiguousarray(
            np.broadcast_to(np.asarray(v.Tavg, dtype=float), (self._ncells,))))

    def prime(self):
        """OpenWQ ignores fluxes on its first interaction step (dt unknown), so
        seed initial conditions one step before the first real step."""
        with self._in_case():
            self._push_volumes()
            y, m, d, h, mi = self._date_tuple(offset_steps=-1)
            self._host.time_start(y, m, d, h, mi)
            self._host.time_end(y, m, d, h, mi)
        self._primed = True

    def step(self):
        """Called once per LISFLOOD time step, AFTER the hydrology is computed
        (hook placed after waterbalance_module.dynamic() in Lisflood_dynamic)."""
        if self._host is None:
            self.setup()            # lazy: soil/GW/channel arrays exist by now
        if not self._primed:
            self.prime()
        with self._in_case():
            self._step_in_case()

    def _step_in_case(self):
        v = self.var
        self._push_volumes()
        y, m, d, h, mi = self._date_tuple()
        self._now = (y, m, d, h, mi)
        self._host.time_start(y, m, d, h, mi)

        I = self._icmp
        # --- vertical soil cascade (mm -> m3) ---------------------------
        self._move(I["SOIL_TOP_A"], I["SOIL_SUB"], self._mm_to_m3(v.SeepTopToSubA))
        self._move(I["SOIL_TOP_B"], I["SOIL_SUB"], self._mm_to_m3(v.SeepTopToSubB))
        self._move(I["SOIL_SUB"],   I["UPPER_ZONE"], self._mm_to_m3(v.SeepSubToGW))
        # --- groundwater ------------------------------------------------
        self._move(I["UPPER_ZONE"], I["LOWER_ZONE"], self._mm_to_m3(v.GwPercUZLZ))
        self._move(I["UPPER_ZONE"], I["CHANNEL"],    self._mm_to_m3(v.UZOutflow))        # interflow
        self._move(I["LOWER_ZONE"], I["CHANNEL"],    self._mm_to_m3(v.LZOutflowToChannel))  # baseflow
        # deep GW loss leaves the domain; sink (TODO: a SINK compartment or drop)
        # --- surface runoff to channel (already m3) ---------------------
        self._move(I["SOIL_TOP_A"], I["CHANNEL"], v.ToChanM3Runoff)  # TODO: correct source store
        # TODO(1): replay the above in LISFLOOD's within-step order with
        #          origin-of-water volumes (see dHBV coupler.py).
        # TODO(2): channel -> downstream channel advection along the LDD w/ ChanQ.
        # TODO(3): external loads: self._host.inflow("DIFFUSE", I["SOIL_TOP_A"], ...)
        #          and inflow("POINT", I["CHANNEL"], ...).

        self._host.time_end(y, m, d, h, mi)

    def finalize(self):
        """End-of-run hook (last step): flush OpenWQ output."""
        # time_end on the last step already written; nothing mandatory here yet.
        pass

    # -- helpers ---------------------------------------------------------
    def _move(self, src, dst, flux_m3):
        """Vectorised water+solute transfer src->dst over all cells.

        Uses the bridge's ``move_array`` (looped in C++), so the host issues ONE
        call per (src,dst) transfer rather than nx -- the path that makes
        LISFLOOD scale (~6.2 M cells) feasible. ``move_array`` signature:
            move_array(src, dst, wflux_m3[], wmass_src_m3[], Y,M,D,h,mi)
        where wmass_src_m3 is the source compartment's start-of-step volume
        (openWQ exports the fraction wflux/wmass_src of the source's solute).
        """
        import numpy as np
        n = self._ncells
        flux = np.ascontiguousarray(np.broadcast_to(
            np.asarray(flux_m3, dtype=float), (n,)))
        srcvol = np.ascontiguousarray(np.broadcast_to(
            np.asarray(self._vol[src], dtype=float), (n,)))
        y, m, d, h, mi = self._now
        self._host.move_array(src, dst, flux, srcvol, y, m, d, h, mi)

    def _date_tuple(self, offset_steps=0):
        """Return (Y, M, D, h, mi) for the current LISFLOOD step (+offset)."""
        # LISFLOOD tracks the calendar date; CalendarDate is a datetime.
        d = self.var.CalendarDate
        if offset_steps:
            import datetime
            d = d + datetime.timedelta(seconds=offset_steps * float(self.var.DtSec))
        return d.year, d.month, d.day, d.hour, d.minute
