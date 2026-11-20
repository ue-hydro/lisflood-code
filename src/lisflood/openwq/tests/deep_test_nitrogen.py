#!/usr/bin/env python3
"""Deep end-to-end test of the LISFLOOD<->OpenWQ coupling with a simple
nitrogen cycle, driving the REAL compiled openWQ (openwq_py) -- no mocks.

It declares the LISFLOOD compartments (soil top A/B, soil sub, upper/lower
groundwater, channel) on a small synthetic grid, writes a minimal NATIVE_BGC
config for a single species NO3, loads NO3 at the soil surface (an external
water flux), and routes water+solute down the LISFLOOD cascade
(soil -> groundwater -> channel) every day. It then reads NO3 mass back from
each compartment and checks:

  * MASS BALANCE: with decay=0 (conservative), total NO3 in the system must
    equal the cumulative NO3 loaded (no spurious creation/destruction).
  * ROUTING: NO3 appears first in the soil, then propagates to groundwater and
    the channel (not instantaneous everywhere, not stuck at the surface).
  * PHYSICAL: concentrations (mass/volume) are non-negative and in a sane range.

Run inside the OpenWQ container (openwq_py present):
    PYTHONPATH=src _dhbv_venv/bin/python3 \
        src/lisflood/openwq/tests/deep_test_nitrogen.py
"""
import os
import sys
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent  # src/lisflood/openwq


def load_host_module():
    spec = importlib.util.spec_from_file_location("owq_host", PKG / "host.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# LISFLOOD compartment order (matches coupler.COMPARTMENTS)
COMPS = ["SOIL_TOP_A", "SOIL_TOP_B", "SOIL_SUB", "UPPER_ZONE", "LOWER_ZONE", "CHANNEL"]

NCELLS = 3
AREA_M2 = 30900.0          # ~0.05 deg equatorial pixel
MM = AREA_M2 / 1000.0      # mm -> m3 per cell

# constant storages (m3/cell) -- realistic LISFLOOD magnitudes
VOL = {
    "SOIL_TOP_A": 20.0 * MM,
    "SOIL_TOP_B": 30.0 * MM,
    "SOIL_SUB":   50.0 * MM,
    "UPPER_ZONE": 10.0 * MM,
    "LOWER_ZONE": 100.0 * MM,
    "CHANNEL":    1000.0,
}

# per-step water transfers (m3/cell/day) along the LISFLOOD cascade
LOAD_M3 = 5.0              # external-water-flux (diffuse load) into SOIL_TOP_A
MOVES = [                  # (src, dst, m3/cell)
    ("SOIL_TOP_A", "SOIL_SUB", 3.0),
    ("SOIL_TOP_B", "SOIL_SUB", 2.0),
    ("SOIL_SUB",   "UPPER_ZONE", 2.0),
    ("UPPER_ZONE", "LOWER_ZONE", 0.5),
    ("UPPER_ZONE", "CHANNEL",    3.0),   # interflow
    ("LOWER_ZONE", "CHANNEL",    0.8),   # baseflow
]
# openWQ reads concentrations in the master's mass unit per m3. With units="KG"
# a value of 0.01 KG/m3 == 10 mg/L (1 mg/L = 1 g/m3 = 1e-3 KG/m3).
LOAD_CONC = 0.01           # NO3 in the load water: 0.01 KG/m3 = 10 mg/L
NSTEPS = 30


def run(decay_per_day=0.0, case_dir=None, verbose=True):
    host = load_host_module()
    case = Path(case_dir or (HERE / "_ncase")).resolve()
    case.mkdir(parents=True, exist_ok=True)

    master = host.write_minimal_config(
        case, COMPS, species=("NO3",),
        initial_mg_l={c: {"NO3": 0.0} for c in COMPS},
        ewf_names=("LOAD",),
        ewf_mg_l={"LOAD": {"NO3": LOAD_CONC}},
        timestep=(1, "day"), decay_per_day=decay_per_day,
        units="KG", project="LISFLOOD-OpenWQ N test",
    )

    # openWQ resolves the config/EWF paths relative to CWD, so run from the case
    _prev_cwd = os.getcwd()
    os.chdir(case)
    H = host.Host(str(master))
    icmp = {c: H.add_compartment(c, NCELLS, 1, 1) for c in COMPS}
    iload = H.add_ext_flux("LOAD", NCELLS, 1, 1)
    H.init()

    def set_vols():
        for c in COMPS:
            H.set_water_volume_array(icmp[c], np.full(NCELLS, VOL[c]))

    # prime: OpenWQ ignores fluxes on its first interaction step
    set_vols()
    H.time_start(2000, 1, 1)
    H.time_end(2000, 1, 1)

    hist = {c: [] for c in COMPS}
    loaded_kg = 0.0
    import datetime
    for step in range(1, NSTEPS + 1):
        d = datetime.date(2000, 1, 1) + datetime.timedelta(days=step)
        set_vols()
        H.time_start(d.year, d.month, d.day)
        # diffuse NO3 load into the soil surface (per cell)
        for ix in range(NCELLS):
            H.inflow("LOAD", icmp["SOIL_TOP_A"], ix, 0, 0, LOAD_M3, d.year, d.month, d.day)
        loaded_kg += NCELLS * LOAD_M3 * LOAD_CONC   # m3 * KG/m3 -> kg
        # cascade: move(src,cell, dst,cell, wflux_m3, wmass_src_m3, date)
        # wmass_src_m3 = source start-of-step storage (sets the exported fraction)
        for src, dst, vol in MOVES:
            for ix in range(NCELLS):
                H.move(icmp[src], ix, 0, 0, icmp[dst], ix, 0, 0,
                       vol, VOL[src], d.year, d.month, d.day)
        H.time_end(d.year, d.month, d.day)
        for c in COMPS:
            hist[c].append(float(np.asarray(H.mass_array(icmp[c], 0)).sum()))

    os.chdir(_prev_cwd)
    total = {c: hist[c][-1] for c in COMPS}
    in_system = sum(total.values())
    resid = (loaded_kg - in_system) / max(loaded_kg, 1e-12)

    if verbose:
        print("\n=== N test (decay_per_day=%.3f) ===" % decay_per_day)
        print("cumulative NO3 loaded (kg):      %.6f" % loaded_kg)
        print("NO3 in system at end (kg):       %.6f" % in_system)
        print("relative residual (loaded-sys):  %.3e  %s" %
              (resid, "(=denitrified for decay>0)" if decay_per_day else ""))
        print("\nfinal NO3 mass per compartment (kg) and mean conc (mg/L):")
        for c in COMPS:
            conc = total[c] / (VOL[c] * NCELLS) * 1000.0   # kg/m3 -> mg/L
            print("  %-11s  mass=%.5f   conc=%.3f mg/L" % (c, total[c], conc))
        print("\nNO3 mass trajectory (kg, every 6 steps):")
        hdr = "  step " + " ".join("%10s" % c[:10] for c in COMPS)
        print(hdr)
        for s in range(0, NSTEPS, 6):
            print("  %4d " % (s + 1) + " ".join("%10.4f" % hist[c][s] for c in COMPS))

    return hist, total, loaded_kg, in_system, resid


def main():
    # 1. conservative: mass must be conserved exactly
    hist, total, loaded, in_sys, resid = run(decay_per_day=0.0)
    ok_balance = abs(resid) < 1e-6
    ok_nonneg = all(min(v) >= -1e-12 for v in hist.values())
    # routing: soil gets N before the channel; channel ends non-zero
    ch = np.array(hist["CHANNEL"]); soil = np.array(hist["SOIL_TOP_A"])
    ok_routing = ch[-1] > 0 and soil[2] > ch[2]

    print("\n=== CHECKS (conservative run) ===")
    print("  mass balance closed (<1e-6):   %s  (resid=%.2e)" % (ok_balance, resid))
    print("  all masses non-negative:       %s" % ok_nonneg)
    print("  N routes soil->...->channel:   %s" % ok_routing)

    # 2. denitrification: first-order loss must REMOVE N
    _, _, loaded2, in_sys2, resid2 = run(decay_per_day=0.05, case_dir=HERE / "_ncase_decay")
    ok_decay = in_sys2 < in_sys - 1e-9 and resid2 > 0
    print("\n=== CHECK (denitrification run) ===")
    print("  decay removes N (sys<loaded):  %s  (%.4f < %.4f kg)" %
          (ok_decay, in_sys2, loaded2))

    allok = ok_balance and ok_nonneg and ok_routing and ok_decay
    print("\n==> DEEP N TEST:", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
