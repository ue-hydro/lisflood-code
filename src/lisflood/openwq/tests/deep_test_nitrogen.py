#!/usr/bin/env python3
"""Deep test of the LISFLOOD <-> OpenWQ coupling with openWQ's official simple
nitrogen template, configured ONLY through openWQ's Python template generator.

Step 1 (host Python with the generator's dependencies) -- generate the inputs:
    python src/lisflood/openwq/model_config_lisflood_nitrogen.py
  = openWQ's supporting_scripts/1_Model_Config/model_config_template.py with
    hostmodel="lisflood" and BGC template individual_chemicals/nitrogen_basic.json
    (ORG_N -> NH4-N -> NO3-N -> N2_loss, pure first-order network, no host
    dependencies). Temperature-dependent templates are also handled (T4).

Step 2 (OpenWQ container, compiled openwq_py from the same openWQ commit):
    PYTHONPATH=src python src/lisflood/openwq/tests/deep_test_nitrogen.py

This test never writes openWQ inputs: it refuses to run unless the case folder
was produced by openWQ's generator. It only drives the compiled engine.

  T1  generator <-> coupler consistency (compartments, species, EWF name)
  T2  initial conditions seeded exactly as generated (1 mg/L x water volume)
  T3  reaction network = template: pure kinetics (no transport, no load) at
      20 C matches an independent Forward-Euler integration of the rate laws
      parsed from the generated BGC file, for every species and compartment
  T4  temperature: no-dependency template -> results identical at 5 C and 25 C;
      temperature-dependent template -> day-1 rates at 25 C / 5 C = theta^20
  T5  full coupled run (advection along LISFLOOD fluxes + NO3-N load +
      reactions, 15 C): exact N budget, non-negative, per-compartment results
  T6  move_array scaling (1e3 .. 1e6 cells) with the same generated inputs
"""
import csv
import datetime
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                                   # src/lisflood/openwq
sys.path.insert(0, str(PKG.parent.parent))          # src/ -> lisflood package
from lisflood.openwq.coupler import COMPARTMENTS    # noqa: E402

CASE = Path(os.environ.get("OWQ_CASE_DIR", HERE / "_case_nitrogen_simple")).resolve()
GEN_MARK = "generated using OpenWQ's supporting scripts"
MM = 30900.0 / 1000.0                               # mm -> m3 (0.05 deg pixel)
VOL = {"SOIL_TOP_A": 20 * MM, "SOIL_TOP_B": 30 * MM, "SOIL_SUB": 50 * MM,
       "UPPER_ZONE": 10 * MM, "LOWER_ZONE": 100 * MM, "CHANNEL": 1000.0}
MOVES = [("SOIL_TOP_A", "SOIL_SUB", 3.0), ("SOIL_TOP_B", "SOIL_SUB", 2.0),
         ("SOIL_SUB", "UPPER_ZONE", 2.0), ("UPPER_ZONE", "LOWER_ZONE", 0.5),
         ("UPPER_ZONE", "CHANNEL", 3.0), ("LOWER_ZONE", "CHANNEL", 0.8)]
LOAD_M3 = 5.0                                       # m3/cell/day into SOIL_TOP_A


# ---------------------------------------------------------------- helpers
def read_owq_json(path):
    """openWQ JSON allows full-line // comments."""
    txt = "\n".join(l for l in Path(path).read_text().splitlines()
                    if not l.lstrip().startswith("//"))
    return json.loads(txt)


def load_host():
    spec = importlib.util.spec_from_file_location("owq_host", PKG / "host.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_dir(name):
    """Copy the generated inputs (unchanged) into a fresh run folder."""
    d = CASE.parent / ("_run_" + name)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy2(CASE / "openWQ_master.json", d / "openWQ_master.json")
    shutil.copytree(CASE / "openwq_in", d / "openwq_in")
    return d


class Run:
    """Drive the compiled engine on the generated inputs (no input writing)."""

    def __init__(self, host_mod, name, n):
        self.dir, self.n = run_dir(name), n
        self.prev = os.getcwd()
        os.chdir(self.dir)                          # openWQ paths are relative
        self.H = host_mod.Host("openWQ_master.json")
        self.ic = {c: self.H.add_compartment(c, n, 1, 1) for c in COMPARTMENTS}
        self.H.add_ext_flux(EWF_NAME, n, 1, 1)
        self.iT = self.H.add_depend("T", n, 1, 1)
        self.H.init()
        self.day = datetime.date(2000, 1, 1)

    def _vols(self, T):
        for c in COMPARTMENTS:
            self.H.set_water_volume_array(self.ic[c], np.full(self.n, VOL[c]))
        self.H.set_depend_array(self.iT, np.full(self.n, float(T)))

    def prime(self, T):                             # seeds ICs, ignores fluxes
        self._vols(T)
        self.H.time_start(self.day.year, self.day.month, self.day.day)
        self.H.time_end(self.day.year, self.day.month, self.day.day)

    def step(self, T, load=False, transport=False):
        self.day += datetime.timedelta(days=1)
        y, m, d = self.day.year, self.day.month, self.day.day
        self._vols(T)
        self.H.time_start(y, m, d)
        if load:
            for ix in range(self.n):
                self.H.inflow(EWF_NAME, self.ic["SOIL_TOP_A"], ix, 0, 0, LOAD_M3, y, m, d)
        if transport:
            for src, dst, v in MOVES:
                self.H.move_array(self.ic[src], self.ic[dst], np.full(self.n, v),
                                  np.full(self.n, VOL[src]), y, m, d)
        self.H.time_end(y, m, d)

    def mass(self):                                 # {comp: {species: total}}
        return {c: {s: float(np.asarray(self.H.mass_array(self.ic[c], k)).sum())
                    for k, s in enumerate(SPECIES)} for c in COMPARTMENTS}

    def close(self):
        os.chdir(self.prev)


def euler_kinetics(state, T, days, simultaneous=True):
    """Independent Forward-Euler integration (dt = 1 day) of the rate laws
    parsed from the generated BGC file. state: {species: mass}."""
    st = dict(state)
    for _ in range(days):
        if simultaneous:
            env = dict(st)
            rates = [(c, p, eval_rate(e, env, T)) for c, p, e in REACTIONS]
            for c, p, r in rates:
                st[c] -= r
                if p in st:
                    st[p] += r
        else:
            for c, p, e in REACTIONS:
                r = eval_rate(e, st, T)
                st[c] -= r
                if p in st:
                    st[p] += r
    return st


def eval_rate(expr, state, T):
    env = {SAFE[s]: v for s, v in state.items()}
    env.update(PARAMS_FOR[expr])
    env["T"] = T
    return eval(PYEXPR[expr], {"__builtins__": {}}, env)


# ------------------------------------------------- T1: generated inputs
def t1_generated_inputs():
    global SPECIES, MOBILE, EWF_NAME, REACTIONS, SAFE, PYEXPR, PARAMS_FOR
    master_txt = (CASE / "openWQ_master.json").read_text()
    assert GEN_MARK in master_txt, (
        "inputs were not produced by openWQ's generator -- run "
        "src/lisflood/openwq/model_config_lisflood_nitrogen.py first")
    master = read_owq_json(CASE / "openWQ_master.json")
    cfg = read_owq_json(CASE / "openwq_in" / "openWQ_config.json")
    bgc = read_owq_json(CASE / "openwq_in" /
                        os.path.basename(master["MODULES"]["BIOGEOCHEMISTRY"]
                                         ["MODULE_CONFIG_FILEPATH"]))
    ewf_file = master["OPENWQ_INPUT"]["EXTERNAL_WATER_FLUXES"]["1"]["FILEPATH"]
    ewf = read_owq_json(CASE / ewf_file)

    gen_comps = list(cfg["BIOGEOCHEMISTRY_CONFIGURATION"].keys())
    out_comps = list(master["OPENWQ_OUTPUT"]["COMPARTMENTS_AND_CELLS"].keys())
    SPECIES = [bgc["CHEMICAL_SPECIES"]["LIST"][k]
               for k in sorted(bgc["CHEMICAL_SPECIES"]["LIST"], key=int)]
    MOBILE = bgc["CHEMICAL_SPECIES"].get("BGC_GENERAL_MOBILE_SPECIES", [])
    EWF_NAME = ewf["1"]["EXTERNAL_INPUTFLUX_NAME"]

    # rate laws exactly as written in the generated BGC file
    SAFE = {s: re.sub(r"\W", "_", s) for s in SPECIES}
    REACTIONS, PYEXPR, PARAMS_FOR = [], {}, {}
    for fw in bgc["CYCLING_FRAMEWORKS"].values():
        for key, tr in fw.items():
            if key == "LIST_TRANSFORMATIONS" or "KINETICS" not in tr:
                continue
            e = tr["KINETICS"][0]
            py = e.replace("^", "**")
            for s in sorted(SPECIES, key=len, reverse=True):   # longest first
                py = re.sub(r"(?<![\w-])" + re.escape(s) + r"(?![\w-])", SAFE[s], py)
            PYEXPR[e], PARAMS_FOR[e] = py, dict(tr["PARAMETER_VALUES"])
            REACTIONS.append((tr["CONSUMED"], tr["PRODUCED"], e))

    print("T1  inputs generated by openWQ's template generator: yes")
    print("    compartments (config) :", gen_comps)
    print("    species (BGC)         :", SPECIES, "| mobile:", MOBILE)
    print("    EWF                   :", EWF_NAME, ewf["1"]["CHEMICAL_NAME"],
          ewf["1"]["DATA"]["1"][-1], ewf["1"]["UNITS"])
    for c, p, e in REACTIONS:
        print("    reaction              : %-6s -> %-8s %s  %s" % (c, p, e, PARAMS_FOR[e]))
    ok = gen_comps == list(COMPARTMENTS) and out_comps == list(COMPARTMENTS)
    print("    coupler COMPARTMENTS == generated compartments (same order):", ok)
    return ok, float(ewf["1"]["DATA"]["1"][-1])


# ------------------------------------------------------------- T2 .. T5
def t2_t3_t4(host):
    # T2 + T3: pure kinetics at 20 C
    r = Run(host, "kinetics_20C", 3)
    r.prime(20.0)
    m0 = r.mass()
    vol_l = {c: VOL[c] * 3 * 1000.0 for c in COMPARTMENTS}      # litres (3 cells)
    unit = m0["SOIL_TOP_A"]["NO3-N"] / (1.0 * vol_l["SOIL_TOP_A"])   # mass per mg
    ic_err = max(abs(m0[c][s] / (unit * vol_l[c]) - 1.0)
                 for c in COMPARTMENTS for s in SPECIES)
    print("\nT2  IC seeding: every species/compartment = 1 mg/L x V ; "
          "max rel err = %.2e  (engine mass unit = %.4g per mg)" % (ic_err, unit))
    ok2 = ic_err < 1e-9

    ndays = 10
    traj = [m0]
    for _ in range(ndays):
        r.step(20.0)
        traj.append(r.mass())
    r.close()
    err_sim = err_seq = 0.0
    for c in COMPARTMENTS:
        for dd in (1, ndays):
            ref_s = euler_kinetics(m0[c], 20.0, dd, simultaneous=True)
            ref_q = euler_kinetics(m0[c], 20.0, dd, simultaneous=False)
            for s in SPECIES:
                scale = max(abs(m0[c][s]), 1e-30)
                err_sim = max(err_sim, abs(traj[dd][c][s] - ref_s[s]) / scale)
                err_seq = max(err_seq, abs(traj[dd][c][s] - ref_q[s]) / scale)
    scheme = "simultaneous" if err_sim <= err_seq else "sequential"
    best = min(err_sim, err_seq)
    print("T3  pure kinetics 20 C, %d days vs independent Forward-Euler of the "
          "template rate laws:" % ndays)
    print("    max rel err  simultaneous=%.2e  sequential=%.2e  -> engine = %s "
          "Forward Euler" % (err_sim, err_seq, scheme))
    c = "SOIL_SUB"
    print("    e.g. %s day %d (engine | reference):" % (c, ndays))
    ref = euler_kinetics(m0[c], 20.0, ndays, simultaneous=(scheme == "simultaneous"))
    for s in SPECIES:
        print("      %-13s %12.6f | %12.6f" % (s, traj[ndays][c][s] / (unit * vol_l[c]),
                                             ref[s] / (unit * vol_l[c])))
    ok3 = best < 1e-9

    # T4: temperature dependence of day-1 rates
    gain = {}
    for T in (5.0, 25.0):
        r = Run(host, "kinetics_%gC" % T, 3)
        r.prime(T)
        a = r.mass()
        r.step(T)
        b = r.mass()
        r.close()
        cc = "SOIL_SUB"
        gain[T] = (a[cc]["ORG_N"] - b[cc]["ORG_N"], b[cc]["N2_loss"] - a[cc]["N2_loss"])
    th = {}
    for _, p, e in REACTIONS:
        mt = re.search(r"([\d.]+)\^\(T - 20\)", e)
        if mt:
            th[p] = float(mt.group(1))
    got_min = gain[25.0][0] / gain[5.0][0]
    got_den = gain[25.0][1] / gain[5.0][1]
    if not th:
        # template has no host dependencies: chemistry must ignore temperature
        print("T4  no host dependencies in the template -> day-1 rates must be "
              "identical at 5 C and 25 C:  mineralization ratio %.15f  "
              "denitrification ratio %.15f" % (got_min, got_den))
        ok4 = got_min == 1.0 and got_den == 1.0
    else:
        exp_min, exp_den = th["NH4-N"] ** 20, th["N2_loss"] ** 20
        print("T4  temperature: day-1 rate ratio 25C/5C   mineralization %.4f "
              "(expected %.4f = %.2f^20)   denitrification %.4f (expected %.4f = %.2f^20)"
              % (got_min, exp_min, th["NH4-N"], got_den, exp_den, th["N2_loss"]))
        ok4 = abs(got_min / exp_min - 1) < 1e-6 and abs(got_den / exp_den - 1) < 1e-6
    return ok2, ok3, ok4, unit


def t5_coupled(host, unit, load_mg_l, ndays=30, T=15.0):
    n = 3
    r = Run(host, "coupled_15C", n)
    r.prime(T)
    m = [r.mass()]
    loaded = [0.0]
    for _ in range(ndays):
        r.step(T, load=True, transport=True)
        m.append(r.mass())
        loaded.append(loaded[-1] + n * LOAD_M3 * 1000.0 * load_mg_l * unit)
    r.close()
    tot = [sum(m[k][c][s] for c in COMPARTMENTS for s in SPECIES) for k in range(ndays + 1)]
    resid = max(abs(tot[k] - (tot[0] + loaded[k])) / (tot[0] + loaded[k])
                for k in range(ndays + 1))
    nonneg = all(m[k][c][s] >= -1e-12 for k in range(ndays + 1)
                 for c in COMPARTMENTS for s in SPECIES)
    n2_mono = all(m[k + 1][c]["N2_loss"] >= m[k][c]["N2_loss"] - 1e-12
                  for k in range(ndays) for c in COMPARTMENTS)

    conc = lambda k, c, s: m[k][c][s] / (unit * VOL[c] * n * 1000.0)     # mg/L
    print("\nT5  coupled run, %g C, %d days: NO3-N load %.1f mg/L x %.1f m3/cell/day "
          "into SOIL_TOP_A, advection along LISFLOOD fluxes, reactions on" %
          (T, ndays, load_mg_l, LOAD_M3))
    print("    N budget  sum(all N species) = initial + loaded : max rel residual %.2e"
          % resid)
    print("    non-negative: %s   N2_loss non-decreasing everywhere: %s" % (nonneg, n2_mono))
    print("    concentrations (mg-N/L), day 0 -> day %d:" % ndays)
    print("      %-11s" % "" + "".join("%15s" % s[:13] for s in SPECIES))
    for c in COMPARTMENTS:
        print("      %-11s" % c + "".join("%7.3f->%-7.3f" % (conc(0, c, s), conc(ndays, c, s))
                                         for s in SPECIES))
    out = CASE.parent / "_results_coupled_15C.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["day", "compartment", "species", "mass_engine_units", "conc_mg_l"])
        for k in range(ndays + 1):
            for c in COMPARTMENTS:
                for s in SPECIES:
                    w.writerow([k, c, s, m[k][c][s], conc(k, c, s)])
    print("    trajectory written to", out)
    no3_up = conc(ndays, "SOIL_TOP_A", "NO3-N") > conc(0, "SOIL_TOP_A", "NO3-N")
    orgn_down = conc(ndays, "SOIL_TOP_B", "ORG_N") < conc(0, "SOIL_TOP_B", "ORG_N")
    print("    loaded topsoil NO3-N rises: %s   ORG_N mineralizes in SOIL_TOP_B: %s"
          % (no3_up, orgn_down))
    return resid < 1e-9 and nonneg and n2_mono and no3_up and orgn_down


def t6_scaling(host):
    print("\nT6  move_array scaling (one SOIL_TOP_A -> SOIL_SUB transfer, generated inputs):")
    for n in (1000, 100000, 1000000):
        try:
            r = Run(host, "scale_%d" % n, n)
            r.prime(15.0)
            r.day += datetime.timedelta(days=1)
            y, mo, d = r.day.year, r.day.month, r.day.day
            r._vols(15.0)
            r.H.time_start(y, mo, d)
            t0 = time.perf_counter()
            r.H.move_array(r.ic["SOIL_TOP_A"], r.ic["SOIL_SUB"], np.full(n, 3.0),
                           np.full(n, VOL["SOIL_TOP_A"]), y, mo, d)
            dt = time.perf_counter() - t0
            r.H.time_end(y, mo, d)
            r.close()
            print("    n=%9d cells  move_array %8.1f ms  (%.2f M cells/s)"
                  % (n, dt * 1e3, n / dt / 1e6))
        except Exception as e:                      # report, don't hide
            os.chdir(str(HERE))
            print("    n=%9d cells  FAILED: %s" % (n, e))
    print("    (a LISFLOOD step issues ~7 transfers; GloFAS domain = 6,221,926 cells)")


def main():
    ok1, load_mg_l = t1_generated_inputs()
    host = load_host()
    ok2, ok3, ok4, unit = t2_t3_t4(host)
    ok5 = t5_coupled(host, unit, load_mg_l)
    t6_scaling(host)
    res = {"T1 generator<->coupler": ok1, "T2 IC seeding": ok2,
           "T3 kinetics = template": ok3, "T4 temperature": ok4,
           "T5 coupled N budget": ok5}
    print("\n" + "\n".join("  %-26s %s" % (k, "PASS" if v else "FAIL") for k, v in res.items()))
    allok = all(res.values())
    print("==> DEEP N TEST (openWQ generator inputs):", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
