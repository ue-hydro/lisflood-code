#!/usr/bin/env python
"""Offline openWQ replay of a recorded LISFLOOD run (option ``OpenWQRecord``).

LISFLOOD -> openWQ coupling is one-way, so replaying the recorded coupling
fields through ``LisfloodOpenWQCoupler.advance`` gives exactly the transfers
the in-process coupling (option ``OpenWQ``) would send.

usage: python replay.py RECORD_DIR MASTER_JSON [--substeps N] [--steps K]
       (needs openwq_py importable: built from OpenWQ_hydrolink + openwq/)

Writes <RECORD_DIR>/replay_diag.csv (per LISFLOOD step: wall time, peak RSS,
openWQ domain mass per species) and replay_summary.json.
"""
import argparse
import glob
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # coupler.py + openwq_py*.so live here

import numpy as np                            # noqa: E402
from coupler import (LisfloodOpenWQCoupler, TRANSFERS, INFLOW_KEY, CASCADE_KEYS,   # noqa: E402
                     read_owq_json)


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6   # bytes on macOS, KB on Linux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir")
    ap.add_argument("master_json")
    ap.add_argument("--substeps", type=int, default=None,
                    help="openWQ sub-steps per LISFLOOD step (default: LISFLOOD NoRoutSteps)")
    ap.add_argument("--steps", type=int, default=None, help="replay only the first K steps")
    ap.add_argument("--no-budget", action="store_true",
                    help="skip the independent N budget (mass that leaves the domain)")
    ap.add_argument("--closed", action="store_true",
                    help="conservation test: replay with the external load and every "
                         "out-of-domain transfer set to zero (river mouths, deep loss, "
                         "abstractions); total N must then stay constant (+ deferred IC)")
    ap.add_argument("--subset", default=None,
                    help=".npy of global pixel indices to replay (whole river basins; see "
                         "replay_partitioned.py)")
    ap.add_argument("--diag-dir", default=None, help="where to write replay_diag.csv/summary (default RECORD_DIR)")
    ap.add_argument("--first-step", type=int, default=1, help="first recorded step to replay (1-based)")
    ap.add_argument("--last-step", type=int, default=None, help="last recorded step to replay (inclusive)")
    ap.add_argument("--state-in", default=None, help="openWQ state saved by a previous chunk (restored after priming)")
    ap.add_argument("--state-out", default=None, help="save the openWQ state after the last step here")
    ap.add_argument("--channel-scheme", choices=["implicit", "explicit"], default="implicit",
                    help="channel transport: implicit backward-Euler cascade (default) or plain explicit "
                         "openWQ transfers (runs made before the implicit scheme)")
    ap.add_argument("--keep-load", action="store_true",
                    help="with --closed: keep the external load (tests the EWF input alone)")
    a = ap.parse_args()

    st = np.load(os.path.join(a.record_dir, "static.npz"))
    steps = sorted(p for p in glob.glob(os.path.join(a.record_dir, "step_*.npz")) if ".tmp" not in p)
    num = lambda p: int(os.path.basename(p)[5:10])
    steps = [p for p in steps if num(p) >= a.first_step and (a.last_step is None or num(p) <= a.last_step)]
    if a.steps:
        steps = steps[:a.steps]
    if not steps:
        raise SystemExit("no step_*.npz in " + a.record_dir)
    nsub = a.substeps if a.substeps else int(st["noroutsteps"])
    ncells_all = int(st["ncells"])
    dst_all = np.asarray(st["dst_ix"]).astype(np.int64)
    if a.subset:
        sub = np.load(a.subset).astype(np.int64)
        lut = np.full(ncells_all, -1, dtype=np.int64)
        lut[sub] = np.arange(sub.size)
        d = dst_all[sub]
        dst_ix = np.where(d < 0, -1, lut[np.where(d < 0, 0, d)])
        leak = int(np.count_nonzero((d >= 0) & (dst_ix < 0)))
        if leak:
            raise SystemExit(f"subset is not made of whole basins: {leak} pixels drain outside it")
    else:
        sub, dst_ix = None, dst_all
    ncells = int(sub.size) if sub is not None else ncells_all
    diag_dir = a.diag_dir or a.record_dir

    t0 = time.time()
    cpl = LisfloodOpenWQCoupler(None, a.master_json, channel_scheme=a.channel_scheme)
    cpl.setup(ncells, dst_ix.astype(np.int32), float(st["dtsec"]), substeps=nsub)
    t_setup = time.time() - t0
    host = cpl._host
    nspec = host.num_species()
    if a.state_in:
        cpl.restore_state_after_prime(a.state_in)
    print(f"[replay] {ncells} pixels x {len(cpl._icmp)} compartments, "
          f"{nspec} species, {nsub} openWQ sub-steps per LISFLOOD step; "
          f"setup {t_setup:.1f} s, peak RSS {peak_rss_gb():.2f} GB", flush=True)

    # Independent budget: BGC only converts between the species (total N is
    # conserved), so  d(total) = EWF load in - mass leaving the domain (+ IC of
    # cells that were dry at the start). Leaving mass is recomputed here from
    # openWQ's own advection rule (exported fraction = min(flux/source volume,1)
    # of the source mass at the start of the sub-step).
    ewf_conc = {}
    for e in read_owq_json(a.master_json).get("OPENWQ_INPUT", {}).get("EXTERNAL_WATER_FLUXES", {}).values():
        ewf = read_owq_json(os.path.join(os.path.dirname(os.path.abspath(a.master_json)), e["FILEPATH"]))
        for k2, blk in ewf.items():
            if k2 != "METADATA" and isinstance(blk, dict) and blk.get("EXTERNAL_INPUTFLUX_NAME") == cpl._topsoil_ewf:
                u = str(blk.get("UNITS", "")).lower()
                v = float(list(blk["DATA"].values())[0][-1])
                ewf_conc[blk["CHEMICAL_NAME"]] = v * {"mg/l": 1.0, "g/m3": 1.0, "ug/l": 1e-3}.get(u, float("nan"))
    out_keys = [(k2, s2) for k2, s2, d2 in TRANSFERS if d2 is None]
    pits = dst_ix < 0
    # mobile species (the only ones openWQ transports) from the generated BGC file
    case = os.path.dirname(os.path.abspath(a.master_json))
    bgc_path = read_owq_json(a.master_json)["MODULES"]["BIOGEOCHEMISTRY"]["MODULE_CONFIG_FILEPATH"]
    bgc = read_owq_json(os.path.join(case, bgc_path))
    species = [bgc["CHEMICAL_SPECIES"]["LIST"][k] for k in sorted(bgc["CHEMICAL_SPECIES"]["LIST"], key=int)]
    mobile = [species.index(s2) for s2 in bgc["CHEMICAL_SPECIES"].get("BGC_GENERAL_MOBILE_SPECIES", species)]
    bud = {"in": 0.0, "out": 0.0, "m_start": None}

    def on_substep(k, vk, flux):
        if bud["m_start"] is None:                   # after IC seeding, before any transfer
            bud["m_start"] = sum(host.total_mass(j) for j in range(nspec))
        bud["in"] += float(np.sum(flux[INFLOW_KEY])) * sum(ewf_conc.values())
        ic = cpl._icmp
        implicit = cpl.channel_scheme == "implicit"
        if implicit and cpl.cascade_out is not None:
            # channel outflows routed by the implicit cascade: the coupler gives, per
            # source compartment, the fraction of its start mass that left the domain
            for icmp, frac in cpl.cascade_out.items():
                for ich in mobile:
                    m = host.mass_array(icmp, ich).ravel()
                    bud["out"] += float(np.sum(frac * np.maximum(m, 0)))
        for key, src in out_keys + [("CH_DOWN", "CHANNEL")]:
            if implicit and key in CASCADE_KEYS:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(vk[src] > 0, flux[key] / vk[src], 1.0)
            frac = np.minimum(frac, 1.0)
            frac[flux[key] <= 0] = 0.0
            if key == "CH_DOWN":
                frac = np.where(pits, frac, 0.0)
            if not frac.any():
                continue
            for ich in mobile:
                m = host.mass_array(ic[src], ich).ravel()
                bud["out"] += float(np.sum(frac * np.maximum(m, 0)))

    rows = []
    t_run = time.time()
    for i, p in enumerate(steps, 1):
        f = {k: (v.astype(np.float64) if v.dtype == np.float32 else v) for k, v in np.load(p).items()}
        if sub is not None:
            f = {k: (v[sub] if v.ndim == 1 and v.size == ncells_all else v) for k, v in f.items()}
        if a.closed:
            for key in ([] if a.keep_load else [INFLOW_KEY]) + [k2 for k2, _, d2 in TRANSFERS if d2 is None]:
                f["F_" + key] = np.zeros_like(f["F_" + key])
            f["F_CH_DOWN"] = np.where(pits, 0.0, f["F_CH_DOWN"])
        ts = time.time()
        cpl.advance(f, on_substep=None if a.no_budget else on_substep)
        dt = time.time() - ts
        mass = [host.total_mass(k) for k in range(nspec)]
        d = cpl.diag[-1]
        resid = (sum(mass) - bud["m_start"] - bud["in"] + bud["out"]) if bud["m_start"] else float("nan")
        rows.append([d["date"], round(dt, 2), round(peak_rss_gb(), 3),
                     d["channel_cfl_gt1_cells"]] + mass + [bud["in"], bud["out"], resid])
        print(f"[replay] step {i}/{len(steps)} {d['date']}: {dt:.1f} s, "
              f"peak RSS {peak_rss_gb():.2f} GB, domain mass (g) "
              + ", ".join(f"{m:.6e}" for m in mass)
              + f" | cum in {bud['in']:.4e} out {bud['out']:.4e} budget resid {resid:.3e} g"
              + (f" ({resid / bud['m_start']:.2e} of start mass)" if bud["m_start"] else ""),
              flush=True)
    t_run = time.time() - t_run
    if a.state_out:
        cpl.save_state(a.state_out)

    hdr = "date,wall_s,peak_rss_gb,channel_cells_flux_gt_volume," + \
          ",".join(f"mass_g_{species[k]}" for k in range(nspec)) + \
          ",cum_ewf_in_g,cum_out_of_domain_g,budget_residual_g"
    with open(os.path.join(diag_dir, "replay_diag.csv"), "w") as fh:
        fh.write(hdr + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(a.master_json)), "openwq_out")
    out_bytes = sum(os.path.getsize(os.path.join(dp, fn))
                    for dp, _, fns in os.walk(out_dir) for fn in fns) if os.path.isdir(out_dir) else 0
    summ = dict(ncells=ncells, compartments=len(cpl._icmp), species=nspec,
                substeps=nsub, steps=len(steps), setup_s=round(t_setup, 1),
                run_s=round(t_run, 1), s_per_step=round(t_run / len(steps), 1),
                peak_rss_gb=round(peak_rss_gb(), 2), openwq_out_bytes=out_bytes,
                n_budget_start_g=bud["m_start"], n_ewf_in_g=bud["in"], n_out_of_domain_g=bud["out"],
                n_budget_residual_g=rows[-1][-1],
                n_budget_residual_rel=(rows[-1][-1] / bud["m_start"]) if bud["m_start"] else None)
    with open(os.path.join(diag_dir, "replay_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    print("[replay] summary", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
