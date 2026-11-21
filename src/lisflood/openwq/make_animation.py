#!/usr/bin/env python
"""Map animation of an openWQ run driven by LISFLOOD (one frame per output time).

Reads openWQ's own HDF5 output (openwq_out/HDF5/<COMPARTMENT>@<SPECIES>#<UNITS>-main.h5:
`concentrations` [time, cell], `timestamps`, `xyz_elements`) and puts every
LISFLOOD compressed pixel back on the LISFLOOD grid with the mask that the
coupling recorder saved (static.npz, LISFLOOD MaskInfo: True = outside domain).

usage: python make_animation.py RECORD_DIR {OPENWQ_OUT_HDF5_DIR | partitions.json} OUT_PREFIX
           [--compartments SOIL_TOP_A,UPPER_ZONE,CHANNEL] [--extent W,E,S,N]
           [--coarsen K] [--fps 1] [--dry-mm 0.01]
Concentrations are not drawn (grey) where the compartment holds less water than
--dry-mm (mm over the pixel area; volumes from the coupling recording): openWQ
reports mass/volume, which is meaningless as the volume tends to zero.
writes OUT_PREFIX.mp4, OUT_PREFIX.gif, OUT_PREFIX_frame<NN>.png
"""
import argparse
import glob
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib import animation                    # noqa: E402

NICE = {"SOIL_TOP_A": "topsoil (LISFLOOD layer 1a)", "SOIL_TOP_B": "soil layer 1b",
        "SOIL_SUB": "subsoil (layer 2)", "UPPER_ZONE": "upper groundwater zone",
        "LOWER_ZONE": "lower groundwater zone", "CHANNEL": "river channel"}


def coarsen(a, k):
    if k <= 1:
        return a
    ny, nx = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:ny, :nx].reshape(ny // k, k, nx // k, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir")
    ap.add_argument("h5_dir")
    ap.add_argument("out_prefix")
    ap.add_argument("--compartments", default="SOIL_TOP_A,UPPER_ZONE,CHANNEL")
    ap.add_argument("--extent", default=None, help="W,E,S,N of the LISFLOOD grid (default: index space)")
    ap.add_argument("--coarsen", type=int, default=1)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--dry-mm", type=float, default=0.01)
    ap.add_argument("--title", default="LISFLOOD x openWQ (nitrogen_basic)")
    argv = sys.argv[1:]
    for i in range(len(argv) - 1):              # "--extent -180,..." would read as an option
        if argv[i] == "--extent":
            argv[i:i + 2] = ["--extent=" + argv[i + 1]]
            break
    a = ap.parse_args(argv)

    st = np.load(os.path.join(a.record_dir, "static.npz"))
    mask = np.asarray(st["mask"], dtype=bool)               # True = outside the domain
    inside = ~mask
    comps = a.compartments.split(",")
    # one openWQ output folder, or the basin groups of replay_partitioned.py
    if a.h5_dir.endswith(".json"):
        import json
        groups = [(os.path.join(g["dir"], "openwq_case", "openwq_out", "HDF5"),
                   np.load(os.path.join(g["dir"], "subset.npy")))
                  for g in json.load(open(a.h5_dir))["groups"]]
    else:
        groups = [(a.h5_dir, None)]
    files = {}
    for p in glob.glob(os.path.join(groups[0][0], "*-main.h5")):
        b = os.path.basename(p)
        comp, rest = b.split("@", 1)
        spec, units = rest.split("#", 1)
        files[(comp, spec)] = (b, units.replace("-main.h5", "").replace("|", "/"))
    species = sorted({s for c, s in files if c in comps},
                     key=lambda s: ["NH4-N", "NO3-N", "ORG_N", "N2_LOSS"].index(s)
                     if s in ["NH4-N", "NO3-N", "ORG_N", "N2_LOSS"] else 99)
    extent = [float(x) for x in a.extent.split(",")] if a.extent else None

    # load all panels: [species][comp] -> (times, cells)
    data, stamps = {}, None
    for s in species:
        for c in comps:
            b, units = files[(c, s)]
            full = None
            for gdir, sub in groups:
                with h5py.File(os.path.join(gdir, b), "r") as h:
                    x = np.asarray(h["xyz_elements"][0], dtype=np.int64) - 1    # 1-based x -> pixel
                    conc = np.asarray(h["concentrations"][...], dtype=np.float32)
                    conc[conc <= -9998] = np.nan                               # NO_WATER_CONC_FLAG
                    if stamps is None:
                        stamps = [t.decode() if isinstance(t, bytes) else str(t) for t in h["timestamps"][...]]
                if full is None:
                    full = np.full((conc.shape[0], int(st["ncells"])), np.nan, dtype=np.float32)
                full[:, x if sub is None else sub[x]] = conc
            data[(s, c)] = (full, units)
    nt = len(stamps)
    # water volume of each compartment at each output time: t=0 is the start of
    # the first LISFLOOD step (openWQ IC seeding), t>=1 the end of step t
    recs = sorted(p for p in glob.glob(os.path.join(a.record_dir, "step_*.npz")) if ".tmp" not in p)
    vthr = a.dry_mm * 1e-3 * np.asarray(st["pixel_area_m2"], dtype=float)
    for c in comps:
        vol = [np.load(recs[0])["V0_" + c]] + [np.load(r)["V1_" + c] for r in recs]
        for s in species:
            full = data[(s, c)][0]
            for t in range(min(nt, len(vol))):
                full[t][vol[t] < vthr] = np.nan

    def grid(vec):
        g = np.full(mask.shape, np.nan, dtype=np.float32)
        g[inside] = vec
        return coarsen(g, a.coarsen)

    # one colour range per species (all compartments/times), robust percentiles
    vr = {}
    for s in species:
        allv = np.concatenate([data[(s, c)][0][np.isfinite(data[(s, c)][0])] for c in comps])
        lo, hi = np.percentile(allv, [1, 99]) if allv.size else (0, 1)
        vr[s] = (float(lo), float(hi if hi > lo else lo + 1e-6))

    nr, nc = len(species), len(comps)
    aspect = mask.shape[1] / mask.shape[0]
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc * min(aspect, 2.4) / 2.4 + 1.5, 2.3 * nr + 0.9),
                             squeeze=False, constrained_layout=True)
    ims = {}
    for i, s in enumerate(species):
        for j, c in enumerate(comps):
            ax = axes[i][j]
            full, units = data[(s, c)]
            ims[(s, c)] = ax.imshow(grid(full[0]), cmap="viridis", vmin=vr[s][0], vmax=vr[s][1],
                                    extent=extent, origin="upper", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#dfe3e8")
            if i == 0:
                ax.set_title(NICE.get(c, c), fontsize=10)
            if j == 0:
                ax.set_ylabel(s, fontsize=10)
        fig.colorbar(ims[(s, comps[-1])], ax=axes[i][:], shrink=0.85, pad=0.01,
                     label=f"{s} [{data[(s, comps[0])][1]}]")
    sup = fig.suptitle(f"{a.title} — {stamps[0]}", fontsize=11)

    def update(t):
        for (s, c), im in ims.items():
            im.set_data(grid(data[(s, c)][0][t]))
        sup.set_text(f"{a.title} — {stamps[t]}")
        return list(ims.values()) + [sup]

    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    for t in range(nt):
        update(t)
        fig.savefig(f"{a.out_prefix}_frame{t:02d}.png", dpi=110)
    anim = animation.FuncAnimation(fig, update, frames=nt, blit=False)
    try:
        anim.save(a.out_prefix + ".mp4", writer=animation.FFMpegWriter(fps=a.fps, bitrate=6000), dpi=110)
    except Exception as e:                                          # noqa: BLE001
        print("mp4 failed:", e)
    anim.save(a.out_prefix + ".gif", writer=animation.PillowWriter(fps=a.fps), dpi=80)
    print("frames:", nt, "->", a.out_prefix + ".mp4/.gif")


if __name__ == "__main__":
    main()
