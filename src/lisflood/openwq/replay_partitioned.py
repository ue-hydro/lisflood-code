#!/usr/bin/env python
"""Replay a recorded LISFLOOD run through openWQ in groups of WHOLE river basins.

LISFLOOD moves water between pixels only along the drainage network (channel
transfer pixel -> downstream pixel; every other transfer stays inside a pixel),
so pixels of different river basins (different outlet/pit) never exchange
solute. Replaying the domain basin-group by basin-group is therefore identical
to one global replay, but each openWQ instance only holds its own pixels -- the
global GloFAS domain (6.2 M pixels x 6 compartments) does not fit in memory as
one instance on a workstation. Groups are run one after another in separate
processes (memory released in between); each gets its own copy of the generated
openWQ inputs (case folder) and writes its own openwq_out/.

usage: python replay_partitioned.py RECORD_DIR CASE_DIR OUT_DIR --groups K [--substeps N]
writes OUT_DIR/group_<g>/{subset.npy, openwq_case/openwq_out/..., replay_*.{csv,json,log}}
       OUT_DIR/partitions.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def basin_of(dst):
    """Outlet (pit) pixel of every pixel: follow dst_ix (pits = -1) by pointer jumping."""
    n = dst.size
    r = np.where(dst < 0, np.arange(n), dst).astype(np.int64)
    while True:
        r2 = r[r]
        if np.array_equal(r2, r):
            return r
        r = r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir")
    ap.add_argument("case_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--substeps", type=int, default=None)
    ap.add_argument("--no-budget", action="store_true")
    a = ap.parse_args()
    rec = os.path.abspath(a.record_dir)
    st = np.load(os.path.join(rec, "static.npz"))
    dst = np.asarray(st["dst_ix"]).astype(np.int64)
    basin = basin_of(dst)
    outlets, inv, counts = np.unique(basin, return_inverse=True, return_counts=True)
    # greedy balancing of whole basins into K groups (largest first)
    order = np.argsort(counts)[::-1]
    a.groups = max(1, min(a.groups, int(outlets.size)))   # never more groups than basins
    load = np.zeros(a.groups, dtype=np.int64)
    gid_of_basin = np.empty(outlets.size, dtype=np.int64)
    for b in order:
        g = int(np.argmin(load))
        gid_of_basin[b] = g
        load[g] += counts[b]
    gid = gid_of_basin[inv]
    os.makedirs(a.out_dir, exist_ok=True)
    parts = {"ncells": int(st["ncells"]), "n_basins": int(outlets.size),
             "largest_basin_pixels": int(counts.max()), "groups": []}
    print(f"[partition] {int(st['ncells'])} pixels, {outlets.size} basins "
          f"(largest {counts.max()} px) -> {a.groups} groups of {load.tolist()} px", flush=True)
    for g in range(a.groups):
        gdir = os.path.abspath(os.path.join(a.out_dir, f"group_{g}"))
        os.makedirs(gdir, exist_ok=True)
        subset = np.nonzero(gid == g)[0].astype(np.int64)
        np.save(os.path.join(gdir, "subset.npy"), subset)
        case = os.path.join(gdir, "openwq_case")
        if os.path.exists(case):
            shutil.rmtree(case)
        shutil.copytree(a.case_dir, case, ignore=shutil.ignore_patterns("openwq_out"))
        cmd = [sys.executable, os.path.join(HERE, "replay.py"), rec,
               os.path.join(case, "openWQ_master.json"), "--subset", os.path.join(gdir, "subset.npy"),
               "--diag-dir", gdir]
        if a.substeps:
            cmd += ["--substeps", str(a.substeps)]
        if a.no_budget:
            cmd += ["--no-budget"]
        t0 = time.time()
        with open(os.path.join(gdir, "replay.log"), "w") as log:
            rc = subprocess.run(["/usr/bin/time", "-l"] + cmd, cwd=gdir, stdout=log, stderr=subprocess.STDOUT).returncode
        summ = {}
        sp = os.path.join(gdir, "replay_summary.json")
        if os.path.exists(sp):
            summ = json.load(open(sp))
        parts["groups"].append({"group": g, "pixels": int(subset.size), "rc": rc,
                                "wall_s": round(time.time() - t0, 1), "dir": gdir, **summ})
        print(f"[partition] group {g}: {subset.size} px, rc={rc}, {time.time() - t0:.0f} s, "
              f"peak RSS {summ.get('peak_rss_gb')} GB", flush=True)
        json.dump(parts, open(os.path.join(a.out_dir, "partitions.json"), "w"), indent=1)
        if rc != 0:
            raise SystemExit(f"group {g} failed (see {gdir}/replay.log)")


if __name__ == "__main__":
    main()
