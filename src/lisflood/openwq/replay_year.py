#!/usr/bin/env python
"""openWQ replay of a LONG recorded LISFLOOD run (e.g. one year, global), in time
chunks x groups of whole river basins, running alongside LISFLOOD.

Why: a global day of coupling fields is ~0.5-1.3 GB and openWQ's global state
does not fit in one process next to LISFLOOD, so the year is processed
chunk by chunk (default 28 days) and, inside each chunk, basin-group by
basin-group (replay_partitioned.py logic). openWQ's state (mass of every
species in every cell + pending-IC flags) is carried from one chunk to the
next through the hydrolink (save_state / restore after priming), which is
bitwise identical to one continuous run (tests: use case, 3 chunks).

Per chunk: wait until LISFLOOD has written its last day -> replay every group
-> merge the openWQ output rows into global frames (float16, near-dry cells
masked; frames/frame_YYYYMMDD.npz, one array per compartment|species) ->
delete the chunk's recordings and raw openWQ HDF5. If more than --max-backlog
recorded days wait on disk beyond the chunk being processed, LISFLOOD is paused
(SIGSTOP) until that backlog falls below --resume-at (SIGCONT).

usage: replay_year.py RECORD_DIR CASE_DIR OUT_DIR --groups 8 --chunk 28 --nsteps 364
                      [--lisflood-match coldstart_global_year] [--dry-mm 0.01]
"""
import argparse
import datetime
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay_partitioned import basin_of           # noqa: E402

MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                                   "SEP", "OCT", "NOV", "DEC"], 1)}


def stamp_to_date(s):                  # "2015JAN09-00:00:00"
    return datetime.date(int(s[:4]), MON[s[4:7]], int(s[7:9]))


def step_files(rec):
    return sorted(p for p in glob.glob(os.path.join(rec, "step_*.npz")) if ".tmp" not in p)


def lisflood_pids(match):
    r = subprocess.run(["pgrep", "-f", match], capture_output=True, text=True)
    pids = []
    for p in r.stdout.split():
        cmd = subprocess.run(["ps", "-o", "command=", "-p", p], capture_output=True, text=True).stdout
        if cmd.strip().startswith("python") and "lisf1.py" in cmd:
            pids.append(int(p))
    return pids


class Throttle(threading.Thread):
    """Pause LISFLOOD while too many recorded days wait on disk BEYOND the chunk that
    openWQ is processing (or waiting for): LISFLOOD keeps running during a chunk's
    replay and only stops when it gets more than `hi` days ahead of it."""
    def __init__(self, rec, match, hi, lo):
        super().__init__(daemon=True)
        self.rec, self.match, self.hi, self.lo = rec, match, hi, lo
        self.paused, self.stop = False, False
        self.horizon = 0                      # last step of the current chunk (set by main)

    def waiting(self):
        n = 0
        for p in step_files(self.rec):
            try:
                if int(os.path.basename(p)[5:10]) > self.horizon:
                    n += 1
            except ValueError:
                pass
        return n

    def run(self):
        while not self.stop:
            n = self.waiting()
            pids = lisflood_pids(self.match)
            if pids and not self.paused and n > self.hi:
                for p in pids:
                    os.kill(p, signal.SIGSTOP)
                self.paused = True
                print(f"[throttle] {n} days ahead of chunk end {self.horizon} -> LISFLOOD paused", flush=True)
            elif self.paused and n < self.lo:
                for p in pids:
                    os.kill(p, signal.SIGCONT)
                self.paused = False
                print(f"[throttle] {n} days ahead of chunk end {self.horizon} -> LISFLOOD resumed", flush=True)
            time.sleep(20)

    def release(self):
        self.stop = True
        if self.paused:
            for p in lisflood_pids(self.match):
                os.kill(p, signal.SIGCONT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir")
    ap.add_argument("case_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=28)
    ap.add_argument("--nsteps", type=int, required=True)
    ap.add_argument("--lisflood-match", default="coldstart_global_year")
    ap.add_argument("--max-backlog", type=int, default=40)
    ap.add_argument("--resume-at", type=int, default=20)
    ap.add_argument("--dry-mm", type=float, default=0.01)
    ap.add_argument("--keep-records", action="store_true")
    a = ap.parse_args()
    rec, out = os.path.abspath(a.record_dir), os.path.abspath(a.out_dir)
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    thr = Throttle(rec, a.lisflood_match, a.max_backlog, a.resume_at)
    thr.start()
    try:
        while not os.path.exists(os.path.join(rec, "static.npz")):
            time.sleep(30)
        time.sleep(5)
        st = np.load(os.path.join(rec, "static.npz"))
        ncells = int(st["ncells"])
        vthr = a.dry_mm * 1e-3 * np.asarray(st["pixel_area_m2"], dtype=float)
        # basin groups (fixed for the whole run)
        basin = basin_of(np.asarray(st["dst_ix"]).astype(np.int64))
        outlets, inv, counts = np.unique(basin, return_inverse=True, return_counts=True)
        K = max(1, min(a.groups, outlets.size))
        load, gob = np.zeros(K, dtype=np.int64), np.empty(outlets.size, dtype=np.int64)
        for b in np.argsort(counts)[::-1]:
            g = int(np.argmin(load)); gob[b] = g; load[g] += counts[b]
        gid = gob[inv]
        subsets = []
        for g in range(K):
            gdir = os.path.join(out, f"group_{g}")
            os.makedirs(gdir, exist_ok=True)
            sub = np.nonzero(gid == g)[0].astype(np.int64)
            np.save(os.path.join(gdir, "subset.npy"), sub)
            subsets.append(sub)
        log = {"ncells": ncells, "groups": [int(s.size) for s in subsets], "chunks": []}
        print(f"[year] {ncells} px, {outlets.size} basins -> {K} groups {load.tolist()}; "
              f"{a.nsteps} steps in chunks of {a.chunk}", flush=True)
        c, first = 0, 1
        while first <= a.nsteps:
            c += 1
            last = min(first + a.chunk - 1, a.nsteps)
            thr.horizon = last
            want = os.path.join(rec, "step_%05d.npz" % last)
            t_wait = time.time()
            while not os.path.exists(want):
                time.sleep(30)
            t0 = time.time()
            ginfo = []
            for g in range(K):
                gdir = os.path.join(out, f"group_{g}")
                cdir = os.path.join(gdir, f"chunk_{c:02d}")
                if os.path.exists(cdir):
                    shutil.rmtree(cdir)
                os.makedirs(cdir)
                shutil.copytree(a.case_dir, os.path.join(cdir, "openwq_case"),
                                ignore=shutil.ignore_patterns("openwq_out"))
                cmd = [sys.executable, os.path.join(HERE, "replay.py"), rec,
                       os.path.join(cdir, "openwq_case", "openWQ_master.json"),
                       "--subset", os.path.join(gdir, "subset.npy"), "--diag-dir", cdir,
                       "--first-step", str(first), "--last-step", str(last),
                       "--state-out", os.path.join(gdir, f"state_{c:02d}.npz")]
                if c > 1:
                    cmd += ["--state-in", os.path.join(gdir, f"state_{c - 1:02d}.npz")]
                with open(os.path.join(cdir, "replay.log"), "w") as lg:
                    rc = subprocess.run(["/usr/bin/time", "-l"] + cmd, cwd=cdir,
                                        stdout=lg, stderr=subprocess.STDOUT).returncode
                summ = json.load(open(os.path.join(cdir, "replay_summary.json"))) \
                    if os.path.exists(os.path.join(cdir, "replay_summary.json")) else {}
                ginfo.append({"group": g, "rc": rc, **summ})
                if rc != 0:
                    raise SystemExit(f"chunk {c} group {g} failed: {cdir}/replay.log")
                if c > 2:                                  # keep the last two states only
                    old = os.path.join(gdir, f"state_{c - 2:02d}.npz")
                    if os.path.exists(old):
                        os.remove(old)
            # merge output rows into global frames (skip the priming row of restarted chunks)
            day_of = {}
            for s in range(first, last + 1):
                f = np.load(os.path.join(rec, "step_%05d.npz" % s))
                d = datetime.date(*[int(x) for x in f["date"][:3]])
                day_of[d] = s
            first_day = datetime.date(*[int(x) for x in np.load(os.path.join(rec, "step_%05d.npz" % first))["date"][:3]]) \
                - datetime.timedelta(days=1)
            h5dir0 = os.path.join(out, "group_0", f"chunk_{c:02d}", "openwq_case", "openwq_out", "HDF5")
            names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(h5dir0, "*-main.h5")))
            frames = {}
            for b in names:
                comp, rest = b.split("@", 1)
                spec = rest.split("#", 1)[0]
                for g in range(K):
                    p = os.path.join(out, f"group_{g}", f"chunk_{c:02d}", "openwq_case", "openwq_out", "HDF5", b)
                    with h5py.File(p, "r") as h:
                        ts = [t.decode() if isinstance(t, bytes) else str(t) for t in h["timestamps"][...]]
                        x = np.asarray(h["xyz_elements"][0], dtype=np.int64) - 1
                        conc = np.asarray(h["concentrations"][...], dtype=np.float32)
                    for j, t in enumerate(ts):
                        if j == 0 and c > 1:
                            continue
                        dd = stamp_to_date(t)
                        arr = frames.setdefault(dd, {}).setdefault(f"{comp}|{spec}",
                                                                   np.full(ncells, np.nan, dtype=np.float32))
                        arr[subsets[g][x]] = conc[j]
            for dd, layers in frames.items():
                if dd == first_day:
                    vols = {k: np.load(os.path.join(rec, "step_%05d.npz" % first))["V0_" + k.split("|")[0]]
                            for k in layers}
                elif dd in day_of:
                    vols = {k: np.load(os.path.join(rec, "step_%05d.npz" % day_of[dd]))["V1_" + k.split("|")[0]]
                            for k in layers}
                else:
                    vols = None
                for k, arr in layers.items():
                    arr[arr <= -9998] = np.nan
                    if vols is not None:
                        arr[np.asarray(vols[k], dtype=float) < vthr] = np.nan
                np.savez_compressed(os.path.join(out, "frames", f"frame_{dd:%Y%m%d}.npz"),
                                    **{k: v.astype(np.float16) for k, v in layers.items()})
            # clean up the chunk
            for g in range(K):
                shutil.rmtree(os.path.join(out, f"group_{g}", f"chunk_{c:02d}", "openwq_case", "openwq_out"),
                              ignore_errors=True)
            if not a.keep_records:
                for s in range(first, last + 1):
                    try:
                        os.remove(os.path.join(rec, "step_%05d.npz" % s))
                    except FileNotFoundError:
                        pass
            ent = {"chunk": c, "steps": [first, last], "wait_s": round(t0 - t_wait, 1),
                   "replay_s": round(time.time() - t0, 1), "frames": sorted(f"{d:%Y-%m-%d}" for d in frames),
                   "groups": ginfo}
            log["chunks"].append(ent)
            json.dump(log, open(os.path.join(out, "year_log.json"), "w"), indent=1)
            print(f"[year] chunk {c} steps {first}-{last}: replay {ent['replay_s']} s "
                  f"(waited {ent['wait_s']} s for LISFLOOD); frames {ent['frames']}", flush=True)
            first = last + 1
    finally:
        thr.release()
    print("[year] DONE", flush=True)


if __name__ == "__main__":
    main()
