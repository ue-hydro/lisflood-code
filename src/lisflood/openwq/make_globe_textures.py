#!/usr/bin/env python
"""Data for the globe viewer (globe/index.html) of an openWQ run driven by LISFLOOD.

Reads openWQ's own HDF5 output (single folder or the basin groups of replay_partitioned.py)
or the global frames of replay_year.py, puts every LISFLOOD pixel back on its grid with the
recorder's mask and writes the VALUES (not colours) so the viewer colours them on the GPU:

  * every layer is stored as 8-bit codes on its own log10 (or linear) scale:
      code 1..254 = value between the 0.5th and 99.9th percentile of the layer over the whole
      period (ends clipped), 255 = no value (cell holds less than --dry-mm of water, or a
      float16 overflow in the frames), 0 = no model cell (raster layers only);
    three output times go in the R, G and B channels of one lossless WebP;
  * soil / groundwater layers: equirectangular raster (lon -180..180, lat 90..-90, --width x
    width/2), each pixel the mean of the wet LISFLOOD cells inside it;
  * rivers: the LISFLOOD channel network itself, one value per model cell with upstream area
    >= --river-min-km2, in drainage order (outlet first, then upstream, depth first);
    tex/network.webp holds each cell's grid row and column, its upstream area and its
    downstream cell (the recorder's routing, dst_ix), so the viewer draws every cell as a
    segment to its downstream cell at the model's own 0.05 deg resolution;
  * a reference per layer (tex/*_ref.webp): the mean over all output times of the log10 value
    (geometric mean; arithmetic mean for --scale linear), for the "departure" view;
  * statistics per output time: weighted histogram over the coding range and the 10th, 50th
    and 90th percentile (soil / groundwater: pixel area over wet cells; rivers: network
    cells, equal weights);
  * base data: hillshade of LISFLOOD's elevation map (tex/relief.webp) and the fraction of
    each pixel covered by model cells (tex/domain.webp, lossless) for the land-mask outline.
Everything the viewer needs to decode and label the maps is written to manifest.json.

usage: python make_globe_textures.py RECORD_DIR {HDF5_DIR | partitions.json | FRAMES_DIR} OUT_DIR
           [--extent -180,180,-60,90] [--width 2048] [--relief-width 4096] [--maps ROOT_FLAT]
           [--river-min-km2 5000] [--compartments SOIL_TOP_A,UPPER_ZONE,CHANNEL] [--scale log]
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

import h5py
import numpy as np
from matplotlib import colormaps
from PIL import Image

FORMAT = "globe-scalar-1"
ORDER = ["NH4-N", "NO3-N", "ORG_N", "N2_LOSS"]
NICE = {"SOIL_TOP_A": "Topsoil", "UPPER_ZONE": "Upper groundwater", "CHANNEL": "Rivers",
        "SOIL_TOP_B": "Soil layer 1b", "SOIL_SUB": "Subsoil", "LOWER_ZONE": "Lower groundwater"}
DEFAULT_CMAP = {"N2_LOSS": "viridis"}        # everything else: plasma
SEQ_CMAPS = ["plasma", "viridis", "magma", "inferno", "cividis"]
DIV_CMAPS = ["RdBu_r", "PuOr_r"]
NBINS = 64                                   # histogram bins over the coding range
PER_FILE = 3                                 # output times per image (R, G, B)
NET_W = 1024                                 # row length of the network images
ENC_PCT = (0.5, 99.9)                        # coding range, percentiles of the layer
DISPLAY_PCT = (2.0, 98.0)                    # default colour range
NICE_ANOM = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]


def load_groups(spec):
    if spec.endswith(".json"):
        return [(os.path.join(g["dir"], "openwq_case", "openwq_out", "HDF5"),
                 np.load(os.path.join(g["dir"], "subset.npy")))
                for g in json.load(open(spec))["groups"]]
    return [(spec, None)]


class Mapper:
    """LISFLOOD grid (ny, nx) over W,E,S,N -> full-globe texture (tw x tw/2) through
    kr x kc source blocks (nearest block for every texture pixel)."""

    def __init__(self, ny, nx, W, E, S, N, tw):
        self.th, self.tw = tw // 2, tw
        lat_t = 90.0 - (np.arange(self.th) + 0.5) * 180.0 / self.th
        lon_t = -180.0 + (np.arange(tw) + 0.5) * 360.0 / tw
        rows = np.floor((N - lat_t) / (N - S) * ny).astype(int)
        cols = np.floor((lon_t - W) / (E - W) * nx).astype(int)
        self.rok, self.cok = (rows >= 0) & (rows < ny), (cols >= 0) & (cols < nx)
        self.kr = max(1, int(round(ny / ((N - S) / 180.0 * self.th))))
        self.kc = max(1, int(round(nx / ((E - W) / 360.0 * tw))))
        self.gy, self.gx = (ny // self.kr) * self.kr, (nx // self.kc) * self.kc
        self.rr = np.clip(rows // self.kr, 0, self.gy // self.kr - 1)
        self.cc = np.clip(cols // self.kc, 0, self.gx // self.kc - 1)

    def blocks(self, g):        # (ny, nx) -> (by, bx, kr*kc)
        b = g[:self.gy, :self.gx].reshape(self.gy // self.kr, self.kr, self.gx // self.kc, self.kc)
        return b.transpose(0, 2, 1, 3).reshape(self.gy // self.kr, self.gx // self.kc, -1)

    def sample(self, blk):      # (by, bx) -> (th, tw), NaN outside the grid
        out = np.full((self.th, self.tw), np.nan, dtype=np.float32)
        sub = blk[np.ix_(self.rr, self.cc)]
        out[np.ix_(self.rok, self.cok)] = sub[np.ix_(self.rok, self.cok)]
        return out

    def mean(self, g):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self.sample(np.nanmean(self.blocks(g), axis=2))


def hillshade(elv, inside, lat):
    """Lambert hillshade (sun azimuth 315 deg, altitude 45 deg, x6 vertical exaggeration
    to read at 0.05 deg) of LISFLOOD's elevation map, contrast-stretched between its 1st
    and 99th percentile inside the domain, 0..1, NaN outside the domain."""
    z = np.where(inside, elv, 0.0).astype(np.float64) * 6.0      # sea level outside the domain
    dy = 0.05 * 111320.0
    dx = dy * np.maximum(np.cos(np.radians(lat)), 0.05)[:, None]
    drow, dcol = np.gradient(z)
    gn, ge = -drow / dy, dcol / dx                  # dz/dnorth (rows run north -> south), dz/deast
    nrm = np.sqrt(ge * ge + gn * gn + 1.0)          # unit surface normal (-ge, -gn, 1) / nrm
    az, alt = np.radians(315.0), np.radians(45.0)
    lx, ly, lz = np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)
    s = (-ge * lx - gn * ly + lz) / nrm
    p1, p99 = np.percentile(s[inside], [1, 99])
    s = (s - p1) / max(p99 - p1, 1e-6)
    return np.where(inside, np.clip(s, 0, 1), np.nan).astype(np.float32)


def encode(f, flo, fhi):
    """transformed values (NaN = no value) -> codes 1..254, 255 for no value"""
    ok = np.isfinite(f)
    c = 1.0 + np.round((np.clip(np.where(ok, f, flo), flo, fhi) - flo) / (fhi - flo) * 253.0)
    return np.where(ok, c, 255).astype(np.uint8)


def decode(c, flo, fhi):
    c = np.asarray(c)
    ok = (c >= 1) & (c <= 254)
    return np.where(ok, flo + (c.astype(np.float32) - 1.0) / 253.0 * (fhi - flo), np.nan).astype(np.float32)


def save_rgb(path, planes):
    """up to 3 equally shaped uint8 planes -> lossless RGB WebP (missing planes = 0)"""
    planes = list(planes) + [np.zeros_like(planes[0])] * (3 - len(planes))
    Image.fromarray(np.stack(planes, axis=-1), "RGB").save(path, "WEBP", lossless=True, quality=100,
                                                            method=4, exact=True)


def net_image(vec, rows):
    """vector in network order -> (rows, NET_W) uint8 image, zero padded"""
    out = np.zeros(rows * NET_W, dtype=np.uint8)
    out[:vec.size] = vec
    return out.reshape(rows, NET_W)


def drainage_order(down, upa):
    """depth-first order from the outlets upstream (larger upstream area first), so the
    downstream cell of every cell comes before it and a river's cells sit next to each other"""
    n = down.size
    has = down >= 0
    kids = np.nonzero(has)[0]
    kids = kids[np.lexsort((-upa[kids], down[kids]))]
    start = np.concatenate([[0], np.cumsum(np.bincount(down[has], minlength=n))])
    outs = np.nonzero(~has)[0]
    outs = outs[np.argsort(-upa[outs])]
    order = np.empty(n, dtype=np.int64)
    k, stack = 0, list(outs[::-1])
    while stack:
        i = stack.pop()
        order[k] = i
        k += 1
        c = kids[start[i]:start[i + 1]]
        if c.size:
            stack.extend(c[::-1].tolist())
    return order[:k]


def stops(name, n):
    cm = colormaps[name]
    return [[round(float(v), 4) for v in cm(i / (n - 1))[:3]] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir")
    ap.add_argument("h5")
    ap.add_argument("out_dir")
    ap.add_argument("--extent", default="-180,180,-60,90", help="W,E,S,N of the LISFLOOD grid")
    ap.add_argument("--width", type=int, default=2048, help="raster width, soil / groundwater")
    ap.add_argument("--relief-width", type=int, default=4096, help="width of the relief / land-mask maps")
    ap.add_argument("--river-min-km2", type=float, default=5000.0)
    ap.add_argument("--maps", default=None, help="folder with upArea / elv maps (default: "
                    "GloFASv5_data/root_flat next to the runs folder)")
    ap.add_argument("--compartments", default="SOIL_TOP_A,UPPER_ZONE,CHANNEL")
    ap.add_argument("--dry-mm", type=float, default=0.01)
    ap.add_argument("--scale", choices=["log", "linear"], default="log")
    ap.add_argument("--units", default="MG/L", help="units label in frames mode")
    argv = sys.argv[1:]
    for i in range(len(argv) - 1):              # "--extent -180,..." would read as an option
        if argv[i] == "--extent":
            argv[i:i + 2] = ["--extent=" + argv[i + 1]]
            break
    a = ap.parse_args(argv)
    t_start = time.time()

    st = np.load(os.path.join(a.record_dir, "static.npz"))
    mask = np.asarray(st["mask"], dtype=bool)
    inside = ~mask
    ny, nx = mask.shape
    W, E, S, N = [float(v) for v in a.extent.split(",")]
    area = np.asarray(st["pixel_area_m2"], dtype=np.float64)
    ncells = int(st["ncells"])
    lat_rows = N - (np.arange(ny) + 0.5) * (N - S) / ny
    soil_map = Mapper(ny, nx, W, E, S, N, a.width)
    dom_soil = np.isfinite(soil_map.mean(np.where(inside, 1.0, np.nan).astype(np.float32)))

    # ---- LISFLOOD static maps (same grid as the recorder's mask) ----
    maps = a.maps or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(a.record_dir)))), "root_flat")

    def read_map(pattern):
        hits = sorted(glob.glob(os.path.join(maps, pattern)))
        if not hits:
            return None
        with h5py.File(hits[0], "r") as h:
            arr = np.asarray(h["Band1"][...], dtype=np.float32)
        if arr.shape != mask.shape:
            print(f"WARNING {hits[0]} shape {arr.shape} != grid {mask.shape}; not used")
            return None
        print("map:", hits[0])
        return arr
    upa_m2 = read_map("upArea*.nc")
    elv = read_map("elv_*.nc")

    os.makedirs(os.path.join(a.out_dir, "tex"), exist_ok=True)
    for old in glob.glob(os.path.join(a.out_dir, "tex", "*.webp")):
        os.remove(old)                                   # no stale files of an earlier build

    # ---- river network: model cells with upstream area >= threshold, in drainage order ----
    net = None
    if upa_m2 is not None and "dst_ix" in st.files:
        upa = upa_m2[inside].astype(np.float64) / 1e6                       # km2, cell vector
        dst = np.asarray(st["dst_ix"], dtype=np.int64)
        sel = np.nonzero(np.isfinite(upa) & (upa >= a.river_min_km2))[0]
        pos = np.full(ncells, -1, dtype=np.int64)
        pos[sel] = np.arange(sel.size)
        d = dst[sel]
        down = np.where(d >= 0, pos[np.clip(d, 0, None)], -1)
        order = drainage_order(down, upa[sel])
        cells = sel[order]                                                   # network order -> cell
        rank = np.empty(sel.size, dtype=np.int64)
        rank[order] = np.arange(order.size)
        dn = down[order]
        dn = np.where(dn >= 0, rank[np.clip(dn, 0, None)], -1)               # downstream, network order
        k = np.arange(cells.size)
        assert np.all(dn < k), "downstream cell must precede its upstream cell"
        rr, cc = np.nonzero(inside)
        r, c = rr[cells], cc[cells]
        u = upa[cells]
        amin, amax = float(a.river_min_km2), float(u.max())
        u16 = np.round((np.log10(u) - np.log10(amin)) / (np.log10(amax) - np.log10(amin)) * 65535).astype(np.int64)
        delta = np.where(dn >= 0, k - dn, 0)
        rows = (cells.size + NET_W - 1) // NET_W
        hi_, lo_ = lambda x: net_image((x >> 8) & 255, rows), lambda x: net_image(x & 255, rows)
        img = np.concatenate([
            np.stack([hi_(r), lo_(r), np.zeros((rows, NET_W), np.uint8)], -1),
            np.stack([hi_(c), lo_(c), np.zeros((rows, NET_W), np.uint8)], -1),
            np.stack([hi_(u16), lo_(u16), np.zeros((rows, NET_W), np.uint8)], -1),
            np.stack([net_image((delta >> 16) & 255, rows), hi_(delta), lo_(delta)], -1)], axis=0)
        Image.fromarray(img, "RGB").save(os.path.join(a.out_dir, "tex", "network.webp"), "WEBP",
                                         lossless=True, quality=100, method=6, exact=True)
        net = dict(cells=cells, rows=rows, n=int(cells.size),
                   meta={"file": "tex/network.webp", "n": int(cells.size), "width": NET_W, "rows": rows,
                         "planes": ["row", "col", "upstream_area_log16", "downstream_delta24"],
                         "min_km2": amin, "max_km2": amax, "outlets": int((dn < 0).sum()),
                         "source": "LISFLOOD upArea map and the recorder's routing (dst_ix)"})
        print(f"network: upstream area >= {amin:g} km2 -> {cells.size} cells, {int((dn < 0).sum())} outlets, "
              f"{os.path.getsize(os.path.join(a.out_dir, 'tex', 'network.webp')) / 1e6:.2f} MB")

    def soil_texture(vec):
        g = np.full(mask.shape, np.nan, dtype=np.float32)
        g[inside] = vec
        return soil_map.mean(g)

    # ---- the layers ----
    comps = a.compartments.split(",")
    vthr = a.dry_mm * 1e-3 * area
    ffiles = sorted(glob.glob(os.path.join(a.h5, "frame_*.npz"))) if os.path.isdir(a.h5) else []
    if ffiles:
        # frames written by replay_year.py (already dry-masked, float16, one file per output date)
        mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        stamps = [f"{b[:4]}{mon[int(b[4:6]) - 1]}{b[6:8]}-00:00:00"
                  for b in (os.path.basename(f)[6:14] for f in ffiles)]
        pairs = [tuple(k.split("|")) for k in np.load(ffiles[0]).files]
        species = sorted({sp for cp, sp in pairs if cp in comps},
                         key=lambda x: ORDER.index(x) if x in ORDER else 99)

        def get_layer(c, sp):
            return np.stack([np.load(f)[f"{c}|{sp}"].astype(np.float32) for f in ffiles]), a.units
    else:
        groups = load_groups(a.h5)
        names = {}
        for p in glob.glob(os.path.join(groups[0][0], "*-main.h5")):
            b = os.path.basename(p)
            comp, rest = b.split("@", 1)
            spec, units = rest.split("#", 1)
            names[(comp, spec)] = (b, units.replace("-main.h5", "").replace("|", "/"))
        species = sorted({sp for c, sp in names if c in comps}, key=lambda x: ORDER.index(x) if x in ORDER else 99)
        recs = sorted(p for p in glob.glob(os.path.join(a.record_dir, "step_*.npz")) if ".tmp" not in p)
        vol = {c: [np.load(recs[0])["V0_" + c]] + [np.load(r)["V1_" + c] for r in recs] for c in comps}
        stamps = None

        def get_layer(c, sp):
            nonlocal stamps
            b, units = names[(c, sp)]
            full = None
            for gdir, sub in groups:
                with h5py.File(os.path.join(gdir, b), "r") as h:
                    x = np.asarray(h["xyz_elements"][0], dtype=np.int64) - 1
                    conc = np.asarray(h["concentrations"][...], dtype=np.float32)
                    if stamps is None:
                        stamps = [t.decode() if isinstance(t, bytes) else str(t) for t in h["timestamps"][...]]
                if full is None:
                    full = np.full((conc.shape[0], ncells), np.nan, dtype=np.float32)
                full[:, x if sub is None else sub[x]] = conc
            full[full <= -9998] = np.nan
            for t in range(min(full.shape[0], len(vol[c]))):
                full[t][vol[c][t] < vthr] = np.nan
            return full, units

    log = a.scale == "log"
    fwd = (lambda v: np.log10(np.maximum(v, 1e-12))) if log else (lambda v: v)

    layers = []
    for c in comps:
        for s in species:
            t0 = time.time()
            full, units = get_layer(c, s)
            full[~np.isfinite(full)] = np.nan            # float16 overflow (inf) -> no value
            is_net = c == "CHANNEL" and net is not None
            sel = net["cells"] if is_net else None
            w = np.ones(sel.size) if is_net else area
            vals = full[:, sel] if is_net else full              # no copy of the full layer
            fin = vals[np.isfinite(vals)]
            pos = fin[fin > 0] if log else fin
            if pos.size > 20_000_000:                    # robust range from a subsample
                pos = pos[:: pos.size // 20_000_000 + 1]
            if pos.size:
                e_lo, e_hi, d_lo, d_hi = np.percentile(pos, [ENC_PCT[0], ENC_PCT[1], DISPLAY_PCT[0], DISPLAY_PCT[1]])
            else:
                e_lo, e_hi, d_lo, d_hi = 1e-3, 1.0, 1e-3, 1.0
            flo, fhi = float(fwd(np.float64(e_lo))), float(fwd(np.float64(e_hi)))
            if fhi - flo < 1e-6:
                flo, fhi = flo - 0.5, fhi + 0.5
            dlo, dhi = float(fwd(np.float64(d_lo))), float(fwd(np.float64(d_hi)))
            if dhi - dlo < 1e-6:
                dlo, dhi = flo, fhi

            # statistics per output time over the coding range
            edges = np.linspace(flo, fhi, NBINS + 1)
            hist, p10, p50, p90 = [], [], [], []
            for t in range(vals.shape[0]):
                x = vals[t]
                ok = np.isfinite(x)
                xs, ws = fwd(x[ok].astype(np.float64)), w[ok]
                if xs.size == 0:
                    hist.append([0.0] * NBINS); p10.append(None); p50.append(None); p90.append(None)
                    continue
                h_, _ = np.histogram(np.clip(xs, flo, fhi), bins=edges, weights=ws)
                hist.append([round(float(v), 5) for v in h_ / ws.sum()])
                o = np.argsort(xs, kind="stable")
                cw = np.cumsum(ws[o]) / ws.sum()
                q = np.interp([0.1, 0.5, 0.9], cw, xs[o])
                p10.append(round(float(q[0]), 5)); p50.append(round(float(q[1]), 5)); p90.append(round(float(q[2]), 5))

            # codes for every output time
            nt = vals.shape[0]
            if is_net:
                codes = np.stack([encode(fwd(vals[t]), flo, fhi) for t in range(nt)])          # (nt, n)
            else:
                codes = np.stack([np.where(dom_soil, encode(fwd(soil_texture(full[t])), flo, fhi), 0).astype(np.uint8)
                                  for t in range(nt)])                                         # (nt, th, tw)
            del full, vals

            # reference: mean of the transformed value over the output times with a value
            ssum = np.zeros(codes.shape[1:], dtype=np.float64)
            scnt = np.zeros(codes.shape[1:], dtype=np.int32)
            for t in range(nt):
                dv = decode(codes[t], flo, fhi)
                ok = np.isfinite(dv)
                ssum[ok] += dv[ok]
                scnt += ok
            with np.errstate(invalid="ignore", divide="ignore"):
                ref = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)
            ref_codes = encode(ref, flo, fhi)
            if not is_net:
                ref_codes = np.where(dom_soil, ref_codes, 0).astype(np.uint8)
            # typical departure from the reference -> default range of the departure view
            dev = []
            for t in range(nt):
                dv = decode(codes[t], flo, fhi) - ref
                dv = np.abs(dv[np.isfinite(dv)])
                if dv.size:
                    dev.append(dv[:: max(1, dv.size // 200_000)])
            p95 = float(np.percentile(np.concatenate(dev), 95)) if dev else 0.5
            if log:
                anom = next((v for v in NICE_ANOM if v >= p95), NICE_ANOM[-1])
            else:
                mag = 10 ** np.floor(np.log10(max(p95, 1e-12)))
                anom = float(next(m * mag for m in (1, 2, 3, 5, 10) if m * mag >= p95))

            # files: three output times per lossless RGB image
            files = []
            tag = f"{c}_{s}"
            for f0 in range(0, nt, PER_FILE):
                fn = f"tex/{tag}_{f0 // PER_FILE:03d}.webp"
                planes = [codes[t] for t in range(f0, min(f0 + PER_FILE, nt))]
                if is_net:
                    planes = [net_image(p, net["rows"]) for p in planes]
                save_rgb(os.path.join(a.out_dir, fn), planes)
                files.append(fn)
            ref_fn = f"tex/{tag}_ref.webp"
            save_rgb(os.path.join(a.out_dir, ref_fn), [net_image(ref_codes, net["rows"]) if is_net else ref_codes])
            nbytes = sum(os.path.getsize(os.path.join(a.out_dir, f)) for f in files + [ref_fn])
            nodata = float((codes == 255).sum() / max(1, (codes > 0).sum()))
            del codes

            ent = {"compartment": c, "compartment_label": NICE.get(c, c), "species": s,
                   "units": units.lower().replace("mg/l", "mg/L"), "kind": "network" if is_net else "raster",
                   "scale": a.scale, "enc": [round(flo, 6), round(fhi, 6)], "display": [round(dlo, 6), round(dhi, 6)],
                   "enc_pct": list(ENC_PCT), "display_pct": list(DISPLAY_PCT),
                   "quant_step": round((fhi - flo) / 253.0, 6), "files": files, "per_file": PER_FILE,
                   "ref": ref_fn, "ref_kind": "geometric mean over all output times" if log else "mean over all output times",
                   "anom_range": anom, "colormap": DEFAULT_CMAP.get(s, "plasma"), "nframes": nt,
                   "hist": hist, "p10": p10, "p50": p50, "p90": p90, "no_value_share": round(nodata, 5),
                   "stats_weight": "network cells, equal weights" if is_net else "pixel area, wet cells",
                   "median": float(np.median(fin)) if fin.size else None,
                   "min": float(e_lo), "max": float(e_hi), "bytes": nbytes}
            if not is_net:
                ent.update(width=soil_map.tw, height=soil_map.th)
            layers.append(ent)
            print(f"{c:11s} {s:8s} {a.scale} code {e_lo:.4g}-{e_hi:.4g}, colour {d_lo:.4g}-{d_hi:.4g} {units}; "
                  f"departure +-{anom:g}; {nt} frames in {len(files)} files, {nbytes / 1e6:.1f} MB, "
                  f"{time.time() - t0:.0f} s", flush=True)

    # ---- base data: hillshade + model-cell fraction per pixel ----
    bm = Mapper(ny, nx, W, E, S, N, a.relief_width)
    frac = bm.mean(inside.astype(np.float32))
    frac = np.where(np.isfinite(frac), frac, 0.0)
    Image.fromarray(np.round(frac * 255).astype(np.uint8), "L").save(
        os.path.join(a.out_dir, "tex", "domain.webp"), "WEBP", lossless=True, quality=100, method=6)
    relief = None
    if elv is not None:
        shade = bm.mean(hillshade(elv, inside, lat_rows))
        g = np.round(np.where(np.isfinite(shade), shade, 0.0) * 255).astype(np.uint8)
        Image.fromarray(np.stack([g, g, g], -1), "RGB").save(os.path.join(a.out_dir, "tex", "relief.webp"),
                                                             "WEBP", quality=86, method=4)
        relief = {"file": "tex/relief.webp", "exaggeration": 6, "azimuth_deg": 315, "altitude_deg": 45,
                  "stretch_pct": [1, 99], "source": "LISFLOOD elevation map (elv)"}

    ri, ci = np.nonzero(inside)
    bbox = [W + ci.min() * (E - W) / nx, W + (ci.max() + 1) * (E - W) / nx,
            N - (ri.max() + 1) * (N - S) / ny, N - ri.min() * (N - S) / ny]
    mi = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
    dates = [f"{s_[:4]}-{mi.get(s_[4:7], 1):02d}-{s_[7:9]}" for s_ in stamps]
    man = {"format": FORMAT, "title": "GloFAS v5 LISFLOOD x openWQ", "timestamps": stamps, "dates": dates,
           "dry_mm": a.dry_mm, "layers": layers, "species": species, "compartments": comps, "ncells": ncells,
           "cell_deg": round((E - W) / nx, 6), "grid": {"W": W, "E": E, "S": S, "N": N, "ny": ny, "nx": nx},
           "data_bbox": [round(float(v), 3) for v in bbox], "hist_bins": NBINS, "per_file": PER_FILE,
           "relief": relief, "domain": {"file": "tex/domain.webp", "width": bm.tw, "height": bm.th},
           "network": net["meta"] if net else None,
           "colormaps": {**{n_: stops(n_, 64) for n_ in SEQ_CMAPS}, **{n_: stops(n_, 65) for n_ in DIV_CMAPS}},
           "diverging": DIV_CMAPS, "sequential": SEQ_CMAPS,
           "codes": {"none": 0, "no_value": 255, "first": 1, "last": 254}}
    json.dump(man, open(os.path.join(a.out_dir, "manifest.json"), "w"), separators=(",", ":"))
    tot = sum(os.path.getsize(p) for p in glob.glob(os.path.join(a.out_dir, "tex", "*.webp")))
    print(f"manifest: {os.path.join(a.out_dir, 'manifest.json')}; tex {tot / 1e6:.1f} MB; "
          f"{time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
