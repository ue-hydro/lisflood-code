#!/usr/bin/env python3
"""Add the terrain heights and the latitude x time summaries to a globe built by
make_globe_textures.py (format globe-scalar-1). Run it after make_globe_textures.py (and
after add_satellite_base.py if used); it only adds files and manifest keys.

  python add_globe_extras.py GLOBE_DIR RECORD_DIR [--maps ROOT_FLAT] [--node-deg 0.25] [--band-deg 1]

Writes
  * tex/terrain.webp: LISFLOOD's elevation map (elv) averaged to the nodes of a
    (360/node) x (180/node) globe mesh, the mesh the viewer displaces. Lossless RGB WebP,
    16-bit metres + 1000 in R (high byte) and G (low byte), (nx+1) x (ny+1) nodes, row 0 at
    90 N, column 0 at 180 W. Outside the model domain (ocean, south of the grid) = 0 m.
    Node value = mean of the four node-centred blocks around it (cells outside the domain
    count as 0 m, so coasts slope down to sea level).
  * manifest["terrain"] = {file, node_deg, nx, ny, offset_m, max_m, source}
  * manifest["hovmoller"] = {lat0, lat1, band_deg, bands, stat, layers: {"COMP|SPECIES": base64}}
    For every layer and output date, the mean of the coded log10 values (or linear values)
    over the pixels / river cells with a value in each latitude band, coded with the
    layer's own codes (1..254, 255 = no value in the band). Bands run from south to north;
    the base64 string holds nframes x bands bytes, date-major. For rasters every pixel of a
    band has the same area, so the band mean is area weighted; rivers: network cells,
    equal weights. Mean of log10 = log10 of the geometric mean.
"""
import argparse
import base64
import glob
import json
import os
import time

import h5py
import numpy as np
from PIL import Image


def read_codes(path):
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globe_dir")
    ap.add_argument("record_dir", help="folder with the recorder's static.npz (model mask)")
    ap.add_argument("--maps", default=None, help="folder with the elv map (default: GloFASv5_data/root_flat)")
    ap.add_argument("--node-deg", type=float, default=0.25)
    ap.add_argument("--band-deg", type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.time()
    mpath = os.path.join(a.globe_dir, "manifest.json")
    man = json.load(open(mpath))
    if man.get("format") != "globe-scalar-1":
        raise SystemExit("needs a globe written by make_globe_textures.py (format globe-scalar-1)")
    g = man["grid"]
    W, E, S, N, ny, nx = g["W"], g["E"], g["S"], g["N"], g["ny"], g["nx"]
    cell = (E - W) / nx

    # ---- terrain at the mesh nodes ----
    st = np.load(os.path.join(a.record_dir, "static.npz"))
    inside = ~np.asarray(st["mask"], dtype=bool)
    assert inside.shape == (ny, nx), (inside.shape, ny, nx)
    maps = a.maps or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(a.record_dir)))), "root_flat")
    hits = sorted(glob.glob(os.path.join(maps, "elv_*.nc")))
    terrain = None
    if hits:
        with h5py.File(hits[0], "r") as h:
            elv = np.asarray(h["Band1"][...], dtype=np.float32)
        assert elv.shape == inside.shape, (elv.shape, inside.shape)
        z = np.where(inside & np.isfinite(elv), elv, 0.0).astype(np.float64)
        k = int(round(a.node_deg / cell))                       # cells per node block
        assert abs(k * cell - a.node_deg) < 1e-9 and ny % k == 0 and nx % k == 0
        blk = z.reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))  # block centres on a node_deg grid
        gny, gnx = int(round(180 / a.node_deg)), int(round(360 / a.node_deg))
        full = np.zeros((gny, gnx))                             # full globe, 0 m outside the grid
        r0 = int(round((90 - N) / a.node_deg))
        c0 = int(round((W + 180) / a.node_deg))
        full[r0:r0 + blk.shape[0], c0:c0 + blk.shape[1]] = blk
        # node (i, j) at lat 90 - i*d, lon -180 + j*d = mean of the four blocks around it
        up = np.vstack([full[:1], full])                         # pad north / south by repeating
        dn = np.vstack([full, full[-1:]])
        rows = 0.5 * (up + dn)                                   # (gny+1, gnx): between block rows
        left = np.roll(rows, 1, axis=1)
        nodes = 0.5 * (left + rows)                              # (gny+1, gnx): between block columns
        nodes = np.hstack([nodes, nodes[:, :1]])                 # column gnx = column 0 (lon 180)
        off = 1000.0
        q = np.clip(np.round(nodes + off), 0, 65535).astype(np.uint16)
        img = np.stack([(q >> 8).astype(np.uint8), (q & 255).astype(np.uint8), np.zeros_like(q, np.uint8)], -1)
        fn = "tex/terrain.webp"
        Image.fromarray(img, "RGB").save(os.path.join(a.globe_dir, fn), "WEBP", lossless=True,
                                         quality=100, method=6, exact=True)
        terrain = {"file": fn, "node_deg": a.node_deg, "nx": gnx, "ny": gny, "offset_m": off,
                   "max_m": float(nodes.max()), "min_m": float(nodes.min()),
                   "source": f"LISFLOOD elevation map ({os.path.basename(hits[0])}), "
                             f"{a.node_deg} deg node means, 0 m outside the model domain"}
        print(f"terrain: {gnx + 1} x {gny + 1} nodes, {nodes.min():.0f}..{nodes.max():.0f} m, "
              f"{os.path.getsize(os.path.join(a.globe_dir, fn)) / 1e6:.2f} MB")
    else:
        print("no elv map found; terrain skipped")

    # ---- latitude x time summaries ----
    lat0, lat1 = max(-90.0, S), min(90.0, N)
    nb = int(round((lat1 - lat0) / a.band_deg))
    net_lat = None
    if man.get("network"):
        M = man["network"]
        A = read_codes(os.path.join(a.globe_dir, M["file"])).astype(np.int64)
        R, n = M["rows"], M["n"]
        row = (A[:R, :, 0] * 256 + A[:R, :, 1]).reshape(-1)[:n]
        net_lat = N - (row + 0.5) * (N - S) / ny
    hov = {}
    for L in man["layers"]:
        nt, P = L["nframes"], L.get("per_file", 3)
        if L["kind"] == "network":
            band = np.floor((net_lat - lat0) / a.band_deg).astype(np.int64)
        else:
            H = L["height"]
            lat_r = 90.0 - (np.arange(H) + 0.5) * 180.0 / H
            band_r = np.floor((lat_r - lat0) / a.band_deg).astype(np.int64)
        out = np.full((nt, nb), 255, dtype=np.uint8)
        for fi, f in enumerate(L["files"]):
            img = read_codes(os.path.join(a.globe_dir, f))
            for ch in range(P):
                t = fi * P + ch
                if t >= nt:
                    break
                if L["kind"] == "network":
                    c = img[:, :, ch].reshape(-1)[:net_lat.size].astype(np.int64)
                    ok = (c >= 1) & (c <= 254) & (band >= 0) & (band < nb)
                    s = np.bincount(band[ok], weights=c[ok], minlength=nb)
                    m = np.bincount(band[ok], minlength=nb)
                else:
                    c = img[:, :, ch].astype(np.int64)
                    okc = (c >= 1) & (c <= 254)
                    s = np.zeros(nb)
                    m = np.zeros(nb)
                    rs, rm = (c * okc).sum(axis=1), okc.sum(axis=1)
                    sel = (band_r >= 0) & (band_r < nb)
                    np.add.at(s, band_r[sel], rs[sel])
                    np.add.at(m, band_r[sel], rm[sel])
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_code = np.where(m > 0, s / np.maximum(m, 1), np.nan)
                # codes are linear in the transformed value, so the mean code is the code of the mean
                out[t] = np.where(np.isfinite(mean_code), np.clip(np.round(mean_code), 1, 254), 255).astype(np.uint8)
        hov[f"{L['compartment']}|{L['species']}"] = base64.b64encode(out.tobytes()).decode("ascii")
    man["hovmoller"] = {"lat0": lat0, "lat1": lat1, "band_deg": a.band_deg, "bands": nb, "order": "date-major, south to north",
                        "stat": "mean of the coded value over the pixels (rasters, area weighted) or river cells "
                                "(equal weights) with a value in each latitude band; log layers: geometric mean",
                        "layers": hov}
    if terrain:
        man["terrain"] = terrain
    json.dump(man, open(mpath, "w"), separators=(",", ":"))
    print(f"hovmoller: {len(hov)} layers x {nb} bands; manifest updated; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
