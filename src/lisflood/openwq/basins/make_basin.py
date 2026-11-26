#!/usr/bin/env python3
"""Pour point -> river basin on the GloFAS v5 / LISFLOOD 0.05 deg network.

The only per-river decision is the pour point. This script snaps it to the
LISFLOOD channel network (upstream-area match), delineates the basin from the
official LDD, and writes everything the later steps need:

  OUT/mask.nc        LISFLOOD MaskMap: the official template on its full global grid,
                     0 (fill) outside the basin (same convention as the Iceland preflight)
  OUT/cells.csv      one row per basin cell in LISFLOOD's own order (row-major over the
                     mask = openWQ cell index ix): ix, row, col, lat, lon, upArea_km2,
                     pixel_area_km2, down_ix (-1 at the pour point)
  OUT/cells.shp      0.05 deg cell polygons with field "cell" = ix (the openWQ cell id,
                     used by the template's land-cover loads as mapping_key)
  OUT/rivers.shp     river network for the templates (river_network_shapefile, key "cell"):
                     channel cells with upstream area >= --river-min-km2
  OUT/pour_point.json

usage:
  python make_basin.py OUT_DIR --pour-point LAT,LON [--area-km2 A] [--search-cells 3]
         [--name NAME] [--root ROOT_FLAT_DIR]
"""
import argparse
import json
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np

ROOT = Path(os.environ.get("GLOFAS_DATA", "/Users/diogocosta/Documents/openwq_code/GloFASv5_data")) / "root_flat"

# pcraster LDD codes -> (drow, dcol); rows increase southwards (lat descending)
LDD_STEP = {1: (1, -1), 2: (1, 0), 3: (1, 1), 4: (0, -1), 6: (0, 1),
            7: (-1, -1), 8: (-1, 0), 9: (-1, 1)}


def read_band(path, var=None):
    d = nc.Dataset(path)
    v = d[var] if var else next(d[k] for k in d.variables if d[k].ndim == 2)
    a = np.ma.filled(v[:], np.nan if v.dtype.kind == "f" else 0)
    lat, lon = d["lat"][:].astype(float), d["lon"][:].astype(float)
    d.close()
    return a, lat, lon


def downstream_flat(ldd):
    ny, nx = ldd.shape
    down = np.full(ldd.size, -1, dtype=np.int64)
    rr, cc = np.nonzero((ldd >= 1) & (ldd <= 9) & (ldd != 5))
    codes = ldd[rr, cc]
    dr = np.zeros_like(rr)
    dc = np.zeros_like(cc)
    for k, (a, b) in LDD_STEP.items():
        m = codes == k
        dr[m], dc[m] = a, b
    r2, c2 = rr + dr, (cc + dc) % nx
    ok = (r2 >= 0) & (r2 < ny)
    down[(rr * nx + cc)[ok]] = r2[ok] * nx + c2[ok]
    return down


def upstream_of(down, p):
    """All cells draining through flat index p (p included)."""
    src = np.nonzero(down >= 0)[0]
    dst = down[src]
    order = np.argsort(dst, kind="stable")
    src, dst = src[order], dst[order]
    seen = [np.array([p])]
    front = np.array([p])
    while front.size:
        lo = np.searchsorted(dst, front, "left")
        hi = np.searchsorted(dst, front, "right")
        n = hi - lo
        if n.sum() == 0:
            break
        idx = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        front = src[idx]
        seen.append(front)
    return np.unique(np.concatenate(seen))


def write_mask(template_path, keep_flat, out):
    src = nc.Dataset(template_path)
    dst = nc.Dataset(out, "w", format="NETCDF4")
    dst.setncatts({a: src.getncattr(a) for a in ("Conventions", "title") if a in src.ncattrs()})
    for name in ("lon", "lat"):
        dst.createDimension(name, src[name].size)
        v = dst.createVariable(name, "f8", (name,), fill_value=np.nan)
        v.setncatts({a: src[name].getncattr(a) for a in src[name].ncattrs() if a != "_FillValue"})
        v[:] = src[name][:]
    t = dst.createVariable("template", "f4", ("lat", "lon"), fill_value=np.float32(0.0), zlib=True)
    t.setncatts({a: src["template"].getncattr(a) for a in src["template"].ncattrs() if a != "_FillValue"})
    a = np.ma.filled(src["template"][:], 0).astype(np.float32)
    keep = np.zeros(a.size, bool)
    keep[keep_flat] = True
    a[~keep.reshape(a.shape)] = 0
    t[:] = np.ma.masked_equal(a, 0)
    n = int((a != 0).sum())
    dst.close()
    src.close()
    return n


def write_polygons(cells, out_shp, half=0.025):
    import geopandas as gpd
    from shapely.geometry import box
    geoms = [box(lo - half, la - half, lo + half, la + half)
             for la, lo in zip(cells["lat"], cells["lon"])]
    gdf = gpd.GeoDataFrame({"cell": cells["ix"].astype(np.int64)}, geometry=geoms, crs="EPSG:4326")
    gdf.to_file(out_shp)


def write_rivers(cells, out_shp, min_km2):
    """River network of the basin as the openWQ templates expect it
    (river_network_shapefile): one line per channel cell with upstream area
    >= min_km2, cell centre -> downstream cell centre, field "cell" = openWQ cell
    id (= LISFLOOD cell index, the id openWQ writes to its HDF5 output)."""
    import geopandas as gpd
    from shapely.geometry import LineString
    c = cells.set_index("ix")
    rows = []
    for ix in c.index[c.upArea_km2 >= min_km2]:
        r = c.loc[ix]
        d = int(r.down_ix)
        end = (c.loc[d].lon, c.loc[d].lat) if d >= 0 else (r.lon, r.lat - 0.02)
        rows.append({"cell": int(ix), "upArea_km2": float(r.upArea_km2),
                     "geometry": LineString([(r.lon, r.lat), end])})
    gpd.GeoDataFrame(rows, crs="EPSG:4326").to_file(out_shp)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--river-min-km2", type=float, default=1000.0,
                    help="channel cells drawn in rivers.shp (the templates' river_network_shapefile)")
    ap.add_argument("--pour-point", required=True, help="LAT,LON")
    ap.add_argument("--area-km2", type=float, default=None,
                    help="reported drainage area; the snapped cell is the one closest to it")
    ap.add_argument("--search-cells", type=int, default=3)
    ap.add_argument("--name", default="basin")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--no-polygons", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lat0, lon0 = [float(v) for v in a.pour_point.split(",")]

    ldd, lat, lon = read_band(root / "ldd_repaired.nc")
    ups, _, _ = read_band(root / "upArea_repaired_correctedmetadata_3000.nc")
    tmpl, _, _ = read_band(root / "template_Global_03min.nc", "template")
    pix, _, _ = read_band(root / "pixarea_Global_03min.nc")
    ny, nx = ldd.shape
    ups_km2 = ups / 1e6

    # snap: cells within the search window, closest upstream area (or largest)
    r0, c0 = int(np.argmin(abs(lat - lat0))), int(np.argmin(abs(lon - lon0)))
    k = a.search_cells
    win = [(r, c % nx) for r in range(max(0, r0 - k), min(ny, r0 + k + 1))
           for c in range(c0 - k, c0 + k + 1)]
    cand = [(r, c, float(ups_km2[r, c])) for r, c in win if np.isfinite(ups_km2[r, c]) and tmpl[r, c] != 0]
    if not cand:
        raise SystemExit("no LISFLOOD cell near the pour point")
    if a.area_km2:
        r, c, A = min(cand, key=lambda t: abs(t[2] - a.area_km2))
    else:
        r, c, A = max(cand, key=lambda t: t[2])
    p = r * nx + c
    print(f"pour point snapped to row {r} col {c} (lat {lat[r]:.3f} lon {lon[c]:.3f}), "
          f"upstream area {A:,.0f} km2" + (f" (reported {a.area_km2:,.0f})" if a.area_km2 else ""))

    down = downstream_flat(ldd)
    basin = upstream_of(down, p)
    basin = basin[np.ma.filled(tmpl.ravel()[basin], 0) != 0]          # model cells only
    print(f"basin: {basin.size:,} cells")

    n = write_mask(root / "template_Global_03min.nc", basin, out / "mask.nc")
    assert n == basin.size, (n, basin.size)

    # LISFLOOD compresses the mask row-major -> openWQ cell index ix = rank in sorted flat order
    basin = np.sort(basin)
    ix_of = {f: i for i, f in enumerate(basin.tolist())}
    rr, cc = basin // nx, basin % nx
    dflat = down[basin]
    down_ix = np.array([ix_of.get(int(d), -1) for d in dflat], dtype=np.int64)
    down_ix[basin == p] = -1
    import pandas as pd
    cells = pd.DataFrame(dict(ix=np.arange(basin.size), row=rr, col=cc,
                              lat=lat[rr].round(5), lon=lon[cc].round(5),
                              upArea_km2=ups_km2[rr, cc].round(3),
                              pixel_area_km2=(pix[rr, cc] / 1e6).round(5),
                              down_ix=down_ix))
    cells.to_csv(out / "cells.csv", index=False)
    print(f"outlet cells (down_ix=-1): {(down_ix < 0).sum()}  (1 expected)")
    if not a.no_polygons:
        write_polygons(cells, out / "cells.shp")
    n_riv = write_rivers(cells, out / "rivers.shp", a.river_min_km2)
    print(f"rivers.shp: {n_riv} channel cells with upstream area >= {a.river_min_km2:g} km2")
    info = dict(name=a.name, pour_point_input=[lat0, lon0], snapped_row=r, snapped_col=c,
                snapped_lat=float(lat[r]), snapped_lon=float(lon[c]), upArea_km2=A,
                reported_area_km2=a.area_km2, n_cells=int(basin.size), outlet_ix=int(ix_of[p]),
                bbox=[float(cells.lon.min()) - 0.025, float(cells.lat.min()) - 0.025,
                      float(cells.lon.max()) + 0.025, float(cells.lat.max()) + 0.025])
    (out / "pour_point.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
