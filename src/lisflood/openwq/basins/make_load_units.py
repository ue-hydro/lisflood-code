#!/usr/bin/env python3
"""Land units for openWQ's land-cover (based_on_lulc) loads on a LISFLOOD basin.

openWQ resolves every source/sink row's cell id with a linear search over the
compartment's cells (OpenWQ_hostModelconfig::find_indices_from_cellid), so one row
per 0.05 deg cell per month (3 M rows on the Mississippi) takes hours to load.
The template's land-cover method works on any land units ("HRUs" with a mapping
key that is an openWQ cell id), so this script groups the basin cells into
BLOCK x BLOCK blocks: each block's polygon is the union of its basin cells, and its
id is the block cell closest to the block centre. The block's land-cover load then
enters the topsoil of that one cell (same mass; placement shifted by < BLOCK cells).

writes BASIN/load_units_<deg>.shp with field "cell" (= representative cell ix) and
"n_cells", plus load_units_<deg>.csv (cell ix -> unit cell ix).

usage: python make_load_units.py BASIN_DIR [--block 5]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("basin_dir")
    ap.add_argument("--block", type=int, default=5, help="cells per block side (5 = 0.25 deg)")
    ap.add_argument("--zones", default=None,
                    help="cascade zone table (id,wq_zone per cell, e.g. wq_zones/zone_reaches.csv): "
                         "blocks are also split along zone boundaries, so every zone has load units")
    ap.add_argument("--tag", default=None, help="file name tag (default: block size in degrees)")
    a = ap.parse_args()
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union
    b = Path(a.basin_dir)
    cells = pd.read_csv(b / "cells.csv")
    k = a.block
    cells["brow"], cells["bcol"] = cells.row // k, cells.col // k
    keys = ["brow", "bcol"]
    if a.zones:
        z = pd.read_csv(a.zones)
        zone_of = dict(zip(z.iloc[:, 0].astype(int), z.iloc[:, 1].astype(int)))
        cells["zone"] = cells.ix.map(zone_of).fillna(0).astype(int)
        keys.append("zone")
    rows = []
    member = np.empty(len(cells), dtype=np.int64)
    for key, g in cells.groupby(keys):
        br, bc = key[0], key[1]
        cr, cc = g.row.mean(), g.col.mean()        # centre of the unit's own cells
        rep = int(g.ix.to_numpy()[np.argmin((g.row - cr) ** 2 + (g.col - cc) ** 2)])
        member[g.ix.to_numpy()] = rep
        geom = unary_union([box(lo - 0.025, la - 0.025, lo + 0.025, la + 0.025)
                            for la, lo in zip(g.lat, g.lon)])
        rows.append({"cell": rep, "n_cells": len(g), "geometry": geom})
    deg = a.tag or f"{k * 0.05:.2f}".replace(".", "p")
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(b / f"load_units_{deg}.shp")
    pd.DataFrame({"ix": cells.ix, "unit_cell": member}).to_csv(b / f"load_units_{deg}.csv", index=False)
    print(f"{len(gdf)} load units of up to {k}x{k} cells -> {b / f'load_units_{deg}.shp'}")


if __name__ == "__main__":
    main()
