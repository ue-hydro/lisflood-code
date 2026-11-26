#!/usr/bin/env python3
"""Preview the cascade zones of a basin for several minimum-sample thresholds.

Stations = cells of OUT/obs/observations_calibration.csv (reach_id = cell index);
network = BASIN/cells.csv (down_ix). For each threshold prints the number of
zones, cascade levels, and the share of the basin area that is gauged.

usage: python preview_zones.py BASIN_DIR OBS_CSV [--min-obs 4 6 10 12] [--period START END]
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("basin_dir")
    ap.add_argument("obs_csv")
    ap.add_argument("--min-obs", nargs="+", type=int, default=[4, 6, 10, 12])
    ap.add_argument("--period", nargs=2, default=None)
    a = ap.parse_args()
    cells = pd.read_csv(f"{a.basin_dir}/cells.csv")
    down = cells.down_ix.to_numpy()
    area = cells.pixel_area_km2.to_numpy()
    n = down.size
    obs = pd.read_csv(a.obs_csv, parse_dates=["datetime"])
    if a.period:
        obs = obs[(obs.datetime >= a.period[0]) & (obs.datetime <= a.period[1])]
    cnt = obs.groupby("reach_id").size()
    # topological order (upstream first) by repeated in-degree peeling
    indeg = np.bincount(down[down >= 0], minlength=n)
    order, front = [], np.nonzero(indeg == 0)[0]
    indeg = indeg.copy()
    while front.size:
        order.append(front)
        d = down[front]
        d = d[d >= 0]
        np.subtract.at(indeg, d, 1)
        front = np.unique(d[indeg[d] == 0])
    order = np.concatenate(order)
    for k in a.min_obs:
        st = set(int(r) for r in cnt[cnt >= k].index)
        # level of a station = 1 + max level of stations directly upstream
        lvl_up = np.zeros(n, dtype=int)     # max station level seen upstream (incl. cell)
        lvl = {}
        for c in order:
            if c in st:
                lvl[c] = lvl_up[c] + 1
                carry = lvl[c]
            else:
                carry = lvl_up[c]
            d = down[c]
            if d >= 0 and carry > lvl_up[d]:
                lvl_up[d] = carry
        # gauged area = cells upstream of any station
        g = np.zeros(n, bool)
        for c in order[::-1]:
            d = down[c]
            g[c] = (c in st) or (d >= 0 and g[d] and False)
        up_any = np.zeros(n, bool)
        mark = np.zeros(n, bool)
        mark[list(st)] = True
        # cell is gauged if some downstream cell (incl. itself) is a station
        for c in order[::-1]:
            d = down[c]
            up_any[c] = mark[c] or (d >= 0 and up_any[d])
        levels = pd.Series(list(lvl.values())).value_counts().sort_index()
        print(f"min_obs {k:>3}: {len(st):>4} zones, {int(levels.index.max()) if len(levels) else 0:>2} levels "
              f"{dict(levels)}, gauged {100 * area[up_any].sum() / area.sum():.1f}% of the basin")


if __name__ == "__main__":
    main()
