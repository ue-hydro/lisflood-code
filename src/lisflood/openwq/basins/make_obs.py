#!/usr/bin/env python3
"""Observations for one LISFLOOD basin -> openWQ calibration CSV (user_csv).

1. Extracts the target species with openWQ's own multi-source module
   (supporting_scripts/1_Model_Config/config_support_lib/Gen_Observation_Sources.py):
   GRQA (local copy), GEMStat (open subset), CAMELS-Chem, and the WQP live API
   queried tile by tile over the basin (one bbox for a continental basin times out).
   WQP's most common USGS nitrate record, "Inorganic nitrogen (nitrate and nitrite)",
   is added to the module's WQP name map at run time (the file is not edited).
2. Harmonizes units to mg N/L (nitrate reported "as NO3" x 14.007/62.004).
3. Snaps every station to the LISFLOOD channel cell whose upstream area best matches
   the station's reported drainage area (GRQA upstream_basin_area or WQP
   DrainageAreaMeasure), within --radius-cells; stations without an area or with no
   cell within --area-tol are dropped (listed in stations_rejected.csv).
4. Writes OUT/observations_calibration.csv (datetime, reach_id, species, value, units,
   source; reach_id = openWQ cell index ix) and OUT/stations.csv.

usage: python make_obs.py BASIN_DIR OUT_DIR --years 2015 2015 [--species NO3-N]
"""
import argparse
import io
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSL = HERE.parent / "openwq" / "supporting_scripts" / "1_Model_Config" / "config_support_lib"
sys.path.insert(0, str(CSL))
import Gen_Observation_Sources as GOS  # noqa: E402

GRQA_LOCAL = os.environ.get("GRQA_LOCAL", "/Users/diogocosta/Documents/GRQA_data_v1.3")
GRQA_CODES = {"NO3-N": "NO3N", "NH4-N": "NH4N", "TN": "TN"}
N_PER_NO3 = 14.0067 / 62.0049
SQMI_KM2 = 2.589988


def basin_outline(basin_dir, out_dir):
    import geopandas as gpd
    p = out_dir / "basin_outline.gpkg"
    if p.exists():
        return gpd.read_file(p)
    cells = gpd.read_file(basin_dir / "cells.shp")
    # dissolve in row bands first (faster than one unary_union over 1e5 boxes)
    cells["band"] = (cells.geometry.centroid.y // 2).astype(int)
    parts = cells.dissolve(by="band")
    # search area only (stations are matched to cells exactly later): a simplified,
    # slightly buffered outline keeps point-in-polygon tests fast
    geom = parts.unary_union.buffer(0.05).simplify(0.05)
    outline = gpd.GeoDataFrame(geometry=[geom], crs=cells.crs)
    outline.to_file(p)
    return outline


def tiles(bbox, step=4.0):
    w, s, e, n = bbox
    for x in np.arange(w, e, step):
        for y in np.arange(s, n, step):
            yield (float(x), float(y), float(min(x + step, e)), float(min(y + step, n)))


def wqp_station_meta(bbox, chars, years):
    params = [("bBox", "%.4f,%.4f,%.4f,%.4f" % bbox), ("mimeType", "csv"), ("zip", "no"),
              ("providers", "NWIS"), ("providers", "STORET"),
              ("startDateLo", f"01-01-{years[0]}"), ("startDateHi", f"12-31-{years[1]}")]
    params += [("characteristicName", c) for c in chars]
    url = "https://www.waterqualitydata.us/data/Station/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310
        df = pd.read_csv(io.StringIO(r.read().decode("utf-8", "replace")), low_memory=False)
    keep = ["MonitoringLocationIdentifier", "MonitoringLocationTypeName",
            "DrainageAreaMeasure/MeasureValue", "DrainageAreaMeasure/MeasureUnitCode",
            "LatitudeMeasure", "LongitudeMeasure", "MonitoringLocationName"]
    return df[[c for c in keep if c in df.columns]]


def to_mgN(df):
    """value in mg N/L from the harmonized (value, units, source) rows."""
    u = df["units"].astype(str).str.lower().str.replace(" ", "")
    v = pd.to_numeric(df["value"], errors="coerce")
    f = pd.Series(np.nan, index=df.index)
    asno3 = u.str.contains("no3")
    mg = u.str.startswith("mg/l") | u.str.startswith("ppm")
    ug = u.str.startswith("ug/l") | u.str.startswith("µg/l")
    f[mg] = 1.0                      # "mg/l", "mg/l as N", "mg/L"
    f[ug] = 1e-3
    f[asno3] = f[asno3] * N_PER_NO3  # reported as nitrate ion
    # GRQA/GEMStat/CAMELS rows are already harmonized to mg/L of N
    pre = df["source"].astype(str).str.upper().str.contains("GRQA|GEMSTAT|CAMELS")
    f[pre & f.isna()] = 1.0
    return v * f, ~f.isna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("basin_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--years", nargs=2, type=int, default=[2015, 2015])
    ap.add_argument("--species", default="NO3-N")
    ap.add_argument("--sources", default="grqa,gemstat,camels_chem,wqp")
    ap.add_argument("--radius-cells", type=int, default=2)
    ap.add_argument("--area-tol", type=float, default=0.35, help="max |ln(A_cell/A_station)|")
    ap.add_argument("--min-area-km2", type=float, default=500.0)
    ap.add_argument("--tile-deg", type=float, default=4.0)
    a = ap.parse_args()
    basin_dir, out = Path(a.basin_dir).resolve(), Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((basin_dir / "pour_point.json").read_text())
    bbox = info["bbox"]
    sp = a.species
    years = tuple(a.years)
    outline = basin_outline(basin_dir, out)
    print(f"basin outline ready; bbox {bbox}", flush=True)

    # WQP: also the USGS nitrate+nitrite record (NO2 is negligible in rivers)
    GOS.WQP_PARAM_MAP["Inorganic nitrogen (nitrate and nitrite)"] = "NO3-N"
    GOS._WQP_CHAR_LOWER["inorganic nitrogen (nitrate and nitrite)"] = "NO3-N"
    # "Nitrate as N" is not a WQP CharacteristicName: WQP answers HTTP 400 to any
    # query that names it (checked 2026-11-23), so drop it from the request list
    GOS.WQP_PARAM_MAP.pop("Nitrate as N", None)
    chars = sorted(c for c, m in GOS.WQP_PARAM_MAP.items() if m == sp)

    ctx = {"search_area_gdf": outline, "bbox": bbox, "species": [sp],
           "grqa_species_mapping": {GRQA_CODES[sp]: sp}, "years": years,
           "cache_dir": str(out / "obs_cache"), "output_dir": str(out),
           "manual_paths": {}, "buffer_m": 0, "grqa_local_data_path": GRQA_LOCAL}
    frames, attempted = [], []
    for key in [s.strip() for s in a.sources.split(",") if s.strip()]:
        if key == "wqp":
            parts, meta = [], []
            for tb in tiles(bbox, a.tile_deg):
                c2 = dict(ctx, bbox=list(tb))
                df, status = GOS.extract_one("wqp", c2)
                print(f"  WQP tile {tb}: {len(df)} rows ({status})", flush=True)
                if len(df):
                    parts.append(df)
                    try:
                        meta.append(wqp_station_meta(tb, chars, years))
                    except Exception as e:  # noqa: BLE001
                        print(f"    station metadata failed: {e}", flush=True)
            df = pd.concat(parts, ignore_index=True) if parts else GOS._empty_harmonized()
            if meta:
                pd.concat(meta, ignore_index=True).drop_duplicates(
                    "MonitoringLocationIdentifier").to_csv(out / "wqp_station_meta.csv", index=False)
            status = "ok" if len(df) else "no_data"
        else:
            df, status = GOS.extract_one(key, ctx)
        print(f"{key}: {len(df)} rows ({status})", flush=True)
        attempted.append((key, len(df), status))
        frames.append(df)
    merged, n_drop = GOS.merge_and_dedup(frames, True)
    merged = GOS._refine_to_polygon(merged, outline)
    merged = merged[merged["parameter"] == sp].copy()
    merged = merged[(merged["year"] >= years[0]) & (merged["year"] <= years[1])]
    merged["value_mgN"], _ = to_mgN(merged)
    bad_units = merged[merged["value_mgN"].isna()]
    if len(bad_units):
        print("dropped rows with unknown units:", bad_units["units"].value_counts().head(10).to_dict())
    merged = merged.dropna(subset=["value_mgN"])
    merged = merged[(merged["value_mgN"] >= 0) & (merged["value_mgN"] < 100)]
    merged.to_csv(out / "observations_harmonized.csv", index=False)
    print(f"harmonized: {len(merged)} rows at {merged.station_id.nunique()} stations "
          f"(removed {n_drop} duplicates)", flush=True)

    # ---- station drainage areas -------------------------------------------------
    area = {}
    try:
        g = pd.read_csv(Path(GRQA_LOCAL) / f"{GRQA_CODES[sp]}_GRQA.csv", sep=";",
                        usecols=["site_id", "upstream_basin_area", "upstream_basin_area_unit"])
        g = g.dropna(subset=["upstream_basin_area"]).drop_duplicates("site_id")
        for sid, A, u in g.itertuples(index=False):
            u = str(u).lower()
            area[str(sid)] = float(A) * (SQMI_KM2 if "mi" in u else (1e-6 if u in ("m2", "m^2") else 1.0))
    except Exception as e:  # noqa: BLE001
        print("GRQA areas unavailable:", e)
    meta_p = out / "wqp_station_meta.csv"
    stype = {}
    if meta_p.exists():
        m = pd.read_csv(meta_p)
        for r in m.itertuples(index=False):
            sid = str(r[0])
            stype[sid] = str(getattr(r, "MonitoringLocationTypeName", ""))
            A, u = r[2], str(r[3]).lower() if len(r) > 3 else ""
            if pd.notna(A):
                area.setdefault(sid, float(A) * (SQMI_KM2 if "mi" in u else (0.01 if "ha" in u else 1.0)))
    # GRQA keeps the WQP ids without the "USGS-" prefix in some versions: try both forms
    def station_area(sid):
        for k in (sid, sid.replace("USGS-", ""), "USGS-" + sid, "USGS-" + sid.zfill(8)):
            if k in area:
                return area[k]
        return np.nan

    # ---- snap to cells ----------------------------------------------------------
    cells = pd.read_csv(basin_dir / "cells.csv")
    key = {(int(r), int(c)): i for i, r, c in zip(cells.ix, cells.row, cells.col)}
    up = cells.upArea_km2.to_numpy()
    st = (merged.groupby("station_id")
          .agg(lat=("lat", "first"), lon=("lon", "first"), source=("source", "first"),
               n_obs=("value_mgN", "size"))
          .reset_index())
    st["area_station_km2"] = [station_area(s) for s in st.station_id]
    st["site_type"] = [stype.get(s, "") for s in st.station_id]
    rows, rej = [], []
    R = a.radius_cells
    for r in st.itertuples(index=False):
        if not np.isfinite(r.area_station_km2) or r.area_station_km2 < a.min_area_km2:
            rej.append(dict(r._asdict(), reason="no_area" if not np.isfinite(r.area_station_km2) else "small"))
            continue
        row0 = int(round((89.975 - r.lat) / 0.05))
        col0 = int(round((r.lon + 179.975) / 0.05))
        best = None
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                ix = key.get((row0 + dr, col0 + dc))
                if ix is None:
                    continue
                d = abs(np.log(up[ix] / r.area_station_km2))
                if best is None or d < best[0]:
                    best = (d, ix)
        if best is None or best[0] > a.area_tol:
            rej.append(dict(r._asdict(), reason="no_cell_match" if best is None else f"area_mismatch_{best[0]:.2f}"))
            continue
        rows.append(dict(r._asdict(), ix=best[1], area_cell_km2=float(up[best[1]]),
                         log_area_ratio=float(best[0])))
    snapped = pd.DataFrame(rows)
    pd.DataFrame(rej).to_csv(out / "stations_rejected.csv", index=False)
    snapped.to_csv(out / "stations.csv", index=False)
    print(f"stations: {len(st)} with data; snapped {len(snapped)}; rejected {len(rej)} "
          f"({pd.Series([x['reason'].split('_')[0] for x in rej]).value_counts().to_dict() if rej else {}})")

    obs = merged.merge(snapped[["station_id", "ix"]], on="station_id")
    dt = pd.to_datetime(dict(year=obs.year, month=obs.month, day=obs.day), errors="coerce")
    cal = pd.DataFrame({"datetime": dt.dt.strftime("%Y-%m-%d 00:00:00"), "reach_id": obs.ix.astype(int),
                        "species": sp, "value": obs.value_mgN.round(5), "units": "mg/l",
                        "source": obs.source.astype(str) + ":" + obs.station_id.astype(str)})
    cal = cal.dropna(subset=["datetime"]).sort_values(["reach_id", "datetime"])
    cal.to_csv(out / "observations_calibration.csv", index=False)
    per = cal.groupby("reach_id").size()
    print(f"calibration CSV: {len(cal)} rows at {per.size} cells; cells with >=6 obs: {(per >= 6).sum()}, "
          f">=12: {(per >= 12).sum()}")
    json.dump({"attempted": attempted, "n_rows": int(len(cal)), "n_cells": int(per.size)},
              open(out / "obs_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
