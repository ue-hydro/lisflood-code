#!/usr/bin/env python3
"""Fast preflight of a GLOBAL settings XML on a small, closed domain (Iceland:
whole island, whole river basins, 11 calibrated stations), so missing files or
broken options show up in minutes instead of after the global cold start's
~3.7 h soil initialisation.

It copies the settings XML and changes ONLY: MaskMap -> the official template
(template_Global_03min.nc) on its full global grid but zeroed outside the box, PathOut / OpenWQRecordDir ->
the preflight folder, StepEnd -> StepStart + N days. Then it runs LISFLOOD for those days.

usage: python preflight_small_domain.py SETTINGS.xml OUT_DIR [--days 3] [--box -25,-13,63,67]
                                        [--path-init PRERUN_DIR]
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

D = Path(os.environ["GLOFAS_DATA"]).resolve()              # data folder (maps/, forcing_global/, runs/)
TEMPLATE = D / "root_flat" / "template_Global_03min.nc"
LISF = Path(os.environ.get("LISFLOOD_REPO", Path(__file__).resolve().parents[4])) / "src" / "lisf1.py"


def cut_mask(box, out):
    """Full global grid (so LISFLOOD's map cutting is the same as in the global run),
    template kept only inside the box, 0 (= fill, outside the domain) elsewhere."""
    W, E, S, N = box
    src = nc.Dataset(TEMPLATE)
    lat, lon = src["lat"][:], src["lon"][:]
    dst = nc.Dataset(out, "w", format="NETCDF4")
    dst.setncatts({a: src.getncattr(a) for a in ("Conventions", "title") if a in src.ncattrs()})
    for name in ("lon", "lat"):
        dst.createDimension(name, src[name].size)
        v = dst.createVariable(name, "f8", (name,), fill_value=np.nan)
        v.setncatts({a: src[name].getncattr(a) for a in src[name].ncattrs() if a != "_FillValue"})
        v[:] = src[name][:]
    t = dst.createVariable("template", "f4", ("lat", "lon"), fill_value=np.float32(0.0), zlib=True)
    t.setncatts({a: src["template"].getncattr(a) for a in src["template"].ncattrs() if a != "_FillValue"})
    keep = ((lat >= S) & (lat <= N))[:, None] & ((lon >= W) & (lon <= E))[None, :]
    a = np.ma.filled(src["template"][:], 0).astype(np.float32)
    a[~keep] = 0
    t[:] = np.ma.masked_equal(a, 0)
    n = int((a != 0).sum())
    dst.close()
    src.close()
    return n, a.shape


def setvar(txt, name, value):
    pat = re.compile(r'(<textvar\s+name="%s"\s+value=\s*")([^"$][^"]*)(")' % name)
    m = pat.search(txt)
    if not m:
        raise SystemExit(f"textvar {name} not found")
    return txt[:m.start()] + m.group(1) + value + m.group(3) + txt[m.end():], m.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("settings")
    ap.add_argument("out_dir")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--box", default="-25,-13,63,67", help="W,E,S,N (default Iceland)")
    ap.add_argument("--path-init", help="other prerun folder (e.g. while the real prerun still runs)")
    a = ap.parse_args()
    out = Path(a.out_dir).resolve()
    (out / "out").mkdir(parents=True, exist_ok=True)
    box = [float(v) for v in a.box.split(",")]
    n, shape = cut_mask(box, out / "mask.nc")
    print(f"mask {shape} with {n} land pixels -> {out / 'mask.nc'}")

    txt = Path(a.settings).read_text()
    txt = txt.replace('name="MaskMap" value="$(PathRoot)/template_Global_03min.nc"',
                      f'name="MaskMap" value="{out / "mask.nc"}"', 1)
    if str(out / "mask.nc") not in txt:
        raise SystemExit("MaskMap not replaced")
    txt, _ = setvar(txt, "PathOut", str(out / "out") + "/")
    if a.path_init:
        txt, _ = setvar(txt, "PathInit", str(Path(a.path_init).resolve()) + "/")
    txt, start = setvar(txt, "StepStart", "PLACEHOLDER")
    d0 = datetime.datetime.strptime(start.split()[0], "%d/%m/%Y")
    txt = txt.replace("PLACEHOLDER", start, 1)
    txt, _ = setvar(txt, "StepEnd", (d0 + datetime.timedelta(days=a.days - 1)).strftime("%d/%m/%Y 00:00"))
    txt = re.sub(r'(<textvar name="OpenWQRecordDir" value=")[^"]*(")', r"\g<1>%s\g<2>" % (out / "openwq_record"), txt)
    xml = out / "preflight.xml"
    xml.write_text(txt)
    print("settings:", xml)
    rc = subprocess.run([sys.executable, str(LISF), str(xml)], cwd=out).returncode
    recs = sorted((out / "openwq_record").glob("step_*.npz")) if (out / "openwq_record").exists() else []
    print(f"LISFLOOD RC={rc}; recorded steps: {len(recs)}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
