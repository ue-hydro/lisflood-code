#!/usr/bin/env python3
"""Run LISFLOOD (official GloFAS v5 settings) on ONE basin with the openWQ coupling recorder.

Copies a global settings XML (e.g. GloFASv5_coldstart_global_year.xml written by
make_week_settings.py) and changes ONLY: MaskMap -> the basin mask from make_basin.py,
PathOut and OpenWQRecordDir -> the basin folder (and, optionally, PathInit, StepStart,
StepEnd). Everything else - parameters, forcing, prerun averages - is the global run's,
so the basin's hydrology is the global run's hydrology on those cells.

usage: python run_basin_lisflood.py SETTINGS.xml BASIN_DIR OUT_DIR [--path-init DIR]
                                    [--start DD/MM/YYYY] [--end DD/MM/YYYY]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

LISF = Path(__file__).resolve().parents[3] / "lisf1.py"


def setvar(txt, name, value):
    pat = re.compile(r'(<textvar\s+name="%s"\s+value=\s*")([^"$][^"]*)(")' % name)
    m = pat.search(txt)
    if not m:
        raise SystemExit(f"textvar {name} not found")
    return txt[:m.start()] + m.group(1) + value + m.group(3) + txt[m.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("settings")
    ap.add_argument("basin_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--path-init")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--record-dtype", choices=("float64", "float32"), default="float64",
                    help="precision of the recorded coupling fields (default float64: the "
                         "replay then sends openWQ exactly the values the in-process "
                         "coupling would; float32 halves the disk but is not identical)")
    a = ap.parse_args()
    basin = Path(a.basin_dir).resolve()
    out = Path(a.out_dir).resolve()
    (out / "out").mkdir(parents=True, exist_ok=True)
    txt = Path(a.settings).read_text()
    new = txt.replace('name="MaskMap" value="$(PathRoot)/template_Global_03min.nc"',
                      f'name="MaskMap" value="{basin / "mask.nc"}"', 1)
    if new == txt:
        raise SystemExit("MaskMap not replaced")
    txt = setvar(new, "PathOut", str(out / "out") + "/")
    if a.path_init:
        txt = setvar(txt, "PathInit", str(Path(a.path_init).resolve()) + "/")
    if a.start:
        txt = setvar(txt, "StepStart", f"{a.start} 00:00")
    if a.end:
        txt = setvar(txt, "StepEnd", f"{a.end} 00:00")
    txt = re.sub(r'(<textvar name="OpenWQRecordDir" value=")[^"]*(")',
                 r"\g<1>%s\g<2>" % (out / "openwq_record"), txt)
    txt, n_dt = re.subn(r'(<textvar name="OpenWQRecordDtype" value=")[^"]*(")',
                        r"\g<1>%s\g<2>" % a.record_dtype, txt)
    if n_dt != 1:
        raise SystemExit("OpenWQRecordDtype not found in the settings")
    xml = out / "basin_settings.xml"
    xml.write_text(txt)
    print("settings:", xml, flush=True)
    rc = subprocess.run([sys.executable, str(LISF), str(xml)], cwd=out).returncode
    recs = sorted((out / "openwq_record").glob("step_*.npz")) if (out / "openwq_record").exists() else []
    print(f"LISFLOOD RC={rc}; recorded steps: {len(recs)}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
