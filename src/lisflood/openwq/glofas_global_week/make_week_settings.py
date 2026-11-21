#!/usr/bin/env python3
"""Build LISFLOOD settings for a short GLOBAL GloFAS v5 run from the OFFICIAL
GloFAS v5 settings XMLs (GloFASv5_prerun.xml / GloFASv5_coldstart.xml, JRC map
pack v2.1.1_OS-LISFLOOD-v5.x), changing ONLY:
  * the Leonardo HPC paths -> one local flat root (symlinks to the public map
    pack + the public GloFAS v5 reanalysis forcing) and local output folders;
  * StepStart / StepEnd (the simulated window);
  * prerun NumDaysSpinUp 1095 -> 0 (unavoidable: a 1-week prerun cannot spin up
    for 3 years; it only provides the long-term averages the cold start reads);
  * coldstart only: option OpenWQRecord=1 + binding OpenWQRecordDir (the
    LISFLOOD->openWQ coupling recorder of the ue-hydro fork; read-only hook,
    inert for the hydrology).
Every change is listed in <out>/CHANGES_vs_official.txt, and every file the
settings reference is checked to exist (else exit 1).

usage: python make_week_settings.py [START DD/MM/YYYY] [END DD/MM/YYYY] [YEAR]
env:   RUN_TAG (default global_week) -> runs/<tag>/, settings_week/GloFASv5_<stage>_<tag>.xml
       DAILY_MAPS=0 -> switch off the daily report maps (rep*Maps; reporting only, the
                       hydrology is unchanged) for long runs (~1 GB/day globally)
       RECORD_DTYPE (default float64) -> binding OpenWQRecordDtype of the recorder
"""
import os
import re
import sys
from pathlib import Path

D = Path(os.environ["GLOFAS_DATA"]).resolve()              # data folder (maps/, forcing_global/, runs/)                       # GloFASv5_data/
START = sys.argv[1] if len(sys.argv) > 1 else "01/06/2015"
END = sys.argv[2] if len(sys.argv) > 2 else "07/06/2015"
YEAR = sys.argv[3] if len(sys.argv) > 3 else "2015"
ROOT = D / "root_flat"
TAG = os.environ.get("RUN_TAG", "global_week")
DAILY_MAPS = os.environ.get("DAILY_MAPS", "1") != "0"
RECORD_DTYPE = os.environ.get("RECORD_DTYPE", "float64")
RUN = D / "runs" / TAG
OUT = D / "settings_week"
OFFICIAL = D / "settings_official"
LEONARDO = {
    "/leonardo_work/EUHPC_D27_008/GloFASv5_meteo_static_maps_reanalysis": str(ROOT),
    "/leonardo_work/EUHPC_D27_008/GloFASv5_longtermrun/param_folder": str(ROOT),
    "/leonardo_work/EUHPC_D27_008/cal_workflow_2025/GloFASv5_tables": str(ROOT),
}
# Files the official settings name that are NOT in the public map pack -> the public
# file that replaces them. Gauges only selects the cells whose discharge / catchment
# averages are written to time series (reporting); the hydrology is unchanged.
NOT_IN_PUBLIC_PACK = {
    "outlets_calibratedstations_regionalizedcatchments.nc": "outlets_calibratedstations.nc",
}
# References that are reported missing but belong to options that are OFF in the
# official GloFAS v5 settings (inflow, polders, dynamic wave, Sites, WFraction maps)
KNOWN_UNUSED = {"PathInflow", "WFractionMaps", "WFracOfDay", "Sites", "PolderSites",
                "TabPolderTotalCapacity", "TabPolderArea", "TabPolderOFWidth",
                "TabPolderBottomLevel", "TabPolderOpeningTime", "TabPolderReleaseTime",
                "ChannelsDynamic", "ChanBottomLevel", "ChanCrossSections", "TabCrossSections",
                "InflowPoints", "QInTS",                       # option inflow = 0
                "Population", "PopulationMaps",                # indicator / TransientLandUseChange off
                "FracMaxWater"}                                # varfractionwater off


def build_flat_root():
    """Symlink every map-pack file + the year's forcing into one folder, as the
    official settings expect (PathRoot = maps AND meteo)."""
    ROOT.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in (D / "maps").rglob("*") if p.is_file())
    srcs += sorted((D / "forcing_global" / YEAR).glob("glofas_*.nc"))
    seen = {}
    for p in srcs:
        if p.name in ("copyright.txt",) or p.name.startswith("README"):
            continue
        if p.name in seen and seen[p.name].stat().st_size != p.stat().st_size:
            raise SystemExit(f"name clash with different content: {p} vs {seen[p.name]}")
        seen[p.name] = p
        link = ROOT / p.name
        if link.is_symlink() and os.readlink(link) == str(p):
            continue                                  # already correct (never break a running model)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(p)
    return len(seen)


def patch(xml_in, xml_out, path_out, path_init, spinup_zero, record_dir=None):
    txt = Path(xml_in).read_text()
    changes = []
    for old, new in LEONARDO.items():
        n = txt.count(old)
        if n:
            txt = txt.replace(old, new)
            changes.append(f"{n:3d}x path  {old}  ->  {new}")
    for old, new in NOT_IN_PUBLIC_PACK.items():
        n = txt.count("/" + old)
        if n:
            txt = txt.replace("/" + old, "/" + new)
            changes.append(f"{n:3d}x file  {old}  ->  {new}  (not in the public map pack; "
                           f"reporting points only)")

    def setvar(name, value):
        nonlocal txt
        pat = re.compile(r'(<textvar\s+name="%s"\s+value=\s*")([^"$][^"]*)(")' % name)
        m = pat.search(txt)                                  # first concrete value (lfuser)
        if not m:
            raise SystemExit(f"{xml_in}: textvar {name} not found")
        changes.append(f"  1x {name:15s} {m.group(2)!r}  ->  {value!r}")
        txt = txt[:m.start()] + m.group(1) + value + m.group(3) + txt[m.end():]

    setvar("PathOut", str(path_out) + "/")
    setvar("PathInit", str(path_init) + "/")
    setvar("StepStart", f"{START} 00:00")
    setvar("StepEnd", f"{END} 00:00")
    if spinup_zero:
        setvar("NumDaysSpinUp", "0")
    if record_dir is not None:
        txt = txt.replace("<lfoptions>", '<lfoptions>\n    <setoption choice="1" name="OpenWQRecord"/>', 1)
        txt = re.sub(r"(<lfbinding>)", r'\1\n    <textvar name="OpenWQRecordDir" value="%s"/>'
                     % record_dir, txt, count=1)
        changes.append(f"  +  option OpenWQRecord=1, binding OpenWQRecordDir={record_dir}")
        if RECORD_DTYPE != "float64":
            txt = re.sub(r"(<lfbinding>)", r'\1\n    <textvar name="OpenWQRecordDtype" value="%s"/>'
                         % RECORD_DTYPE, txt, count=1)
            changes.append(f"  +  binding OpenWQRecordDtype={RECORD_DTYPE}")
    if not DAILY_MAPS:
        n_off = len(re.findall(r'<setoption\s+choice="1"\s+name="rep[A-Za-z0-9]*Maps"\s*/>', txt))
        txt = re.sub(r'(<setoption\s+choice=")1("\s+name="rep[A-Za-z0-9]*Maps"\s*/>)', r"\g<1>0\g<2>", txt)
        changes.append(f"  -  {n_off} daily report-map options (rep*Maps) set to 0 (reporting only)")
    left = re.findall(r"/leonardo_work[^\"<]*", txt)
    if left:
        raise SystemExit(f"{xml_in}: unresolved Leonardo paths remain: {left[:3]}")
    Path(xml_out).write_text(txt)
    return changes


def resolve(xml):
    """Return (missing, checked) for every file referenced via the root paths."""
    txt = Path(xml).read_text()
    vals = {}
    for k, v in re.findall(r'<textvar\s+name="([^"]+)"\s+value=\s*"([^"]*)"', txt):
        if v.strip() != f"$({k})":          # lfbinding X="$(X)" keeps the lfuser value
            vals[k] = v
    def expand(s, depth=0):
        return re.sub(r"\$\(([^)]+)\)", lambda m: expand(vals.get(m.group(1), ""), depth + 1)
                      if depth < 20 else "", s)
    missing, checked = [], 0
    for name, v in vals.items():
        e = expand(v)
        if str(ROOT) not in e or "*" in e and "glofas_" not in e:
            continue
        if e.rstrip("/") in (str(ROOT),):
            continue
        checked += 1
        if "*" in e:                                          # forcing glob
            if not list(Path(e).parent.glob(Path(e).name)):
                missing.append((name, e))
            continue
        if not any(Path(e + ext).exists() for ext in ("", ".nc", ".map", ".txt", ".csv")):
            missing.append((name, e))
    return missing, checked


def main():
    n = build_flat_root()
    print(f"flat root: {ROOT}  ({n} files: map pack + {YEAR} forcing)")
    OUT.mkdir(parents=True, exist_ok=True)
    (RUN / "prerun").mkdir(parents=True, exist_ok=True)
    (RUN / "coldstart").mkdir(parents=True, exist_ok=True)
    log = [f"Window {START} .. {END} (inclusive); forcing year {YEAR}\n"]
    for name, src, pout, pinit, sp, rec in [
            ("prerun", "GloFASv5_prerun.xml", RUN / "prerun", RUN / "prerun", True, None),
            ("coldstart", "GloFASv5_coldstart.xml", RUN / "coldstart", RUN / "prerun", False,
             str(RUN / "openwq_record"))]:
        dst = OUT / f"GloFASv5_{name}_{TAG}.xml"
        ch = patch(OFFICIAL / src, dst, pout, pinit, sp, rec)
        log += [f"== {dst.name}  (from official {src}) =="] + ch + [""]
        miss, chk = resolve(dst)
        unused = [m for m in miss if m[0] in KNOWN_UNUSED]
        miss = [m for m in miss if m[0] not in KNOWN_UNUSED]
        print(f"{dst.name}: {len(ch)} change groups; referenced files checked: {chk}; "
              f"missing: {len(miss)} (+{len(unused)} of options that are off)")
        for m in miss[:25]:
            print("   MISSING", m)
        if unused:
            log.append(f"absent, options off ({len(unused)}): " + ", ".join(a for a, _ in unused))
        if miss:
            log.append(f"MISSING ({len(miss)}): " + "; ".join(f"{a}={b}" for a, b in miss))
    (OUT / f"CHANGES_vs_official_{TAG}.txt").write_text("\n".join(log) + "\n")
    print("change log:", OUT / f"CHANGES_vs_official_{TAG}.txt")
    if any("MISSING" in l for l in log):
        sys.exit(1)


if __name__ == "__main__":
    main()
