#!/usr/bin/env python
"""Settings for the LISFLOOD -> openWQ check on LISFLOOD's own test catchment
(tests/data/LF_lat_lon_UseCase, shipped with OS-LISFLOOD v5.0.0).

Copies run_lat_lon.xml changing ONLY: PathRoot (absolute, so the copy can live
elsewhere), PathOut, StepStart/StepEnd, and adds option OpenWQRecord=1 +
textvar OpenWQRecordDir in <lfbinding> (the coupling recorder of lisflood/openwq/coupler.py).

usage: python make_usecase_settings.py OUT_DIR [START] [END]
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CASE = os.path.join(REPO, "tests", "data", "LF_lat_lon_UseCase")


def main():
    out = os.path.abspath(sys.argv[1])
    start = sys.argv[2] if len(sys.argv) > 2 else "02/01/1986 00:00"
    end = sys.argv[3] if len(sys.argv) > 3 else "31/01/1986 00:00"
    os.makedirs(os.path.join(out, "lisflood_out"), exist_ok=True)
    txt = open(os.path.join(CASE, "run_lat_lon.xml")).read()

    def setvar(name, value):
        nonlocal txt
        pat = re.compile(r'(<textvar\s+name="%s"\s+value=")([^"]*)(")' % name)
        if not pat.search(txt):
            raise SystemExit("textvar not found: " + name)
        txt = pat.sub(lambda m: m.group(1) + value + m.group(3), txt, count=1)

    setvar("PathRoot", CASE)
    setvar("PathOut", os.path.join(out, "lisflood_out"))
    setvar("StepStart", start)
    setvar("StepEnd", end)
    txt = txt.replace("<lfoptions>", '<lfoptions>\n        <setoption choice="1" name="OpenWQRecord"/>', 1)
    # bindings are read from <lfbinding> (lfuser only holds substitution variables)
    txt = re.sub(r"(<lfbinding>)", r'\1\n        <textvar name="OpenWQRecordDir" value="%s"/>'
                 % os.path.join(out, "openwq_record"), txt, count=1)
    dst = os.path.join(out, "run_lat_lon_openwq_record.xml")
    open(dst, "w").write(txt)
    print(dst)


if __name__ == "__main__":
    main()
