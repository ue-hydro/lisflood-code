#!/usr/bin/env python
"""Add a satellite base map to a globe folder made by make_globe_textures.py.

Takes 1 or 12 full-globe equirectangular images (lon -180..180, lat 90..-90), e.g. the
NASA Blue Marble Next Generation monthly composites (world.topo.bathy.2004MM.3x5400x2700.jpg,
public domain), resizes them to WIDTH x WIDTH/2, writes GLOBE_DIR/tex/sat_MM.webp and adds a
"satellite" entry to GLOBE_DIR/manifest.json. With 12 images the viewer shows the composite of
the calendar month of the date on screen.

usage: python add_satellite_base.py GLOBE_DIR IMG [IMG ...] [--width 4096] [--quality 80]
           [--credit "..."] [--year 2004]
"""
import argparse
import json
import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = 400_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globe_dir")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--year", type=int, default=2004)
    ap.add_argument("--credit", default="NASA Earth Observatory, Blue Marble Next Generation with topography "
                    "and bathymetry (Stöckli et al. 2005)")
    a = ap.parse_args()
    if len(a.images) not in (1, 12):
        raise SystemExit("give 1 image or 12 monthly images (January .. December)")
    man_p = os.path.join(a.globe_dir, "manifest.json")
    man = json.load(open(man_p))
    os.makedirs(os.path.join(a.globe_dir, "tex"), exist_ok=True)
    files = []
    for i, src in enumerate(a.images):
        im = Image.open(src).convert("RGB")
        w, h = im.size
        if abs(w / h - 2.0) > 0.01:
            raise SystemExit(f"{src}: {w}x{h} is not a 2:1 full-globe equirectangular image")
        im = im.resize((a.width, a.width // 2), Image.LANCZOS)
        fn = f"tex/sat_{i + 1:02d}.webp" if len(a.images) == 12 else "tex/sat.webp"
        im.save(os.path.join(a.globe_dir, fn), "WEBP", quality=a.quality, method=4)
        files.append(fn)
        print(f"{src} -> {fn} ({os.path.getsize(os.path.join(a.globe_dir, fn)) / 1e6:.2f} MB)", flush=True)
    man["satellite"] = {"files": files, "credit": a.credit, "year": a.year,
                        "sources": [os.path.basename(s) for s in a.images]}
    json.dump(man, open(man_p, "w"), separators=(",", ":"))
    print("manifest:", man_p)


if __name__ == "__main__":
    main()
