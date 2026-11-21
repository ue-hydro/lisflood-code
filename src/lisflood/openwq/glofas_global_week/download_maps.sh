#!/bin/bash
# Mirror the public GloFAS v5 LISFLOOD static/parameter map pack (JRC open data,
# CC BY 4.0, DOI 10.2905/JRC.J2J3DQR). Resumable; skips complete files.
# --- paths (set GLOFAS_DATA; LISFLOOD_REPO defaults to this fork) ---
: "${GLOFAS_DATA:?set GLOFAS_DATA to the GloFAS data folder (maps/, forcing_global/, runs/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
LISFLOOD_REPO=${LISFLOOD_REPO:-$(cd "$HERE/../../../.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}
set -u
BASE=https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/LISFLOOD_static_and_parameter_maps_for_GloFAS/v2.1.1_OS-LISFLOOD-v5.x
OUT=${GLOFAS_DATA}/maps
mirror() {  # $1 = relative folder ('' for root)
  local rel="$1"; local url="$BASE/${rel}"; mkdir -p "$OUT/$rel"
  local html; html=$(curl -sL --max-time 120 "$url")
  for f in $(echo "$html" | grep -oE 'href="[^"?/][^"]*"' | sed -E 's/href="//;s/"//' | grep -v '^\.\./'); do
    if [[ "$f" == */ ]]; then mirror "${rel}${f}"; else
      echo "$(date +%T) ${rel}${f}"
      curl -sSL -C - --retry 5 --retry-delay 10 -o "$OUT/${rel}${f}" "$url$f" || echo "  FAILED ${rel}${f}"
    fi
  done
}
mirror ""
echo "DONE $(date)"; du -sh "$OUT"
