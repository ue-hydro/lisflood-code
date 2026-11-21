#!/bin/bash
# Parallel, resumable mirror of the public GloFAS v5 LISFLOOD static/parameter map
# pack (JRC open data, CC BY 4.0, DOI 10.2905/JRC.J2J3DQR). Same files as
# download_maps.sh, NP concurrent streams (the JRC server caps each stream at
# ~0.5-1 MB/s). Complete files (local size == remote Content-Length) are skipped.
# --- paths (set GLOFAS_DATA; LISFLOOD_REPO defaults to this fork) ---
: "${GLOFAS_DATA:?set GLOFAS_DATA to the GloFAS data folder (maps/, forcing_global/, runs/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
LISFLOOD_REPO=${LISFLOOD_REPO:-$(cd "$HERE/../../../.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}
set -u
BASE=https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/LISFLOOD_static_and_parameter_maps_for_GloFAS/v2.1.1_OS-LISFLOOD-v5.x
OUT=${GLOFAS_DATA}/maps
LIST=${GLOFAS_DATA}/logs/maps_filelist.txt
NP=${NP:-8}
: > "$LIST"
list() {  # $1 = relative folder ('' for root)
  local rel="$1"; local html; html=$(curl -sL --max-time 120 "$BASE/${rel}")
  for f in $(echo "$html" | grep -oE 'href="[^"?/][^"]*"' | sed -E 's/href="//;s/"//' | grep -v '^\.\./'); do
    if [[ "$f" == */ ]]; then list "${rel}${f}"; else echo "${rel}${f}" >> "$LIST"; fi
  done
}
list ""
echo "$(date +%T) $(wc -l < "$LIST") files listed"
fetch() {
  local rel="$1" url="$BASE/$1" dst="$OUT/$1"
  mkdir -p "$(dirname "$dst")"
  local rsz; rsz=$(curl -sIL --max-time 60 "$url" | awk 'tolower($1)=="content-length:"{v=$2} END{gsub("\r","",v); print v}')
  local lsz; lsz=$(stat -f %z "$dst" 2>/dev/null || echo 0)
  if [ -n "$rsz" ] && [ "$lsz" = "$rsz" ]; then return 0; fi
  curl -sSL -C - --retry 8 --retry-delay 15 -o "$dst" "$url" && echo "$(date +%T) OK $rel" || echo "$(date +%T) FAILED $rel"
}
export -f fetch; export BASE OUT
xargs -P "$NP" -I{} bash -c 'fetch "$@"' _ {} < "$LIST"
echo "DONE $(date)"; du -sh "$OUT"
