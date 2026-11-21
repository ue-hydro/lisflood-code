#!/bin/bash
# Official GloFAS v5 reanalysis forcing (JRC open data) for one year, 5 variables in parallel.
# --- paths (set GLOFAS_DATA; LISFLOOD_REPO defaults to this fork) ---
: "${GLOFAS_DATA:?set GLOFAS_DATA to the GloFAS data folder (maps/, forcing_global/, runs/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
LISFLOOD_REPO=${LISFLOOD_REPO:-$(cd "$HERE/../../../.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}
set -u; Y=${1:-2015}
B=https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/meteorological_forcings/GloFASv5_reanalysis
O=${GLOFAS_DATA}/forcing_global/$Y; mkdir -p $O
for v in tp ta e0 eS0 eT0; do
  ( curl -sSL -C - --retry 5 --retry-delay 10 -o $O/glofas_${v}_$Y.nc $B/$v/glofas_${v}_$Y.nc \
      && echo "$(date +%T) done $v" || echo "$(date +%T) FAILED $v" ) &
done; wait; echo "ALL DONE $(date)"; ls -la $O
