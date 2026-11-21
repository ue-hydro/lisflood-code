#!/bin/bash
# Global GloFAS v5 LISFLOOD run, 1 week of simulated time (1-7 June 2015), from the
# official GloFAS v5 settings (settings_week/*.xml, see CHANGES_vs_official.txt).
#   ./run_global_week.sh prerun     -> long-term averages for the cold start
#   ./run_global_week.sh coldstart  -> cold start + LISFLOOD->openWQ coupling recorder
# LISFLOOD source: the ue-hydro fork (= OS-LISFLOOD v5.0.0 + inert openWQ hooks).
# Logs wall time / peak memory (/usr/bin/time -l) and RSS every 30 s.
# --- paths (set GLOFAS_DATA; LISFLOOD_REPO defaults to this fork) ---
: "${GLOFAS_DATA:?set GLOFAS_DATA to the GloFAS data folder (maps/, forcing_global/, runs/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
LISFLOOD_REPO=${LISFLOOD_REPO:-$(cd "$HERE/../../../.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}
STAGE=$1
TAG=${2:-global_week}
D=${GLOFAS_DATA}
SRC=${LISFLOOD_REPO}/src/lisf1.py
XML=$D/settings_week/GloFASv5_${STAGE}_${TAG}.xml
OUT=$D/runs/$TAG/$STAGE
LOG=$D/logs/${TAG}_${STAGE}.log
source "$CONDA_SH"
conda activate lisflood_v5
set -u   # (after conda: its activate scripts use unset variables)
mkdir -p "$OUT"
echo "start $(date -u)  stage=$STAGE  xml=$XML" > "$LOG"
( /usr/bin/time -l python "$SRC" "$XML" ) >> "$LOG" 2>&1 &
PID=$!
( while kill -0 $PID 2>/dev/null; do
    R=$(for PP in $(pgrep -f "lisf1.py $XML"); do ps -o rss= -p "$PP" 2>/dev/null; done | sort -n | tail -1 | tr -d ' ')
    echo "$(date -u +%T) rss_kb=${R:-?} free_pages=$(vm_stat | awk '/Pages free/{print $3}')" >> "$D/logs/${TAG}_${STAGE}_mem.log"
    sleep 30; done ) &
wait $PID
RC=$?
echo "end $(date -u) RC=$RC" >> "$LOG"
du -sh "$OUT" >> "$LOG"
[ "$STAGE" = coldstart ] && du -sh "$D/runs/$TAG/openwq_record" >> "$LOG" 2>/dev/null
exit $RC
