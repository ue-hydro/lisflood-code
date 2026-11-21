#!/bin/bash
# Global GloFAS v5 LISFLOOD x openWQ for one year (2 Jan - 31 Dec 2015, 364 daily steps):
#  1) year prerun (long-term averages for the cold start)
#  2) year cold start with the coupling recorder (float32, compressed), and, alongside it,
#     replay_year.py: openWQ in 28-day chunks x 8 groups of whole river basins, state carried
#     between chunks, weekly global frames, recordings deleted after use, LISFLOOD paused if
#     unprocessed days pile up
#  3) globe textures (52 weekly frames x 12 layers) for the rotating-globe viewer
# --- paths (set GLOFAS_DATA; LISFLOOD_REPO defaults to this fork) ---
: "${GLOFAS_DATA:?set GLOFAS_DATA to the GloFAS data folder (maps/, forcing_global/, runs/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
LISFLOOD_REPO=${LISFLOOD_REPO:-$(cd "$HERE/../../../.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}
D=${GLOFAS_DATA}
OWQ=${LISFLOOD_REPO}/src/lisflood/openwq
Y=$D/runs/global_year
LOG=$D/logs/global_year_pipeline.log
echo "pipeline start $(date -u)" > $LOG

$HERE/run_global.sh prerun global_year
RC=$?; echo "prerun end $(date -u) RC=$RC" >> $LOG
[ $RC -ne 0 ] && exit $RC

# openWQ driver (waits for the recorder's files), then the LISFLOOD cold start
( source "$CONDA_SH"; conda activate owq_arm64
  python $OWQ/replay_year.py $Y/openwq_record $Y/openwq_case $Y/openwq_year \
      --groups 8 --chunk 28 --nsteps 364 --lisflood-match coldstart_global_year \
      --max-backlog 30 --resume-at 15 > $D/logs/global_year_openwq.log 2>&1
  echo "openwq driver end $(date -u) RC=$?" >> $LOG ) &
DRV=$!
$HERE/run_global.sh coldstart global_year
RC=$?; echo "coldstart end $(date -u) RC=$RC" >> $LOG
wait $DRV
grep -q "\[year\] DONE" $D/logs/global_year_openwq.log || { echo "openWQ driver did not finish" >> $LOG; exit 1; }

source "$CONDA_SH"; conda activate owq_arm64
python $OWQ/make_globe_textures.py $Y/openwq_record $Y/openwq_year/frames $Y/globe \
    --extent=-180,180,-60,90 --width 2048 >> $D/logs/global_year_textures.log 2>&1
echo "textures end $(date -u) RC=$?" >> $LOG
# optional satellite base (NASA Blue Marble NG 2004 monthly composites, if downloaded to basemaps/)
BM=$D/basemaps/bmng_2004_topo_bathy
if ls $BM/world.topo.bathy.2004*.jpg >/dev/null 2>&1; then
  python -I $OWQ/add_satellite_base.py $Y/globe $(ls $BM/world.topo.bathy.2004*.jpg | sort) >> $D/logs/global_year_textures.log 2>&1
  echo "satellite end $(date -u) RC=$?" >> $LOG
fi
# terrain heights (LISFLOOD elv at the viewer's mesh nodes) + latitude x time band means
python -I $OWQ/add_globe_extras.py $Y/globe $Y/openwq_record >> $D/logs/global_year_textures.log 2>&1
echo "extras end $(date -u) RC=$?" >> $LOG
cp $OWQ/globe/index.html $Y/globe/index.html
du -sh $Y/* >> $LOG
echo "pipeline DONE $(date -u)" >> $LOG
