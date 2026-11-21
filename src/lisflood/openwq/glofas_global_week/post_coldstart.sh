#!/bin/bash
# After the global coldstart week (LISFLOOD + coupling recorder):
#  1) openWQ replay of the recorded global week, in groups of whole river basins
#     (exactly equal to one global replay; fits workstation memory)
#  2) global map animation of the simulated species (topsoil, upper groundwater, rivers)
# --- paths (set GLOFAS_DATA; LISFLOOD_REPO defaults to this fork) ---
: "${GLOFAS_DATA:?set GLOFAS_DATA to the GloFAS data folder (maps/, forcing_global/, runs/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
LISFLOOD_REPO=${LISFLOOD_REPO:-$(cd "$HERE/../../../.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}
D=${GLOFAS_DATA}
OWQ=${LISFLOOD_REPO}/src/lisflood/openwq
G=$D/runs/global_week
NGROUPS=${NGROUPS:-8}
source "$CONDA_SH"
conda activate owq_arm64
set -u
LOG=$D/logs/global_week_openwq.log
echo "start $(date -u) groups=$NGROUPS" > $LOG
( /usr/bin/time -l python $OWQ/replay_partitioned.py $G/openwq_record $G/openwq_case $G/openwq_groups \
    --groups $NGROUPS ) >> $LOG 2>&1
RC=$?
echo "replay end $(date -u) RC=$RC" >> $LOG
[ $RC -ne 0 ] && exit $RC
du -sh $G/openwq_groups >> $LOG
( /usr/bin/time -l python $OWQ/make_animation.py $G/openwq_record $G/openwq_groups/partitions.json \
    $G/animation/global_week_N --extent=-180,180,-60,90 --coarsen 4 --fps 1 \
    --title "GloFAS v5 LISFLOOD x openWQ nitrogen_basic, 1-7 June 2015" ) >> $LOG 2>&1
echo "animation end $(date -u) RC=$?" >> $LOG
( /usr/bin/time -l python $OWQ/make_globe_textures.py $G/openwq_record $G/openwq_groups/partitions.json \
    $G/globe --extent=-180,180,-60,90 --width 2048 ) >> $LOG 2>&1
echo "globe textures end $(date -u) RC=$?" >> $LOG
cp $OWQ/globe/index.html $G/globe/index.html
