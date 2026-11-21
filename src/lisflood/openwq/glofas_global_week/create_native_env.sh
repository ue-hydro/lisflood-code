#!/bin/bash
# Native macOS-arm64 mirror of OS-LISFLOOD v5.0.0's official environment.yml:
# same conda pins (python/pcraster/gdal/libnetcdf) + same pip pins for the runtime.
# --- paths (set GLOFAS_DATA; LISFLOOD_REPO defaults to this fork) ---
: "${GLOFAS_DATA:?set GLOFAS_DATA to the GloFAS data folder (maps/, forcing_global/, runs/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
LISFLOOD_REPO=${LISFLOOD_REPO:-$(cd "$HERE/../../../.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}
set -e
source "$CONDA_SH"
SOLVER=""   # classic solver (libmamba not installed in this conda)
# This Mac's Anaconda is x86_64 (Rosetta); force NATIVE arm64 packages
# (llvmlite >= 0.46 ships no macOS-x86_64 wheels; arm64 also avoids Rosetta).
export CONDA_SUBDIR=osx-arm64
conda create -y -n lisflood_v5 $SOLVER --override-channels -c conda-forge \
    python=3.12.13 pcraster=4.4.2 gdal=3.12.4 libnetcdf=4.10.0 pip
conda activate lisflood_v5
conda config --env --set subdir osx-arm64
python -c "import platform; assert platform.machine()=='arm64', platform.machine()"
pip install --no-cache-dir numpy==2.2.6 numba==0.65.1 llvmlite==0.47.0 netcdf4==1.7.4 \
  xarray==2026.4.0 dask==2026.3.0 pandas==2.3.3 scipy==1.17.1 cftime==1.6.5 pyproj==3.7.2 \
  numexpr==2.14.1 lxml==6.1.0 nine==1.2.0 future==1.0.0 python-dateutil==2.9.0.post0 \
  python-dotenv==1.2.2 lisflood-utilities==1.0.1 pyflwdir==0.5.11 toolz==1.1.0 \
  cloudpickle==3.1.2 partd==1.4.2 fsspec==2026.4.0 pytest==9.0.3 pytest-env==1.6.0 pytest-mock==3.15.1
python - <<'PY'
import sys, numpy, numba, netCDF4, xarray, pcraster
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| numba", numba.__version__,
      "| netCDF4", netCDF4.__version__, "| xarray", xarray.__version__, "| pcraster OK")
PY
echo ENV_READY
