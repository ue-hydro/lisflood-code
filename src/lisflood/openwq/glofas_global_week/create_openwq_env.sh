#!/bin/bash
# Native macOS-arm64 environment to BUILD and RUN openWQ's Python module (openwq_py)
# for the LISFLOOD x openWQ global run. Mirrors the openWQ container's library
# versions where it matters (SUNDIALS 7.1.1; Armadillo with HDF5; PhreeqcRM built
# from source by build_openwq_py_macos.sh). The global run is too large for the
# shared Docker VM (16 GB), hence a native build. openWQ source is NOT modified.
# --- paths (set GLOFAS_DATA; LISFLOOD_REPO defaults to this fork) ---
: "${GLOFAS_DATA:?set GLOFAS_DATA to the GloFAS data folder (maps/, forcing_global/, runs/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
LISFLOOD_REPO=${LISFLOOD_REPO:-$(cd "$HERE/../../../.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}
set -e
source "$CONDA_SH"
export CONDA_SUBDIR=osx-arm64          # this Anaconda is x86_64; force native arm64
conda create -y -n owq_arm64 --override-channels -c conda-forge \
    python=3.12 numpy pybind11 cmake make cxx-compiler c-compiler llvm-openmp \
    sundials=7.1.1 armadillo hdf5 h5py matplotlib ffmpeg git
conda activate owq_arm64
conda config --env --set subdir osx-arm64
python -c "import platform, numpy, h5py, matplotlib; assert platform.machine()=='arm64'; print('numpy', numpy.__version__, 'h5py', h5py.__version__)"
echo OWQ_ENV_READY
