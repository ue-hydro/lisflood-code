#!/bin/bash
# Native macOS-arm64 build of openWQ's Python module (openwq_py) for the LISFLOOD
# host, from the UNMODIFIED openWQ submodule (CMake target python_openwq) + the
# LISFLOOD hydrolink files (src/lisflood/openwq/OpenWQ_hydrolink.{h,cpp}).
# Out-of-source build (the submodule may hold a Linux in-source CMake cache) and
# a macOS PhreeqcRM v3.8.0 (same version as the openWQ container build) passed
# explicitly via -DPhreeqcRM_DIR, so the Linux _deps/ install is never picked up.
# --- paths (set GLOFAS_DATA; LISFLOOD_REPO defaults to this fork) ---
: "${GLOFAS_DATA:?set GLOFAS_DATA to the GloFAS data folder (maps/, forcing_global/, runs/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
LISFLOOD_REPO=${LISFLOOD_REPO:-$(cd "$HERE/../../../.." && pwd)}
CONDA_SH=${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}
set -e
source "$CONDA_SH"
conda activate owq_arm64
# openWQ is written for GCC/libstdc++ (as in its container): it names std::__cxx11
# and relies on Armadillo's HDF5 wrapper (ARMA_USE_HDF5); conda's HDF5 CMake config
# leaves HDF5_C_LIBRARY_hdf5 empty, so libhdf5 is passed to the link explicitly. Build with Homebrew GCC,
# without conda's clang flags; libraries (Armadillo/HDF5/SUNDIALS/Python) are C-ABI.
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS
export CC=/opt/homebrew/bin/gcc-15 CXX=/opt/homebrew/bin/g++-15
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
PKG=${LISFLOOD_REPO}/src/lisflood/openwq
DEPS=${OWQ_DEPS:-$HOME/owq_macos_deps}
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
NJ=8

# 1) PhreeqcRM v3.8.0, shared (a Python module cannot link the static library)
if [ ! -f $DEPS/phreeqcrm-install-gcc/lib/cmake/PhreeqcRM/PhreeqcRMConfig.cmake ]; then
  cmake -S $DEPS/phreeqcrm-src -B $DEPS/phreeqcrm-build-gcc -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON -DPHREEQCRM_BUILD_TESTS=OFF -DPHREEQCRM_FORTRAN_TESTING=OFF \
        -DPHREEQCRM_BUILD_MPI=OFF -DPHREEQCRM_BUILD_PYTHON=OFF \
        -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
        -DCMAKE_INSTALL_PREFIX=$DEPS/phreeqcrm-install-gcc
  cmake --build $DEPS/phreeqcrm-build-gcc -j $NJ
  cmake --install $DEPS/phreeqcrm-build-gcc
fi
PRM_CFG=$(dirname $(find $DEPS/phreeqcrm-install-gcc -name PhreeqcRMConfig.cmake | head -1))
echo "PhreeqcRM_DIR=$PRM_CFG"

# 2) openwq_py (written by CMake into $PKG as openwq_py.cpython-312-darwin.so)
cmake -S $PKG/openwq -B $DEPS/build_openwq_py_gcc -DHOST_MODEL_TARGET=python_openwq \
      -DCMAKE_BUILD_TYPE=Release -DPhreeqcRM_DIR=$PRM_CFG \
      -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -DARMA_USE_HDF5" \
      -DHDF5_C_LIBRARY_hdf5=$CONDA_PREFIX/lib/libhdf5.dylib \
      -DCMAKE_MODULE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -lhdf5 -Wl,-rpath,$CONDA_PREFIX/lib" \
      -DPython_EXECUTABLE=$(which python) -DCMAKE_INSTALL_RPATH=$CONDA_PREFIX/lib \
      -DCMAKE_BUILD_RPATH="$CONDA_PREFIX/lib;$DEPS/phreeqcrm-install-gcc/lib"
cmake --build $DEPS/build_openwq_py_gcc -j $NJ
ls -la $PKG/openwq_py*darwin*.so
cd $PKG && python -c "import openwq_py; print('openwq_py OK:', openwq_py.__file__)"
echo OPENWQ_PY_BUILD_OK
