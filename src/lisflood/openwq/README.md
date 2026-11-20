# OpenWQ coupling for OS-LISFLOOD

One-way (forward) coupling: the LISFLOOD water balance drives the OpenWQ solute
balance. LISFLOOD hydrology is **not** modified — a single optional hook reads
the end-of-step state/fluxes and advances OpenWQ.

This follows the UE-hydro pattern used for the differentiable-HBV host
(`hydroDL2-openwq`): a generic pybind11 bridge (`openwq_py`) exposing OpenWQ's
coupler to Python, plus a host-specific `coupler.py` mapping layer.

## Layout
```
src/lisflood/openwq/
  OpenWQ_hydrolink.cpp / .h   generic pybind11 bridge (openwq_py) — reused unchanged
  openwq/                     git submodule -> github.com/ue-hydro/openwq.git
  CMakeLists.txt              builds the openwq_py extension
  coupler.py                  LISFLOOD <-> OpenWQ compartment/flux mapping
  host.py                     thin Python wrapper around openwq_py.Host
  tests/                      pytest unit tests (run without the compiled module)
```

## Enabling it
Set the `OpenWQ` option and the master-JSON path in the LISFLOOD settings file.
When the option is **off (default)** the hooks are inert and LISFLOOD runs are
byte-identical (hooks in `Lisflood_initial.py` and `Lisflood_dynamic.py` are
guarded by `option.get('OpenWQ', False)`).

## Build (compiled bridge)
```bash
git submodule update --init src/lisflood/openwq/openwq      # fetch OpenWQ
cmake -S src/lisflood/openwq -B build/openwq -DCMAKE_BUILD_TYPE=Release
cmake --build build/openwq --target openwq_py -j
# put the resulting openwq_py*.so on PYTHONPATH (or in this folder)
```
Dependencies: a C++17 compiler, pybind11, Armadillo (and OpenWQ's own deps, incl.
optionally PhreeqcRM). See the OpenWQ build docs.

## Compartment ↔ LISFLOOD variable map (verified against v5.0.0)

| OpenWQ compartment | LISFLOOD store (`self.var`) | outgoing transfers |
|---|---|---|
| SOIL_TOP_A | `W1a` (mm) | `Infiltration` in; `SeepTopToSubA` |
| SOIL_TOP_B | `W1b` (mm) | `SeepTopToSubB` |
| SOIL_SUB | `W2` (mm) | `SeepSubToGW` → UZ |
| UPPER_ZONE | `UZ` (mm) | `UZOutflow` (interflow→CHANNEL); `GwPercUZLZ` → LZ |
| LOWER_ZONE | `LZ` (mm) | `LZOutflowToChannel` (baseflow); `GwLoss` (deep, out) |
| CHANNEL | `ChanM3` (m³) | `ChanQ` routing (downstream, LDD); `ToChanM3Runoff` in |

Conversions: mm → m³ via `self.var.MMtoM3` (pixel area / 1000); `ChanM3` and
`...M3...` fluxes are already volumetric. OpenWQ cells = LISFLOOD *compressed*
(masked) 1-D cells (~6.2 M for the global GloFAS domain).

## Status — what is done vs. what remains
**Done (verified against OS-LISFLOOD v5.0.0 source):** package scaffold; generic
bridge vendored; compartment↔variable mapping; lifecycle (declare → init →
prime → per-step `time_start`/`move`/`time_end`); guarded hooks; unit tests that
pass without the compiled module.

**Remaining (flagged `# TODO` in coupler.py — the real implementation/research):**
1. **Within-step flux ordering** — LISFLOOD updates several stores within one
   step; OpenWQ applies transfers from start-of-step state. Port the dHBV
   coupler's origin-of-water replay so no store exports more than it held.
2. **In-channel advection** — move channel solute cell → downstream cell along
   the LDD using `ChanQ` each step (needs the drainage-direction index).
3. **External loads** — diffuse (HaNi/HaPi) into the soil surface and point
   sources (HydroWASTE) into the channel, via `inflow`.
4. **Scaling** — ~6.2 M cells: the coupler must use the array API
   (`set_water_volume_array`, `move_array`); add `move_array` to the bridge if
   absent. OpenWQ's many-cell scaling is unproven (OpenMP-only) — **prototype on
   one catchment and profile before any global run.**
