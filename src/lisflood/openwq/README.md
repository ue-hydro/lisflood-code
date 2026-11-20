# OpenWQ coupling for OS-LISFLOOD

One-way (forward) coupling: the LISFLOOD water balance drives the OpenWQ solute
balance. LISFLOOD hydrology is **not** modified — one optional hook reads the
end-of-step state/fluxes and advances OpenWQ through the generic `openwq_py`
pybind11 bridge (the UE-hydro pattern used for the differentiable-HBV host).

## Layout
```
src/lisflood/openwq/
  openwq/                          git submodule -> github.com/ue-hydro/openwq
                                   (branch feature/lisflood-host, base a312148):
                                   OpenWQ engine + its Python template generator
                                   (supporting_scripts/1_Model_Config/), which
                                   knows hostmodel = "lisflood"
  model_config_lisflood_nitrogen.py  openWQ's model_config_template.py, unchanged,
                                   plus a LISFLOOD settings block (generates inputs)
  OpenWQ_hydrolink.cpp / .h        pybind11 bridge (openwq_py), incl. move_array
  coupler.py                       LISFLOOD <-> OpenWQ compartment/flux mapping
  host.py                          thin Python wrapper around openwq_py.Host
  tests/                           unit tests + deep nitrogen test
```

## Reproducing a run (three steps)

**1. Generate the openWQ input files with openWQ's own template generator** — never
by hand. This writes `openWQ_master.json` + `openwq_in/` (config, NATIVE_BGC_FLEX,
NATIVE_TD_ADV, external fluxes) for `hostmodel = "lisflood"`:
```bash
git submodule update --init src/lisflood/openwq/openwq
python src/lisflood/openwq/model_config_lisflood_nitrogen.py
```
The script is the official `model_config_template.py` with only a marked
settings block changed (diff it against
`openwq/supporting_scripts/1_Model_Config/model_config_template.py`). It uses the
official BGC template `individual_chemicals/nitrogen_simple.json`. Paths can be
overridden with `OWQ_CASE_DIR` (output folder) and `LISFLOOD_SETTINGS_XML`.
Needs a Python with the generator's dependencies (the one you use for openWQ
`model_config_*.py` scripts).

**2. Build the bridge from the same openWQ commit** (inside the OpenWQ container,
Python env with pybind11 + numpy active):
```bash
cd src/lisflood/openwq/openwq
cmake -DHOST_MODEL_TARGET=python_openwq -DCMAKE_BUILD_TYPE=Release . && make
```
writes `openwq_py*.so` into `src/lisflood/openwq/` (a shared PhreeqcRM is built
once into the git-ignored `src/lisflood/openwq/_deps/`).

**3. Run** — LISFLOOD with the `OpenWQ` option on and `OpenWQMasterJson` pointing to
the generated master (hooks in `Lisflood_initial.py` / `Lisflood_dynamic.py`;
inert when the option is off, so default runs are byte-identical). Or the tests:
```bash
PYTHONPATH=src python -m pytest src/lisflood/openwq/tests/test_openwq_coupler.py
PYTHONPATH=src python src/lisflood/openwq/tests/deep_test_nitrogen.py   # in container
```

## Compartment ↔ LISFLOOD variable map (verified against v5.0.0)

| OpenWQ compartment | LISFLOOD store (`self.var`) | outgoing transfers |
|---|---|---|
| SOIL_TOP_A | `W1a` (mm) | `SeepTopToSubA` → SOIL_SUB |
| SOIL_TOP_B | `W1b` (mm) | `SeepTopToSubB` → SOIL_SUB |
| SOIL_SUB | `W2` (mm) | `SeepSubToGW` → UPPER_ZONE |
| UPPER_ZONE | `UZ` (mm) | `UZOutflow` (interflow) → CHANNEL; `GwPercUZLZ` → LOWER_ZONE |
| LOWER_ZONE | `LZ` (mm) | `LZOutflowToChannel` (baseflow) → CHANNEL; `GwLoss` (deep, out) |
| CHANNEL | `ChanM3` (m³) | `ChanQ` routing (downstream, LDD); `ToChanM3Runoff` in |

mm → m³ via `self.var.MMtoM3`. The temperature dependency `T` (°C, used by the BGC
templates' `θ^(T−20)` terms) is LISFLOOD's daily mean air temperature `Tavg`.
External fluxes are declared from the generated master (e.g. `LOAD`). The same
compartment list is defined in openWQ's generator
(`Gen_Input_Driver.LISFLOOD_COMPARTMENTS`); the deep test checks they agree.

## Status
**Done and tested:** generator support for `hostmodel = "lisflood"` (compartments,
sim-period from the LISFLOOD settings XML); input generation through the official
template; bridge incl. vectorised `move_array`; coupler (compartments, fluxes,
`T` from `Tavg`, external fluxes from the generated master, CWD-safe); unit tests;
deep nitrogen test against the compiled engine (see its docstring).

**Remaining (`# TODO` in coupler.py):**
1. Within-step flux ordering (port the dHBV coupler's origin-of-water replay).
2. In-channel advection cell → downstream cell along the LDD with `ChanQ`.
3. Feeding real loads (HaNi/HaPi diffuse, HydroWASTE point) through the EWF/SS
   options of the generator.
4. A full LISFLOOD run (needs LISFLOOD's runtime: PCRaster + its conda env) and
   profiling at GloFAS scale (6,221,926 cells).
5. `Gen_Report` has no LISFLOOD support yet (`generate_report = False`).
