# Global GloFAS v5 LISFLOOD × openWQ runs

Scripts used for the global coupling runs (one simulated week, 1–7 Jun 2015, and one year,
2 Jan – 31 Dec 2015). Every script reads its paths from the environment:

| variable | meaning | default |
|---|---|---|
| `GLOFAS_DATA` | data folder holding `maps/`, `forcing_global/`, `settings_week/`, `runs/`, `logs/` | required |
| `LISFLOOD_REPO` | this fork | four levels above this folder |
| `CONDA_SH` | conda's `profile.d/conda.sh` | `/opt/homebrew/anaconda3/etc/profile.d/conda.sh` |
| `OWQ_DEPS` | macOS build dependencies for the openWQ Python module | `$HOME/owq_macos_deps` |

Order:

1. `create_native_env.sh`, `create_openwq_env.sh`, `build_openwq_py_macos.sh`: conda environments
   (pcraster for LISFLOOD; `owq_arm64` for openWQ) and the openWQ Python module.
2. `download_maps.sh` / `download_maps_parallel.sh`, `download_forcing_year.sh YEAR`: the public
   GloFAS v5 static maps and the ERA5-based v5 forcing (JRC open data, CC BY 4.0).
3. `make_week_settings.py`: copies the official GloFAS v5 settings XMLs and changes only paths,
   dates and spin-up (plus the reporting-only substitutions it prints). `RUN_TAG`, `DAILY_MAPS`,
   `RECORD_DTYPE` select the run variant.
4. `preflight_small_domain.py`: a few-day run on a small domain to check the settings.
5. `run_global.sh STAGE TAG` (`prerun`, `coldstart`), then `post_coldstart.sh` (week) or
   `year_pipeline.sh` (year: prerun → cold start with the coupling recorder, openWQ replay in
   28-day chunks × 8 basin groups via `replay_year.py`, globe textures, satellite base, terrain and
   latitude × time extras, viewer).

The viewer is `../globe/index.html`; serve the run's `globe/` folder over HTTP to open it.
