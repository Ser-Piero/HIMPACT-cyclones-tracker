# HIMPACT — HIgh-resolution Multilevel Python-based Algorithm for Cyclones' Centroid Tracking

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19695732.svg)](https://doi.org/10.5281/zenodo.19695732)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**HIMPACT** is a multi-level cyclone tracking tool for atmospheric model output.  
It locates a cyclone centre by finding the centroid and minimum of geopotential height
(or mean sea-level pressure) inside a moving search circle, simultaneously across
multiple pressure levels, and combines them into a robust weighted-mean track.

---

## Supported models

| Model | Grid type | Data format | Python back-end |
|-------|-----------|-------------|-----------------|
| **ICON** | Unstructured | NetCDF (`_ML_` / `_PL_` files) | `xarray` + `MetPy` |
| **WRF**  | Structured | NetCDF (`wrfout*` files) | `netCDF4` + `wrf-python` |
| **MPAS** | Unstructured | NetCDF (`mpasout*` / `diag*` files) | `xarray` + `MetPy` |
| **ERA5** | Regular lat-lon | GRIB (`.grib` / `.grb`) | `xarray` + `cfgrib` |

---

## Quick start

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ser-Piero/HIMPACT-cyclones-tracker.git
   cd HIMPACT-cyclones-tracker
   ```

2. **Install dependencies**

   ```bash
   conda create -n himpact -c conda-forge python=3.10
   conda activate himpact
   conda install -c conda-forge numpy pandas matplotlib cartopy scipy metpy xarray netcdf4 cfgrib wrf-python pillow shapely
   ```

3. **Edit the user configuration** in `HIMPACT_v1_6.py`:
   - Set `MODEL` (ICON / WRF / MPAS / ERA5)
   - Fill in `infolder`, `outfolder`
   - Set `CYCLONE`, `START_DATE`, `S0LAT`, `S0LON`
   - Choose tracking parameters (`SEARCH_RADIUS_KM`, `INTERP_LEVELS_HPA`, …)

4. **Run**

   ```bash
   python HIMPACT_v1_6.py
   ```

---

## Repository contents

| File | Purpose |
|------|---------|
| `HIMPACT_v1_6.py` | Main tracking algorithm |
| `HIMPACT_SENS.py` | Sensitivity-analysis runner (parallel parameter sweep) |
| `HIMPACT_SENS_EVALUATION.py` | Post-processing: compare sensitivity runs, compute error metrics, produce evaluation figures |

---

## Output

Running `HIMPACT_v1_6.py` produces:

- **`<cyclone>_<model>_<sim>_track_multilevelz.csv`** — weighted-mean track with SLP minimum and optional diagnostic variables
- **`<cyclone>_<model>_<sim>_track_multilevelz_smooth.csv`** — same track after rolling-mean smoothing
- **Per-level CSV files** (when `SAVE_ALL_TRACKS = True`) — centroid and minimum at each pressure level
- **PNG plots** (when `PLOT = True`) — running track preview, final track map with SLP colouring, per-level cloud plot

---

## How the algorithm works

1. At each time step, a circular search mask (radius `SEARCH_RADIUS_KM`) is centred on
   the cyclone position estimated at the previous step.
2. Inside the circle, the algorithm isolates the low-value core using the
   `PERCENTILE_THRESHOLD`-th percentile of the field.
3. The **centroid** of the convex hull of the core points and the **absolute minimum**
   are computed at every requested pressure level (plus SLP).
4. The running centre estimate is updated as the mean across all levels and methods,
   and the search circle moves with it to the next time step.
5. After all time steps, a **weighted-mean track** is computed from all per-level
   centroid and minimum positions (equal weights by default; customisable).
6. Diagnostic variables (SST, winds, heat fluxes, precipitation, PV, …) can be
   extracted inside the search circle and appended to the output CSV.

### Key features

- **Robust to noisy pixels** — convex-hull centroid is less sensitive to isolated
  extremes than a plain minimum
- **Multi-level** — tracking across pressure levels catches vertical tilts of the
  cyclone column
- **Landfall detection** — automatic land/sea classification of the centre location
  using Natural Earth shapefiles
- **Smoothing** — optional rolling-mean filter on the final track

---

## Sensitivity analysis

`HIMPACT_SENS.py` launches a parallel parameter sweep over:

- `SEARCH_RADIUS_KM`
- `INTERP_LEVELS_HPA`
- `PERCENTILE_THRESHOLD`
- Tracking mode: centroid-only, minimum-only, or both

Each combination runs as an independent subprocess of `HIMPACT_v1_6.py`.
The number of parallel workers defaults to (physical CPU cores − 1).

```bash
python HIMPACT_SENS.py
```

`HIMPACT_SENS_EVALUATION.py` then reads all `HIMPACT_SENS_*` output folders and
produces comparison plots and an error-metrics table (mean displacement, RMSE, …)
against an observation track.

```bash
# Edit trackfolder, cyclone, model, and obs path inside the script, then:
python HIMPACT_SENS_EVALUATION.py
```

---

## Citation

If you use HIMPACT in a publication, please cite:

> Serafini P. (2026). *HIMPACT — HIgh-resolution Multilevel Python-based Algorithm
> for Cyclones' Centroid Tracking*. Zenodo.
> [https://doi.org/10.5281/zenodo.19695732](https://doi.org/10.5281/zenodo.19695732)

---

## Author

**Piero Serafini** — PhD student in Atmospheric Physics  
University of L'Aquila (UNIVAQ) — CETEMPS  
piero.serafini@graduate.univaq.it

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).  
See the `LICENSE` file for details.
