#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIMPACT_v1_6.py
=======================================================================================
HIgh-resolution Multilevel Python-based Algorithm for cyclones' Centroid Tracking

Multi-level cyclone tracking tool for ICON, WRF, MPAS and ERA5 model output.

The algorithm tracks a cyclone center by locating the minimum and centroid
of geopotential height (or sea-level pressure) within a search radius at
multiple pressure levels.  A weighted-mean track is then derived from all
levels and optionally enriched with diagnostic variables (SLP, SST, winds,
heat fluxes, precipitation, potential vorticity, ...).
==============================================================================================================================================================================
Version update notes
---------------
v1.6 (2026-07-07)
    * Added support for ERA5 reanalysis data MSLP tracking (via xarray + MetPy)
    * Added support for ERA5 reanalysis data geopotential tracking (via xarray + MetPy)
    * Refactored the code to remove duplication across model back-ends, fixed several
      bugs (per-level CSV columns, MPAS SHF variable, ICON variable-extraction level
      mapping) and made every function receive its inputs explicitly for easier debugging.
v1.5 (2026-06-16)
    * Added JSON-based sensitivity override mechanism for tracking parameters and model configuration, 
      to be used by HIMPACT_SENS.py without modifying the code (see Section 1b).
    * Corrected bug on plotting final track
    * Corrected bug in saving raw track before smoothing (previously it was saving the smoothed track twice)
    * Corrected some aspects of the check plots
    * Implemented END_DATE to limit the analysis to a specific time window
v1.4 (2026-06-01)
    * Added a figure about timeseries plots of diagnostic variables along the track, 
      to check the physical consistency of the track and its relation with cyclone intensity
v1.3 (2026-05-26)
    * Added support for plot temporarily estimated track and SLP at each time step, to check the track quality
    * Improved plotting functionality with markers for first and last valid points
v1.2 (2026-05-24)
    * Added MPAS support 
    * Added level interpolation for MPAS
v1.1 (2026-05-19)
    * Added Landfall Detection
v1.0 (2025-07-15)
    * Initial release
==============================================================================================================================================================================
Supported model back-ends
--------------------------
  MODEL = "ICON"  -  ICON unstructured-grid NetCDF output (via xarray + MetPy)
  MODEL = "WRF"   -  WRF structured-grid output          (via netCDF4 + wrf-python)
  MODEL = "MPAS"  -  MPAS unstructured-grid NetCDF output (via xarray + MetPy)
  MODEL = "ERA5"  -  ERA5 reanalysis data (via xarray + cfgrib)

Quick-start
-----------
  1. Set MODEL to "ICON" or "WRF" or "MPAS" or "ERA5" in the USER CONFIGURATION section below.
  2. Fill in infolders, sims, outfolder and the initial cyclone position/date.
  3. Toggle export_variables flags as needed.
  4. Run:  python HIMPACT_v1_6.py

Output
------
  * <cyclone>_<model>_<sim>_track_multilevelz.csv  - weighted-mean track + diagnostics
  * (optional) per-level CSV files                 - when save_all_tracks = True
  * (optional) PNG plots per level and timestep    - when plot = True

Authors
-------
  Piero Serafini
    PhD student in Atmospheric Physics
      University of L'Aquila (UNIVAQ)
      Center of Excellence in Telesensing of Environment and Model Prediction of Severe Events (CETEMPS)
        Via Vetoio, Edificio Renato Ricamo, L'Aquila (AQ), Italy, 67100
          piero.serafini@graduate.univaq.it

License
-------
  MIT License - see LICENSE file.

Citation / DOI
--------------
  This code is archived on Zenodo.
  If you use this tool in a publication, please cite it as:
    Serafini P. (2026). HIMPACT - HIgh-resolution Multilevel Python-based Algorithm for cyclones' Centroid Tracking. Zenodo.
    https://doi.org/10.5281/zenodo.19695732  

==============================================================================================================================================================================
==============================================================================================================================================================================
==============================================================================================================================================================================
"""
# =============================================================================
# SECTION 0 - STANDARD LIBRARY / CROSS-MODEL IMPORTS
# =============================================================================

import os           # path and permission checks
import time         # wall-clock timing
import locale       # set locale so month names are always in English
import traceback    # detailed tracebacks so errors always report the exact line

import numpy as np  # type: ignore[reportMissingImports]                # numerical arrays
import pandas as pd  # type: ignore[reportMissingModuleSource]          # DataFrames, CSV I/O, datetime parsing
import matplotlib  # type: ignore[reportMissingModuleSource]            # backend must be set before pyplot import
import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
import cartopy.crs as ccrs  # type: ignore[reportMissingImports]         # map projections
from cartopy.feature import NaturalEarthFeature  # type: ignore[reportMissingImports]  # land/ocean shading
from cartopy.feature import COLORS  # type: ignore[reportMissingImports]  # predefined map colors
from shapely.geometry import Point, Polygon  # type: ignore[reportMissingModuleSource]  # convex-hull centroid
from scipy.spatial import ConvexHull  # type: ignore[reportMissingImports]  # convex-hull computation
import metpy  # type: ignore # noqa: F401  # for unit handling and interpolation of meteorological variables (ICON)
from shapely.geometry import Point as ShapelyPoint # type: ignore[reportMissingModuleSource]  # for land mask generation
import cartopy.io.shapereader as shpreader  # type: ignore[reportMissingImports]  # for land mask generation
from PIL import Image  # type: ignore[reportMissingImports]  # for land mask generation

# Force English month names in timestamps (works on Linux/macOS)
try:
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
except locale.Error:
    print(
        "Could not set locale to 'en_US.UTF-8'. \n Month names in timestamps may be in the system language.",
    )

# Use a non-interactive Matplotlib backend (safe for servers without a display)
matplotlib.use('Agg')

# ==========================================================================================================================================================================================
# ==========================================================================================================================================================================================
# ==========================================================================================================================================================================================
# SECTION 1 - USER CONFIGURATION  ← EDIT THIS SECTION BEFORE RUNNING
# =============================================================================

# --- 1.0  Simulation configurations ----------------------
# Example configuration 
# MODELS_CONFIG = [
#    {
#        "model": "<name of model, supported values: 'ICON', 'WRF', 'MPAS', 'ERA5'>",
#        "sim": "<short label for the simulation, e.g. 'SH' or 'EXP'>",
#        "infolder": "<absolute path to the folder containing the model output files for this simulation>",
            # ICON: separate model-level (_ML_) and pressure-level (_PL_) sub-folders
            #       can live in the SAME parent folder; the script filters by filename tag.
            # WRF:  one folder per simulation containing *wrfout* files.
            # MPAS: one folder per simulation containing *mpasout* and *diag* files.
#        "outfolder": "<absolute path to the folder where output CSV and PNG files will be written for this simulation>",
#    },
#]

MODELS_CONFIG = [
    # Example ICON configuration:
    # {
    #     "model": "ICON",
    #     "sim": "MY_SIM",
    #     "infolder": "/path/to/your/ICON/data",
    #     "outfolder": "/path/to/your/output",
    # },
    # Example WRF configuration:
    # {
    #     "model": "WRF",
    #     "sim": "MY_SIM",
    #     "infolder": "/path/to/your/WRF/data",
    #     "outfolder": "/path/to/your/output",
    # },
    # Example MPAS configuration:
    # {
    #     "model": "MPAS",
    #     "sim": "MY_SIM",
    #     "infolder": "/path/to/your/MPAS/data",
    #     "outfolder": "/path/to/your/output",
    # },
    # Example ERA5 configuration:
    # {
    #     "model": "ERA5",
    #     "sim": "MY_SIM",
    #     "infolder": "/path/to/your/ERA5/data",
    #     "outfolder": "/path/to/your/output",
    # },
]

# name of the cyclone (used in output filenames and plot titles)
CYCLONE = "DANIEL"

# ==== 1.2  Tracking parameters =======================================================================================================

# Some example cyclones with their approximate center coordinates at the start date:
# DANIEL '04-Sep-2023 12:00 UTC' LAT: 38.1 LON: 20.5
# QENDRESA '07-Nov-2014 00:00 UTC' LAT: 36.8 LON: 12.0
# IANOS '15-Sep-2020 00:00 UTC' LAT: 32.9 LON: 15.7
# HENRY '16-Jan-2026 12:00 UTC' LAT: 34.4 LON: 7.8

# Date and time of the FIRST timestep you want to analyse.
# Files before this date are silently skipped.
# Format: 'DD-Mon-YYYY HH:MM UTC'  (e.g. '04-Sep-2023 12:00 UTC')
START_DATE = '04-Sep-2023 12:00 UTC'  # required: set to the first timestep you want to analyse

END_DATE   = '11-Sep-2023 12:00 UTC'  # optional: set to None to process all available files after START_DATE

# Approximate cyclone center at START_DATE (degrees North / East)
# This is just a first guess, the algorithm will correct it at the first time step.
# Use any kind of map to estimate these coordinates (they don't have to be exact, but they should be inside the SEARCH_RADIUS_KM at START_DATE to avoid losing the track).
S0LAT = 38.1   # initial latitude  [°N]
S0LON = 20.5  # initial longitude [°E]

# Search radius for the tracking algorithm [km]
# Recommended: 50-200 km (Mediterranean, suggested 150km), 300-500 km (Atlantic/large systems, lower R for higher-resolution models)
# This is the radius of the circle within which the algorithm looks for the cyclone center at each time step.
# If the cyclone moves more than this distance between time steps, the track will be lost.
# So choose a value that is larger than the expected maximum displacement of the cyclone between time steps, but not too large to avoid confusion with other nearby systems or noise.
SEARCH_RADIUS_KM = 150

# Pressure levels at which geopotential height is tracked [hPa]
# INTERP_LEVELS_HPA = np.array([800, 850, 900, 950])
# Sea level pressure is always included as level 0 (SLP).
# Leave empty to track only SLP.
#INTERP_LEVELS_HPA = None
INTERP_LEVELS_HPA = np.array([800, 850, 900, 950])
#INTERP_LEVELS_HPA = np.arange(500, 1025, 25)  # track every 25 hPa from 500 to 1000 hPa (inclusive)

# Toggle which position estimate(s) to use for tracking.
# At least one of these must be True.
TRACK_CENTROID = True
TRACK_MINIMUM = False

# percentile threshold for core isolation in locate_center (suggested: 5th percentile)
PERCENTILE_THRESHOLD = 5  

# number of time steps for rolling mean smoothing of the track (odd integer, recommended: 3-5)
# if SMOOTHING_WINDOW = 1, no smoothing is applied and the raw track is exported.
SMOOTHING_WINDOW = 3  

# add a "land_sea" column to the output CSV with "land" or "sea" for each time step, 
# based on the cyclone center location and a land mask generated from Natural Earth shapefiles.
LANDFALL_DETECTION = True  # True → create land mask

# --- 1.3  Diagnostic variables to export ------------------------------------
# Set each flag to True to include that variable in the output CSV.
# Variables that are unavailable for a given model are silently skipped
# (a warning is issued instead of crashing the program).
EXPORT_VARIABLES = {
    # minimum sea-level pressure inside search circle  [hPa] always available
    "max_sst":      True,   # maximum sea-surface temperature                  [K]
    "mean_sst":     True,   # mean    sea-surface temperature                  [K]
    "max_wind10m":  True,   # maximum 10-m wind speed                          [m/s]
    "max_lhf":      True,   # maximum latent heat flux                         [W/m²]
    "mean_lhf":     True,   # mean    latent heat flux                         [W/m²]
    "max_shf":      True,   # maximum sensible heat flux                       [W/m²]
    "mean_shf":     True,   # mean    sensible heat flux                       [W/m²]
    "max_qvf":      True,   # maximum water-vapour flux                        [kg/m²/s]
    "mean_qvf":     True,   # mean    water-vapour flux                        [kg/m²/s]
    "mean_pw":      True,   # mean    precipitable water                       [mm]
    "max_pvo":      True,   # maximum potential vorticity at 300 hPa           [PVU]
    "max_rh":       True,   # maximum 2-m relative humidity                    [%]
    "max_rain":     True,   # maximum hourly accumulated rainfall              [mm/h]
    "mean_rain":    True,   # mean    hourly accumulated rainfall              [mm/h]
}

# --- 1.4  Output options -----------------------------------------------------
PLOT            = True    # True → generate PNG maps at each time step to evaluate the estimated track and the cyclone structure (overwrites previous plots for the same simulation)
DO_EXPORT_VARIABLES = False    # True → compute and export diagnostic variables in the output CSV
SAVE_ALL_TRACKS = True   # True → also save per-level CSV files (larger output)
CHECK_PLOTS     = False   # True → generate per-timestep SLP and GPH check plots
##########################################################################
#### BE AWARE ############################################################
##########################################################################
#### Activating CHECK_PLOTS will generate a large number of PNG files ####
#### for each simulation you may generate up to 2GB of files #############
#### It also significantly increases the runtime of the script ###########
#### Use it only for debugging purposes ##################################
##########################################################################


# =============================================================================
# SECTION 1b – SENSITIVITY OVERRIDE  (populated by HIMPACT_SENS.py; no-op otherwise)
# When the environment variable HIMPACT_SENS_JSON points to a valid JSON file
# written by the sensitivity runner, the tracking parameters and MODELS_CONFIG
# are overridden here before Section 3 processes them.  Running the script
# standalone (without the env var) is completely unaffected.
# =============================================================================
import json as _json
_sens_json = os.environ.get("HIMPACT_SENS_JSON", "")
if _sens_json and os.path.isfile(_sens_json):
    with open(_sens_json) as _f:
        _ov = _json.load(_f)
    SEARCH_RADIUS_KM     = _ov.get("SEARCH_RADIUS_KM",     SEARCH_RADIUS_KM)
    _lvls                = _ov.get("INTERP_LEVELS_HPA")       # None  or  list[int]
    INTERP_LEVELS_HPA    = np.array(_lvls, dtype=int) if _lvls is not None else None
    PERCENTILE_THRESHOLD = _ov.get("PERCENTILE_THRESHOLD", PERCENTILE_THRESHOLD)
    TRACK_CENTROID       = _ov.get("TRACK_CENTROID",       TRACK_CENTROID)
    TRACK_MINIMUM        = _ov.get("TRACK_MINIMUM",        TRACK_MINIMUM)
    if "MODELS_CONFIG" in _ov:
        MODELS_CONFIG    = [_ov["MODELS_CONFIG"]]
    del _ov, _lvls
del _json, _sens_json
# =============================================================================


# ==========================================================================================================================================================================================
# ==========================================================================================================================================================================================
# ==========================================================================================================================================================================================

# =============================================================================
# SECTION 2 - HELPER FUNCTIONS
# =============================================================================

# ANSI 24-bit color tags for log prefixes (defined here so helper functions can use them)
COLOR_OK = "\033[38;2;0;170;0m"
COLOR_INFO = "\033[38;2;0;126;126m"
COLOR_WARN = "\033[38;2;170;170;0m"
COLOR_ERR = "\033[38;2;170;0;0m"
COLOR_RESET = "\033[0m"
INFO_TAG = f"[{COLOR_INFO}INFO{COLOR_RESET}]"
WARNING_TAG = f"[{COLOR_WARN}WARNING{COLOR_RESET}]"
ERROR_TAG = f"[{COLOR_ERR}ERROR{COLOR_RESET}]"


def log_exception(context_message):
    """
    Print a standardized error block that always shows the exact file, line
    number and function where the exception occurred (via traceback).

    Parameters
    ----------
    context_message : str
        A short human-readable description of what the code was trying to do.
    """
    # traceback.format_exc() gives the full stack, including the exact line
    # inside the function where the error was raised, which greatly simplifies debugging.
    print(
        f"{ERROR_TAG}\n"
        f"{context_message}\n"
        f"{traceback.format_exc()}"
    )


def check_and_create_folder(folder_path):
    """
    Ensure a directory exists, creating it (including parents) if necessary.

    Parameters
    ----------
    folder_path : str
        Absolute or relative path to the directory.

    Raises
    ------
    OSError
        If the directory cannot be created (e.g. insufficient permissions).
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)   # makedirs creates all intermediate dirs
            print(f"{INFO_TAG} Created output folder: {folder_path}")
        except OSError:
            # log_exception prints the exact failing line for easier debugging
            log_exception(
                f"Could not create folder '{folder_path}'. "
                "Check that the parent directory exists and is writable."
            )
            raise   # re-raise so the calling code can decide whether to abort


def haversine(lat_array, lon_array, lat_center, lon_center):
    """
    Compute the great-circle distance [km] from a single center point to every
    point in a 1-D or 2-D lat/lon array.

    Uses the Haversine formula instead of geodesic distance because it is
    ~100 times faster and gives indistinguishable results at the scales used here.

    Parameters
    ----------
    lat_array, lon_array : np.ndarray
        Arrays of latitude and longitude (degrees) for all grid points.
        Shapes must match.
    lat_center, lon_center : float
        Center point (degrees).

    Returns
    -------
    np.ndarray
        Distance in kilometres, same shape as lat_array / lon_array.
    """
    EARTH_RADIUS_KM = 6371.0   # mean radius of a spherical Earth

    lat1 = np.radians(lat_center)
    lon1 = np.radians(lon_center)
    lat2 = np.radians(lat_array)
    lon2 = np.radians(lon_array)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def normalize_longitudes(longitudes):
    """Normalize longitudes to the range [-180, 180] (used by MPAS and ERA5)."""
    return np.where(longitudes > 180, longitudes - 360, longitudes)


def locate_center(var, mask, lats, lons,
                  varname=None, step_date=None, model=None, sim=None,
                  t=None, checkout=None, make_check_plots=False):
    """
    Find the centroid and minimum of *var* inside the search circle defined by
    *mask*.

    The centroid is computed as the geometric centroid of the convex hull
    formed by the points that fall below the PERCENTILE_THRESHOLD-th percentile
    of *var* within the mask.  This makes the algorithm robust to isolated
    noisy pixels.

    Parameters
    ----------
    var : array-like
        2-D field to analyse (e.g. geopotential height or SLP).
        Must be convertible to a NumPy array.
    mask : np.ndarray of bool
        True where the grid point lies inside the search circle.
    lats, lons : np.ndarray
        Latitude and longitude arrays with the same shape as *var*.
    varname : str, optional
        Name of the field (used only for the check plots and log messages).
    step_date : str, optional
        Timestamp string (used only for the check plots and log messages).
    model, sim : str, optional
        Model and simulation labels (used only for the check-plot filenames).
    t : int, optional
        Time-step index (used only for the check-plot filenames).
    checkout : str, optional
        Output directory for the check plots.
    make_check_plots : bool
        When True, generate the debug check plots.

    Returns
    -------
    lat_centroid, lon_centroid : float
        Centroid of the convex hull built from the lowest values.
    lat_minimum, lon_minimum : float
        Location of the absolute minimum of *var* inside the mask.
    minimum_value, threshold : float
        Absolute minimum value and the percentile threshold used.

    Raises
    ------
    ValueError
        If the ConvexHull computation fails.
    """

    # Convert to a plain NumPy array regardless of whether var is an
    # xarray.DataArray (ICON) or a wrf-python masked array (WRF)
    var_np = np.asarray(var)
    flat_mask = np.asarray(mask).flatten()
    flat_var = var_np.flatten()[flat_mask]  # only values inside the mask
    flat_lons = np.asarray(lons).flatten()[flat_mask]
    flat_lats = np.asarray(lats).flatten()[flat_mask]

    # Mask points outside the search circle with NaN
    var_masked = np.where(mask, var_np, np.nan)

    # Find the percentile threshold to isolate the low-value core
    threshold = np.nanpercentile(var_masked, PERCENTILE_THRESHOLD)

    # Build a secondary mask for the core region (below threshold)
    core_mask = var_masked < threshold

    # Extract latitudes and longitudes of core points
    lats_core = np.where(core_mask, lats, np.nan)
    lons_core = np.where(core_mask, lons, np.nan)

    valid = ~np.isnan(lons_core) & ~np.isnan(lats_core)
    lon_valid = lons_core[valid].flatten()
    lat_valid = lats_core[valid].flatten()
    var_valid = var_masked[valid].flatten()

    points = np.column_stack((lon_valid, lat_valid))
    points_unique = np.unique(points, axis=0)  # drop exact duplicates

    if points_unique.shape[0] < 3:
        print(f"{WARNING_TAG} locate_center: Not enough unique points ({points_unique.shape[0]}) to compute a convex hull; "
              f"falling back to arithmetic mean.")
        lat_centroid = float(np.nanmean(lat_valid))
        lon_centroid = float(np.nanmean(lon_valid))
        hull = None
    else:
        center = points_unique.mean(axis=0)
        rank = np.linalg.matrix_rank(points_unique - center)
        if rank < 2:
            print(f"{WARNING_TAG} locate_center: Points are collinear (rank={rank}); "
                  f"falling back to arithmetic mean.")
            lat_centroid = float(np.nanmean(lat_valid))
            lon_centroid = float(np.nanmean(lon_valid))
            hull = None
        else:
            try:
                hull = ConvexHull(points_unique)
                poly = Polygon(points_unique[hull.vertices])
                lat_centroid = poly.centroid.y
                lon_centroid = poly.centroid.x
            except Exception as exc:
                # Numerically degenerate even though rank check passed —
                # fall back instead of aborting process_levels for this level/time
                print(f"{WARNING_TAG} locate_center: ConvexHull failed ({type(exc).__name__}: {exc}); "
                    f"falling back to arithmetic mean.")
                lat_centroid = float(np.nanmean(lat_valid))
                lon_centroid = float(np.nanmean(lon_valid))
                hull = None

    # Location of the absolute minimum inside the full search circle
    min_idx = np.unravel_index(np.nanargmin(var_masked), var_masked.shape)
    lat_minimum = float(lats[min_idx])
    lon_minimum = float(lons[min_idx])

    minimum_value = float(np.nanmin(var_masked))  # for diagnostic output

    if make_check_plots:
        print(
            f"Variable: {varname}\n"
            f"Step date: {step_date}\n"
            f"locate_center debug info:\n"
            f"  Number of core points: {len(lon_valid)}\n"
            f"  {PERCENTILE_THRESHOLD}th percentile threshold: {threshold:.2f}\n"
            f"  Centroid: (LAT: {lat_centroid:.2f}, LON: {lon_centroid:.2f})\n"
            f"  Minimum: (LAT: {lat_minimum:.2f}, LON: {lon_minimum:.2f}), value={minimum_value:.2f}\n"
        )
        # Plot the variable with the full search circle (not masked), to check the context of the minimum and centroid
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        dl = 0.1  # degree padding for map extent
        ax.set_extent([np.nanmin(flat_lons) - dl, np.nanmax(flat_lons) + dl, np.nanmin(flat_lats) - dl, np.nanmax(flat_lats) + dl])
        im = ax.tricontourf(flat_lons, flat_lats, flat_var, levels=np.linspace(np.nanmin(flat_var), np.nanmax(flat_var), 101), cmap='nipy_spectral', transform=ccrs.PlateCarree())
        # Add polygon outline
        if len(lon_valid) >= 3 and hull is not None:
            # attach first point to the end to close the polygon
            points_hull = np.vstack([points[hull.vertices], points[hull.vertices[0]]])
            ax.plot(points_hull[:, 0], points_hull[:, 1], color='orange', linestyle='--', linewidth=3, label='Convex Hull')
        plt.colorbar(im, ax=ax, label=f'{varname}')
        ax.scatter(lon_centroid, lat_centroid, edgecolors='#FFAAAA', facecolors='none', linewidths=1.5, marker='X', s=70, label='Centroid')
        ax.scatter(lon_minimum, lat_minimum, edgecolors='#AAFFFF', facecolors='none', linewidths=2, marker='D', s=50, label='Minimum')
        ax.set_title(f"locate_center debug {varname} \n {step_date}")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        # lat and lon ticks every 0.1 degree
        ax.set_xticks(np.arange(np.nanmin(flat_lons)-dl, np.nanmax(flat_lons)+dl, 0.1), crs=ccrs.PlateCarree(), minor=True)
        ax.set_yticks(np.arange(np.nanmin(flat_lats)-dl, np.nanmax(flat_lats)+dl, 0.1), crs=ccrs.PlateCarree(), minor=True)
        # end outside on the left of the plot area 
        #ax.legend(loc='center left', bbox_to_anchor=(-0.25, 0.5))
        ax.legend(loc='best', fontsize=8)
        # constrain the axes to be equal to avoid distortion of the search circle
        ax.set_aspect('equal', adjustable='datalim')
        plt.savefig(f"{checkout}/debug_check_locate_center/debug_check_locate_center_{model}_{sim}_{varname}_t{t:03d}.png", dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)

        # Plot the variable only where percentile threshold is met, with the search circle outline
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        dl = 0.1  # degree padding for map extent
        ax.set_extent([np.nanmin(lon_valid) - dl, np.nanmax(lon_valid) + dl, np.nanmin(lat_valid) - dl, np.nanmax(lat_valid) + dl])
        im = ax.tricontourf(lon_valid, lat_valid, var_valid, levels=np.linspace(np.nanmin(var_valid), np.nanmax(var_valid), 101), cmap='nipy_spectral', transform=ccrs.PlateCarree())
        # Add polygon outline
        if len(lon_valid) >= 3 and hull is not None:
            # attach first point to the end to close the polygon
            points_hull = np.vstack([points[hull.vertices], points[hull.vertices[0]]])
            ax.plot(points_hull[:, 0], points_hull[:, 1], color='orange', linestyle='--', linewidth=3, label='Convex Hull')
        plt.colorbar(im, ax=ax, label=f'{varname} (hPa)')
        ax.set_title(f"Convex Hull Polygon \n {PERCENTILE_THRESHOLD} percentile debug {varname} \n {step_date}")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        # lat and lon ticks every 0.1 degree
        ax.set_xticks(np.arange(np.nanmin(lon_valid)-dl, np.nanmax(lon_valid)+dl, 0.1), crs=ccrs.PlateCarree(), minor=True)
        ax.set_yticks(np.arange(np.nanmin(lat_valid)-dl, np.nanmax(lat_valid)+dl, 0.1), crs=ccrs.PlateCarree(), minor=True)
        # end outside on the left of the plot area 
        #ax.legend(loc='center left', bbox_to_anchor=(-0.25, 0.5))
        plt.savefig(f"{checkout}/debug_check_percentile/debug_check_percentile_{model}_{sim}_{varname}_t{t:03d}.png", dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)

    return lat_centroid, lon_centroid, lat_minimum, lon_minimum, minimum_value, threshold


def plot_check_slp(slp, lats, lons, slat, slon, latc, lonc, latm, lonm,
                   interp_levels, mask, step_date, model, sim, t, checkout):
    """
    Generate the per-timestep SLP check plot with the estimated center and the
    per-level centroid/minimum positions overlaid.  A separate legend image is
    generated and vertically merged onto the main plot.
    """
    interp_levels = np.atleast_1d(interp_levels)
    latc = np.atleast_1d(latc)
    lonc = np.atleast_1d(lonc)
    latm = np.atleast_1d(latm)
    lonm = np.atleast_1d(lonm)

    mask = np.asarray(mask).flatten()
    slp = np.asarray(slp).flatten()[mask]
    lons = np.asarray(lons).flatten()[mask]
    lats = np.asarray(lats).flatten()[mask]
    # PLOT CLOUD TRACKS
    dl = 0.1  # degree padding for map extent (rough conversion from km to degrees at mid-latitudes)
    min_lon = np.nanmin(lons) - dl
    max_lon = np.nanmax(lons) + dl
    min_lat = np.nanmin(lats) - dl
    max_lat = np.nanmax(lats) + dl
    # set colormap
    # levcolors = ['#887777cc','#cc0000cc','#dddd00cc','#00aa00cc','#00ffffff','#1111dddd','#880088cc','#000000cc']
    cmap = plt.get_cmap("nipy_spectral")
    # Extract 12 evenly-spaced colors with alpha=0.75
    num_colors = len(interp_levels)
    denom = max(num_colors - 1, 1)
    levcolors = np.flipud([(*cmap(i / denom)[:3], 1) for i in range(num_colors)])
    proj = ccrs.PlateCarree()
    # set figure
    fig, ax = plt.subplots(figsize=(10, 7.5), subplot_kw={'projection': proj})
    # Add coastlines and set extent
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=proj)
    ax.coastlines(resolution='10m', color='black', linewidth=2, zorder=3)
    cf = ax.tricontourf(lons, lats, slp, levels=100, cmap='Greys', transform=proj, zorder=2)
    # lat and lon ticks every 0.1 degree also on top and right of the plot
    ax.set_xticks(np.arange(min_lon, max_lon, 0.1), crs=proj, minor=True)
    ax.set_yticks(np.arange(min_lat, max_lat, 0.1), crs=proj, minor=True)
    plt.colorbar(cf, ax=ax, label='Sea-level pressure (hPa)')
    # Add track
    for lv in interp_levels:
        idx = int(np.argwhere(interp_levels == lv)[0, 0])
        if lv == 0:
            z = "slp"
        else:
            z = f"z{lv}"
        lcolor = levcolors[idx]
        if TRACK_CENTROID:
            ax.scatter(lonc[idx], latc[idx], transform=proj, color=lcolor, marker="o", s=25, label=f'avg {z}', zorder=4)
        if TRACK_MINIMUM:
            ax.scatter(lonm[idx], latm[idx], transform=proj, color=lcolor, marker="x" , s=25, label=f'min {z}', zorder=4)
    ax.scatter(slon, slat, transform=proj, color="#ee4400ff", marker="*", s=50, label=f'Center estimate', zorder=5)
    # legend outside on the left of the plot area 
    #ax.legend(loc='center left', bbox_to_anchor=(-0.25, 0.5))
    title = f"Cloud check \n {CYCLONE}_{model}_{sim} \n {step_date}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    # Save figure
    figname = f"{checkout}/debug_check_track/debug_check_track_{CYCLONE}_{model}_{sim}_t{t:03d}.png"
    plt.savefig(figname, dpi=300, format='png') #bbox_inches='tight', 
    plt.close(fig)

    # Plot ONLY legend
    handles, labels = ax.get_legend_handles_labels()
    fig_leg, ax_leg = plt.subplots(figsize=(10, 2.5))
    ax_leg.axis("off")
    ax_leg.legend(handles, labels, loc="center", ncol=4)
    figname_leg = f"{checkout}/debug_check_track/debug_check_track_{CYCLONE}_{model}_{sim}_t{t:03d}_legend.png"
    plt.savefig(figname_leg, dpi=300, format='png') #bbox_inches='tight', 
    plt.close(fig_leg)

    # than vertical merge of track_plot_all_cloud and its legend based on debug_check_track_ width
    img_plot = Image.open(figname)
    img_leg = Image.open(figname_leg)
    total_height = img_plot.height + img_leg.height
    merged_img = Image.new("RGB", (img_plot.width, total_height), (255, 255, 255))
    merged_img.paste(img_plot, (0, 0))
    merged_img.paste(img_leg, (0, img_plot.height))
    merged_img.save(figname)

    # Remove the separate legend file to save space
    os.remove(figname_leg)


def interplevel_hpa_native(field_cells_lev, pressure_pa_cells_lev, target_hpa):
    """Linear interpolation of a native MPAS field (nCells, nVertLevels) to hPa."""
    field = np.asarray(field_cells_lev, dtype=float)
    pressure_hpa = np.asarray(pressure_pa_cells_lev, dtype=float) / 100.0

    n_cells = field.shape[0]
    out = np.full(n_cells, np.nan, dtype=float)

    for i in range(n_cells):
        pp = pressure_hpa[i, :]
        ff = field[i, :]
        valid = np.isfinite(pp) & np.isfinite(ff)
        if valid.sum() < 2:
            continue

        pp = pp[valid]
        ff = ff[valid]
        order = np.argsort(pp)
        pp = pp[order]
        ff = ff[order]

        if target_hpa < pp[0] or target_hpa > pp[-1]:
            continue

        out[i] = np.interp(target_hpa, pp, ff)

    return out


def plot_track_timestep(plot_lats, plot_lons, plot_slp, actual_date, max_dom_ext,
                        model, sim, outfolder):
    """
    Plot the temporarily estimated track (colored by SLP) up to the current
    time step, so the user can monitor track quality while the run proceeds.
    """
    # find first step with valid data for plotting (in case of missing data at the first steps)
    lat_zero = plot_lats[np.isfinite(plot_lats)][0]
    lon_zero = plot_lons[np.isfinite(plot_lons)][0]
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={'projection': proj})
    # Add coastlines and set extent
    ax.set_extent(max_dom_ext, crs=proj)
    # Add the lat-lon grid
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='#777777ee', linestyle='--', x_inline=False, y_inline=False)
    # Customize gridline labels size
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    # Add background with ocean and land colors
    ax.add_feature(NaturalEarthFeature('physical', 'ocean', '10m', facecolor=COLORS['water'])) #'#87CEEB'
    ax.add_feature(NaturalEarthFeature('physical', 'land', '10m', facecolor=COLORS['land']))
    # Add track
    scatter = ax.scatter(
        plot_lons,
        plot_lats,
        transform=proj,
        c=plot_slp,
        s=70,  # Size inversely proportional to min_slp
        cmap="hot", 
        edgecolor="black", 
        linewidth=0.1, 
        label="Track CSLP",
        zorder=30,
        vmin=960, vmax=1010,  # Set color limits for SLP
    )
    ax.plot(plot_lons, plot_lats, transform=proj, color='#ee4400ff', linewidth=1.5, label='Track', zorder=25)
    ax.scatter(lon_zero, lat_zero, transform=proj, color="#009900ff", marker="^", s=40, zorder=35)
    # Set colorbar limits
    cbar_min = 960 # typical lower bound for cyclone SLP in hPa (adjust if needed)
    cbar_max = 1010 # typical upper bound for cyclone SLP in hPa (adjust if needed)
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_ticks(np.arange(cbar_min, cbar_max + 1, 2))  # ticks every 2 hPa
    cbar.set_label("CSLP [hPa]", fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    title = f"{CYCLONE} - {model} {sim} - Dummy Track and SLP until {actual_date}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    # Save figure
    figname = f"{outfolder}/{CYCLONE}_{model}_{sim}_track_plot_slp.png"
    plt.savefig(figname, dpi=100, bbox_inches='tight', format='png')
    plt.close(fig)  # Close the figure to free memory


# ---------------------------------------------------------------------------
# ERA5 GRIB helpers
# ---------------------------------------------------------------------------

def open_era5_surface_grib(grib_path):
    """Open the ERA5 surface GRIB file, returning (instantaneous_ds, accumulated_ds)."""
    base_kwargs = {"indexpath": ""}
    instant_kwargs = {"filter_by_keys": {"typeOfLevel": ["surface","meanSea","heightAboveGround","meansea","height"], "shortName": ["10u", "10v", "msl", "sst", "10wdir", "10si"], "dataType": "an"}}
    accum_kwargs = {"filter_by_keys": {"typeOfLevel": "surface", "shortName": "tp", "dataType": "fc"}}

    try:
        instant_ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={**base_kwargs, **instant_kwargs})
    except Exception:
        instant_ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={**base_kwargs, "filter_by_keys": {"typeOfLevel": ["surface","meanSea","heightAboveGround"]}})

    try:
        accum_ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={**base_kwargs, **accum_kwargs})
    except Exception:
        accum_ds = instant_ds

    return instant_ds, accum_ds


def open_era5_pressure_grib(grib_path):
    """Open the ERA5 pressure-level GRIB file and return the geopotential dataset."""
    base_kwargs = {"indexpath": ""}
    pressure_kwargs = {"filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "z", "dataType": "an"}}

    try:
        pressure_ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={**base_kwargs, **pressure_kwargs})
    except Exception:
        pressure_ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={**base_kwargs, "filter_by_keys": {"typeOfLevel": ["isobaricInhPa","pressure"]}})

    return pressure_ds


def era5_time_values(ds):
    """Return the 1-D array of time values for an ERA5 dataset."""
    if "time" in ds.coords:
        return np.atleast_1d(np.asarray(ds["time"].values))
    if "valid_time" in ds.coords:
        return np.atleast_1d(np.asarray(ds["valid_time"].values))
    raise KeyError("Could not find a time coordinate in the ERA5 dataset.")


def era5_select_field(ds, var_name, time_index=0, level_hpa=None):
    """Select a single time (and optional pressure level) slice from an ERA5 dataset."""
    field = ds[var_name]

    if "time" in field.dims:
        time_index = min(int(time_index), field.sizes["time"] - 1)
        field = field.isel(time=time_index)
    elif "valid_time" in field.dims:
        time_index = min(int(time_index), field.sizes["valid_time"] - 1)
        field = field.isel(valid_time=time_index)

    if level_hpa is not None:
        level_coord_name = None
        for candidate in ("isobaricInhPa", "level", "pressure"):
            if candidate in field.dims or candidate in field.coords:
                level_coord_name = candidate
                break

        if level_coord_name is not None:
            available_levels = np.asarray(ds[level_coord_name].values, dtype=float).reshape(-1)
            if available_levels.size > 0:
                level_index = int(np.argmin(np.abs(available_levels - float(level_hpa))))
                field = field.isel({level_coord_name: level_index})

    return np.asarray(field, dtype=float)


def extract_era5_tp_hourly_field(tp_da, target_time):
    """Select the ERA5 tp slice matching target_time and convert accumulations to hourly totals."""
    target_time = np.datetime64(pd.to_datetime(target_time).to_datetime64())

    if "valid_time" not in tp_da.coords:
        return np.asarray(tp_da, dtype=float)

    valid_time = np.asarray(tp_da["valid_time"].values).astype("datetime64[ns]")

    if valid_time.ndim == 2:
        matches = np.argwhere(valid_time == target_time)
        if matches.size == 0:
            flat_idx = int(np.argmin(np.abs(valid_time - target_time)))
            time_idx, step_idx = np.unravel_index(flat_idx, valid_time.shape)
        else:
            time_idx, step_idx = matches[0]

        current_tp = tp_da.isel(time=int(time_idx), step=int(step_idx))
        if int(step_idx) > 0 and "step" in tp_da.dims:
            previous_tp = tp_da.isel(time=int(time_idx), step=int(step_idx) - 1)
            return np.asarray(current_tp - previous_tp, dtype=float)
        return np.asarray(current_tp, dtype=float)

    if valid_time.ndim == 1:
        step_idx = int(np.argmin(np.abs(valid_time - target_time)))
        if "step" in tp_da.dims:
            current_tp = tp_da.isel(step=step_idx)
            if step_idx > 0:
                previous_tp = tp_da.isel(step=step_idx - 1)
                return np.asarray(current_tp - previous_tp, dtype=float)
            return np.asarray(current_tp, dtype=float)
        if "time" in tp_da.dims:
            return np.asarray(tp_da.isel(time=step_idx), dtype=float)

    return np.asarray(tp_da, dtype=float)

def plot_era5_diagnostics(timestep_str, min_slp, max_wind, max_tp, mean_tp, outfolder, CYCLONE, model, sim):
    """
    Generate a diagnostic plot for ERA5 data, showing minimum SLP, maximum wind speed,
    maximum total precipitation, and mean total precipitation over time.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot minimum SLP
    ax1.plot(timestep_str, min_slp, color='blue', marker='o', label='Min SLP (hPa)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Min SLP (hPa)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for wind speed
    ax2 = ax1.twinx()
    ax2.plot(timestep_str, max_wind, color='red', marker='x', label='Max Wind Speed (m/s)')
    ax2.set_ylabel('Max Wind Speed (m/s)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Create a third y-axis for total precipitation
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
    ax3.plot(timestep_str, max_tp, color='green', marker='s', label='Max Total Precipitation (mm)')
    ax3.plot(timestep_str, mean_tp, color='orange', marker='d', label='Mean Total Precipitation (mm)')
    ax3.set_ylabel('Total Precipitation (mm)', color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    
    lines = lines_1 + lines_2 + lines_3
    labels = labels_1 + labels_2 + labels_3
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f"ERA5 Diagnostics for {CYCLONE} - {model} {sim}")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the figure
    figname = f"{outfolder}/{CYCLONE}_{model}_{sim}_era5_diagnostics.png"
    plt.savefig(figname, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Shared tracking helpers (used by every model back-end)
# ---------------------------------------------------------------------------

def init_tracking_arrays(n_timesteps, n_levels):
    """
    Pre-allocate the per-level result arrays with NaN so that skipped time steps
    (before START_DATE) appear as NaN rather than zero.

    Returns a dictionary of arrays plus the timestep-string array, and (optionally)
    the landfall and dummy-plot arrays if the corresponding flags are enabled.
    """
    arrays = {
        "lat_centroid":     np.full((n_timesteps, n_levels), np.nan),
        "lon_centroid":     np.full((n_timesteps, n_levels), np.nan),
        "lat_minimum":      np.full((n_timesteps, n_levels), np.nan),
        "lon_minimum":      np.full((n_timesteps, n_levels), np.nan),
        "value_minimum":    np.full((n_timesteps, n_levels), np.nan),
        "value_percentile": np.full((n_timesteps, n_levels), np.nan),
        "timestep_str":     np.empty(n_timesteps, dtype="U21"),   # 21-char string
    }
    if LANDFALL_DETECTION:
        arrays["landsea"] = np.empty(n_timesteps, dtype="U4")  # will hold "land" or "sea"
    if PLOT:
        arrays["dummy_lats"] = np.full((n_timesteps,), np.nan)
        arrays["dummy_lons"] = np.full((n_timesteps,), np.nan)
        arrays["dummy_slp"]  = np.full((n_timesteps,), np.nan)
    return arrays


def update_center_estimate(arrays, t):
    """
    Update the running cyclone center estimate as the mean of the enabled
    position estimates (centroid and/or minimum) across all levels.  This makes
    the tracker more robust to outliers at individual levels.

    Returns
    -------
    slat, slon : float
        The new center estimate.
    """
    lat_components = []
    lon_components = []
    if TRACK_CENTROID:
        lat_components.append(arrays["lat_centroid"][t, :])
        lon_components.append(arrays["lon_centroid"][t, :])
    if TRACK_MINIMUM:
        lat_components.append(arrays["lat_minimum"][t, :])
        lon_components.append(arrays["lon_minimum"][t, :])
    slat = float(np.nanmean(np.concatenate(lat_components)))
    slon = float(np.nanmean(np.concatenate(lon_components)))
    return slat, slon


def process_levels(z_fields, interp_levels, mask, lats_grid, lons_grid,
                   arrays, t, step_date, model, sim, checkout):
    """
    Run locate_center for every level, store the results in *arrays* and return
    the updated center estimate.  This is the shared core of the four model
    back-ends.

    Parameters
    ----------
    z_fields : dict
        Mapping {level_value: 2-D field}.  Level 0 must hold the SLP field.
    interp_levels : np.ndarray
        Sorted array of levels to process (includes 0 for SLP).
    mask : np.ndarray of bool
        True inside the search circle.
    lats_grid, lons_grid : np.ndarray
        Latitude/longitude grids.
    arrays : dict
        Result arrays created by init_tracking_arrays.
    t : int
        Time-step index.
    step_date, model, sim, checkout :
        Metadata passed through to locate_center / check plots.

    Returns
    -------
    slat, slon, last_val_min : float
        Updated center estimate and the SLP minimum value at this step.
    """
    last_val_min = np.nan
    for lv in interp_levels:
        idx = int(np.argwhere(interp_levels == lv)[0, 0])
        z_field = z_fields[lv]  # field already prepared by the model branch

        # Variable name used only for the check-plot filenames and log messages
        varname = f"z{lv}" if lv != 0 else "slp"

        try:
            (lat_c,
             lon_c,
             lat_m,
             lon_m,
             val_min,
             val_perc) = locate_center(
                z_field, mask, lats_grid, lons_grid,
                varname=varname, step_date=step_date, model=model, sim=sim,
                t=t, checkout=checkout, make_check_plots=CHECK_PLOTS,
            )
        except Exception:
            # log_exception prints the exact failing line inside locate_center
            log_exception(
                f"locate_center failed at t={t + 1}, "
                f"level={'SLP' if lv == 0 else f'{lv} Pa/hPa'}."
            )
            raise

        # Blank out the estimate(s) that are not requested by the user
        if not TRACK_CENTROID:
            lat_c = np.nan
            lon_c = np.nan
        if not TRACK_MINIMUM:
            lat_m = np.nan
            lon_m = np.nan

        arrays["lat_centroid"][t, idx] = lat_c
        arrays["lon_centroid"][t, idx] = lon_c
        arrays["lat_minimum"][t, idx] = lat_m
        arrays["lon_minimum"][t, idx] = lon_m
        arrays["value_minimum"][t, idx] = val_min
        arrays["value_percentile"][t, idx] = val_perc
        last_val_min = val_min

    # -- Update the cyclone center estimate --
    slat, slon = update_center_estimate(arrays, t)
    return slat, slon, last_val_min


def finalize_timestep(arrays, t, slat, slon, val_min, step_date, model, sim,
                      max_dom_ext, outfolder, slp_for_plot, lats_grid, lons_grid,
                      interp_levels, mask, checkout, plot_arrays=None):
    """
    Common per-time-step post-processing: dummy plot, landfall detection, NaN
    guard and CHECK_PLOTS.  *plot_arrays* is an optional (lats, lons, slp) tuple
    of growing lists for models that collect data dynamically (ERA5).
    """
    # Update the running dummy track plot (if enabled)
    if PLOT:
        if plot_arrays is None:
            arrays["dummy_lats"][t] = slat
            arrays["dummy_lons"][t] = slon
            arrays["dummy_slp"][t] = val_min
            plot_track_timestep(arrays["dummy_lats"], arrays["dummy_lons"],
                                arrays["dummy_slp"], step_date, max_dom_ext,
                                model, sim, outfolder)
        else:
            plot_lats, plot_lons, plot_slp = plot_arrays
            plot_track_timestep(np.asarray(plot_lats, dtype=float),
                                np.asarray(plot_lons, dtype=float),
                                np.asarray(plot_slp, dtype=float),
                                step_date, max_dom_ext, model, sim, outfolder)

    # Land/sea classification of the center location
    if LANDFALL_DETECTION:
        point = Point(slon, slat)
        arrays["landsea"][t] = "land" if any(geom.contains(point) for geom in land_geoms) else "sea"

    # Abort if the center could not be located at all
    if np.isnan(slat) or np.isnan(slon):
        print(
            f"{ERROR_TAG}\n"
            f"All levels returned NaN at t={t + 1}. "
        )
        raise ValueError("All levels returned NaN for cyclone center.")

    # Full SLP check plot (if enabled)
    if CHECK_PLOTS:
        plot_check_slp(slp_for_plot, lats_grid, lons_grid, slat, slon,
                       arrays["lat_centroid"][t, :].squeeze(),
                       arrays["lon_centroid"][t, :].squeeze(),
                       arrays["lat_minimum"][t, :].squeeze(),
                       arrays["lon_minimum"][t, :].squeeze(),
                       interp_levels, mask, step_date, model, sim, t, checkout)


def compute_weighted_mean_track(arrays, interp_levels):
    """
    Combine the per-level centroid/minimum positions into a single weighted-mean
    track (equal weights by default).  Returns (mlat, mlon, lats_all, lons_all).
    """
    # Stack centroid/minimum latitude/longitude arrays based on enabled options
    track_lat_arrays = []
    track_lon_arrays = []
    if TRACK_CENTROID:
        track_lat_arrays.append(arrays["lat_centroid"])
        track_lon_arrays.append(arrays["lon_centroid"])
    if TRACK_MINIMUM:
        track_lat_arrays.append(arrays["lat_minimum"])
        track_lon_arrays.append(arrays["lon_minimum"])

    lats_all = np.concatenate(track_lat_arrays, axis=1)
    lons_all = np.concatenate(track_lon_arrays, axis=1)

    # Define weights (all equal by default; customise here if needed)
    weightss = np.ones(lats_all.shape[1])
    # If both centroid and minimum are enabled, centroid weights occupy
    # the first n_levels and minimum weights the next n_levels.
    # Example: give double weight to centroid levels:
    #   weightss[:n_levels] = 2
    # Example: give triple weight to the SLP minimum (when minimum is enabled):
    #   idx_slp = int(np.argwhere(interp_levels == 0)[0, 0])
    #   weightss[n_levels + idx_slp] = 3

    weights = weightss / np.sum(weightss)   # normalise so weights sum to 1

    # Weighted mean latitude and longitude at each time step
    mlat = np.dot(lats_all, weights.reshape(-1, 1)).flatten()   # shape (n_t,)
    mlon = np.dot(lons_all, weights.reshape(-1, 1)).flatten()
    return mlat, mlon, lats_all, lons_all


def detect_landfall(track_csv_path):
    """
    Identify the first landfall (3 sea points followed by 3 land points) in a
    saved track CSV.  Returns (landfall_date, landfall_id) or (None, None).
    """
    track = pd.read_csv(track_csv_path)
    land_sea_series = track["land_sea"]
    for i in range(3, len(land_sea_series) - 3):
        if (land_sea_series.iloc[i] == "land" and
            all(land_sea_series.iloc[i-j] == "sea" for j in range(1, 4)) and
            all(land_sea_series.iloc[i+j] == "land" for j in range(1, 4))):
            landfall_date = track["date"].iloc[i]
            print(f"{INFO_TAG}    Landfall detected at {landfall_date}")
            return landfall_date, i
    print(f"{INFO_TAG}    No clear landfall detected based on the 3-sea-3-land criterion.")
    return None, None


def plot_track_and_slp(track_csv_path, outfolder, model, sim, proj,
                       title_suffix, figname_suffix, landfall_id, landfall_date):
    """
    Plot a track colored by min_slp (used for both the raw and the smoothed
    tracks; the two differ only by the input CSV, the title and the filename).
    """
    track = pd.read_csv(track_csv_path)
    lons = track["lon"].values
    lats = track["lat"].values
    slp = track["min_slp"].values
    if np.all(np.isnan(slp)):
        print(f"{WARNING_TAG} min_slp not available; skipping SLP-colored track plot.")
        return

    fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={'projection': proj})
    # Add coastlines and set extent
    ax.set_extent([np.nanmin(lons)-1, np.nanmax(lons)+1, np.nanmin(lats)-1, np.nanmax(lats)+1], crs=proj)
    # Add the lat-lon grid
    ax.gridlines(draw_labels=True, linewidth=0.5, color='#777777ee', linestyle='--', x_inline=False, y_inline=False)
    # Add background with ocean and land colors
    ax.add_feature(NaturalEarthFeature('physical', 'ocean', '10m', facecolor=COLORS['water'])) #'#87CEEB'
    ax.add_feature(NaturalEarthFeature('physical', 'land', '10m', facecolor=COLORS['land']))
    # Add track
    scatter = ax.scatter(
        lons,
        lats,
        transform=proj,
        c=slp,
        s=80,  # Size inversely proportional to min_slp
        cmap="hot", 
        edgecolor="black", 
        linewidth=0.1, 
        label="Track CSLP",
        zorder=30
    )
    ax.plot(lons, lats, transform=proj, color='#ee4400ff', linewidth=2, label='Track', zorder=25)
    ax.scatter(lons[0], lats[0]+0.03, transform=proj, color="#009900ff", marker="^", s=60, label=f'First valid point', zorder=35)
    ax.scatter(lons[-1], lats[-1], transform=proj, color="#000099ff", marker="s", s=60, label=f'Last valid point', zorder=35)
    if LANDFALL_DETECTION and landfall_id is not None:
        # white box at 0.5 alpha and black text with date
        ax.text(lons[landfall_id]+0.1, lats[landfall_id]+0.1, f'Landfall\n{landfall_date}', fontsize=8, fontweight='bold', ha='left', va='bottom', zorder=40, bbox=dict(facecolor='white', alpha=0.4, edgecolor=None, boxstyle='round'))
    # Set colorbar limits
    cbar_min = int(np.floor(np.nanmin(slp)))
    cbar_max = int(np.ceil(np.nanmax(slp)))
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_ticks(np.arange(cbar_min, cbar_max+1, dtype=np.int32))
    cbar.set_label("CSLP [hPa]", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    title = f"{CYCLONE} - {model} {sim} - Track and SLP{title_suffix}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    # Save figure
    figname = f"{outfolder}/{CYCLONE}_{model}_{sim}_track_plot_slp{figname_suffix}.png"
    plt.savefig(figname, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)  # Close the figure to free memory

    print(f"{COLOR_OK}Track plot saved: {figname}{COLOR_RESET}")


# ====================================================================================
# SECTION 3 - INTERNAL CONFIGURATION (DO NOT EDIT UNLESS YOU KNOW WHAT YOU ARE DOING)
# ====================================================================================

if INTERP_LEVELS_HPA is None:
    INTERP_LEVELS_HPA = np.array([], dtype=int)
    print(f"{INFO_TAG} No pressure levels specified for tracking. Only SLP will be tracked (level 0).")
else:
    INTERP_LEVELS_HPA = np.array(INTERP_LEVELS_HPA, dtype=int)
    print(f"{INFO_TAG} Pressure levels specified for tracking: {INTERP_LEVELS_HPA} hPa (plus 0 hPa for SLP).")


if TRACK_CENTROID and TRACK_MINIMUM:
    print(f"{INFO_TAG} Both centroid and minimum tracking enabled. The output CSV will include both estimates for each level.")
elif TRACK_CENTROID and not TRACK_MINIMUM:
    print(f"{INFO_TAG} Only centroid tracking enabled. The output CSV will include only the centroid estimate for each level.")
elif TRACK_MINIMUM and not TRACK_CENTROID:
    print(f"{INFO_TAG} Only minimum tracking enabled. The output CSV will include only the minimum estimate for each level.")
else:
    # This case is already ruled out by the earlier check, but we include it for completeness.
    print(f"{ERROR_TAG} Invalid tracking configuration. Both TRACK_CENTROID and TRACK_MINIMUM cannot be False.")
    raise ValueError("Invalid tracking configuration. Set TRACK_CENTROID and/or TRACK_MINIMUM to True.")

MODELS = [cfg["model"] for cfg in MODELS_CONFIG]
SIMS = [cfg["sim"] for cfg in MODELS_CONFIG]
INFOLDERS = [cfg["infolder"] for cfg in MODELS_CONFIG]
OUTFOLDERS = [cfg["outfolder"] for cfg in MODELS_CONFIG]

if "ICON" in MODELS:
    try:
        import xarray as xr  # type: ignore # noqa: F401
        import metpy.calc as mpcalc  # type: ignore # noqa: F401
        from metpy.units import units  # type: ignore # noqa: F401
    except ImportError as e:
        print(f"{ERROR_TAG} Failed to import ICON dependencies: {e}")
        raise

if "WRF" in MODELS:
    try:
        from netCDF4 import Dataset  # type: ignore # noqa: F401
        from wrf import getvar, vinterp, latlon_coords  # type: ignore # noqa: F401
    except ImportError as e:
        print(f"{ERROR_TAG} Failed to import WRF dependencies: {e}")
        raise

if "MPAS" in MODELS:
    try:
        import xarray as xr  # type: ignore # noqa: F401
        import metpy.calc as mpcalc  # type: ignore # noqa: F401
        from metpy.units import units  # type: ignore # noqa: F401
    except ImportError as e:
        print(f"{ERROR_TAG} Failed to import MPAS dependencies: {e}")
        raise

if "ERA5" in MODELS:
    try:
        import xarray as xr  # type: ignore # noqa: F401
        import cfgrib  # type: ignore # noqa: F401
    except ImportError as e:
        print(f"{ERROR_TAG} Failed to import ERA5 dependencies: {e}")
        raise

if LANDFALL_DETECTION:
    land_shp = shpreader.natural_earth(resolution='10m', category='physical', name='land')
    land_geoms = list(shpreader.Reader(land_shp).geometries())


def check_and_create_outfolder(folder):
    """Create the output folder (and, if CHECK_PLOTS is on, the debug sub-folders)."""
    check_and_create_folder(f"{folder}")
    checkout = None
    if CHECK_PLOTS:
        checkout = os.path.join(folder, "check_plots")
        check_and_create_folder(checkout)
        check_and_create_folder(f"{checkout}/debug_check_locate_center")
        check_and_create_folder(f"{checkout}/debug_check_percentile")
        check_and_create_folder(f"{checkout}/debug_check_track")
    return checkout


# =============================================================================
# SECTION 4 - MAIN BRANCH
# =============================================================================

for sim_idx, (model, sim, infolder, outfolder) in enumerate(zip(MODELS, SIMS, INFOLDERS, OUTFOLDERS)):
    checkout = check_and_create_outfolder(outfolder)

    if model == "ICON":

        print(f"{INFO_TAG}    Processing simulation '{sim}' with model '{model}'")

        use_pressure_levels = INTERP_LEVELS_HPA.size > 0

        # Generate the filelist for Model Levels
        print(f"{INFO_TAG}    Generating ml_filelist")
        ml_filelist = sorted([f"{infolder}/{f}"
                            for f in os.listdir(infolder) if "_ML_" in f])
        if not ml_filelist:
            print(f"{ERROR_TAG} No *_ML_* files found for in '{infolder}'.")
            raise FileNotFoundError(f"No *_ML_* files found for in '{infolder}'.")

        if use_pressure_levels:
            # Generate the filelist for Pressure Levels
            print(f"{INFO_TAG}    Generating pl_filelist")
            pl_filelist = sorted([f"{infolder}/{f}"
                                for f in os.listdir(infolder) if "_PL_" in f])
            if not pl_filelist:
                print(f"{ERROR_TAG} No *_PL_* files found for in '{infolder}'.")
                raise FileNotFoundError(f"No *_PL_* files found for in '{infolder}'.")

            # find levels in ICON included in desired interp_levels
            try:
                interp_levels = INTERP_LEVELS_HPA.copy()
                ncfile_pl = xr.open_dataset(pl_filelist[1])
                ncfile_pl = ncfile_pl.metpy.parse_cf().squeeze()
                plev = ncfile_pl["plev"].values/100 # convert from Pa to hPa
                p_indices = np.unique(np.array([np.argmin(np.abs(plev - level)) for level in interp_levels])).astype(int)
                interp_levels = plev[p_indices]
                interp_levels = np.append(interp_levels, 0) # add 0 back as the SLP placeholder
            except Exception:
                log_exception(
                    f"Could not read ICON pressure levels from '{pl_filelist[1]}'."
                )
                raise
        else:
            interp_levels = np.array([0], dtype=int)
            p_indices = np.array([], dtype=int)

        interp_levels = np.unique(interp_levels)  # ensure unique levels
        print(f"{INFO_TAG}    ICON pressure levels mapped (hPa) - {len(interp_levels)}: levels{interp_levels.astype(int)}")

        # define max domain extent for plotting
        max_dom_ext = np.empty((4,), dtype=float)
        max_dom_ext[0] = np.rad2deg(np.nanmin(xr.open_dataset(ml_filelist[1])["clon"].values))
        max_dom_ext[1] = np.rad2deg(np.nanmax(xr.open_dataset(ml_filelist[1])["clon"].values))
        max_dom_ext[2] = np.rad2deg(np.nanmin(xr.open_dataset(ml_filelist[1])["clat"].values))
        max_dom_ext[3] = np.rad2deg(np.nanmax(xr.open_dataset(ml_filelist[1])["clat"].values))

        # ---- Initialise result arrays ----------------------------
        n_timesteps = len(ml_filelist)
        n_levels = len(interp_levels)
        arrays = init_tracking_arrays(n_timesteps, n_levels)

        # Running cyclone center estimate (updated each time step)
        slat, slon = S0LAT, S0LON

        tic = time.perf_counter()
        for t in range(n_timesteps):
            # Read the date from the current file (assumes time is a coordinate in the NetCDF file)
            step_date = pd.to_datetime(xr.open_dataset(ml_filelist[t])["time"].values).strftime("%d-%b-%Y %H:%M UTC")
            step_date = step_date[-1]
            # Skip until reaching the start date
            if pd.to_datetime(step_date) < pd.to_datetime(START_DATE):
                continue
            if END_DATE is not None:
                if pd.to_datetime(step_date) > pd.to_datetime(END_DATE):
                    break

            print(f"{INFO_TAG}    Model '{model}', simulation '{sim}'")
            print(f"{INFO_TAG}    Processing time step {t+1}/{n_timesteps} - {step_date}")
            # Add printing information on the total percentage progress of the whole script
            if n_timesteps > 0:
                sim_progress = (t + 1) / n_timesteps * 100
                total_progress = (sim_idx + sim_progress / 100) / len(MODELS) * 100
                print(f"{INFO_TAG}    Simulation progress: {sim_progress:.1f}%")
                print(f"{INFO_TAG}    Total progress: {total_progress:.1f}%")

            try:
                # Open files
                ncfile_ml = xr.open_dataset(ml_filelist[t])
                ncfile_ml = ncfile_ml.metpy.parse_cf().squeeze()

                if use_pressure_levels:
                    ncfile_pl = xr.open_dataset(pl_filelist[t])
                    ncfile_pl = ncfile_pl.metpy.parse_cf().squeeze()
            except Exception:
                log_exception(f"Could not open files for time step {t} ('{step_date}').")
                continue  # skip to next time step

            # Extract the variables and prepare the per-level field dictionary
            z_fields = {}
            if use_pressure_levels:
                try:
                    gph = ncfile_pl["geopot"] # Geopotential for the Mass Grid [m2/s2]
                except Exception:
                    log_exception(f"Could not extract Geopotential for time step {t} ('{step_date}').")
                    continue  # skip to next time step
                gph_levels = np.array(gph[p_indices, :])
                # Map each requested level to its interpolated geopotential field
                for lv in interp_levels:
                    if lv == 0:
                        try:
                            slp = ncfile_ml["pres_msl"]/100 # Sea-level pressure (hPa)
                            z_fields[lv] = slp
                        except Exception:
                            raise ValueError(f"Could not extract Sea-level pressure for time step {t} ('{step_date}').")
                    else:
                        idx = int(np.argwhere(interp_levels[interp_levels != 0] == lv)[0, 0])
                        z_fields[lv] = gph_levels[idx, :].squeeze()
            else:
                try:
                    slp = ncfile_ml["pres_msl"]/100 # Sea-level pressure (hPa)
                    z_fields[0] = slp
                except Exception:
                    raise ValueError(f"Could not extract Sea-level pressure for time step {t} ('{step_date}').")
            try:
                lons_grid, lats_grid = np.rad2deg(ncfile_ml["clon"]), np.rad2deg(ncfile_ml["clat"])
            except Exception:
                log_exception(f"Could not extract cyclone center coordinates for time step {t} ('{step_date}').")
                continue  # skip to next time step

            # Extract time
            arrays["timestep_str"][t] = pd.to_datetime(ncfile_ml["time"].values).strftime("%d-%b-%Y %H:%M UTC")

            # -- Compute haversine distances and build search mask --
            distances = haversine(lats_grid, lons_grid, slat, slon)
            mask = distances <= SEARCH_RADIUS_KM   # True inside the circle

            # -- Locate the cyclone center at each level and update the estimate --
            slat, slon, val_min = process_levels(
                z_fields, interp_levels, mask, lats_grid, lons_grid,
                arrays, t, step_date, model, sim, checkout
            )

            # -- Common per-time-step post-processing --
            finalize_timestep(
                arrays, t, slat, slon, val_min, step_date, model, sim,
                max_dom_ext, outfolder, slp, lats_grid, lons_grid,
                interp_levels, mask, checkout
            )

    elif model == "WRF":

        print(f"{INFO_TAG}    Processing simulation '{sim}' with model '{model}'")

        # Generate the filelist for WRF output files
        print(f"{INFO_TAG}    Generating filelist")
        filelist = sorted([f"{infolder}/{f}"
                            for f in os.listdir(infolder) if "wrfout" in f])
        if not filelist:
            print(f"{ERROR_TAG}    No 'wrfout' files found for in '{infolder}'.")
            raise FileNotFoundError(f"No 'wrfout' files found for in '{infolder}'.")

        interp_levels = INTERP_LEVELS_HPA.copy()
        # add 0 back as the SLP placeholder
        interp_levels = np.append(interp_levels, 0)
        pressure_levels = interp_levels[interp_levels != 0]

        # define max domain extent for plotting
        max_dom_ext = np.empty((4,), dtype=float)
        max_dom_ext[0] = np.nanmin(getvar(Dataset(filelist[1]), "XLONG").values)
        max_dom_ext[1] = np.nanmax(getvar(Dataset(filelist[1]), "XLONG").values)
        max_dom_ext[2] = np.nanmin(getvar(Dataset(filelist[1]), "XLAT").values)
        max_dom_ext[3] = np.nanmax(getvar(Dataset(filelist[1]), "XLAT").values)

        # ---- Initialise result arrays ----------------------------
        n_timesteps = len(filelist)
        n_levels = len(interp_levels)
        arrays = init_tracking_arrays(n_timesteps, n_levels)

        # Running cyclone center estimate (updated each time step)
        slat, slon = S0LAT, S0LON

        tic = time.perf_counter()
        for t in range(n_timesteps):
            # Read the date from the current file (assumes time is a coordinate in the NetCDF file)
            ncfile = Dataset(filelist[t])
            step_date = pd.to_datetime(getvar(ncfile, "times").values).strftime("%d-%b-%Y %H:%M UTC")
            # Skip until reaching the start date
            if pd.to_datetime(step_date) < pd.to_datetime(START_DATE):
                continue
            if END_DATE is not None:
                if pd.to_datetime(step_date) > pd.to_datetime(END_DATE):
                    break

            print(f"{INFO_TAG}    Model '{model}', simulation '{sim}'")
            print(f"{INFO_TAG}    Processing time step {t+1}/{n_timesteps} - {step_date}")
            # Add printing information on the total percentage progress of the whole script
            if n_timesteps > 0:
                sim_progress = (t + 1) / n_timesteps * 100
                total_progress = (sim_idx + sim_progress / 100) / len(MODELS) * 100
                print(f"{INFO_TAG}    Simulation progress: {sim_progress:.1f}%")
                print(f"{INFO_TAG}    Total progress: {total_progress:.1f}%")

            try:
                ncfile = Dataset(filelist[t])
                lats_grid, lons_grid = latlon_coords(getvar(ncfile, "slp"))
            except Exception:
                log_exception(
                    f"Could not open file or extract cyclone center coordinates for time step {t} ('{step_date}')."
                )
                continue  # skip to next time step

            # Extract time
            arrays["timestep_str"][t] = pd.to_datetime(getvar(ncfile, "times").values).strftime("%d-%b-%Y %H:%M UTC")

            # -- Compute haversine distances and build search mask --
            distances = haversine(lats_grid, lons_grid, slat, slon)
            mask = distances <= SEARCH_RADIUS_KM   # True inside the circle

            # Interpolate geopotential to the desired pressure levels (if any)
            if pressure_levels.size > 0:
                gph_levels = vinterp(
                    ncfile,
                    getvar(ncfile, "geopotential"),
                    'pressure',
                    pressure_levels,
                    extrapolate=True
                )

            # Prepare the per-level field dictionary
            slp_field = getvar(ncfile, "slp").metpy.unit_array
            z_fields = {}
            for lv in interp_levels:
                if lv == 0:
                    # SLP level: use sea-level pressure field
                    z_fields[lv] = slp_field
                else:
                    idx_level = np.argwhere(pressure_levels == lv)[0, 0]
                    # Geopotential level
                    z_fields[lv] = gph_levels[idx_level, :].squeeze().metpy.unit_array

            # -- Locate the cyclone center at each level and update the estimate --
            slat, slon, val_min = process_levels(
                z_fields, interp_levels, mask, lats_grid, lons_grid,
                arrays, t, step_date, model, sim, checkout
            )

            # -- Common per-time-step post-processing --
            finalize_timestep(
                arrays, t, slat, slon, val_min, step_date, model, sim,
                max_dom_ext, outfolder, slp_field, lats_grid, lons_grid,
                interp_levels, mask, checkout
            )

    elif model == "MPAS":

        print(f"{INFO_TAG}    Processing simulation '{sim}' with model '{model}'")

        use_pressure_levels = INTERP_LEVELS_HPA.size > 0

        # Generate the filelist for Model Levels
        print(f"{INFO_TAG}    Generating mpas_filelist")
        mpas_filelist = sorted([f"{infolder}/{f}"
                            for f in os.listdir(infolder) if "mpasout" in f])
        diag_filelist = sorted([f"{infolder}/{f}"
                            for f in os.listdir(infolder) if "diag." in f])

        if not mpas_filelist or not diag_filelist:
            print(f"{ERROR_TAG} No *mpasout* or *diag* files found for in '{infolder}'.")
            raise FileNotFoundError(f"No *mpasout* or *diag* files found for in '{infolder}'.")

        interp_levels = INTERP_LEVELS_HPA.copy()
        # add 0 back as the SLP placeholder
        interp_levels = np.append(interp_levels, 0)
        pressure_levels = interp_levels[interp_levels != 0]

        # ---- Initialise result arrays ----------------------------
        n_timesteps = len(diag_filelist)
        n_levels = len(interp_levels)
        arrays = init_tracking_arrays(n_timesteps, n_levels)

        # Running cyclone center estimate (updated each time step)
        slat, slon = S0LAT, S0LON

        # define max domain extent for plotting
        max_dom_ext = np.empty((4,), dtype=float)
        ncfile = xr.open_dataset(mpas_filelist[1])
        lons_grid, lats_grid = np.rad2deg(ncfile["lonCell"]), np.rad2deg(ncfile["latCell"])
        lons_grid = normalize_longitudes(lons_grid)
        max_dom_ext[0] = np.nanmin(lons_grid)
        max_dom_ext[1] = np.nanmax(lons_grid)
        max_dom_ext[2] = np.nanmin(lats_grid)
        max_dom_ext[3] = np.nanmax(lats_grid)

        tic = time.perf_counter()
        for t in range(n_timesteps):
            # Read the date from the current file (assumes time is a coordinate in the NetCDF file)
            step_date = pd.to_datetime(xr.open_dataset(diag_filelist[t])["xtime"].values[0].decode().strip(), format="%Y-%m-%d_%H:%M:%S").strftime("%d-%b-%Y %H:%M UTC")
            # Skip until reaching the start date
            if pd.to_datetime(step_date) < pd.to_datetime(START_DATE):
                continue
            if END_DATE is not None:
                if pd.to_datetime(step_date) > pd.to_datetime(END_DATE):
                    break

            print(f"{INFO_TAG}    Model '{model}', simulation '{sim}'")
            print(f"{INFO_TAG}    Processing time step {t+1}/{n_timesteps} - {step_date}")
            # Add printing information on the total percentage progress of the whole script
            if n_timesteps > 0:
                sim_progress = (t + 1) / n_timesteps * 100
                total_progress = (sim_idx + sim_progress / 100) / len(MODELS) * 100
                print(f"{INFO_TAG}    Simulation progress: {sim_progress:.1f}%")
                print(f"{INFO_TAG}    Total progress: {total_progress:.1f}%")

            try:
                # Open files
                ncfile_diag = xr.open_dataset(diag_filelist[t])
                ncfile_diag = ncfile_diag.metpy.parse_cf().squeeze()

                ncfile = xr.open_dataset(mpas_filelist[t])
                ncfile = ncfile.metpy.parse_cf().squeeze()
            except Exception:
                log_exception(f"Could not open files for time step {t} ('{step_date}').")
                continue  # skip to next time step

            # Extract the variables and prepare the per-level field dictionary
            z_fields = {}
            if 0 in interp_levels:
                try:
                    slp = ncfile_diag["mslp"]/100 # Sea-level pressure (hPa)
                    z_fields[0] = slp
                except Exception:
                    log_exception(f"Could not extract Sea-level pressure for time step {t} ('{step_date}').")
                    continue  # skip to next time step
            if use_pressure_levels:
                try:
                    z_mid_m = 0.5 * (ncfile["zgrid"][:, :-1] + ncfile["zgrid"][:, 1:])
                    gph = metpy.calc.height_to_geopotential(z_mid_m * units.m) # Geopotential for the Mass Grid [m2/s2]
                except Exception:
                    log_exception(f"Could not extract Geopotential for time step {t} ('{step_date}').")
                    continue  # skip to next time step
                # Interpolate geopotential to the desired pressure levels using the native MPAS vertical coordinate (which is hybrid sigma-pressure)
                print(f"{INFO_TAG}    Interpolating geopotential to pressure levels")
                gph_levels = np.array([interplevel_hpa_native(gph, ncfile["pressure"], level) for level in pressure_levels])
                for lv in interp_levels:
                    if lv == 0:
                        continue
                    idx = int(np.argwhere(interp_levels == lv)[0, 0])
                    z_fields[lv] = gph_levels[idx, :].squeeze()
            try:
                lons_grid, lats_grid = np.rad2deg(ncfile["lonCell"]), np.rad2deg(ncfile["latCell"])
                lons_grid = normalize_longitudes(lons_grid)
            except Exception:
                log_exception(f"Could not extract cyclone center coordinates for time step {t} ('{step_date}').")
                continue  # skip to next time step

            # Extract time
            arrays["timestep_str"][t] = pd.to_datetime(xr.open_dataset(diag_filelist[t])["xtime"].values[0].decode().strip(), format="%Y-%m-%d_%H:%M:%S").strftime("%d-%b-%Y %H:%M UTC")

            # -- Compute haversine distances and build search mask --
            distances = haversine(lats_grid, lons_grid, slat, slon)
            mask = distances <= SEARCH_RADIUS_KM   # True inside the circle

            # -- Locate the cyclone center at each level and update the estimate --
            slat, slon, val_min = process_levels(
                z_fields, interp_levels, mask, lats_grid, lons_grid,
                arrays, t, step_date, model, sim, checkout
            )

            # -- Common per-time-step post-processing --
            finalize_timestep(
                arrays, t, slat, slon, val_min, step_date, model, sim,
                max_dom_ext, outfolder, slp, lats_grid, lons_grid,
                interp_levels, mask, checkout
            )

    elif model == "ERA5":
        
        print(f"{INFO_TAG}    Processing simulation '{sim}' with model '{model}'")

        print(f"{INFO_TAG}    Generating grib filelist")
        filelist = sorted([
            f"{infolder}/{f}"
            for f in os.listdir(infolder)
            if f.lower().endswith((".grib", ".grib2", ".grb", ".grb2"))
        ])
        if not filelist:
            print(f"{ERROR_TAG} No GRIB files found for in '{infolder}'.")
            raise FileNotFoundError(f"No GRIB files found for in '{infolder}'.")

        use_pressure_levels = INTERP_LEVELS_HPA.size > 0

        # Identify the surface file (msl/winds/sst/tp) and the pressure-level file (z)
        surface_file = None
        pressure_file = None
        for grib_file in filelist:
            if surface_file is None:
                try:
                    instant_probe, _ = open_era5_surface_grib(grib_file)
                    if any(name in instant_probe.data_vars for name in ("msl", "10u", "10v", "sst", "tp")):
                        surface_file = grib_file
                except Exception:
                    pass

            if use_pressure_levels and pressure_file is None:
                try:
                    pressure_probe = open_era5_pressure_grib(grib_file)
                    if "z" in pressure_probe.data_vars:
                        pressure_file = grib_file
                except Exception:
                    pass

            if surface_file is not None and (not use_pressure_levels or pressure_file is not None):
                break

        if surface_file is None:
            print(f"{ERROR_TAG} Could not identify an ERA5 surface GRIB file in '{infolder}'.")
            raise FileNotFoundError(f"Could not identify an ERA5 surface GRIB file in '{infolder}'.")

        if use_pressure_levels and pressure_file is None:
            print(f"{ERROR_TAG} Could not identify an ERA5 pressure-level GRIB file in '{infolder}'.")
            raise FileNotFoundError(f"Could not identify an ERA5 pressure-level GRIB file in '{infolder}'.")

        try:
            instant_ds, accum_ds = open_era5_surface_grib(surface_file)
            pressure_ds = open_era5_pressure_grib(pressure_file) if use_pressure_levels else None
        except Exception:
            log_exception("Could not open ERA5 GRIB datasets.")
            raise

        try:
            time_values = era5_time_values(instant_ds)
        except Exception:
            log_exception(f"Could not read ERA5 time coordinates from '{surface_file}'.")
            raise

        # Map the requested levels onto the available ERA5 pressure levels
        if use_pressure_levels:
            pressure_coord_name = None
            for candidate in ("isobaricInhPa", "level", "pressure"):
                if candidate in pressure_ds.coords or candidate in pressure_ds.dims:
                    pressure_coord_name = candidate
                    break
            if pressure_coord_name is None:
                raise KeyError("Could not find an ERA5 pressure coordinate in the pressure-level dataset.")

            available_pressure_levels = np.asarray(pressure_ds[pressure_coord_name].values, dtype=float).reshape(-1)
            mapped_pressure_levels = np.array(
                [available_pressure_levels[np.argmin(np.abs(available_pressure_levels - level))] for level in INTERP_LEVELS_HPA],
                dtype=float,
            )
            interp_levels = np.append(mapped_pressure_levels.astype(int), 0)
        else:
            interp_levels = np.array([0], dtype=int)

        interp_levels = np.unique(interp_levels)
        print(f"{INFO_TAG}    ERA5 pressure levels mapped (Pa): {interp_levels[:-1].astype(int)}")
        n_levels = len(interp_levels)

        # ---- Initialise result arrays ----------------------------
        n_timesteps = len(time_values)
        arrays = init_tracking_arrays(n_timesteps, n_levels)
        # ERA5-specific diagnostic arrays (available directly from the surface file)
        era5_max_wind = np.full((n_timesteps,), np.nan)
        era5_max_tp = np.full((n_timesteps,), np.nan)
        era5_mean_tp = np.full((n_timesteps,), np.nan)

        # Growing lists used only to drive the running dummy plot for ERA5
        era5_dates = []
        era5_plot_lats = []
        era5_plot_lons = []
        era5_plot_slp = []

        # Domain extent for plotting (constant for ERA5)
        max_dom_ext = np.empty((4,), dtype=float)
        lats_grid = np.asarray(instant_ds["latitude"].values)
        lons_grid = np.asarray(instant_ds["longitude"].values)
        lons_grid = normalize_longitudes(lons_grid)
        max_dom_ext[0] = np.nanmin(lons_grid)
        max_dom_ext[1] = np.nanmax(lons_grid)
        max_dom_ext[2] = np.nanmin(lats_grid)
        max_dom_ext[3] = np.nanmax(lats_grid)

        slat, slon = S0LAT, S0LON

        tic = time.perf_counter()
        for t in range(n_timesteps):
            step_date = pd.to_datetime(time_values[t]).strftime("%d-%b-%Y %H:%M UTC")
            if pd.to_datetime(step_date) < pd.to_datetime(START_DATE):
                continue
            if END_DATE is not None and pd.to_datetime(step_date) > pd.to_datetime(END_DATE):
                break

            arrays["timestep_str"][t] = step_date

            print(f"{INFO_TAG}    Model '{model}', simulation '{sim}'")
            print(f"{INFO_TAG}    Processing time step {len(era5_dates)+1} - {step_date}")
            if n_timesteps > 0:
                sim_progress = (t + 1) / n_timesteps * 100
                total_progress = (sim_idx + sim_progress / 100) / len(MODELS) * 100
                print(f"{INFO_TAG}    Simulation progress: {sim_progress:.1f}%")
                print(f"{INFO_TAG}    Total progress: {total_progress:.1f}%")

            # Sea-level pressure (mandatory field)
            try:
                slp_field = era5_select_field(instant_ds, "msl", time_index=t) / 100.0
            except Exception:
                log_exception(f"Could not extract mean sea-level pressure for time step {t} ('{step_date}').")
                continue

            # 10-m wind speed (optional diagnostic)
            if "10u" in instant_ds.data_vars or "u10" in instant_ds.data_vars:
                u10_name = "10u" if "10u" in instant_ds.data_vars else "u10"
                v10_name = "10v" if "10v" in instant_ds.data_vars else "v10"
                u10 = era5_select_field(instant_ds, u10_name, time_index=t)
                v10 = era5_select_field(instant_ds, v10_name, time_index=t)
                wind10m = np.sqrt(np.asarray(u10, dtype=float) ** 2 + np.asarray(v10, dtype=float) ** 2)
            elif "10si" in instant_ds.data_vars or "si10" in instant_ds.data_vars:
                wind_name = "10si" if "10si" in instant_ds.data_vars else "si10"
                wind10m = era5_select_field(instant_ds, wind_name, time_index=t)
            else:
                if t == 0:
                    log_exception(f"Could not extract 10m wind fields for time step {t} ('{step_date}').")
                wind10m = np.full_like(slp_field, np.nan)

            # Total precipitation (optional diagnostic)
            if "tp" in accum_ds.data_vars:
                tp_source = accum_ds
                try:
                    tp_field = extract_era5_tp_hourly_field(tp_source["tp"], time_values[t]) * 1000.0
                except Exception:
                    log_exception(f"Could not extract ERA5 total precipitation for time step {t} ('{step_date}').")
                    tp_field = np.full_like(slp_field, np.nan)
            elif "tp" in instant_ds.data_vars:
                tp_source = instant_ds
                try:
                    tp_field = extract_era5_tp_hourly_field(tp_source["tp"], time_values[t]) * 1000.0
                except Exception:
                    log_exception(f"Could not extract ERA5 total precipitation for time step {t} ('{step_date}').")
                    tp_field = np.full_like(slp_field, np.nan)
            else:
                if t == 0:
                    log_exception(f"Could not find total precipitation field in ERA5 datasets !")
                tp_source = None

            # -- Compute haversine distances and build search mask --
            distances = haversine(lats_grid, lons_grid, slat, slon)
            mask = distances <= SEARCH_RADIUS_KM

            # Prepare the per-level field dictionary
            z_fields = {}
            for lv in interp_levels:
                if lv == 0:
                    z_fields[lv] = slp_field
                else:
                    z_fields[lv] = era5_select_field(pressure_ds, "z", time_index=t, level_hpa=lv)

            # -- Locate the cyclone center at each level and update the estimate --
            slat, slon, val_min = process_levels(
                z_fields, interp_levels, mask, lats_grid, lons_grid,
                arrays, t, step_date, model, sim, checkout
            )

            # Collect the running dummy-plot values (ERA5 grows lists dynamically)
            if PLOT:
                era5_plot_lats.append(slat)
                era5_plot_lons.append(slon)
                era5_plot_slp.append(float(np.nanmin(np.where(mask, slp_field, np.nan))))

            # -- Common per-time-step post-processing --
            finalize_timestep(
                arrays, t, slat, slon, val_min, step_date, model, sim,
                max_dom_ext, outfolder, slp_field, lats_grid, lons_grid,
                interp_levels, mask, checkout,
                plot_arrays=(era5_plot_lats, era5_plot_lons, era5_plot_slp) if PLOT else None
            )

            # Store the ERA5-specific diagnostics inside the search circle
            era5_dates.append(step_date)
            era5_max_wind[t] = float(np.nanmax(np.where(mask, wind10m, np.nan)))
            if tp_source is not None:
                era5_max_tp[t] = float(np.nanmax(np.where(mask, tp_field, np.nan)))
                era5_mean_tp[t] = float(np.nanmean(np.where(mask, tp_field, np.nan)))

        if not era5_dates:
            print(f"{ERROR_TAG}    Something went wrong for simulation '{sim}'")
            raise ValueError(f"No valid ERA5 timesteps were processed for simulation '{sim}'.")

        tac = time.perf_counter()
        total_time = tac - tic
        print(
            f"{COLOR_OK} \n Simulation {model}_{sim} completed in {total_time:.1f} s ({total_time / 60:.1f} min).\n {COLOR_RESET}"
        )

    else:
        print(f"{ERROR_TAG} Unsupported model '{model}' for simulation '{sim}'. Skipping.")
        continue  # skip to next simulation

    # -- Convenience references so the shared post-processing reads cleanly --
    lat_centroid = arrays["lat_centroid"]
    lon_centroid = arrays["lon_centroid"]
    lat_minimum = arrays["lat_minimum"]
    lon_minimum = arrays["lon_minimum"]
    value_minimum = arrays["value_minimum"]
    value_percentile = arrays["value_percentile"]
    timestep_str = arrays["timestep_str"]
    landsea = arrays.get("landsea")

    # -- Verify at least one valid position estimate exists --
    track_ok = False
    if TRACK_CENTROID and not (np.all(np.isnan(lat_centroid)) and np.all(np.isnan(lon_centroid))):
        track_ok = True
    if TRACK_MINIMUM and not (np.all(np.isnan(lat_minimum)) and np.all(np.isnan(lon_minimum))):
        track_ok = True

    if model in ["ICON", "WRF", "MPAS", "ERA5"] and track_ok:
        print(f"{INFO_TAG}    Computing weighted-mean track ...")

        # Combine per-level positions into a single weighted-mean track
        mlat, mlon, lats_all, lons_all = compute_weighted_mean_track(arrays, interp_levels)

        # Build the output dictionary (min_slp is always available at level 0)
        if 0 in interp_levels:
            idx_slp = int(np.argwhere(interp_levels == 0)[0, 0])
            min_slp = value_minimum[:, idx_slp].flatten()

        if model == "ERA5":
            mtrack_dict = {
                "date": timestep_str,
                "lat":  mlat,
                "lon":  mlon,
                "min_slp": min_slp,
                "max_wind": era5_max_wind,
                "max_tp": era5_max_tp if tp_source is not None else np.full_like(min_slp, np.nan),
                "mean_tp": era5_mean_tp if tp_source is not None else np.full_like(min_slp, np.nan),
            }
            plot_era5_diagnostics(
                timestep_str, min_slp, era5_max_wind, era5_max_tp, era5_mean_tp,
                outfolder, CYCLONE, model, sim
            )
            if LANDFALL_DETECTION:
                mtrack_dict["land_sea"] = landsea
        else:
            mtrack_dict = {
                "date": timestep_str,
                "lat":  mlat,
                "lon":  mlon,
                "min_slp": min_slp
            }
            if LANDFALL_DETECTION:
                mtrack_dict["land_sea"] = landsea

        # Ensure elements have same length
        for key, value in mtrack_dict.items():
            if len(value) != len(timestep_str):
                raise ValueError(f"Length mismatch for key '{key}': expected {len(timestep_str)}, got {len(value)}")

        mtrack_df = pd.DataFrame(mtrack_dict)
        # Drop rows where lat or lon is NaN (time steps that were fully skipped)
        mtrack_df = mtrack_df.dropna(subset=["lat", "lon"])
        mtrack_df = mtrack_df.reset_index(drop=True)

        fname = os.path.join(
            outfolder, f"{CYCLONE}_{model}_{sim}_track_multilevelz.csv"
        )
        mtrack_df.to_csv(fname, sep=",", index=False)
        print(f"{INFO_TAG}    Weighted-mean raw track saved: {fname}")

        if SMOOTHING_WINDOW > 1:
            mtrack_df["lat"] = mtrack_df["lat"].rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean()
            mtrack_df["lon"] = mtrack_df["lon"].rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean()
            fname = os.path.join(
                outfolder, f"{CYCLONE}_{model}_{sim}_track_multilevelz_smooth.csv"
            )
            mtrack_df.to_csv(fname, sep=",", index=False)
            print(f"{INFO_TAG}    Weighted-mean smoothed track saved: {fname}")

        # ---- End of inner loop ----------------------------------------
        tac = time.perf_counter()
        total_time = tac - tic
        print(
            f"{COLOR_OK} \n Simulation {model}_{sim} completed in {total_time:.1f} s ({total_time / 60:.1f} min).\n {COLOR_RESET}"
        )
    else:
        print(f"{ERROR_TAG}    Something went wrong for simulation '{sim}'")
        raise ValueError(f"All latitudes and longitudes are NaN for simulation '{sim}'.")

    # IDENTIFY LANDFALL
    landfall_date = None
    landfall_id = None
    if LANDFALL_DETECTION and track_ok:
        landfall_date, landfall_id = detect_landfall(
            os.path.join(outfolder, f"{CYCLONE}_{model}_{sim}_track_multilevelz.csv")
        )

    # =============================================================================
    # SECTION 5 - PLOT TRACK
    # =============================================================================

    # PLOT CLOUD TRACKS
    min_lon = np.nanmin(lons_all)
    max_lon = np.nanmax(lons_all)
    min_lat = np.nanmin(lats_all)
    max_lat = np.nanmax(lats_all)
    dl = 0.5
    # set colormap
    # levcolors = ['#887777cc','#cc0000cc','#dddd00cc','#00aa00cc','#00ffffff','#1111dddd','#880088cc','#000000cc']
    cmap = plt.get_cmap("nipy_spectral")
    # Extract 12 evenly-spaced colors with alpha=0.75
    num_colors = len(interp_levels)
    denom = max(num_colors - 1, 1)
    levcolors = np.flipud([(*cmap(i / denom)[:3], 0.75) for i in range(num_colors)])  # Imposta alpha = 0.75
    proj = ccrs.PlateCarree()
    # set figure
    fig, ax = plt.subplots(figsize=(15, 13), subplot_kw={'projection': proj})
    # Add coastlines and set extent
    ax.set_extent([min_lon-dl, max_lon+dl, min_lat-dl, max_lat+dl], crs=proj)
    ax.coastlines(resolution='10m', color='black', linewidth=2)
    # Add track
    for lv in interp_levels:
        idx = int(np.argwhere(interp_levels == lv)[0, 0])
        if lv == 0:
            z = "slp"
        else:
            z = f"z{lv}"
        lcolor = levcolors[idx]
        if TRACK_CENTROID:
            ax.scatter(lon_centroid[:,idx], lat_centroid[:,idx], transform=proj, color=lcolor, marker="o", s=25, label=f'avg {z}')
        if TRACK_MINIMUM:
            ax.scatter(lon_minimum[:,idx], lat_minimum[:,idx], transform=proj, color=lcolor, marker="x" , s=25, label=f'min {z}')
    ax.plot(mlon, mlat, transform=proj, color='#ee4400ff', linewidth=2, label='Mean track', zorder=30)
    #ax.legend(loc='upper right')
    # st = strack["date"][0]
    # et = strack["date"][-1]
    # title = f"Track {sim}    {st} - {et}"
    title = f"Cloud tracking at multiple levels {CYCLONE}_{model}_{sim}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    # Save figure
    figname = f"{outfolder}/{CYCLONE}_{model}_{sim}_track_plot_all_cloud.png"
    plt.savefig(figname, dpi=300, format='png') #bbox_inches='tight', 

    # Plot ONLY legend
    handles, labels = ax.get_legend_handles_labels()
    fig_leg, ax_leg = plt.subplots(figsize=(2, 15))
    ax_leg.axis("off")
    ax_leg.legend(handles, labels, loc="center", ncol=1)
    figname_leg = f"{outfolder}/{CYCLONE}_{model}_{sim}_track_plot_all_cloud_legend.png"
    plt.savefig(figname_leg, dpi=300, format='png') #bbox_inches='tight', 
    plt.close(fig_leg)

    plt.close(fig)

    # vertical merge of track_plot_all_cloud and its legend based on track_plot_all_cloud width
    img_plot = Image.open(figname)
    img_leg = Image.open(figname_leg)
    total_height = img_plot.height + img_leg.height
    merged_img = Image.new("RGB", (img_plot.width, total_height), (255, 255, 255))
    merged_img.paste(img_plot, (0, 0))
    merged_img.paste(img_leg, (0, img_plot.height))
    merged_img.save(f"{outfolder}/{CYCLONE}_{model}_{sim}_track_plot_all_cloud_with_legend.png")

    print(f"{COLOR_OK}Cloud plot saved: {figname}{COLOR_RESET}")

    # PLOT TRACK and SLP (smoothed track, if smoothing is enabled)
    if SMOOTHING_WINDOW > 1:
        plot_track_and_slp(
            os.path.join(outfolder, f"{CYCLONE}_{model}_{sim}_track_multilevelz_smooth.csv"),
            outfolder, model, sim, proj,
            title_suffix=" (Smoothed)", figname_suffix="_smooth",
            landfall_id=landfall_id, landfall_date=landfall_date
        )

    # PLOT TRACK and SLP (raw track)
    plot_track_and_slp(
        os.path.join(outfolder, f"{CYCLONE}_{model}_{sim}_track_multilevelz.csv"),
        outfolder, model, sim, proj,
        title_suffix="", figname_suffix="",
        landfall_id=landfall_id, landfall_date=landfall_date
    )

    # =============================================================================
    # SECTION 6 - SAVE ALL TRACKS DATA (OPTIONAL)
    # =============================================================================

    if SAVE_ALL_TRACKS:
        # Save all track data to separate CSV files, one per level.
        # Each level gets its own dictionary so columns are NOT overwritten.
        for idx, lv in enumerate(interp_levels):
            level_str = f"z{lv}" if lv != 0 else "slp"
            level_data_dict = {
                "date": timestep_str,
                "lat_centroid": lat_centroid[:, idx],
                "lon_centroid": lon_centroid[:, idx],
                "lat_minimum": lat_minimum[:, idx],
                "lon_minimum": lon_minimum[:, idx],
                "min_value": value_minimum[:, idx],
                f"{PERCENTILE_THRESHOLD}th_percentile": value_percentile[:, idx],
            }
            level_data_df = pd.DataFrame(level_data_dict)
            level_data_df.to_csv(
                os.path.join(outfolder, f"{CYCLONE}_{model}_{sim}_track_level_{level_str}.csv"),
                sep=",", index=False
            )

        print(f"{COLOR_OK} All track data saved for each level in separate CSV files. {COLOR_RESET}")


# =============================================================================
# SECTION 7 - EXTRACT ALL VARIABLES AT TRACK POINTS (OPTIONAL)
# =============================================================================

def masked_max_mean(field, mask, variables_list, selected_variables, t,
                    max_key=None, mean_key=None):
    """
    Apply the search-circle mask to *field* and store the requested max/mean
    statistics.  Keeps the per-model extraction code compact and consistent.
    """
    field_masked = np.where(mask, np.array(field), np.nan)
    if max_key is not None and max_key in variables_list:
        selected_variables[max_key][t] = np.nanmax(field_masked)
    if mean_key is not None and mean_key in variables_list:
        selected_variables[mean_key][t] = np.nanmean(field_masked)


if DO_EXPORT_VARIABLES:
    variables_list = [var for var, export in EXPORT_VARIABLES.items() if export]
    if len(variables_list) > 0:
        print(f"{INFO_TAG} Extracting variables at track points: {', '.join(variables_list)}")
        for sim_idx, (model, sim, infolder, outfolder) in enumerate(zip(MODELS, SIMS, INFOLDERS, OUTFOLDERS)):
            track = pd.read_csv(os.path.join(outfolder, f"{CYCLONE}_{model}_{sim}_track_multilevelz.csv"))
            if track.empty:
                print(f"{ERROR_TAG} Track file is empty for simulation '{sim}'. Skipping variable extraction.")
                continue
            timestep_str = track["date"].values
            # create empty dictionary to store selected variables at track points
            selected_variables = {var: np.full(len(timestep_str), np.nan) for var in variables_list}
            if model == "WRF":
                print(f"{INFO_TAG}    Processing simulation '{sim}' with model '{model}' for variable extraction")
                filelist = sorted([f"{infolder}/{f}" for f in os.listdir(infolder) if "wrfout" in f])
                lons, lats = np.array(getvar(Dataset(filelist[0]), "XLONG")), np.array(getvar(Dataset(filelist[0]), "XLAT"))
                # Find the file index that matches the first track date
                st = 0
                for t in range(len(filelist)):
                    step_date = pd.to_datetime(getvar(Dataset(filelist[t]), "times").values).strftime("%d-%b-%Y %H:%M UTC")
                    if pd.to_datetime(step_date) == pd.to_datetime(track["date"][0]):
                        st = t
                        break
                for t in range(len(track)):
                    print(f"{INFO_TAG}    Extracting variables for {model} {sim} at track point {t+1}/{len(track)}")
                    file_T = st + t
                    slat = track["lat"].values[t]
                    slon = track["lon"].values[t]
                    distances = haversine(lats, lons, slat, slon)
                    mask = distances <= SEARCH_RADIUS_KM
                    ncfile = Dataset(filelist[file_T])
                    # Calculate and store selected variables
                    if "max_sst" in variables_list or "mean_sst" in variables_list:
                        try:
                            sst = getvar(ncfile, "SST")  # Sea Surface Temperature
                            masked_max_mean(sst, mask, variables_list, selected_variables, t, "max_sst", "mean_sst")
                        except Exception:
                            print(f"{WARNING_TAG} 'SST' variable not found")
                            # remove 'max_sst' and 'mean_sst' from variables_list to avoid trying to extract them in future time steps
                            variables_list = [var for var in variables_list if var not in ["max_sst", "mean_sst"]]

                    if "max_wind10m" in variables_list:
                        try:
                            wind10m = getvar(ncfile, "uvmet10_wspd")  # Wind speed at 10m
                            masked_max_mean(wind10m, mask, variables_list, selected_variables, t, "max_wind10m")
                        except Exception:
                            print(f"{WARNING_TAG} 'uvmet10_wspd' variable not found")
                            variables_list = [var for var in variables_list if var != "max_wind10m"]

                    if "max_lhf" in variables_list or "mean_lhf" in variables_list:
                        try:
                            lhf = getvar(ncfile, "LH")  # Latent Heat Flux
                            masked_max_mean(lhf, mask, variables_list, selected_variables, t, "max_lhf", "mean_lhf")
                        except Exception:
                            print(f"{WARNING_TAG} 'LH' variable not found")
                            variables_list = [var for var in variables_list if var not in ["max_lhf", "mean_lhf"]]

                    if "max_shf" in variables_list or "mean_shf" in variables_list:
                        try:
                            shf = getvar(ncfile, "HFX")  # Sensible Heat Flux
                            masked_max_mean(shf, mask, variables_list, selected_variables, t, "max_shf", "mean_shf")
                        except Exception:
                            print(f"{WARNING_TAG} 'HFX' variable not found")
                            variables_list = [var for var in variables_list if var not in ["max_shf", "mean_shf"]]

                    if "max_qvf" in variables_list or "mean_qvf" in variables_list:
                        try:
                            qvf = getvar(ncfile, "QFX")  # Water Vapor Flux
                            masked_max_mean(qvf, mask, variables_list, selected_variables, t, "max_qvf", "mean_qvf")
                        except Exception:
                            print(f"{WARNING_TAG} 'QFX' variable not found")
                            variables_list = [var for var in variables_list if var not in ["max_qvf", "mean_qvf"]]

                    if "mean_pw" in variables_list:
                        try:
                            pw = getvar(ncfile, "pw")  # Precipitable Water
                            masked_max_mean(pw, mask, variables_list, selected_variables, t, mean_key="mean_pw")
                        except Exception:
                            print(f"{WARNING_TAG} 'pw' variable not found")
                            variables_list = [var for var in variables_list if var != "mean_pw"]

                    if "max_pvo" in variables_list:
                        try:
                            pvo_plev = getvar(ncfile, "pvo")  # Potential Vorticity
                            pvo = vinterp(ncfile, pvo_plev, 'pressure', [300], extrapolate=True)
                            pvo = pvo.squeeze()
                            masked_max_mean(pvo, mask, variables_list, selected_variables, t, "max_pvo")
                        except Exception:
                            print(f"{WARNING_TAG} 'pvo' variable not found")
                            variables_list = [var for var in variables_list if var != "max_pvo"]

                    if "max_rh" in variables_list:
                        try:
                            rh = getvar(ncfile, "rh2")  # Relative Humidity
                            masked_max_mean(rh, mask, variables_list, selected_variables, t, "max_rh")
                        except Exception:
                            print(f"{WARNING_TAG} 'rh2' variable not found")
                            variables_list = [var for var in variables_list if var != "max_rh"]

                    if "max_rain" in variables_list or "mean_rain" in variables_list:
                        try:
                            rain = np.array(getvar(ncfile, "RAINNC"))-np.array(getvar(Dataset(filelist[file_T-1]), "RAINNC"))  # Hourly rainfall
                            masked_max_mean(rain, mask, variables_list, selected_variables, t, "max_rain", "mean_rain")
                        except Exception:
                            print(f"{WARNING_TAG} 'RAINNC' variable not found or previous time step missing for rainfall calculation")
                            variables_list = [var for var in variables_list if var not in ["max_rain", "mean_rain"]]

            elif model == "ICON":
                print(f"{INFO_TAG}    Processing simulation '{sim}' with model '{model}'")
                ml_filelist = sorted([f"{infolder}/{f}" for f in os.listdir(infolder) if "_ML_" in f])
                pl_filelist = sorted([f"{infolder}/{f}" for f in os.listdir(infolder) if "_PL_" in f])
                # Map the requested pressure levels onto the ICON grid, starting from
                # the user-defined INTERP_LEVELS_HPA (not a stale outer-scope value).
                ncfile_pl = xr.open_dataset(pl_filelist[1])
                ncfile_pl = ncfile_pl.metpy.parse_cf().squeeze()
                plev = ncfile_pl["plev"].values/100  # convert from Pa to hPa
                p_indices = np.unique(np.array([np.argmin(np.abs(plev - level)) for level in INTERP_LEVELS_HPA])).astype(int)
                interp_levels = plev[p_indices]
                interp_levels = np.append(interp_levels, 0) # add 0 back as the SLP placeholder
                # Find the file index that matches the first track date
                st = 0
                for t in range(len(ml_filelist)):
                    ncfile_ml = xr.open_dataset(ml_filelist[t])
                    ncfile_ml = ncfile_ml.metpy.parse_cf().squeeze()
                    lons_grid, lats_grid = np.rad2deg(ncfile_ml["clon"]), np.rad2deg(ncfile_ml["clat"])
                    step_date = pd.to_datetime(ncfile_ml["time"].values).strftime("%d-%b-%Y %H:%M UTC")
                    if len(step_date) > 1 and len(step_date) < 21:
                        step_date = step_date[0]
                    if pd.to_datetime(step_date) == pd.to_datetime(track["date"][0]):
                        st = t
                        break
                for t in range(len(track)):
                    print(f"{INFO_TAG}    Extracting variables for {model} {sim} at track point {t+1}/{len(track)}")
                    file_T = st + t
                    slat = track["lat"].values[t]
                    slon = track["lon"].values[t]
                    distances = haversine(lats_grid, lons_grid, slat, slon)
                    mask = distances <= SEARCH_RADIUS_KM
                    ncfile_ml = xr.open_dataset(ml_filelist[file_T])
                    ncfile_ml = ncfile_ml.metpy.parse_cf().squeeze()
                    ncfile_pl = xr.open_dataset(pl_filelist[file_T])
                    ncfile_pl = ncfile_pl.metpy.parse_cf().squeeze()
                    # Calculate and store selected variables
                    if "max_sst" in variables_list or "mean_sst" in variables_list:
                        try:
                            sst = ncfile_ml["sst"]  # Sea Surface Temperature
                            masked_max_mean(sst, mask, variables_list, selected_variables, t, "max_sst", "mean_sst")
                        except Exception:
                            print(f"{WARNING_TAG} 'sst' variable not found in ML files ... trying PL files")
                            try:
                                sst = ncfile_pl["sst"]  # Alternative variable name for Sea Surface Temperature
                                masked_max_mean(sst, mask, variables_list, selected_variables, t, "max_sst", "mean_sst")
                            except Exception:
                                print(f"{WARNING_TAG} 'sst' variable not found")
                                variables_list = [var for var in variables_list if var not in ["max_sst", "mean_sst"]]

                    if "max_wind10m" in variables_list:
                        try:
                            wind10m = np.sqrt((ncfile_ml["u_10m"].values)**2 + (ncfile_ml["v_10m"].values)**2)  # Wind speed at 10m
                            masked_max_mean(wind10m, mask, variables_list, selected_variables, t, "max_wind10m")
                        except Exception:
                            print(f"{WARNING_TAG} 'u_10m' and 'v_10m' variables not found in ML files ... trying PL files")
                            try:
                                wind10m = np.sqrt((ncfile_pl["u_10m"].values)**2 + (ncfile_pl["v_10m"].values)**2)  # Alternative variable names for Wind speed at 10m
                                masked_max_mean(wind10m, mask, variables_list, selected_variables, t, "max_wind10m")
                            except Exception:
                                print(f"{WARNING_TAG} 'u_10m' and 'v_10m' variables not found")
                                variables_list = [var for var in variables_list if var != "max_wind10m"]

                    if "max_lhf" in variables_list or "mean_lhf" in variables_list:
                        try:
                            lhf = ncfile_ml["lhfl_s"]  # Latent Heat Flux
                            masked_max_mean(lhf, mask, variables_list, selected_variables, t, "max_lhf", "mean_lhf")
                        except Exception:
                            print(f"{WARNING_TAG} 'lhfl_s' variable not found in ML files ... trying PL files")
                            try:
                                lhf = ncfile_pl["lhfl_s"]  # Alternative variable name for Latent Heat Flux
                                masked_max_mean(lhf, mask, variables_list, selected_variables, t, "max_lhf", "mean_lhf")
                            except Exception:
                                print(f"{WARNING_TAG} 'lhfl_s' variable not found")
                                variables_list = [var for var in variables_list if var not in ["max_lhf", "mean_lhf"]]

                    if "max_shf" in variables_list or "mean_shf" in variables_list:
                        try:
                            shf = ncfile_ml["shfl_s"]  # Sensible Heat Flux
                            masked_max_mean(shf, mask, variables_list, selected_variables, t, "max_shf", "mean_shf")
                        except Exception:
                            print(f"{WARNING_TAG} 'shfl_s' variable not found in ML files ... trying PL files")
                            try:
                                shf = ncfile_pl["shfl_s"]  # Alternative variable name for Sensible Heat Flux
                                masked_max_mean(shf, mask, variables_list, selected_variables, t, "max_shf", "mean_shf")
                            except Exception:
                                print(f"{WARNING_TAG} 'shfl_s' variable not found")
                                variables_list = [var for var in variables_list if var not in ["max_shf", "mean_shf"]]

                    if "max_qvf" in variables_list or "mean_qvf" in variables_list:
                        try:
                            qvf = ncfile_ml["qvf"]  # Water Vapor Flux
                            masked_max_mean(qvf, mask, variables_list, selected_variables, t, "max_qvf", "mean_qvf")
                        except Exception:
                            print(f"{WARNING_TAG} 'qvf' variable not found in ML files ... trying PL files")
                            try:
                                qvf = ncfile_pl["qvf"]  # Alternative variable name for Water Vapor Flux
                                masked_max_mean(qvf, mask, variables_list, selected_variables, t, "max_qvf", "mean_qvf")
                            except Exception:
                                print(f"{WARNING_TAG} 'qvf' variable not found")
                                variables_list = [var for var in variables_list if var not in ["max_qvf", "mean_qvf"]]

                    if "mean_pw" in variables_list:
                        try:
                            dewpoint = metpy.calc.dewpoint_from_relative_humidity(
                                        ncfile_pl["temp"]*units.kelvin,
                                        ncfile_pl["rh"]/ 100.0)
                            dewpoint = np.where(mask, np.array(dewpoint), 1)*units.degC
                            plev = ncfile_pl["plev"]/100 * units.hPa
                            pw = np.full(dewpoint.shape, np.nan)
                            idx_i, idx_j = np.where(mask)
                            for ii, jj in zip(idx_i, idx_j):
                                dp = dewpoint[:, ii, jj].squeeze()
                                pw[:, ii, jj] = metpy.calc.precipitable_water(plev, dp)
                            selected_variables["mean_pw"][t] = np.nanmean(pw)
                        except Exception as exc:
                            print(f"{WARNING_TAG} Could not calculate precipitable water: {exc}")
                            variables_list = [var for var in variables_list if var != "mean_pw"]

                    if "max_pvo" in variables_list:
                        try:
                            plev = ncfile_pl["plev"]/100
                            idx300 = np.unique(np.argmin(np.abs(plev - 300))).astype(int)
                            pvo = ncfile_pl["pv"][idx300,:].squeeze()
                            masked_max_mean(pvo, mask, variables_list, selected_variables, t, "max_pvo")
                        except Exception as exc:
                            print(f"{WARNING_TAG} Could not calculate max PV: {exc}")
                            variables_list = [var for var in variables_list if var != "max_pvo"]

                    if "max_rh" in variables_list:
                        try:
                            rh = ncfile_ml["rh_2m"]  # Relative Humidity
                            masked_max_mean(rh, mask, variables_list, selected_variables, t, "max_rh")
                        except Exception:
                            print(f"{WARNING_TAG} 'rh_2m' variable not found in ML files ... trying PL files")
                            try:
                                rh = ncfile_pl["rh_2m"]  # Alternative variable name for Relative Humidity
                                masked_max_mean(rh, mask, variables_list, selected_variables, t, "max_rh")
                            except Exception:
                                print(f"{WARNING_TAG} 'rh_2m' variable not found")
                                variables_list = [var for var in variables_list if var != "max_rh"]

                    if "max_rain" in variables_list or "mean_rain" in variables_list:
                        try:
                            nc0 = xr.open_dataset(ml_filelist[file_T-1])
                            nc0 = nc0.metpy.parse_cf().squeeze()
                            rain = ncfile_ml["rain_gsp"].values - nc0["rain_gsp"].values  # Hourly rainfall
                            masked_max_mean(rain, mask, variables_list, selected_variables, t, "max_rain", "mean_rain")
                        except Exception:
                            print(f"{WARNING_TAG} 'rain_gsp' variable not found in ML files or previous time step missing for rainfall calculation ... trying PL files")
                            try:
                                nc0 = xr.open_dataset(pl_filelist[file_T-1])
                                nc0 = nc0.metpy.parse_cf().squeeze()
                                rain = ncfile_pl["rain_gsp"].values - nc0["rain_gsp"].values  # Alternative variable name for Hourly rainfall
                                masked_max_mean(rain, mask, variables_list, selected_variables, t, "max_rain", "mean_rain")
                            except Exception:
                                print(f"{WARNING_TAG} 'rain_gsp' variable not found or previous time step missing for rainfall calculation")
                                variables_list = [var for var in variables_list if var not in ["max_rain", "mean_rain"]]

            elif model == "MPAS":
                print(f"{INFO_TAG}    Processing simulation '{sim}' with model '{model}'")
                mpas_filelist = sorted([f"{infolder}/{f}" for f in os.listdir(infolder) if "mpasout" in f])
                diag_filelist = sorted([f"{infolder}/{f}" for f in os.listdir(infolder) if "diag" in f])
                # Find the file index that matches the first track date
                st = 0
                for t in range(len(mpas_filelist)):
                    ncfile = xr.open_dataset(mpas_filelist[t])
                    ncfile = ncfile.metpy.parse_cf().squeeze()
                    lons_grid, lats_grid = np.rad2deg(ncfile["lonCell"]), np.rad2deg(ncfile["latCell"])
                    step_date = pd.to_datetime(ncfile["xtime"].values).strftime("%d-%b-%Y %H:%M UTC")
                    if len(step_date) > 1 and len(step_date) < 21:
                        step_date = step_date[0]
                    if pd.to_datetime(step_date) == pd.to_datetime(track["date"][0]):
                        st = t
                        break
                for t in range(len(track)):
                    print(f"{INFO_TAG}    Extracting variables for {model} {sim} at track point {t+1}/{len(track)}")
                    file_T = st + t
                    slat = track["lat"].values[t]
                    slon = track["lon"].values[t]
                    distances = haversine(lats_grid, lons_grid, slat, slon)
                    mask = distances <= SEARCH_RADIUS_KM
                    ncfile_diag = xr.open_dataset(diag_filelist[file_T])
                    ncfile_diag = ncfile_diag.metpy.parse_cf().squeeze()
                    ncfile = xr.open_dataset(mpas_filelist[file_T])
                    ncfile = ncfile.metpy.parse_cf().squeeze()
                    # Calculate and store selected variables
                    if "max_sst" in variables_list or "mean_sst" in variables_list:
                        try:
                            sst = ncfile["sst"]  # Sea Surface Temperature
                            masked_max_mean(sst, mask, variables_list, selected_variables, t, "max_sst", "mean_sst")
                        except Exception:
                            print(f"{WARNING_TAG} 'sst' variable not found in mpas files ... trying diag files")
                            try:
                                sst = ncfile_diag["sst"]  # Alternative variable name for Sea Surface Temperature
                                masked_max_mean(sst, mask, variables_list, selected_variables, t, "max_sst", "mean_sst")
                            except Exception:
                                print(f"{WARNING_TAG} 'sst' variable not found")
                                variables_list = [var for var in variables_list if var not in ["max_sst", "mean_sst"]]

                    if "max_wind10m" in variables_list:
                        try:
                            wind10m = np.sqrt((ncfile["u10"].values)**2 + (ncfile["v10"].values)**2)  # Wind speed at 10m
                            masked_max_mean(wind10m, mask, variables_list, selected_variables, t, "max_wind10m")
                        except Exception:
                            print(f"{WARNING_TAG} 'u10' and 'v10' variables not found in mpas files ... trying diag files")
                            try:
                                wind10m = np.sqrt((ncfile_diag["u10"].values)**2 + (ncfile_diag["v10"].values)**2)  # Alternative variable names for Wind speed at 10m
                                masked_max_mean(wind10m, mask, variables_list, selected_variables, t, "max_wind10m")
                            except Exception:
                                print(f"{WARNING_TAG} 'u10' and 'v10' variables not found")
                                variables_list = [var for var in variables_list if var != "max_wind10m"]

                    if "max_lhf" in variables_list or "mean_lhf" in variables_list:
                        try:
                            lhf = ncfile["lh"]  # Latent Heat Flux
                            masked_max_mean(lhf, mask, variables_list, selected_variables, t, "max_lhf", "mean_lhf")
                        except Exception:
                            print(f"{WARNING_TAG} 'lh' variable not found in mpas files ... trying diag files")
                            try:
                                lhf = ncfile_diag["lh"]  # Alternative variable name for Latent Heat Flux
                                masked_max_mean(lhf, mask, variables_list, selected_variables, t, "max_lhf", "mean_lhf")
                            except Exception:
                                print(f"{WARNING_TAG} 'lh' variable not found")
                                variables_list = [var for var in variables_list if var not in ["max_lhf", "mean_lhf"]]

                    if "max_shf" in variables_list or "mean_shf" in variables_list:
                        try:
                            shf = ncfile["hfx"]  # Sensible Heat Flux (was mistakenly reading ncfile_ml before the fix)
                            masked_max_mean(shf, mask, variables_list, selected_variables, t, "max_shf", "mean_shf")
                        except Exception:
                            print(f"{WARNING_TAG} 'hfx' variable not found in mpas files ... trying diag files")
                            try:
                                shf = ncfile_diag["hfx"]  # Alternative variable name for Sensible Heat Flux
                                masked_max_mean(shf, mask, variables_list, selected_variables, t, "max_shf", "mean_shf")
                            except Exception:
                                print(f"{WARNING_TAG} 'hfx' variable not found")
                                variables_list = [var for var in variables_list if var not in ["max_shf", "mean_shf"]]

                    if "max_qvf" in variables_list or "mean_qvf" in variables_list:
                        try:
                            qvf = ncfile["qfx"]  # Water Vapor Flux
                            masked_max_mean(qvf, mask, variables_list, selected_variables, t, "max_qvf", "mean_qvf")
                        except Exception:
                            print(f"{WARNING_TAG} 'qfx' variable not found in mpas files ... trying diag files")
                            try:
                                qvf = ncfile_diag["qfx"]  # Alternative variable name for Water Vapor Flux
                                masked_max_mean(qvf, mask, variables_list, selected_variables, t, "max_qvf", "mean_qvf")
                            except Exception:
                                print(f"{WARNING_TAG} 'qfx' variable not found")
                                variables_list = [var for var in variables_list if var not in ["max_qvf", "mean_qvf"]]

                    if "mean_pw" in variables_list:
                        try:
                            pw = ncfile["pw"]  # Precipitable Water
                            masked_max_mean(pw, mask, variables_list, selected_variables, t, mean_key="mean_pw")
                        except Exception:
                            print(f"{WARNING_TAG} 'pw' variable not found in mpas files ... trying diag files")
                            try:
                                pw = ncfile_diag["pw"]  # Alternative variable name for Precipitable Water
                                masked_max_mean(pw, mask, variables_list, selected_variables, t, mean_key="mean_pw")
                            except Exception:
                                print(f"{WARNING_TAG} 'pw' variable not found")
                                variables_list = [var for var in variables_list if var != "mean_pw"]

                    if "max_pvo" in variables_list:
                        try:
                            pvo_300 = interplevel_hpa_native(ncfile["ertel_pv"].squeeze(), ncfile["pressure"].squeeze(), 300)  # Interpolate to 300 hPa
                            masked_max_mean(pvo_300, mask, variables_list, selected_variables, t, "max_pvo")
                        except Exception as exc:
                            print(f"{WARNING_TAG} Could not calculate max PV: {exc}")
                            variables_list = [var for var in variables_list if var != "max_pvo"]

                    if "max_rh" in variables_list:
                        try:
                            rh = metpy.calc.relative_humidity_from_specific_humidity(
                                        ncfile["pressure"][:,0].squeeze()*units.Pa,
                                        ncfile["t2m"]*units.degC,
                                        ncfile["q2"]/1000).to('percent')
                            masked_max_mean(rh, mask, variables_list, selected_variables, t, "max_rh")
                        except Exception:
                            print(f"{WARNING_TAG} error computing 'rh_2m' with mpas files ... trying diag files")
                            try:
                                rh = metpy.calc.relative_humidity_from_specific_humidity(
                                        ncfile_diag["pressure"][:,0].squeeze()*units.Pa,
                                        ncfile_diag["t2m"]*units.degC,
                                        ncfile_diag["q2"]/1000).to('percent')
                                masked_max_mean(rh, mask, variables_list, selected_variables, t, "max_rh")
                            except Exception:
                                print(f"{WARNING_TAG} 'rh_2m' failed to compute")
                                variables_list = [var for var in variables_list if var != "max_rh"]

                    if "max_rain" in variables_list or "mean_rain" in variables_list:
                        try:
                            nc0 = xr.open_dataset(mpas_filelist[file_T-1])
                            nc0 = nc0.metpy.parse_cf().squeeze()
                            rain = ncfile["rainnc"].values - nc0["rainnc"].values  # Hourly rainfall
                            masked_max_mean(rain, mask, variables_list, selected_variables, t, "max_rain", "mean_rain")
                        except Exception:
                            print(f"{WARNING_TAG} 'rainnc' variable not found in mpas files or previous time step missing for rainfall calculation ... trying diag files")
                            try:
                                nc0 = xr.open_dataset(diag_filelist[file_T-1])
                                nc0 = nc0.metpy.parse_cf().squeeze()
                                rain = ncfile_diag["rainnc"].values - nc0["rainnc"].values  # Alternative variable name for Hourly rainfall
                                masked_max_mean(rain, mask, variables_list, selected_variables, t, "max_rain", "mean_rain")
                            except Exception:
                                print(f"{WARNING_TAG} 'rainnc' variable not found or previous time step missing for rainfall calculation")
                                variables_list = [var for var in variables_list if var not in ["max_rain", "mean_rain"]]

            elif model == "ERA5":
                print(f"{INFO_TAG}    ERA5 track already contains the requested diagnostics. Skipping extra extraction.")
                continue

            # add variables to track dataframe and save
            for var in variables_list:
                track[var] = selected_variables[var]
            track.to_csv(os.path.join(outfolder, f"{CYCLONE}_{model}_{sim}_track_multilevelz.csv"), sep=",", index=False)

            print(
                f"{INFO_TAG} Variable extraction completed for simulation '{sim}'. \n"
                f"{COLOR_OK}Updated track file saved: {os.path.join(outfolder, f'{CYCLONE}_{model}_{sim}_track_multilevelz.csv')}{COLOR_RESET}"
                )

            # quick plot of some of the main variables 
            # check if min_slp, max_wind10m, mean_rain, max_rain, max_sst, max_shf, max_lhf
            # are in the variables list and plot them is stacked plots if they are
            available_vars = variables_list + ["min_slp"]  # add min_slp back for plotting if it was removed due to not found
            plt_vars = ["min_slp", "max_wind10m", "mean_rain", "max_rain", "max_sst", "max_shf", "max_lhf"]
            plt_cols = ["#000099ff", "#009900ff", "#005555ff", "#00AAAAff", "#990099ff", "#990000ff", "#994400ff"]
            plot_variables = [var for var in ["min_slp", "max_wind10m", "mean_rain", "max_rain", "max_sst", "max_shf", "max_lhf"] if var in variables_list or var == "min_slp"]
            plot_colors = [plt_cols[i] for i, var in enumerate(plt_vars) if var in plot_variables]
            x_tick_indices = np.linspace(0, len(track["date"])-1, num=11, dtype=int)
            if len(plot_variables) > 0:
                fig, axes = plt.subplots(len(plot_variables), 1, figsize=(18, 5*len(plot_variables)), sharex=True)
                for i, var in enumerate(plot_variables):
                    axes[i].plot(track["date"], track[var], linewidth=3, color=plot_colors[i], label=var)
                    axes[i].set_ylabel(var, fontsize=14, fontweight="bold")
                    axes[i].grid()
                    #axes[i].legend(loc="upper left", fontsize=12)
                    m = np.floor(np.nanmin(track[var]))
                    M = np.ceil(np.nanmax(track[var]))
                    ticks = ((np.round(np.linspace(m, M, num=5))).astype(int)).tolist()
                    axes[i].set_yticks(ticks)
                    axes[i].set_yticklabels(ticks, fontsize=12)
                    axes[i].set_ylim(m - 0.1*(M-m), M + 0.1*(M-m))
                    # x tick labels only for the last subplot
                    if i == len(plot_variables) - 1: 
                        x_ticks_lab = pd.to_datetime(track["date"].iloc[x_tick_indices]).dt.strftime("%d  %H:%M")
                        axes[i].set_xlabel("Date")
                        axes[i].set_xticks(track["date"].iloc[x_tick_indices])
                        axes[i].set_xticklabels(x_ticks_lab, rotation=90, ha="right", fontsize=12, fontweight="bold")
                    else:
                        axes[i].set_xticks(track["date"].iloc[x_tick_indices])
                        axes[i].set_xticklabels([])
                    if i == 0:
                        axes[i].set_title(f"{CYCLONE} - {model} - {sim} - Extracted Variables", fontsize=12, fontweight="bold")
                plt.savefig(os.path.join(outfolder, f"{CYCLONE}_{model}_{sim}_track_vars_plot.png"), bbox_inches="tight", dpi=300)
                plt.close()
                print(f"{INFO_TAG} Quick plot of extracted variables saved: {os.path.join(outfolder, f'{CYCLONE}_{model}_{sim}_track_vars_plot.png')}")

# ====================================================================================================================================================================================================================================================================================================================
# ==== END OF SCRIPT =================================================================================================================================================================================================================================================================================================
# ====================================================================================================================================================================================================================================================================================================================

# PRINT BOLD LARGE CHARACTERS IN #EE4400 "THAT'S ALL FOLKS !!!!"
print("\033[1m\033[38;2;238;68;0m" + "\n" + "="*73 + "\033[0m")
print("\033[1m\033[38;2;238;68;0m" + "        T H A T ' S    A L L    F O L K S    ! ! ! ! " + "\033[0m")
print("\033[1m\033[38;2;238;68;0m" + "="*73 + "\n" + "\033[0m")
