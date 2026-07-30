#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIMPACT_SENS_EVALUATION.py
===========================================================================
Post-processing tool for HIMPACT sensitivity-analysis output.

Reads all HIMPACT_SENS_* subdirectories produced by HIMPACT_SENS.py,
compares every track against a reference (observation or ensemble mean),
computes displacement statistics (mean error, RMSE, max/min), and
produces:
  - Track comparison map (all sensitivity runs + mean + observations)
  - Displacement time-series plot
  - RMSE bar chart ranked by simulation
  - Mean-and-spread map
  - CSV table of error metrics
  - Updated mean-track CSV enriched with min_slp / max_wind from the
    best-performing simulation

Author: Piero Serafini
License: MIT
"""

# from datetime import datetime
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
from datetime import datetime
import xarray as xr
import matplotlib  # to change backend
from cartopy.feature import NaturalEarthFeature
from cartopy.feature import COLORS

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import locale
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')  # On Linux/macOS

matplotlib.use('Agg')  # Use a non-interactive backend

###########################################################################################################################################
outfolder = "/path/to/your/output/postprocess"
os.makedirs(outfolder, exist_ok=True)
cyclone = "MY_CYCLONE"  # e.g. "QENDRESA"
model = "WRF"  # "ICON", "WRF", "MPAS", or "ERA5"
LEGEND_COLS = 4  # Number of columns in the legend
MERGE = "vertical"  # "vertical" or "horizontal" for legend merging
OBS_REF = False  # If True, use OBS as reference track; if False, use MEAN as reference track

from pathlib import Path
import pandas as pd
import re

trackfolder = Path("/path/to/your/sensitivity/output")  # Folder containing HIMPACT_SENS_* subdirectories

tracks = []
labels = []

pattern = re.compile(
    r'(?P<v>\d+)v_'
    r'(?P<km>\d+)km_'
    r'(?P<p>\d+)p_'
    r'C(?P<C>[ft])_'
    r'M(?P<M>[tf])'
)

def sort_key(path):
    label = path.name.replace("HIMPACT_SENS_", "")
    m = pattern.fullmatch(label)

    # directory with unexpected name → at the bottom
    if m is None:
        return (999, 999, 999, 999, 999)

    return (
        int(m.group("v")),
        int(m.group("km")),
        int(m.group("p")),
        0 if m.group("C") == "f" else 1,  # Cf, Ct
        0 if m.group("M") == "t" else 1   # Mt, Mf
    )

dirs = sorted(trackfolder.glob("HIMPACT_SENS_*"), key=sort_key)

for d in dirs:

    label = d.name.replace("HIMPACT_SENS_", "")
    csv_file = d / f"{cyclone}_{model}_{label}_track_multilevelz.csv"

    if not csv_file.exists():
        print(f"Missing: {csv_file}")
        continue

    try:
        tracks.append(pd.read_csv(csv_file))
        labels.append(label)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        continue

print(f"Loaded {len(tracks)} tracks")

# mode of len of tracks
lengths = [len(track) for track in tracks]
mode_length = max(set(lengths), key=lengths.count)
for i,track in enumerate(tracks):
    if len(track) != mode_length:
        print(f"[WARNING] Track {labels[i]} has length {len(track)} which is different from mode length {mode_length}. It will be deleted.")
        tracks.pop(i)
        labels.pop(i)


# add in the first slot of tracks the mean track of all tracks, and in the first slot of labels "MEAN"
mean_track = pd.DataFrame({
    'date': tracks[0]['date'],  # assuming all tracks have the same date array
    'lat': np.nanmean([track['lat'] for track in tracks], axis=0),
    'lon': np.nanmean([track['lon'] for track in tracks], axis=0)
})
tracks.insert(0, mean_track)
labels.insert(0, 'MEAN')

# save mean track to csv
mean_track.to_csv(f"{outfolder}/{cyclone}_{model}_MEAN_track_multilevelz.csv", index=False)
ref_track = tracks[0]  # assuming the first track is the reference (OBS)

#obs = pd.read_csv('/path/to/your/observations/track_obs.csv')
#obs = pd.read_csv('/path/to/your/observations/track_obs_v2.csv')
obs = pd.read_csv('/path/to/your/observations/observed_track.csv')
#obs = pd.read_csv('/path/to/your/observations/observed_track_from_BT.csv')
# transform obs date from 20260317140000 to 17-Mar-2026 14:00 UTC
# if date is a column in obs, and the format type is not datetime, then convert it to datetime
#if 'date' in obs.columns and not isinstance(obs['date'].iloc[0], pd.Timestamp):
#    obs['date'] = pd.to_datetime(obs['date'], format='%Y%m%d%H%M%S')
tracks.insert(0, obs)
labels.insert(0, 'OBS')

if OBS_REF:
    ref_track = tracks[0]  # assuming the first track is the reference (OBS)


###########################################################################################################################################
# DON'T USE GEODESIC BECAUSE WE HAVE SAME RESULT BUT 100x COMPUTING TIME
def haversine(lat_array, lon_array, lat_center, lon_center):
    """
    Calculate the great-circle distance (in km) from a center to all grid points.
    """
    R = 6371  # Radius of Spherical Earth in km
    lat1 = np.radians(lat_center)
    lon1 = np.radians(lon_center)
    lat2 = np.radians(lat_array)
    lon2 = np.radians(lon_array)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def find_date_idx(obs_date, track_date):
    """
    Find the index of the closest date in track_date to obs_date.
    """
    # Convert to datetime objects
    dt1 = pd.to_datetime(obs_date, utc=True)
    dt2 = pd.to_datetime(track_date, utc=True)

    idx = np.full(len(dt1), np.nan)
    for t in range(len(dt1)):
        # Calculate the absolute difference between the two dates
        diff = (dt2 - dt1[t]).dt.total_seconds().abs()
        # Find the index of the minimum difference
        idx[t] = diff.idxmin()

    return idx

###########################################################################################################################################

N_T = len(tracks)
cmap = plt.get_cmap("nipy_spectral_r")
num_colors = N_T+1
alpha = 1 # set transparency
c = np.flipud([(*cmap(i / (num_colors - 1))[:3], alpha) for i in range(num_colors)])

###########################################################################################################################################
# PLOT TRACKS
min_lon, max_lon, min_lat, max_lat = np.min([track["lon"].min() for track in tracks]) - 1, np.max([track["lon"].max() for track in tracks]) + 1, np.min([track["lat"].min() for track in tracks]) - 1, np.max([track["lat"].max() for track in tracks]) + 1
# set colormap
proj = ccrs.PlateCarree()
# set figure
fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'projection': proj})
# Add coastlines and set extent
ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=proj)
# Add the lat-lon grid
ax.gridlines(draw_labels=True, linewidth=0.5, color='#777777ee', linestyle='--', x_inline=False, y_inline=False)
# Add background with ocean and land colors
# ax.add_feature(NaturalEarthFeature('physical', 'ocean', '10m', facecolor=COLORS['water'])) #'#87CEEB'
# ax.add_feature(NaturalEarthFeature('physical', 'land', '10m', facecolor=COLORS['land']))
ax.add_feature(NaturalEarthFeature('physical', 'ocean', '10m', facecolor='#dddddd')) #'#87CEEB'
ax.add_feature(NaturalEarthFeature('physical', 'land', '10m', facecolor='#bbbbbb'))
if 'OBS' in labels:
    # find OBS idx in labels
    idxo = labels.index('OBS')
idxm = labels.index('MEAN')  # find MEAN idx in labels

for i in range(N_T):
    track = tracks[i]
    if i == idxm:
        ax.plot(track["lon"], track["lat"], transform=proj, color='#000000', linewidth=3, linestyle='--', label=labels[i], zorder=N_T+2)
    elif 'OBS' in labels[i] and i == idxo:
        ax.plot(track["lon"], track["lat"], transform=proj, color='#00000099', marker='o', linewidth=3, label=labels[i], zorder=N_T+1)
    else:
        ax.plot(track["lon"], track["lat"], transform=proj, color=c[i], linewidth=1, label=labels[i], zorder=N_T+1-i)
#ax.legend(loc='upper right')
#ax.set_title("(a)\n ", loc='left', fontsize=20, fontweight='bold')
# Save figure
figname = f"{outfolder}/{cyclone}_{model}_tracks_comparison.png"
plt.savefig(figname, dpi=300, bbox_inches='tight', format='png')
plt.close(fig)  # Close the figure to free memory

# Create a separate figure for the legend
fig_leg, ax_leg = plt.subplots(figsize=(5, 15))
ax_leg.axis('off')
legend_elements = []
for i in range(N_T):
    if i == idxm:
        # dashed line for MEAN
        legend_elements.append(Line2D([0], [0], color='#000000', linestyle='--', linewidth=3, label=labels[i]))
    elif 'OBS' in labels[i] and i == idxo:
        # Add a line with a marker for OBS
        legend_elements.append(Line2D([0], [0], color='#00000099', marker='o', linewidth=3, label=labels[i]))
    else:
        legend_elements.append(Patch(facecolor=c[i], label=labels[i]))
ax_leg.legend(handles=legend_elements, loc='upper left', fontsize=14, frameon=False, ncol=LEGEND_COLS)
figname_leg = f"{outfolder}/{cyclone}_{model}_tracks_comparison_legend.png"
plt.savefig(figname_leg, dpi=300, bbox_inches='tight', format='png')
plt.close(fig_leg)
print(f"[INFO] Saved separate legend: {figname_leg}")

# horizontal merge of track_plot_all_cloud and its legend based on track_plot_all_cloud width
if MERGE == "horizontal":
    img_plot = Image.open(figname)
    img_leg = Image.open(figname_leg)
    total_width = img_plot.width + img_leg.width
    merged_img = Image.new("RGB", (total_width, img_plot.height), (255, 255, 255))
    merged_img.paste(img_plot, (0, 0))
    merged_img.paste(img_leg, (img_plot.width, 0))
    merged_img.save(f"{outfolder}/{cyclone}_{model}_tracks_comparison_with_legend.png")
elif MERGE == "vertical":
    img_plot = Image.open(figname)
    img_leg = Image.open(figname_leg)
    total_height = img_plot.height + img_leg.height
    merged_img = Image.new("RGB", (img_plot.width, total_height), (255, 255, 255))
    merged_img.paste(img_plot, (0, 0))
    merged_img.paste(img_leg, (0, img_plot.height))
    merged_img.save(f"{outfolder}/{cyclone}_{model}_tracks_comparison_with_legend.png")

print(f"[INFO] Saved figure : \n{figname}")
###########################################################################################################################################
# Compute displacement in km

displacements = []
for i in range(1, len(tracks)):  # Start from 1 to skip ref_track
    # Find corresponding date in ref_track
    idx = find_date_idx(ref_track["date"], tracks[i]["date"])
    # Compute displacement
    ds = np.full(len(idx), np.nan)
    for j in range(len(idx)):
        ds[j] = haversine(ref_track["lat"][j], ref_track["lon"][j], tracks[i]["lat"][idx[j]], tracks[i]["lon"][idx[j]])
    displacements.append(ds)

###########################################################################################################################################
# PLOT displacement
# Set figure
fig, ax = plt.subplots(figsize=(18, 10))

# Plot displacements and fill areas
for i in range(1, len(tracks)):  # Start from 1 to skip ref_track
    mu = np.nanmean(displacements[i - 1])
    std = np.nanstd(displacements[i - 1])
    lbl = f"{labels[i]}: mean = {mu:.0f} km, std = {std:.0f} km"
    if 'MEAN' in labels[i] and i == idxm:
        ax.plot(ref_track["date"], displacements[i - 1], color='#000000', linewidth=3, linestyle='--', label=lbl, zorder=1)
        # spread area for MEAN+-std
        ax.fill_between(ref_track["date"], displacements[i - 1] - std, displacements[i - 1] + std, color='#000000', alpha=0.1, zorder=N_T)
    
    ax.plot(ref_track["date"], displacements[i - 1], color=c[i], linewidth=2, label=lbl, zorder=N_T - i)

#    ax.fill_between(ref_track["date"], 0, displacements[i - 1], color=c[i], alpha=0.7, zorder=N_T - len(tracks) - i)

# Add vertical line at landfall
#landfall_date = ref_track["date"][44]
#ax.axvline(x=landfall_date, color='#000000ff', linestyle='--', linewidth=4, label='Landfall', zorder=1)

# Set y-axis limits
max_displacement = max(np.nanmax(ds) for ds in displacements)
ax.set_ylim(0, max_displacement + 20)

# Set x-axis ticks
num_ticks = 12  # First, last, and 10 in between
tick_indices = np.linspace(0, len(ref_track["date"]) - 1, num_ticks, dtype=int)
ax.set_xticks(ref_track["date"].iloc[tick_indices])
ax.tick_params(axis='x', labelrotation=45)

# Set x-axis and y-axis labels and legend
ax.set_xlabel("Date", fontsize=14, fontweight='bold')
ax.set_ylabel("Displacement (km)", fontsize=14, fontweight='bold')

# Set title
title = f"{cyclone} tracks displacement w.r.t. obs"
ax.set_title(title, fontsize=14, fontweight='bold')

# Save figure
figname = f"{outfolder}/{cyclone}_{model}_tracks_comparison_displacement.png"
plt.savefig(figname, dpi=300, bbox_inches='tight', format='png')

# separate legend in a new figure
fig_leg, ax_leg = plt.subplots(figsize=(5, 10))
ax_leg.axis('off')
legend_elements = []
for i in range(1, len(tracks)):
    mu = np.nanmean(displacements[i - 1])
    std = np.nanstd(displacements[i - 1])
    lbl = f"{labels[i]}: mean = {mu:.0f} km, std = {std:.0f} km"
    if 'MEAN' in labels[i] and i == idxm:
        legend_elements.append(Line2D([0], [0], color='#000000', linestyle='--', linewidth=3, label=lbl))
    else:
        legend_elements.append(Line2D([0], [0], color=c[i], linewidth=2, label=lbl))
ax_leg.legend(handles=legend_elements, loc='upper left', fontsize=14, frameon=False, ncol=3)
figname_leg = f"{outfolder}/{cyclone}_{model}_tracks_comparison_displacement_legend.png"
plt.savefig(figname_leg, dpi=300, bbox_inches='tight', format='png')
plt.close(fig_leg)

plt.close(fig)  # Close the figure to free memory

#### TABLE of ERRORS ####
# Calcola media e deviazione standard
rows = []
for i in range(1, len(labels)):  # Salta OBS
    mu = np.nanmean(displacements[i - 1])
    std = np.nanstd(displacements[i - 1])
    rmse = np.sqrt(mu**2 + std**2)
    max_error = np.nanmax(displacements[i - 1])
    min_error = np.nanmin(displacements[i - 1])
    rows.append([labels[i], mu, std, rmse, min_error, max_error])

# Create DataFrame
df = pd.DataFrame(rows, columns=["SIMULATION", "MEAN ERROR (km)", "STANDARD DEVIATION (km)", "RMSE (km)", "MIN ERROR (km)" , "MAX ERROR (km)" ])
# Sort by RMSE ascending, then by MEAN ERROR, then by STANDARD DEVIATION
df_sorted = df.sort_values(by=["RMSE (km)", "MEAN ERROR (km)", "STANDARD DEVIATION (km)"], ascending=[True, True, True])
# Round for output while keeping correct numerical ordering
df_sorted["MEAN ERROR (km)"] = df_sorted["MEAN ERROR (km)"].round(1).astype(float)
df_sorted["STANDARD DEVIATION (km)"] = df_sorted["STANDARD DEVIATION (km)"].round(1).astype(float)
df_sorted["RMSE (km)"] = df_sorted["RMSE (km)"].round(1).astype(float)
df_sorted["MIN ERROR (km)"] = df_sorted["MIN ERROR (km)"].round(1).astype(float)
df_sorted["MAX ERROR (km)"] = df_sorted["MAX ERROR (km)"].round(1).astype(float)
# Save to CSV
df_sorted.to_csv(f"{outfolder}/{cyclone}_displacement_errors.csv", index=False)
# (Optional) Display the result
print("Sorted by RMSE")
print(df_sorted)

# Red and blue gradient palette
reds = plt.cm.Reds_r(np.linspace(0.4, 0.8, len(df_sorted)))
blues = plt.cm.Blues_r(np.linspace(0.4, 0.8, len(df_sorted)))

bar_colors = []
text_colors = []
ytick_colors = []

for sim in df_sorted["SIMULATION"]:
    if sim.startswith("WRF"):
        bar_colors.append(reds[len(bar_colors) % len(reds)])
        text_colors.append("#8B0000")  # rosso scuro
        ytick_colors.append("#8B0000")
    elif sim.startswith("ICON"):
        bar_colors.append(blues[len(text_colors) % len(blues)])
        text_colors.append("#00008B")  # blu scuro
        ytick_colors.append("#00008B")
    else:
        bar_colors.append("#777777")
        text_colors.append("#333333")
        ytick_colors.append("#333333")

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    df_sorted["SIMULATION"],
    df_sorted["RMSE (km)"],
    color=bar_colors,
    edgecolor="black"
)

for i, (bar, mean, std, txt_color) in enumerate(zip(bars, df_sorted["MEAN ERROR (km)"], df_sorted["STANDARD DEVIATION (km)"], text_colors)):
    width = bar.get_width()
    ax.text(
        width + 1,
        bar.get_y() + bar.get_height() / 2,
        f"μ = {mean} km, σ = {std} km",
        va='center',
        fontsize=16,
        fontweight='bold',
        color=txt_color
    )

ax.set_title("(c)\n ", loc='left', fontsize=20, fontweight='bold')
#ax.set_title(f"Tracks displacement Errors for {cyclone}", fontsize=20, weight='bold', pad=20)
#ax.text(
#    0.5,  # posizione orizzontale (0=sinistra, 1=destra)
#    1.0,  # posizione verticale appena sopra il titolo
#    "( lower is better )",
#    fontsize=12,
#    fontweight='normal',
#    ha='center',
#    va='bottom',
#    transform=ax.transAxes
#)
ax.set_xlabel("RMSE (km)", fontsize=20, fontweight='bold')
ax.set_ylabel("Simulation", fontsize=20, fontweight='bold')
ax.set_xlim(0, df_sorted["RMSE (km)"].max() * 1.2)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Color and bold the y-axis tick labels (simulation names)
for ticklabel, color in zip(ax.get_yticklabels(), ytick_colors):
    ticklabel.set_color(color)
    ticklabel.set_fontweight('bold')
    ticklabel.set_fontsize(18)
for ticklabel in ax.get_xticklabels():
    ticklabel.set_fontweight('bold')
    ticklabel.set_fontsize(18)

figname = f"{outfolder}/{cyclone}_displacement_errors_RMSE.png"
plt.savefig(figname, dpi=300, bbox_inches='tight', format='png')
plt.close(fig)

# Plot mean and spread track
# PLOT TRACKS
# set colormap
proj = ccrs.PlateCarree()
# set figure
fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={'projection': proj})
# Add coastlines and set extent
ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=proj)
# Add the lat-lon grid
ax.gridlines(draw_labels=True, linewidth=0.5, color='#777777ee', linestyle='--', x_inline=False, y_inline=False)
# Add background with ocean and land colors
# ax.add_feature(NaturalEarthFeature('physical', 'ocean', '10m', facecolor=COLORS['water'])) #'#87CEEB'
# ax.add_feature(NaturalEarthFeature('physical', 'land', '10m', facecolor=COLORS['land']))
ax.add_feature(NaturalEarthFeature('physical', 'ocean', '10m', facecolor='#dddddd')) #'#87CEEB'
ax.add_feature(NaturalEarthFeature('physical', 'land', '10m', facecolor='#bbbbbb'))
# Compute mean track (across models) and spread, aligned to obs times
n_times = len(tracks[3]["date"])
n_models = len(tracks) - 1  # exclude OBS

lats = np.full((n_models, n_times), np.nan)
lons = np.full((n_models, n_times), np.nan)

for mi in range(n_models):
    lats[mi, :] = tracks[mi + 1]["lat"]
    lons[mi, :] = tracks[mi + 1]["lon"]

# Mean position per timestep
mean_lat = np.nanmean(lats, axis=0)
mean_lon = np.nanmean(lons, axis=0)

# Spread per timestep: std of great-circle distances from mean position
spread_km = np.full(n_times, np.nan)
for t in range(len(mean_lat)):
    dists = haversine(lats[:, t], lons[:, t], mean_lat[t], mean_lon[t])
    spread_km[t] = np.nanmean(dists)

# Plot spread as semi-transparent filled discs around the mean position
first = True
for t in range(len(mean_lat)):
    lat_c = mean_lat[t]
    lon_c = mean_lon[t]
    r = spread_km[t]
    dlat = r / 111.0
    coslat = np.cos(np.radians(lat_c))
    coslat = coslat if coslat > 1e-8 else 1e-8
    dlon = r / (111.0 * coslat)
    theta = np.linspace(0, 2 * np.pi, 100)
    poly_lats = lat_c + dlat * np.sin(theta)
    poly_lons = lon_c + dlon * np.cos(theta)
    ax.fill(
        poly_lons,
        poly_lats,
        transform=proj,
        color='#ee4400',
        alpha=0.1,
        linewidth=0,
        zorder=5,
        label='Mean spread' if first else None
    )
    first = False

# Plot mean track as dotted line with points
ax.plot(
    mean_lon,
    mean_lat,
    transform=proj,
    color='#000000',
    linestyle=':',
    marker='o',
    markersize=4,
    linewidth=2,
    label='Mean track',
    zorder=10
)

ax.legend(loc='upper right')
# st = strack["date"][0]
# et = strack["date"][-1]
# title = f"Track {sim}    {st} - {et}"
title = f"{cyclone} track prediction : mean and spread"
ax.set_title(title, fontsize=14, fontweight='bold')
# Save figure
figname = f"{outfolder}/{cyclone}_{model}_tracks_mean_spread.png"
plt.savefig(figname, dpi=300, bbox_inches='tight', format='png')
plt.close(fig)  # Close the figure to free memory
print(f"[INFO] Saved figure : \n{figname}")

#######
# take the best track based on RMSE, extract min_slp and max_wind from the corresponding track file, and add to mean track file, and save to csv
# avoid MEAN 
if 'MEAN' in df_sorted.iloc[0]["SIMULATION"]:
    best_simulation = df_sorted.iloc[1]["SIMULATION"]
else:
    best_simulation = df_sorted.iloc[0]["SIMULATION"]
best_track_file = trackfolder / f"HIMPACT_SENS_{best_simulation}" /f"{cyclone}_{model}_{best_simulation}_track_multilevelz.csv"
best_track = pd.read_csv(best_track_file)
try:
    mean_track["min_slp"] = best_track["min_slp"]
except KeyError:
    print(f"[WARNING] min_slp not found in {best_track_file}. Skipping.")
try:
    mean_track["max_wind"] = best_track["max_wind"]
except KeyError:
    print(f"[WARNING] max_wind not found in {best_track_file}. Skipping.")
mean_track.to_csv(f"{outfolder}/{cyclone}_{model}_MEAN_track_multilevelz.csv", index=False)
print(f"[INFO] Updated mean track with min_slp and max_wind from best simulation ({best_simulation}) and saved to {outfolder}/{cyclone}_{model}_MEAN_track_multilevelz.csv")
      