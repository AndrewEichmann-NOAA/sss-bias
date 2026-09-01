#!/usr/bin/env python3
"""
Geographic snapshot of raw (uncorrected) SMAP CAP salinity for a chosen date
window -- e.g. one week -- using the same per-file loader as
build_raw_smap_matchups.py, but restricted by filename date to just the
requested window rather than loading the entire archive.

No Argo comparison here -- this is just the retrieved SSS field itself,
binned and averaged geographically, like a typical satellite SSS snapshot
map (cf. Fig. 1 in Tang et al. 2017).
"""

import argparse
import glob
import re
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_raw_smap_matchups import load_raw_smap_file
from plot_geographic_errors import BIN_DEG, MIN_COUNT

RAW_DIR = '/Users/afeman/Desktop/work/sss-bias/data/raw_smap_cap'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=None, help='Plot one specific raw .h5 swath file instead of a date range.')
    parser.add_argument('--start-date', default='2024-01-01', help='YYYY-MM-DD')
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--bin-deg', type=float, default=BIN_DEG)
    parser.add_argument('--min-count', type=int, default=MIN_COUNT)
    parser.add_argument('--min-salinity', type=float, default=20.0)
    parser.add_argument('--max-salinity', type=float, default=42.0)
    parser.add_argument('--scatter', action='store_true',
                         help='Plot every individual QC-pass pixel directly (no gridding/averaging) -- '
                              'shows the actual swath/orbit-track structure, at the cost of a much '
                              'heavier plot and overlapping points where orbits cross.')
    parser.add_argument('--point-size', type=float, default=0.2, help='Marker size for --scatter.')
    parser.add_argument('--zoom', action='store_true',
                         help='Set axis limits to the data\'s own bounding box (with a small pad) '
                              'instead of the fixed -180/180/-90/90 globe -- much more detail for a '
                              'single swath or short window that only covers a small area.')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    if args.file:
        files = [args.file]
        label = args.file.split('/')[-1]
        print(f"1 raw file: {label}")
    else:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = start + timedelta(days=args.days)
        start_str, end_str = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')

        all_files = sorted(glob.glob(f'{RAW_DIR}/*.h5'))
        files = [f for f in all_files if start_str <= re.search(r'_(\d{8})T', f).group(1) < end_str]
        label = f'{args.start_date}_{args.days}d'
        print(f"{len(files)} raw files in [{args.start_date}, +{args.days}d)")

    dfs = []
    for f in files:
        try:
            df = load_raw_smap_file(f)
        except Exception as e:
            print(f"  error loading {f}: {e}")
            continue
        df = df[(df['quality_flag'] == 0) & df['sss'].between(args.min_salinity, args.max_salinity)]
        if not df.empty:
            dfs.append(df[['lat', 'lon', 'sss']])

    if not dfs:
        print("No QC-pass obs found in this window.")
        return
    full = pd.concat(dfs, ignore_index=True)
    print(f"{len(full)} QC-pass ocean obs")

    title_range = label if args.file else f'{args.start_date} to {end.strftime("%Y-%m-%d")}'

    if args.zoom:
        lon_pad = max(0.5, (full['lon'].max() - full['lon'].min()) * 0.05)
        lat_pad = max(0.5, (full['lat'].max() - full['lat'].min()) * 0.05)
        xlim = (full['lon'].min() - lon_pad, full['lon'].max() + lon_pad)
        ylim = (full['lat'].min() - lat_pad, full['lat'].max() + lat_pad)
    else:
        xlim, ylim = (-180, 180), (-90, 90)

    if args.scatter:
        out_path = args.out or f'/Users/afeman/Desktop/work/sss-bias/data/matchups/raw_smap_sss_{label}_scatter.png'
        vmin, vmax = np.nanpercentile(full['sss'], [2, 98])

        fig, ax = plt.subplots(figsize=(14, 6))
        # rasterized=True keeps the saved file a reasonable size despite millions
        # of points -- otherwise each point stays a separate vector object.
        sc = ax.scatter(full['lon'], full['lat'], c=full['sss'], s=args.point_size,
                         cmap='viridis', vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.colorbar(sc, ax=ax, label='PSU', fraction=0.025, pad=0.02)
        ax.set_title(f'Raw SMAP CAP salinity, {title_range} (every QC-pass pixel, no gridding, n={len(full):,})')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
        return

    lat_edges = np.arange(-90, 90 + args.bin_deg, args.bin_deg)
    lon_edges = np.arange(-180, 180 + args.bin_deg, args.bin_deg)
    n_lat, n_lon = len(lat_edges) - 1, len(lon_edges) - 1
    lat_idx = np.clip(np.digitize(full['lat'].to_numpy(), lat_edges) - 1, 0, n_lat - 1)
    lon_idx = np.clip(np.digitize(full['lon'].to_numpy(), lon_edges) - 1, 0, n_lon - 1)

    count_grid = np.zeros((n_lat, n_lon))
    sum_grid = np.zeros((n_lat, n_lon))
    np.add.at(count_grid, (lat_idx, lon_idx), 1)
    np.add.at(sum_grid, (lat_idx, lon_idx), full['sss'].to_numpy())
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_grid = sum_grid / count_grid
    mean_grid[count_grid < args.min_count] = np.nan

    n_cells_filled = np.isfinite(mean_grid).sum()
    print(f"{n_cells_filled}/{mean_grid.size} cells have >= {args.min_count} obs "
          f"({100 * n_cells_filled / mean_grid.size:.1f}%)")

    out_path = args.out or f'/Users/afeman/Desktop/work/sss-bias/data/matchups/raw_smap_sss_{label}.png'

    fig, ax = plt.subplots(figsize=(14, 6))
    vmin, vmax = np.nanpercentile(mean_grid, [2, 98])
    mesh = ax.pcolormesh(lon_edges, lat_edges, mean_grid, cmap='viridis', vmin=vmin, vmax=vmax, shading='flat')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.colorbar(mesh, ax=ax, label='PSU', fraction=0.025, pad=0.02)
    ax.set_title(f'Raw SMAP CAP salinity, {title_range} '
                 f'({args.bin_deg:g}deg bins, min {args.min_count} obs/cell, n={len(full):,})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
