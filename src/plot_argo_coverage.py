#!/usr/bin/env python3
"""
Compare geographic density of ALL near-surface Argo profiles in the local
archive against the density of profiles that actually matched a SMAP or
SMOS observation -- to check whether the "gaps" in plot_geographic_errors.py
reflect genuine Argo data scarcity, or just losses from the satellite
matching/QC process on top of otherwise-dense oceanographic coverage.
"""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_geographic_errors import bin_stats, BIN_DEG


def density_grid(lat, lon):
    _, _, _, _, count = bin_stats(lat, lon, np.zeros(len(lat)), min_count=1)
    return np.where(count > 0, np.log10(count), np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-dir', default='/Users/afeman/Desktop/work/sss-bias/data/matchups')
    parser.add_argument('--all-argo', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/all_argo_profiles.parquet')
    parser.add_argument('--out', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/argo_coverage.png')
    args = parser.parse_args()

    all_argo = pd.read_parquet(args.all_argo)
    smap = pd.read_parquet(f'{args.matchups_dir}/smap_argo_matchups.parquet', columns=['argo_lat', 'argo_lon'])
    smos = pd.read_parquet(f'{args.matchups_dir}/smos_argo_matchups.parquet', columns=['argo_lat', 'argo_lon'])

    lon_e, lat_e, _, _, _ = bin_stats(all_argo['lat'].to_numpy(), all_argo['lon'].to_numpy(),
                                       np.zeros(len(all_argo)), min_count=1)

    panels = [
        (all_argo['lat'].to_numpy(), all_argo['lon'].to_numpy(),
         f'ALL near-surface Argo profiles, local archive (n={len(all_argo):,})'),
        (smap['argo_lat'].to_numpy(), smap['argo_lon'].to_numpy(),
         f'SMAP-matched Argo profiles (n={len(smap):,})'),
        (smos['argo_lat'].to_numpy(), smos['argo_lon'].to_numpy(),
         f'SMOS-matched Argo profiles (n={len(smos):,})'),
    ]

    grids = [density_grid(lat, lon) for lat, lon, _ in panels]
    vmax = max(np.nanmax(g) for g in grids)

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True, sharey=True)
    for ax, (lat, lon, title), grid in zip(axes, panels, grids):
        mesh = ax.pcolormesh(lon_e, lat_e, grid, cmap='viridis', vmin=0, vmax=vmax, shading='flat')
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_ylabel('Latitude')
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('log10(profiles/cell)')

    axes[-1].set_xlabel('Longitude')
    fig.suptitle(f'Argo coverage: all profiles vs. satellite-matched subset ({BIN_DEG:.0f}deg bins)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
