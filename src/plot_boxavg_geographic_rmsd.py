#!/usr/bin/env python3
"""
Geographic RMSD map for the Schanze et al. (2020) box-average validation
matchup (build_raw_smap_matchups.py --box-average, DESIGN.md 28): bins the
satellite-box-average-minus-Argo difference by the Argo report's own
location (the natural bin center here, since each matchup is already an
average over a space-time box centered on that report -- there's no single
"sat_lat/sat_lon" the way the nearest-neighbor matchup has).
"""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_geographic_errors import bin_stats, MIN_COUNT

DEFAULT_PATH = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_boxavg_12wk.parquet'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups', default=DEFAULT_PATH)
    parser.add_argument('--bin-deg', type=float, default=5.0)
    parser.add_argument('--min-count', type=int, default=MIN_COUNT)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.matchups)
    diff = (df['sat_sss_mean'] - df['argo_salinity']).to_numpy()
    lon_e, lat_e, rmse, bias, count = bin_stats(df['argo_lat'].to_numpy(), df['argo_lon'].to_numpy(), diff,
                                                 bin_deg=args.bin_deg, min_count=args.min_count)

    n_filled = np.isfinite(rmse).sum()
    print(f"{len(df)} matchups; {n_filled}/{rmse.size} cells have >= {args.min_count} matches "
          f"({100 * n_filled / rmse.size:.1f}%)")

    out_path = args.out or '/Users/afeman/Desktop/work/sss-bias/data/matchups/geo_rmsd_boxavg.png'
    rmse_scale = np.nanpercentile(rmse[~np.isnan(rmse)], 95) if n_filled else 1.0

    fig, ax = plt.subplots(figsize=(14, 6))
    mesh = ax.pcolormesh(lon_e, lat_e, rmse, cmap='YlOrRd', vmin=0, vmax=rmse_scale, shading='flat')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.colorbar(mesh, ax=ax, label='PSU', fraction=0.025, pad=0.02)
    ax.set_title(f'RMSD, box-average SMAP CAP vs. Argo (Schanze et al. 50km/+-3.5day matchup) '
                 f'({args.bin_deg:g}deg bins, min {args.min_count} matches/cell, n={len(df):,})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
