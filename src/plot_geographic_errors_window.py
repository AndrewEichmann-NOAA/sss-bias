#!/usr/bin/env python3
"""
Geographic bias/RMSE for the raw (uncorrected) SMAP CAP vs. Argo comparison,
for one of the wider match-window tables (+/-12h or +/-24h, built alongside
the default +/-3h table) -- see the Vernieres et al. (2014) match-window
discussion (DESIGN.md): a wider window recovers more matches (173,711 for
+/-12h, 243,041 for +/-24h, vs. 54,787 for +/-3h, same 50km/2022-06-01 to
2025-05-01 span), at the cost of pairing satellite obs with Argo profiles
that are less contemporaneous.

Raw-only (no FFANN panel) since no model has been trained on these wider
windows yet -- this is a first look at whether the extra matches shift the
raw bias/RMSE picture geographically, before deciding whether to also
retrain a model on one of them.
"""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_geographic_errors import bin_stats, MIN_COUNT

MATCHUPS_DIR = '/Users/afeman/Desktop/work/sss-bias/data/matchups'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', choices=['12h', '24h'], default='12h',
                         help='Which wider match-window table to plot.')
    parser.add_argument('--bin-deg', type=float, default=5.0,
                         help='Grid cell size in degrees -- smaller means finer resolution but fewer '
                              'matches/cell, so more cells fall below --min-count and get masked blank.')
    parser.add_argument('--min-count', type=int, default=MIN_COUNT,
                         help='Minimum matches per cell before it is masked blank.')
    parser.add_argument('--out', default=None,
                         help='Defaults to geo_errors_smap_cap_<window>_<bin_deg>deg.png')
    args = parser.parse_args()
    matchups_path = f'{MATCHUPS_DIR}/smap_cap_argo_matchups_{args.window}.parquet'
    out_path = args.out or f'{MATCHUPS_DIR}/geo_errors_smap_cap_{args.window}_{args.bin_deg:g}deg.png'

    df = pd.read_parquet(matchups_path, columns=['sat_lat', 'sat_lon', 'sat_sss', 'argo_salinity'])
    diff = (df['sat_sss'] - df['argo_salinity']).to_numpy()
    lon_e, lat_e, rmse, bias, count = bin_stats(df['sat_lat'].to_numpy(), df['sat_lon'].to_numpy(), diff,
                                                 bin_deg=args.bin_deg, min_count=args.min_count)
    n_cells_total = np.isfinite(rmse).size
    n_cells_filled = np.isfinite(rmse).sum()
    print(f"{n_cells_filled}/{n_cells_total} cells have >= {args.min_count} matches "
          f"({100*n_cells_filled/n_cells_total:.1f}%)")

    bias_scale = np.nanpercentile(np.abs(bias[~np.isnan(bias)]), 95)
    rmse_scale = np.nanpercentile(rmse[~np.isnan(rmse)], 95)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

    mesh0 = axes[0].pcolormesh(lon_e, lat_e, bias, cmap='RdBu_r', vmin=-bias_scale, vmax=bias_scale, shading='flat')
    axes[0].set_title(f'Bias, raw SMAP CAP vs. Argo (+/-{args.window} window, n={len(df):,})', fontsize=10)
    fig.colorbar(mesh0, ax=axes[0], label='PSU', fraction=0.03, pad=0.02)

    mesh1 = axes[1].pcolormesh(lon_e, lat_e, rmse, cmap='YlOrRd', vmin=0, vmax=rmse_scale, shading='flat')
    axes[1].set_title(f'RMSE, raw SMAP CAP vs. Argo (+/-{args.window} window)', fontsize=10)
    fig.colorbar(mesh1, ax=axes[1], label='PSU', fraction=0.03, pad=0.02)

    for ax in axes:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')

    fig.suptitle(f'Raw JPL CAP SMAP vs. Argo: +/-{args.window} match window '
                 f'({args.bin_deg:g}deg bins, min {args.min_count} obs/cell)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
