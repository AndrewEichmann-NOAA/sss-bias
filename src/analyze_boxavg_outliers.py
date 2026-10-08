#!/usr/bin/env python3
"""
For each geographic bin in the box-average SMAP-vs-Argo matchup, compare RMSE
to a robust (outlier-resistant) measure of the same spread -- the median
absolute deviation, scaled to be comparable to a standard deviation. If a
bin's RMSE is close to its robust spread, the error there is a consistent
offset across most matches. If RMSE is much larger than the robust spread,
a small number of outlier matches are inflating the bin's RMSE while most
matches in that bin are actually fine.
"""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_COUNT = 5
MAD_TO_STD = 1.4826  # scales MAD to be a consistent estimator of std for normal data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_boxavg_35mo.parquet')
    parser.add_argument('--bin-deg', type=float, default=2.0)
    parser.add_argument('--min-count', type=int, default=MIN_COUNT)
    parser.add_argument('--top-n', type=int, default=8)
    parser.add_argument('--out', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/boxavg_outlier_analysis.png')
    args = parser.parse_args()

    df = pd.read_parquet(args.matchups)
    df['diff'] = df['sat_sss_mean'] - df['argo_salinity']
    df['lat_bin'] = (np.floor(df['argo_lat'] / args.bin_deg) * args.bin_deg).astype(int)
    df['lon_bin'] = (np.floor(df['argo_lon'] / args.bin_deg) * args.bin_deg).astype(int)

    rows = []
    for (lat_bin, lon_bin), g in df.groupby(['lat_bin', 'lon_bin']):
        n = len(g)
        if n < args.min_count:
            continue
        diff = g['diff'].to_numpy()
        rmse = np.sqrt(np.mean(diff ** 2))
        median_abs = np.median(np.abs(diff))
        robust_std = MAD_TO_STD * np.median(np.abs(diff - np.median(diff)))
        max_abs = np.abs(diff).max()
        frac_gt2 = np.mean(np.abs(diff) > 2.0)
        rows.append(dict(lat_bin=lat_bin, lon_bin=lon_bin, n=n, rmse=rmse,
                          robust_std=robust_std, median_abs=median_abs,
                          max_abs=max_abs, frac_gt2=frac_gt2,
                          ratio=rmse / robust_std if robust_std > 1e-6 else np.nan))
    cells = pd.DataFrame(rows)
    print(f"{len(cells)} cells with >= {args.min_count} matches")
    print(f"Correlation(RMSE, robust std) across cells: {cells['rmse'].corr(cells['robust_std']):.3f}")
    print(f"Median RMSE/robust-std ratio across cells: {cells['ratio'].median():.2f}")
    print()

    top = cells.sort_values('rmse', ascending=False).head(args.top_n)
    print(f"Top {args.top_n} highest-RMSE cells:")
    print(top[['lat_bin', 'lon_bin', 'n', 'rmse', 'robust_std', 'median_abs', 'max_abs', 'frac_gt2']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    sc = ax.scatter(cells['robust_std'], cells['rmse'], c=cells['n'], cmap='viridis',
                     s=14, alpha=0.7, norm=matplotlib.colors.LogNorm())
    lims = [0, max(cells['rmse'].max(), cells['robust_std'].max()) * 1.05]
    ax.plot(lims, lims, 'k--', lw=1, label='RMSE = robust std (consistent error)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Robust std (1.4826 x MAD) [PSU]')
    ax.set_ylabel('RMSE [PSU]')
    ax.set_title(f'Per-cell RMSE vs. robust std ({args.bin_deg:g}deg bins)')
    ax.legend(fontsize=8, loc='upper left')
    fig.colorbar(sc, ax=ax, label='matches/cell')

    ax = axes[1]
    sc = ax.scatter(cells['rmse'], cells['ratio'], c=cells['n'], cmap='viridis',
                     s=14, alpha=0.7, norm=matplotlib.colors.LogNorm())
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xlabel('Cell RMSE [PSU]')
    ax.set_ylabel('RMSE / robust std (higher = more outlier-driven)')
    ax.set_title('Outlier-driven-ness vs. cell RMSE')
    fig.colorbar(sc, ax=ax, label='matches/cell')

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == '__main__':
    main()
