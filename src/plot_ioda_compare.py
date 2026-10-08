#!/usr/bin/env python3
"""
Plots the original-vs-bias-corrected IODA comparison (DESIGN.md 39.3).
n=590 over a 41-day window doesn't support gridded geographic bins (most
5deg cells would have 0-2 points) -- instead: a distribution comparison
(histogram) and a geographic scatter (no binning), point-by-point.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MATCHUPS_DIR = '/Users/afeman/Desktop/work/sss-bias/data/matchups'
OUT_PATH = f'{MATCHUPS_DIR}/ioda_compare_dec2025_jan2026.png'


def main():
    orig = pd.read_parquet(f'{MATCHUPS_DIR}/ioda_compare_original.parquet')
    corr = pd.read_parquet(f'{MATCHUPS_DIR}/ioda_compare_corrected.parquet')

    diff_orig = (orig['sat_sss'] - orig['argo_salinity']).to_numpy()
    diff_corr = (corr['sat_sss'] - corr['argo_salinity']).to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    ax = axes[0]
    bins = np.linspace(-3, 3, 61)
    ax.hist(diff_orig, bins=bins, alpha=0.5, label=f'Original (RMSE {np.sqrt((diff_orig**2).mean()):.3f})',
            color='#d62728')
    ax.hist(diff_corr, bins=bins, alpha=0.5, label=f'Corrected (RMSE {np.sqrt((diff_corr**2).mean()):.3f})',
            color='#1f77b4')
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Satellite - Argo salinity (PSU)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error distribution (n={len(orig)})')
    ax.legend(fontsize=9)

    vmax = np.percentile(np.abs(np.concatenate([diff_orig, diff_corr])), 95)
    for ax, diff, label in [(axes[1], diff_orig, 'Original'), (axes[2], diff_corr, 'Bias-corrected')]:
        sc = ax.scatter(orig['argo_lon'], orig['argo_lat'], c=diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                         s=18, edgecolors='none')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude')
        ax.set_title(f'{label}: satellite - Argo (PSU)')
        fig.colorbar(sc, ax=ax, label='PSU', fraction=0.03, pad=0.02)
    axes[1].set_ylabel('Latitude')

    fig.suptitle('IODA SMAP SSS: original vs. bias-corrected, matched to Argo '
                  '(2025-12-10 to 2026-01-19)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == '__main__':
    main()
