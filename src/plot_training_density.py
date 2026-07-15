#!/usr/bin/env python3
"""
Compare training-set vs. test-set observation density geographically, to
check whether the white gaps in plot_geographic_errors.py's corrected
panels reflect genuine low-confidence (sparse training) regions or just
test-window validation blind spots over otherwise well-sampled areas.
"""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import add_features, split_data_naive
from plot_geographic_errors import bin_stats, BIN_DEG

MIN_COUNT = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-dir', default='/Users/afeman/Desktop/work/sss-bias/data/matchups')
    parser.add_argument('--out', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/training_density.png')
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), sharex=True, sharey=True)

    for row, sensor in enumerate(['smap', 'smos']):
        df = pd.read_parquet(f'{args.matchups_dir}/{sensor}_argo_matchups.parquet')
        df = add_features(df)
        train, val, test = split_data_naive(df)

        lon_e, lat_e, _, _, train_count = bin_stats(
            train['sat_lat'].to_numpy(), train['sat_lon'].to_numpy(), np.zeros(len(train)), min_count=1)
        _, _, _, _, test_count = bin_stats(
            test['sat_lat'].to_numpy(), test['sat_lon'].to_numpy(), np.zeros(len(test)), min_count=1)

        train_log = np.where(train_count > 0, np.log10(train_count), np.nan)
        test_log = np.where(test_count > 0, np.log10(test_count), np.nan)
        vmax = max(np.nanmax(train_log), np.nanmax(test_log))

        for col, (grid, label) in enumerate([(train_log, 'train'), (test_log, 'test')]):
            ax = axes[row, col]
            mesh = ax.pcolormesh(lon_e, lat_e, grid, cmap='viridis', vmin=0, vmax=vmax, shading='flat')
            ax.set_title(f'{sensor.upper()} {label} density (n={len(train) if label=="train" else len(test):,})',
                         fontsize=10)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            cbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label('log10(obs/cell)')

    for ax in axes[1, :]:
        ax.set_xlabel('Longitude')
    for ax in axes[:, 0]:
        ax.set_ylabel('Latitude')

    fig.suptitle(f'Training vs. test observation density ({BIN_DEG:.0f}deg bins) -- '
                 f'MIN_COUNT={MIN_COUNT} is the display threshold used in plot_geographic_errors.py',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
