#!/usr/bin/env python3
"""
Global maps of RMSE and bias (satellite - Argo) on a lat/lon grid.

Two different data sources are used deliberately:
  - Raw (uncorrected) panels use the FULL matchup table (all years). Raw
    satellite SSS involves no fitting, so there's no leakage risk in using
    every matchup -- this gives the most complete, least noisy picture of
    where the satellite retrieval itself disagrees with Argo.
  - FFANN panels use ONLY the held-out test-set predictions
    (phase1_test_predictions_<sensor>.parquet from train_baseline.py) to
    keep the correction's spatial performance honestly out-of-sample. Using
    training-period data here would show optimistic, memorized error rather
    than true generalization.

No coastline basemap is used -- these are ocean-only observations, so land
shows up naturally as empty (masked) cells against the surrounding data.
"""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BIN_DEG = 5.0
MIN_COUNT = 5


def bin_stats(lat, lon, diff, bin_deg=BIN_DEG, min_count=MIN_COUNT):
    """Grid (lat, lon, diff=pred-actual) into bin_deg x bin_deg cells.

    Returns (lon_edges, lat_edges, rmse_grid, bias_grid, count_grid), grids
    shaped (n_lat_bins, n_lon_bins), NaN where count < min_count.
    """
    lat_edges = np.arange(-90, 90 + bin_deg, bin_deg)
    lon_edges = np.arange(-180, 180 + bin_deg, bin_deg)
    n_lat, n_lon = len(lat_edges) - 1, len(lon_edges) - 1

    lat_idx = np.clip(np.digitize(lat, lat_edges) - 1, 0, n_lat - 1)
    lon_idx = np.clip(np.digitize(lon, lon_edges) - 1, 0, n_lon - 1)

    count_grid = np.zeros((n_lat, n_lon))
    sum_diff = np.zeros((n_lat, n_lon))
    sum_sq = np.zeros((n_lat, n_lon))
    np.add.at(count_grid, (lat_idx, lon_idx), 1)
    np.add.at(sum_diff, (lat_idx, lon_idx), diff)
    np.add.at(sum_sq, (lat_idx, lon_idx), diff ** 2)

    with np.errstate(invalid='ignore', divide='ignore'):
        bias_grid = sum_diff / count_grid
        rmse_grid = np.sqrt(sum_sq / count_grid)

    mask = count_grid < min_count
    bias_grid[mask] = np.nan
    rmse_grid[mask] = np.nan

    return lon_edges, lat_edges, rmse_grid, bias_grid, count_grid


def plot_sensor(sensor, matchups_path, predictions_path, out_path):
    full = pd.read_parquet(matchups_path, columns=['sat_lat', 'sat_lon', 'sat_sss', 'argo_salinity'])
    raw_diff = (full['sat_sss'] - full['argo_salinity']).to_numpy()
    lon_e, lat_e, raw_rmse, raw_bias, raw_count = bin_stats(
        full['sat_lat'].to_numpy(), full['sat_lon'].to_numpy(), raw_diff)

    test = pd.read_parquet(predictions_path)
    ffann_diff = (test['pred_ffann'] - test['argo_salinity']).to_numpy()
    _, _, ffann_rmse, ffann_bias, ffann_count = bin_stats(
        test['sat_lat'].to_numpy(), test['sat_lon'].to_numpy(), ffann_diff)

    bias_scale = np.nanpercentile(np.abs(np.concatenate([raw_bias[~np.isnan(raw_bias)],
                                                           ffann_bias[~np.isnan(ffann_bias)]])), 95)
    rmse_scale = np.nanpercentile(np.concatenate([raw_rmse[~np.isnan(raw_rmse)],
                                                    ffann_rmse[~np.isnan(ffann_rmse)]]), 95)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), sharex=True, sharey=True)

    panels = [
        (axes[0, 0], raw_bias, f'Bias, raw {sensor.upper()} (all years, n={len(full):,})',
         'RdBu_r', -bias_scale, bias_scale),
        (axes[0, 1], ffann_bias, f'Bias, FFANN-corrected (test set only, n={len(test):,})',
         'RdBu_r', -bias_scale, bias_scale),
        (axes[1, 0], raw_rmse, 'RMSE, raw (all years)', 'YlOrRd', 0, rmse_scale),
        (axes[1, 1], ffann_rmse, 'RMSE, FFANN-corrected (test set only)', 'YlOrRd', 0, rmse_scale),
    ]

    for ax, grid, title, cmap, vmin, vmax in panels:
        mesh = ax.pcolormesh(lon_e, lat_e, grid, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        fig.colorbar(mesh, ax=ax, label='PSU', fraction=0.025, pad=0.02)

    for ax in axes[1, :]:
        ax.set_xlabel('Longitude')
    for ax in axes[:, 0]:
        ax.set_ylabel('Latitude')

    fig.suptitle(f'{sensor.upper()} vs. Argo bulk salinity: geographic error ({BIN_DEG:.0f}deg bins, '
                 f'min {MIN_COUNT} obs/cell)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', choices=['smap', 'smos'], default='smap')
    parser.add_argument('--matchups-dir', default='/Users/afeman/Desktop/work/sss-bias/data/matchups')
    args = parser.parse_args()

    matchups_path = f'{args.matchups_dir}/{args.sensor}_argo_matchups.parquet'
    predictions_path = f'{args.matchups_dir}/phase1_test_predictions_{args.sensor}.parquet'
    out_path = f'{args.matchups_dir}/geo_errors_{args.sensor}.png'

    plot_sensor(args.sensor, matchups_path, predictions_path, out_path)


if __name__ == '__main__':
    main()
