#!/usr/bin/env python3
"""
Geographic error maps for the raw-SMAP-CAP rich-feature pipeline, analogous
to plot_geographic_errors.py's maps for the IODA-based pipeline.

Same two-source-data design as plot_geographic_errors.py:
  - Raw panel uses the FULL matchup table (all 2 years currently downloaded)
    -- raw satellite SSS involves no fitting, so there's no leakage risk in
    using every matchup for the least-noisy picture of the retrieval itself.
  - FFANN panels use ONLY cap_test_predictions.parquet's held-out test rows
    (train_rich_features_poc.py's chronological split -- see DESIGN.md 24/26)
    to keep the correction's spatial performance honestly out-of-sample.

Three columns instead of plot_geographic_errors.py's two: raw, baseline-
feature FFANN, and rich-feature FFANN, so the spatial pattern of the rich-
feature gain (DESIGN.md 25/26) is visible directly rather than only as an
aggregate RMSE number.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_geographic_errors import bin_stats, BIN_DEG, MIN_COUNT

MATCHUPS_PATH = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups.parquet'
PREDICTIONS_PATH = '/Users/afeman/Desktop/work/sss-bias/data/matchups/cap_test_predictions.parquet'
OUT_PATH = '/Users/afeman/Desktop/work/sss-bias/data/matchups/geo_errors_smap_cap.png'


def main():
    full = pd.read_parquet(MATCHUPS_PATH, columns=['sat_lat', 'sat_lon', 'sat_sss', 'argo_salinity'])
    raw_diff = (full['sat_sss'] - full['argo_salinity']).to_numpy()
    lon_e, lat_e, raw_rmse, raw_bias, _ = bin_stats(
        full['sat_lat'].to_numpy(), full['sat_lon'].to_numpy(), raw_diff)

    test = pd.read_parquet(PREDICTIONS_PATH)
    baseline_diff = (test['pred_ffann_baseline'] - test['argo_salinity']).to_numpy()
    _, _, baseline_rmse, baseline_bias, _ = bin_stats(
        test['sat_lat'].to_numpy(), test['sat_lon'].to_numpy(), baseline_diff)
    rich_diff = (test['pred_ffann_rich'] - test['argo_salinity']).to_numpy()
    _, _, rich_rmse, rich_bias, _ = bin_stats(
        test['sat_lat'].to_numpy(), test['sat_lon'].to_numpy(), rich_diff)

    all_bias = np.concatenate([g[~np.isnan(g)] for g in (raw_bias, baseline_bias, rich_bias)])
    all_rmse = np.concatenate([g[~np.isnan(g)] for g in (raw_rmse, baseline_rmse, rich_rmse)])
    bias_scale = np.nanpercentile(np.abs(all_bias), 95)
    rmse_scale = np.nanpercentile(all_rmse, 95)

    fig, axes = plt.subplots(2, 3, figsize=(19, 8.5), sharex=True, sharey=True)

    panels = [
        (axes[0, 0], raw_bias, f'Bias, raw SMAP CAP (all matches, n={len(full):,})', 'RdBu_r', -bias_scale, bias_scale),
        (axes[0, 1], baseline_bias, f'Bias, baseline-feature FFANN (test only, n={len(test):,})', 'RdBu_r', -bias_scale, bias_scale),
        (axes[0, 2], rich_bias, f'Bias, rich-feature FFANN (test only, n={len(test):,})', 'RdBu_r', -bias_scale, bias_scale),
        (axes[1, 0], raw_rmse, 'RMSE, raw (all matches)', 'YlOrRd', 0, rmse_scale),
        (axes[1, 1], baseline_rmse, 'RMSE, baseline-feature FFANN (test only)', 'YlOrRd', 0, rmse_scale),
        (axes[1, 2], rich_rmse, 'RMSE, rich-feature FFANN (test only)', 'YlOrRd', 0, rmse_scale),
    ]

    for ax, grid, title, cmap, vmin, vmax in panels:
        mesh = ax.pcolormesh(lon_e, lat_e, grid, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        fig.colorbar(mesh, ax=ax, label='PSU', fraction=0.03, pad=0.02)

    for ax in axes[1, :]:
        ax.set_xlabel('Longitude')
    for ax in axes[:, 0]:
        ax.set_ylabel('Latitude')

    fig.suptitle(f'Raw JPL CAP SMAP vs. Argo bulk salinity: geographic error ({BIN_DEG:.0f}deg bins, '
                 f'min {MIN_COUNT} obs/cell)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == '__main__':
    main()
