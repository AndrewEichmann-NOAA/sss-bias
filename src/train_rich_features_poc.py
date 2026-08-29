#!/usr/bin/env python3
"""
Proof-of-concept: does the rich JPL CAP feature set (DESIGN.md 21's "research
upper bound" question) do anything for bias correction, trained on the rich
matchup table (data/matchups/smap_cap_argo_matchups.parquet -- see DESIGN.md
23/24 for how this table's size grew from 310 rows/1 week to 14,705 rows/1
year)?

Trains two FFANNs on an identical split: one restricted to the same feature
set as the operational baseline (sat_sss, sat_lat, lon, season, basin), one
with the rich per-pixel fields added, so any difference is attributable to
the extra fields rather than to different data/splits. Defaults to a
chronological split (earlier months train, later months test) once the table
spans multiple seasons -- the whole reason for pulling a full year was to
check the rich-feature gain isn't just an artifact of training and testing
within the same season's conditions.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from train_baseline import FFANN, train_ffann, compute_metrics

MATCHUPS_PATH = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups.parquet'
TARGET_COLUMN = 'argo_salinity'
BASIN_CODES = [0, 1, 2, 3, 4, 5]

BASELINE_FEATURES = ['sat_sss', 'sat_lat', 'lon_sin', 'lon_cos', 'doy_sin', 'doy_cos'] + \
    [f'basin_{c}' for c in BASIN_CODES]

# Excludes sat_anc_swh (100% fill in this product, see DESIGN.md 23) and
# sat_quality_flag (constant 0 -- already QC-filtered to pass-only).
RICH_EXTRA_FEATURES = [
    'sat_smap_sss_uncertainty', 'sat_anc_sst', 'sat_anc_spd', 'sat_anc_dir', 'sat_anc_sss',
    'sat_inc_fore', 'sat_inc_aft', 'sat_azi_fore', 'sat_azi_aft', 'sat_antazi_fore', 'sat_antazi_aft',
    'sat_ice_concentration', 'sat_land_fraction_fore', 'sat_land_fraction_aft',
    'sat_tb_h_fore', 'sat_tb_h_aft', 'sat_tb_v_fore', 'sat_tb_v_aft',
    'sat_tb_h_bias_adj', 'sat_tb_v_bias_adj',
    'sat_nedt_h_fore', 'sat_nedt_h_aft', 'sat_nedt_v_fore', 'sat_nedt_v_aft',
    'sat_smap_spd', 'sat_smap_high_spd', 'sat_smap_high_dir', 'sat_smap_high_dir_smooth',
]
RICH_FEATURES = BASELINE_FEATURES + RICH_EXTRA_FEATURES


class Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Threshold, not exact-zero: sat_ice_concentration is ~constant 0 at
        # these mid-latitude float locations, but not bit-identically zero,
        # so std_ came out ~5.6e-19 -- dividing by that blew up any val/test
        # row differing from train's near-constant value by even float noise.
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def add_features(df):
    df = df.copy()
    day_of_year = df['sat_datetime'].dt.dayofyear.astype(float)
    df['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    lon_rad = np.radians(df['sat_lon'].astype(float))
    df['lon_sin'] = np.sin(lon_rad)
    df['lon_cos'] = np.cos(lon_rad)

    # No sat_oceanBasin in the raw-CAP table (unlike the IODA-based pipeline's
    # features.py) -- argo_oceanBasin is an equally valid stand-in since
    # matches are colocated within 50km, almost always the same basin.
    for code in BASIN_CODES:
        df[f'basin_{code}'] = (df['argo_oceanBasin'] == code).astype(float)

    return df


def chronological_split(df, train_end='2023-01-31', val_end='2023-03-31', date_col='sat_datetime'):
    """Split by date, train on the earlier months and test on later ones --
    the point of pulling a full year (DESIGN.md 24) was specifically to test
    whether the rich-feature gain survives a train/test split across
    different seasons, not just a bigger random split of a single season.
    Default boundaries put summer-through-early-winter 2022 in train,
    late-winter 2023 in val, and spring 2023 (a season absent from train) in
    test -- roughly a 68/15/16 split of the full-year table.
    """
    dates = df[date_col]
    train = df[dates <= train_end].reset_index(drop=True)
    val = df[(dates > train_end) & (dates <= val_end)].reset_index(drop=True)
    test = df[dates > val_end].reset_index(drop=True)
    return train, val, test


def random_split(df, train_frac=0.7, val_frac=0.15, seed=0):
    """70/15/15 random split. Not chronological/float-aware like features.py's
    split_data() -- with all 310 rows from a single week, there's no meaningful
    date axis to split on, and no float-ID leakage concern either (see
    DESIGN.md 23: table is already one row per unique Argo profile).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(len(df) * train_frac)
    n_val = int(len(df) * val_frac)
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))


def fit_and_eval_ffann(feature_cols, train, val, test, label):
    scaler = Standardizer()
    X_train = scaler.fit_transform(train[feature_cols].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val[feature_cols].to_numpy(dtype=np.float64))
    X_test = scaler.transform(test[feature_cols].to_numpy(dtype=np.float64))

    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_val = val[TARGET_COLUMN].to_numpy(dtype=np.float64)

    resid_train = y_train - train['sat_sss'].to_numpy(dtype=np.float64)
    resid_val = y_val - val['sat_sss'].to_numpy(dtype=np.float64)

    print(f"\nTraining FFANN ({label}, {len(feature_cols)} features)...")
    model = train_ffann(X_train, resid_train, X_val, resid_val, n_features=X_train.shape[1])

    model.eval()
    with torch.no_grad():
        resid_pred_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    pred = test['sat_sss'].to_numpy(dtype=np.float64) + resid_pred_test
    return compute_metrics(pred, test[TARGET_COLUMN])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['chronological', 'random'], default='chronological',
                         help='chronological (default): train on earlier months, test on later ones, '
                              'to check the rich-feature gain survives a season change (DESIGN.md 24). '
                              'random: 70/15/15 random split (only meaningful for a single-season table).')
    args = parser.parse_args()

    df = pd.read_parquet(MATCHUPS_PATH)
    df = add_features(df)
    print(f"Loaded {len(df)} matchups, {df['sat_datetime'].min()} to {df['sat_datetime'].max()}")

    if args.split == 'chronological':
        train, val, test = chronological_split(df)
    else:
        train, val, test = random_split(df)
    print(f"split={args.split}  train={len(train)} val={len(val)} test={len(test)}")

    results = {}

    # --- Baseline 1: raw satellite SSS, no correction ---
    results['raw'] = compute_metrics(test['sat_sss'], test[TARGET_COLUMN])

    # --- Baseline 2: constant bias correction (train-set mean offset) ---
    train_bias = float((train['sat_sss'] - train[TARGET_COLUMN]).mean())
    results['constant_bias'] = compute_metrics(test['sat_sss'] - train_bias, test[TARGET_COLUMN])

    # --- Baseline 3: linear regression on the operational feature set ---
    scaler = Standardizer()
    X_train = scaler.fit_transform(train[BASELINE_FEATURES].to_numpy(dtype=np.float64))
    X_test = scaler.transform(test[BASELINE_FEATURES].to_numpy(dtype=np.float64))
    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    lr_model = LinearRegression().fit(X_train, y_train)
    results['linear_regression'] = compute_metrics(lr_model.predict(X_test), test[TARGET_COLUMN])

    # --- FFANN, operational (baseline) feature set ---
    results['ffann_baseline_features'] = fit_and_eval_ffann(BASELINE_FEATURES, train, val, test, 'baseline features')

    # --- FFANN, rich feature set (research upper bound) ---
    results['ffann_rich_features'] = fit_and_eval_ffann(RICH_FEATURES, train, val, test, 'rich features')

    print(f"\n=== Rich-feature POC test-set results (n={len(test)}, {args.split} split) ===")
    print(f"{'method':<26}{'n':>6}{'rmse':>10}{'bias':>10}{'corr':>10}")
    for name, m in results.items():
        print(f"{name:<26}{m['n']:>6}{m['rmse']:>10.4f}{m['bias']:>10.4f}{m['corr']:>10.4f}")


if __name__ == '__main__':
    main()
