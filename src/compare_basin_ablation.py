#!/usr/bin/env python3
"""
Basin-input ablation (DESIGN.md 40): trains the FFANN with and without the
basin_0..5 one-hot inputs, on the same chronological split, over several
random initializations each. The basin flags are a pure function of lat/lon in
the GDAC-direct table (classify_ocean_basin.py), so they add no information
beyond what the model already has; this measures whether they help anyway.

A single run per config isn't enough: training has no fixed seed and earlier
identical runs differed by ~0.005 PSU. Full-batch training is deterministic
given the initialization, so seeding torch before each run is all that varies.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from train_baseline import compute_metrics, train_ffann
from train_rich_features_poc import (
    BASELINE_FEATURES, BASELINE_FEATURES_WITH_BASIN, RICH_FEATURES, RICH_FEATURES_WITH_BASIN, TARGET_COLUMN,
    Standardizer, add_features, chronological_split,
)

DEFAULT_MATCHUPS = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups_gdacdirect.parquet'
DEFAULT_OUT = '/Users/afeman/Desktop/work/sss-bias/data/matchups/basin_ablation_results.parquet'


def drop_basin(features):
    return [f for f in features if not f.startswith('basin_')]


def run(features, train, val, test, seed):
    scaler = Standardizer()
    X_train = scaler.fit_transform(train[features].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val[features].to_numpy(dtype=np.float64))
    X_test = scaler.transform(test[features].to_numpy(dtype=np.float64))

    resid_train = (train[TARGET_COLUMN] - train['sat_sss']).to_numpy(dtype=np.float64)
    resid_val = (val[TARGET_COLUMN] - val['sat_sss']).to_numpy(dtype=np.float64)

    torch.manual_seed(seed)
    model = train_ffann(X_train, resid_train, X_val, resid_val, n_features=X_train.shape[1])
    model.eval()
    with torch.no_grad():
        resid_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    pred = test['sat_sss'].to_numpy(dtype=np.float64) + resid_pred
    return compute_metrics(pred, test[TARGET_COLUMN])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-path', default=DEFAULT_MATCHUPS)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--out', default=DEFAULT_OUT)
    args = parser.parse_args()

    df = add_features(pd.read_parquet(args.matchups_path))
    df = df.dropna(subset=RICH_FEATURES_WITH_BASIN).reset_index(drop=True)
    train, val, test = chronological_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)

    configs = {
        'rich, with basin': RICH_FEATURES_WITH_BASIN,
        'rich, no basin': RICH_FEATURES,
        'baseline, with basin': BASELINE_FEATURES_WITH_BASIN,
        'baseline, no basin': BASELINE_FEATURES,
    }

    rows = []
    for name, feats in configs.items():
        for seed in range(args.seeds):
            m = run(feats, train, val, test, seed)
            rows.append({'config': name, 'n_features': len(feats), 'seed': seed, **m})
            print(f"  {name:<22} seed={seed}  rmse={m['rmse']:.4f}  bias={m['bias']:+.4f}  corr={m['corr']:.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_parquet(args.out, index=False)

    summary = res.groupby('config', sort=False).agg(
        n_features=('n_features', 'first'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'), rmse_min=('rmse', 'min'), rmse_max=('rmse', 'max'),
        bias_mean=('bias', 'mean'), corr_mean=('corr', 'mean'))
    print(f"\n=== test-set summary over {args.seeds} seeds (n={len(test)}) ===")
    print(summary.to_string(float_format=lambda v: f'{v:.4f}'))
    print(f"\nSaved per-run results to {args.out}")


if __name__ == '__main__':
    main()
