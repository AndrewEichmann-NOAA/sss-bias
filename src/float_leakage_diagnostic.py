#!/usr/bin/env python3
"""
Empirically test whether float-level train/test leakage is material for
this model, rather than assuming it (see DESIGN.md 18.3 discussion).

Trains on the pure date-based split (split_data_naive -- ignores float ID,
recovers the full-size test set), then splits the test set into three
groups by float membership in train:
  - 'seen'    : float ID known, and that float also has rows in train
  - 'unseen'  : float ID known, and that float has NO rows in train
  - 'unknown' : float ID not recovered (can't tell either way)

If 'seen' and 'unseen' have comparably good error, that's evidence this
model doesn't benefit from having trained on the same physical instrument
before -- i.e. float ID doesn't matter here and the float-aware split in
features.py::split_data() is unnecessary overhead. If 'seen' is notably
better than 'unseen', that's evidence of real instrument-level leakage.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from features import add_features, split_data_naive, Standardizer, FEATURE_COLUMNS, TARGET_COLUMN
from train_baseline import train_ffann, compute_metrics

torch.manual_seed(0)
np.random.seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', choices=['smap', 'smos'], default='smap')
    parser.add_argument('--matchups', default=None)
    args = parser.parse_args()

    matchups_path = args.matchups or f'/Users/afeman/Desktop/work/sss-bias/data/matchups/{args.sensor}_argo_matchups.parquet'

    print(f"Loading {args.sensor} matchup table...")
    df = pd.read_parquet(matchups_path)
    df = add_features(df)
    train, val, test = split_data_naive(df)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  (pure date-based, no float partitioning)")

    train_floats = set(train['wmo'].dropna())
    print(f"{len(train_floats)} unique floats in train")

    def group_of(row):
        if pd.isna(row):
            return 'unknown'
        return 'seen' if row in train_floats else 'unseen'

    test = test.copy()
    test['float_group'] = test['wmo'].apply(group_of)
    print("\nTest-set composition by float group:")
    print(test['float_group'].value_counts())

    scaler = Standardizer()
    X_train = scaler.fit_transform(train[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    X_test = scaler.transform(test[FEATURE_COLUMNS].to_numpy(dtype=np.float64))

    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_val = val[TARGET_COLUMN].to_numpy(dtype=np.float64)

    resid_train = y_train - train['sat_sss'].to_numpy(dtype=np.float64)
    resid_val = y_val - val['sat_sss'].to_numpy(dtype=np.float64)

    print("\nTraining FFANN on the date-based (non-float-partitioned) train set...")
    model = train_ffann(X_train, resid_train, X_val, resid_val, n_features=X_train.shape[1])

    model.eval()
    with torch.no_grad():
        resid_pred_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    test['pred_ffann'] = test['sat_sss'].to_numpy(dtype=np.float64) + resid_pred_test

    print(f"\n=== {args.sensor.upper()}: FFANN error by float-group membership ===")
    overall = compute_metrics(test['pred_ffann'], test[TARGET_COLUMN])
    print(f"{'group':<12}{'n':>8}{'rmse':>10}{'bias':>10}{'corr':>10}")
    print(f"{'overall':<12}{overall['n']:>8}{overall['rmse']:>10.4f}{overall['bias']:>10.4f}{overall['corr']:>10.4f}")
    for group in ['seen', 'unseen', 'unknown']:
        sub = test[test['float_group'] == group]
        if len(sub) == 0:
            print(f"{group:<12}{'--':>8}")
            continue
        m = compute_metrics(sub['pred_ffann'], sub[TARGET_COLUMN])
        print(f"{group:<12}{m['n']:>8}{m['rmse']:>10.4f}{m['bias']:>10.4f}{m['corr']:>10.4f}")

    # Also check raw (uncorrected) satellite error by group, as a control --
    # if raw error already differs by group, that's a population difference
    # unrelated to the model (e.g. seen-floats sit in different regions),
    # not evidence of the model specifically exploiting train exposure.
    print(f"\n=== {args.sensor.upper()}: RAW (uncorrected) satellite error by float-group, for comparison ===")
    for group in ['seen', 'unseen', 'unknown']:
        sub = test[test['float_group'] == group]
        if len(sub) == 0:
            continue
        m = compute_metrics(sub['sat_sss'], sub[TARGET_COLUMN])
        print(f"{group:<12}{m['n']:>8}{m['rmse']:>10.4f}{m['bias']:>10.4f}{m['corr']:>10.4f}")


if __name__ == '__main__':
    main()
