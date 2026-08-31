#!/usr/bin/env python3
"""
Walk-forward test of training-window length vs. recency, using the 2-year
raw-SMAP-CAP matchup table currently available (2022-06-01 to 2024-06-01;
the archive is still being extended further -- see DESIGN.md 26). Answers
the question raised in 26: for an operational, continuously-updated bias-
correction model, does "use all available history" actually outperform a
shorter, more recent trailing window, or does secular drift (instrument
recalibration, ENSO-scale variability, the safe-mode/gap discontinuities
already found in 22/25/26.1) argue for the opposite?

For each of several training-cutoff ("origin") dates, trains the same
baseline-feature FFANN used throughout this project on several different
trailing window lengths ending at that origin, and evaluates ALL of them on
the SAME fixed-length test period immediately following the origin -- holding
the test period fixed isolates the effect of window length/recency from the
effect of an easier or harder test period (see 26's note on why a single
train/test split can't separate these).

Only tests the operational baseline feature set (not the rich JPL CAP
fields) to keep the grid small enough to run now, on the data already
downloaded and validated; repeating with rich features, and adding more
origins once the archive extends further, is a natural follow-up.
"""

import numpy as np
import pandas as pd
import torch

from train_baseline import compute_metrics, train_ffann
from train_rich_features_poc import add_features, Standardizer, BASELINE_FEATURES, TARGET_COLUMN

MATCHUPS_PATH = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups.parquet'

TEST_SPAN_MONTHS = 2
# Origins chosen so every window length (including "all available history")
# has enough trailing data, and the 2-month test period after each stays
# inside the currently-downloaded archive (ends 2024-06-01).
ORIGINS = ['2023-09-01', '2023-12-01', '2024-03-01']
WINDOW_MONTHS = [3, 6, 12, None]  # None = all available history before the origin


def fit_eval(train, test, seed=0):
    scaler = Standardizer()
    X_train_full = scaler.fit_transform(train[BASELINE_FEATURES].to_numpy(dtype=np.float64))
    X_test = scaler.transform(test[BASELINE_FEATURES].to_numpy(dtype=np.float64))
    y_train_full = train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    resid_full = y_train_full - train['sat_sss'].to_numpy(dtype=np.float64)

    # Hold out 15% of the training window itself for early-stopping (val),
    # same role val plays in train_baseline.py -- not used for the reported
    # test metric, just to decide when to stop training.
    n_val = max(1, int(len(train) * 0.15))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    model = train_ffann(X_train_full[fit_idx], resid_full[fit_idx],
                         X_train_full[val_idx], resid_full[val_idx],
                         n_features=X_train_full.shape[1])
    model.eval()
    with torch.no_grad():
        resid_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    pred = test['sat_sss'].to_numpy(dtype=np.float64) + resid_pred
    return compute_metrics(pred, test[TARGET_COLUMN])


def main():
    df = pd.read_parquet(MATCHUPS_PATH)
    df = add_features(df)
    df = df.sort_values('sat_datetime').reset_index(drop=True)
    print(f"{len(df)} matchups, {df['sat_datetime'].min()} to {df['sat_datetime'].max()}\n")

    results = []
    for origin_str in ORIGINS:
        origin = pd.Timestamp(origin_str)
        test_end = origin + pd.DateOffset(months=TEST_SPAN_MONTHS)
        test = df[(df['sat_datetime'] >= origin) & (df['sat_datetime'] < test_end)]
        if test.empty:
            print(f"origin {origin_str}: no test data in [{origin}, {test_end}), skipping")
            continue

        print(f"--- origin {origin_str}  (test = [{origin.date()}, {test_end.date()}), n_test={len(test)}) ---")
        for window in WINDOW_MONTHS:
            if window is None:
                train = df[df['sat_datetime'] < origin]
                label = 'all-history'
            else:
                train_start = origin - pd.DateOffset(months=window)
                train = df[(df['sat_datetime'] >= train_start) & (df['sat_datetime'] < origin)]
                label = f'{window}mo'

            if len(train) < 50:
                print(f"  window {label:>12}: too little training data ({len(train)}), skipping")
                continue

            m = fit_eval(train, test)
            m.update(origin=origin_str, window=label, n_train=len(train))
            results.append(m)
            print(f"  window {label:>12}  n_train={len(train):6d}  n_test={m['n']:5d}  "
                  f"rmse={m['rmse']:.4f}  bias={m['bias']:+.4f}  corr={m['corr']:.4f}")
        print()

    results_df = pd.DataFrame(results)
    out_path = '/Users/afeman/Desktop/work/sss-bias/data/matchups/window_sweep_results.parquet'
    results_df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}\n")
    print("RMSE by window length x origin:")
    print(results_df.pivot(index='window', columns='origin', values='rmse').to_string())


if __name__ == '__main__':
    main()
