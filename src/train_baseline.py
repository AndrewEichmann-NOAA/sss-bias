#!/usr/bin/env python3
"""
Phase 1 baseline: train a small PyTorch FFANN to correct satellite SSS
(SMAP or SMOS) to Argo bulk salinity, and compare against raw, constant-
bias, and linear regression baselines on the held-out test split.

Inputs: sat_sss, sat_lat, lon_sin/cos, doy_sin/cos, oceanBasin one-hot.
Target: argo_salinity. Model predicts a residual (argo_salinity - sat_sss),
not salinity directly -- see DESIGN.md section 7.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from features import add_features, split_data_naive as split_data, Standardizer, FEATURE_COLUMNS, TARGET_COLUMN
# Uses the plain date-based split, not the float-aware split_data() -- see
# DESIGN.md 18.3: tested empirically whether float-level train/test leakage
# is material for this model and found no evidence of it, so the float-aware
# version (kept in features.py for reference) isn't worth its cost in
# val/test size.

torch.manual_seed(0)
np.random.seed(0)


class FFANN(nn.Module):
    def __init__(self, n_features, hidden=(32, 16)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_metrics(pred, actual):
    diff = np.asarray(pred) - np.asarray(actual)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    corr = float(np.corrcoef(pred, actual)[0, 1]) if len(actual) > 1 else float('nan')
    return {'rmse': rmse, 'bias': bias, 'corr': corr, 'n': int(len(actual))}


def train_ffann(X_train, y_resid_train, X_val, y_resid_val, n_features,
                 hidden=(32, 16), lr=1e-3, max_epochs=300, patience=20):
    model = FFANN(n_features, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_resid_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_resid_val, dtype=torch.float32)

    best_val_loss = float('inf')
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch % 20 == 0 or epochs_since_improvement == 0:
            print(f"  epoch {epoch:3d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

        if epochs_since_improvement >= patience:
            print(f"  early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    return model


def breakdown(df, pred_col, label):
    print(f"\n  -- {label} --")
    lat_bins = [(-90, -60), (-60, -30), (-30, 30), (30, 60), (60, 90)]
    for lo, hi in lat_bins:
        sub = df[(df['sat_lat'] >= lo) & (df['sat_lat'] < hi)]
        if len(sub) == 0:
            continue
        m = compute_metrics(sub[pred_col], sub[TARGET_COLUMN])
        print(f"    lat [{lo:4d},{hi:4d}): n={m['n']:6d} rmse={m['rmse']:.3f} bias={m['bias']:+.3f}")
    for basin, sub in df.groupby('sat_oceanBasin'):
        m = compute_metrics(sub[pred_col], sub[TARGET_COLUMN])
        print(f"    basin {basin:.0f}: n={m['n']:6d} rmse={m['rmse']:.3f} bias={m['bias']:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', choices=['smap', 'smos'], default='smap',
                         help='Used only to pick default --matchups path and label output')
    parser.add_argument('--matchups', default=None,
                         help='Defaults to data/matchups/<sensor>_argo_matchups.parquet')
    parser.add_argument('--out-dir', default='/Users/afeman/Desktop/work/sss-bias/data/matchups')
    args = parser.parse_args()

    matchups_path = args.matchups or f'/Users/afeman/Desktop/work/sss-bias/data/matchups/{args.sensor}_argo_matchups.parquet'

    print(f"Loading {args.sensor} matchup table...")
    df = pd.read_parquet(matchups_path)
    df = add_features(df)
    train, val, test = split_data(df)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    scaler = Standardizer()
    X_train = scaler.fit_transform(train[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    X_test = scaler.transform(test[FEATURE_COLUMNS].to_numpy(dtype=np.float64))

    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_val = val[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=np.float64)

    results = {}

    # --- Baseline 1: raw satellite SSS, no correction ---
    test = test.assign(pred_raw=test['sat_sss'])
    results['raw'] = compute_metrics(test['pred_raw'], test[TARGET_COLUMN])

    # --- Baseline 2: constant bias correction (train-set mean offset) ---
    train_bias = float((train['sat_sss'] - train[TARGET_COLUMN]).mean())
    test = test.assign(pred_constant=test['sat_sss'] - train_bias)
    results['constant_bias'] = compute_metrics(test['pred_constant'], test[TARGET_COLUMN])

    # --- Baseline 3: linear regression on the same feature set ---
    lr_model = LinearRegression().fit(X_train, y_train)
    test = test.assign(pred_linreg=lr_model.predict(X_test))
    results['linear_regression'] = compute_metrics(test['pred_linreg'], test[TARGET_COLUMN])

    # --- Model: FFANN, residual formulation ---
    resid_train = y_train - train['sat_sss'].to_numpy(dtype=np.float64)
    resid_val = y_val - val['sat_sss'].to_numpy(dtype=np.float64)

    print("\nTraining FFANN...")
    model = train_ffann(X_train, resid_train, X_val, resid_val, n_features=X_train.shape[1])

    model.eval()
    with torch.no_grad():
        resid_pred_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    test = test.assign(pred_ffann=test['sat_sss'].to_numpy(dtype=np.float64) + resid_pred_test)
    results['ffann'] = compute_metrics(test['pred_ffann'], test[TARGET_COLUMN])

    # --- Report ---
    print(f"\n=== {args.sensor.upper()} test-set results (2023-07-15 to 2023-12-31) ===")
    print(f"{'method':<20}{'n':>8}{'rmse':>10}{'bias':>10}{'corr':>10}")
    for name, m in results.items():
        print(f"{name:<20}{m['n']:>8}{m['rmse']:>10.4f}{m['bias']:>10.4f}{m['corr']:>10.4f}")

    for name, col in [('raw', 'pred_raw'), ('constant_bias', 'pred_constant'),
                       ('linear_regression', 'pred_linreg'), ('ffann', 'pred_ffann')]:
        breakdown(test, col, name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f'phase1_results_{args.sensor}.json'
    model_path = out_dir / f'phase1_ffann_{args.sensor}.pt'
    predictions_path = out_dir / f'phase1_test_predictions_{args.sensor}.parquet'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), model_path)
    pred_cols = ['sat_lat', 'sat_lon', 'sat_sss', 'sat_oceanBasin', TARGET_COLUMN,
                 'pred_raw', 'pred_constant', 'pred_linreg', 'pred_ffann']
    test[pred_cols].to_parquet(predictions_path, index=False)
    print(f"\nSaved results to {results_path}")
    print(f"Saved model to {model_path}")
    print(f"Saved test-set predictions to {predictions_path}")


if __name__ == '__main__':
    main()
