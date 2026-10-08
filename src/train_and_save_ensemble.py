#!/usr/bin/env python3
"""
Trains an N-seed ensemble of the no-basin rich-feature FFANN on the 'recent'
split (DESIGN.md 41: train through 2025-09-30, validate on 2025-10-01 up to the
day before the IODA window) and saves all members in one checkpoint, for
generate_bias_corrected_ioda.py to average. Seed-averaging the predictions
lowered held-out RMSE by ~0.01-0.02 PSU relative to a single network.

Members share an identical train set, so they share one Standardizer; they
differ only in weight initialization (training has no other randomness).
Everything after the validation end (default 2025-12-10) is held out.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from compare_basin_ablation import drop_basin
from train_baseline import FFANN, compute_metrics, train_ffann
from train_rich_features_poc import RICH_FEATURES, TARGET_COLUMN, Standardizer, add_features

DEFAULT_MATCHUPS = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups_gdacdirect.parquet'
DEFAULT_OUT = '/Users/afeman/Desktop/work/sss-bias/data/matchups/rich_correction_ensemble_recent_nobasin.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-path', default=DEFAULT_MATCHUPS)
    parser.add_argument('--train-end', default='2025-09-30', help='last sat_datetime date included in training')
    parser.add_argument('--heldout-start', default='2025-12-10',
                         help='validation covers (train-end, heldout-start); everything from here on is held out')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--out', default=DEFAULT_OUT)
    args = parser.parse_args()

    df = add_features(pd.read_parquet(args.matchups_path)).dropna(subset=RICH_FEATURES).reset_index(drop=True)
    features = drop_basin(RICH_FEATURES)
    t = df['sat_datetime']
    train = df[t <= args.train_end].reset_index(drop=True)
    val = df[(t > args.train_end) & (t < pd.Timestamp(args.heldout_start))].reset_index(drop=True)
    heldout = df[t >= pd.Timestamp(args.heldout_start)].reset_index(drop=True)
    print(f"train={len(train)} val={len(val)} heldout={len(heldout)} features={len(features)}", flush=True)

    scaler = Standardizer()
    X_train = scaler.fit_transform(train[features].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val[features].to_numpy(dtype=np.float64))
    X_held = scaler.transform(heldout[features].to_numpy(dtype=np.float64))
    resid_train = (train[TARGET_COLUMN] - train['sat_sss']).to_numpy(dtype=np.float64)
    resid_val = (val[TARGET_COLUMN] - val['sat_sss']).to_numpy(dtype=np.float64)
    y_held = heldout[TARGET_COLUMN].to_numpy(dtype=np.float64)

    states, member_preds = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = train_ffann(X_train, resid_train, X_val, resid_val, n_features=X_train.shape[1])
        model.eval()
        with torch.no_grad():
            resid = model(torch.tensor(X_held, dtype=torch.float32)).numpy()
        pred = heldout['sat_sss'].to_numpy(dtype=np.float64) + resid
        member_preds.append(pred)
        states.append(model.state_dict())
        print(f"  member seed={seed}: held-out rmse={compute_metrics(pred, y_held)['rmse']:.4f}", flush=True)

    ens = compute_metrics(np.mean(member_preds, axis=0), y_held)
    print(f"ensemble (mean of {args.seeds}) held-out: {ens}", flush=True)

    torch.save({
        'model_state_dicts': states,
        'hidden': (32, 16),
        'n_features': len(features),
        'rich_features': features,
        'scaler_mean': scaler.mean_,
        'scaler_std': scaler.std_,
        'target_column': TARGET_COLUMN,
        'seeds': list(range(args.seeds)),
        'train_end': args.train_end,
        'heldout_start': args.heldout_start,
        'heldout_metrics': ens,
    }, args.out)
    print(f"Saved ensemble checkpoint to {args.out}")


if __name__ == '__main__':
    main()
