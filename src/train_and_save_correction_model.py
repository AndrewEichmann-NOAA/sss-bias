#!/usr/bin/env python3
"""
Trains the rich-feature FFANN bias-correction model (same architecture/split
as train_rich_features_poc.py, on the full GDAC-direct matchup table,
DESIGN.md 37) and saves everything needed for later inference on new raw
SMAP files: model weights, the feature Standardizer's mean/std, and the
RICH_FEATURES column order -- so a downstream script can reproduce
predictions without re-running the training pipeline.

Built for DESIGN.md 39: generating bias-corrected IODA SSS files to replace
the ones in common_obsForge for a specific cycle range.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from train_baseline import FFANN, train_ffann
from train_rich_features_poc import (
    add_features, chronological_split, RICH_FEATURES, TARGET_COLUMN, Standardizer, compute_metrics,
)

DEFAULT_MATCHUPS = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups_gdacdirect.parquet'
# Default output is the no-basin file. The older with-basin checkpoint,
# rich_correction_model.pt, is kept as-is and must not be overwritten (DESIGN.md 44.1).
DEFAULT_OUT = '/Users/afeman/Desktop/work/sss-bias/data/matchups/rich_correction_model_nobasin.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-path', default=DEFAULT_MATCHUPS)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=0,
                         help='torch seed for weight initialization (training has no other randomness).')
    args = parser.parse_args()
    out_path = args.out
    features = list(RICH_FEATURES)

    df = pd.read_parquet(args.matchups_path)
    df = add_features(df)
    df = df.dropna(subset=RICH_FEATURES).reset_index(drop=True)
    train, val, test = chronological_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    scaler = Standardizer()
    X_train = scaler.fit_transform(train[features].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val[features].to_numpy(dtype=np.float64))
    X_test = scaler.transform(test[features].to_numpy(dtype=np.float64))

    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_val = val[TARGET_COLUMN].to_numpy(dtype=np.float64)
    resid_train = y_train - train['sat_sss'].to_numpy(dtype=np.float64)
    resid_val = y_val - val['sat_sss'].to_numpy(dtype=np.float64)

    torch.manual_seed(args.seed)
    model = train_ffann(X_train, resid_train, X_val, resid_val, n_features=X_train.shape[1])

    model.eval()
    with torch.no_grad():
        resid_pred_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    pred = test['sat_sss'].to_numpy(dtype=np.float64) + resid_pred_test
    metrics = compute_metrics(pred, test[TARGET_COLUMN])
    print(f"Test metrics (sanity check, should match train_rich_features_poc.py): {metrics}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden': (32, 16),
        'n_features': len(features),
        'rich_features': features,
        'seed': args.seed,
        'scaler_mean': scaler.mean_,
        'scaler_std': scaler.std_,
        'target_column': TARGET_COLUMN,
        'test_metrics': metrics,
    }, out_path)
    print(f"Saved model checkpoint to {out_path}")


if __name__ == '__main__':
    main()
