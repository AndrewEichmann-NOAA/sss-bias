#!/usr/bin/env python3
"""
Permutation feature importance for the rich-feature FFANN (train_rich_features_poc.py,
DESIGN.md 21-26): trains the same model on the same chronological split, then for each
of the 40 rich input features, shuffles just that column in the test set and measures
how much test RMSE degrades relative to the unshuffled baseline. A feature the model
actually relies on will show a large RMSE increase when scrambled; a feature it ignores
(or that's redundant with another) will show little to no increase.

Chosen over weight-magnitude inspection because a 2-layer MLP's input-layer weights
don't account for interactions or downstream layers, and over ablation/retraining
because that would require retraining 40 separate models. Permutation importance
needs only the one already-trained model, evaluated repeatedly.

Averaged over N_REPEATS independent shufflings per feature to smooth out the shuffle's
own randomness -- a single shuffle can over/under-state a feature's importance by luck.
"""

import numpy as np
import pandas as pd
import torch

from train_baseline import compute_metrics, train_ffann
from train_rich_features_poc import add_features, chronological_split, Standardizer, RICH_FEATURES, TARGET_COLUMN

MATCHUPS_PATH = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups.parquet'
OUT_PATH = '/Users/afeman/Desktop/work/sss-bias/data/matchups/rich_feature_importance.parquet'
N_REPEATS = 10
SEED = 0


def main():
    df = pd.read_parquet(MATCHUPS_PATH)
    df = add_features(df)
    n_before = len(df)
    df = df.dropna(subset=RICH_FEATURES).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with NaN in a rich feature")

    train, val, test = chronological_split(df)
    print(f"{len(df)} matchups; train={len(train)} val={len(val)} test={len(test)}")

    scaler = Standardizer()
    X_train = scaler.fit_transform(train[RICH_FEATURES].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val[RICH_FEATURES].to_numpy(dtype=np.float64))
    X_test = scaler.transform(test[RICH_FEATURES].to_numpy(dtype=np.float64))

    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_val = val[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=np.float64)
    sat_sss_test = test['sat_sss'].to_numpy(dtype=np.float64)

    resid_train = y_train - train['sat_sss'].to_numpy(dtype=np.float64)
    resid_val = y_val - val['sat_sss'].to_numpy(dtype=np.float64)

    print("\nTraining rich-feature FFANN...")
    model = train_ffann(X_train, resid_train, X_val, resid_val, n_features=X_train.shape[1])
    model.eval()

    def predict(X):
        with torch.no_grad():
            resid = model(torch.tensor(X, dtype=torch.float32)).numpy()
        return sat_sss_test + resid

    baseline_rmse = compute_metrics(predict(X_test), y_test)['rmse']
    print(f"Baseline (unshuffled) test RMSE: {baseline_rmse:.4f}\n")

    rng = np.random.default_rng(SEED)
    rows = []
    for j, feat in enumerate(RICH_FEATURES):
        deltas = []
        for _ in range(N_REPEATS):
            X_perm = X_test.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_rmse = compute_metrics(predict(X_perm), y_test)['rmse']
            deltas.append(perm_rmse - baseline_rmse)
        rows.append({'feature': feat, 'delta_rmse_mean': np.mean(deltas), 'delta_rmse_std': np.std(deltas)})
        print(f"  {feat:<28} delta_rmse = {np.mean(deltas):+.4f} (+/-{np.std(deltas):.4f})")

    imp_df = pd.DataFrame(rows).sort_values('delta_rmse_mean', ascending=False).reset_index(drop=True)
    print(f"\n=== Ranked by mean RMSE increase when shuffled ({N_REPEATS} repeats) ===")
    print(imp_df.to_string(index=False))

    imp_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == '__main__':
    main()
