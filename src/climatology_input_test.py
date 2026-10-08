#!/usr/bin/env python3
"""
Does a climatology input improve the existing correction model, separately from
how the target is defined? (DESIGN.md 42)

The ladder's top rung (R6, 36 inputs incl. sat_sss) beat the existing model
(35 inputs, no climatology) by ~0.02 PSU, but differed in two ways at once:
it had the climatology as an input AND predicted Argo - climatology instead of
Argo - sat_sss. Three configurations, same split, same 5 seeds:

  C0  existing:            35 inputs,            target = Argo - sat_sss
  C1  + climatology input: 35 + clim,            target = Argo - sat_sss
  C2  ladder R6:           35 + clim,            target = Argo - clim

C1 - C0 isolates the climatology input; C2 - C1 isolates the target definition.
The climatology is hierarchical cell means of train-split Argo salinity (leave-
one-out for train rows), exactly as in ladder_tb_retrieval.py.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from compare_basin_ablation import drop_basin
from ladder_tb_retrieval import climatology
from train_baseline import compute_metrics, train_ffann
from train_rich_features_poc import (
    RICH_FEATURES, TARGET_COLUMN, Standardizer, add_features, chronological_split,
)

DEFAULT_MATCHUPS = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups_gdacdirect.parquet'
OUT = '/Users/afeman/Desktop/work/sss-bias/data/matchups/climatology_input_test_results.parquet'


def fit_predict(features, baseline, train, val, test, seed):
    scaler = Standardizer()
    X_train = scaler.fit_transform(train[features].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val[features].to_numpy(dtype=np.float64))
    X_test = scaler.transform(test[features].to_numpy(dtype=np.float64))
    r_train = (train[TARGET_COLUMN] - train[baseline]).to_numpy(dtype=np.float64)
    r_val = (val[TARGET_COLUMN] - val[baseline]).to_numpy(dtype=np.float64)
    torch.manual_seed(seed)
    model = train_ffann(X_train, r_train, X_val, r_val, n_features=X_train.shape[1])
    model.eval()
    with torch.no_grad():
        r = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    return test[baseline].to_numpy(dtype=np.float64) + r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-path', default=DEFAULT_MATCHUPS)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    df = add_features(pd.read_parquet(args.matchups_path)).dropna(subset=RICH_FEATURES).reset_index(drop=True)
    train, val, test = chronological_split(df)
    train, val, test = (d.assign(clim=climatology(train, d, d is train)) for d in (train, val, test))
    print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)

    base = drop_basin(RICH_FEATURES)
    configs = {
        'C0 existing (35 in, target-sat_sss)': (base, 'sat_sss'),
        'C1 +clim input (36 in, target-sat_sss)': (['clim'] + base, 'sat_sss'),
        'C2 +clim input, target-clim (= ladder R6)': (['clim'] + base, 'clim'),
    }
    y = test[TARGET_COLUMN].to_numpy(dtype=np.float64)
    preds = {k: [] for k in configs}
    rows = []
    for name, (feats, baseline) in configs.items():
        for seed in range(args.seeds):
            p = fit_predict(feats, baseline, train, val, test, seed)
            preds[name].append(p)
            m = compute_metrics(p, y)
            rows.append({'config': name, 'seed': seed, **m})
            print(f"  {name:<44} seed={seed}  rmse={m['rmse']:.4f}  bias={m['bias']:+.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_parquet(OUT, index=False)
    print(f"\n=== over {args.seeds} seeds (test n={len(test)}) ===")
    for name in configs:
        v = res.loc[res['config'] == name, 'rmse']
        print(f"{name:<44} {v.mean():.4f} +/- {v.std():.4f}  ({v.min():.4f}-{v.max():.4f})")

    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    ens = {k: np.mean(v, axis=0) for k, v in preds.items()}
    names = list(configs)
    print("\nseed-averaged predictions; paired row-bootstrap of RMSE(first) - RMSE(second) (positive = second better):")
    rng = np.random.default_rng(0)
    idxs = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        ea, eb = ens[names[a]] - y, ens[names[b]] - y
        d = np.array([rmse(ea[i]) - rmse(eb[i]) for i in idxs])
        print(f"   {names[a][:2]} - {names[b][:2]}: {rmse(ea):.4f} vs {rmse(eb):.4f}  diff {d.mean():+.4f}, "
              f"95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
    print(f"\nSaved per-run results to {OUT}")


if __name__ == '__main__':
    main()
