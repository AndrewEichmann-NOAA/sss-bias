#!/usr/bin/env python3
"""
Does more recent training data help the period being corrected? (DESIGN.md 41)

Both models use the no-basin rich feature set (DESIGN.md 40) and are scored on
the same held-out rows, taken straight from the raw-SMAP/GDAC-Argo matchup table
(no IODA files involved): everything with sat_datetime >= 2025-12-10, split into
the 'window' (2025-12-10..2026-01-19, the period the bias-corrected IODA files
cover, ~2k rows) and 'later' (2026-01-20 onward, ~10k rows -- more statistical
power, and a longer gap from training, i.e. a staleness test).

  A (old split):    train <= 2023-12-31, val to 2024-02-29  (what generated the IODA files)
  B (recent split): train <= 2025-09-30, val 2025-10-01 .. 2025-12-09

All of it is after both models' validation sets, so it is held out for both.
N seeds per model (training has no randomness beyond weight initialization);
per-seed RMSE on the window, plus a paired row-bootstrap on the seed-averaged
predictions to show sampling uncertainty at this sample size.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from compare_basin_ablation import drop_basin
from train_baseline import compute_metrics, train_ffann
from train_rich_features_poc import (
    RICH_FEATURES, TARGET_COLUMN, Standardizer, add_features, chronological_split,
)

DEFAULT_MATCHUPS = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups_gdacdirect.parquet'
OUT_DIR = '/Users/afeman/Desktop/work/sss-bias/data/matchups'
WINDOW = (pd.Timestamp('2025-12-10'), pd.Timestamp('2026-01-20'))


def fit_predict(features, train, val, evalset, seed):
    scaler = Standardizer()
    X_train = scaler.fit_transform(train[features].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val[features].to_numpy(dtype=np.float64))
    X_win = scaler.transform(evalset[features].to_numpy(dtype=np.float64))
    resid_train = (train[TARGET_COLUMN] - train['sat_sss']).to_numpy(dtype=np.float64)
    resid_val = (val[TARGET_COLUMN] - val['sat_sss']).to_numpy(dtype=np.float64)

    torch.manual_seed(seed)
    model = train_ffann(X_train, resid_train, X_val, resid_val, n_features=X_train.shape[1])
    model.eval()
    with torch.no_grad():
        resid = model(torch.tensor(X_win, dtype=torch.float32)).numpy()
    return evalset['sat_sss'].to_numpy(dtype=np.float64) + resid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-path', default=DEFAULT_MATCHUPS)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    df = add_features(pd.read_parquet(args.matchups_path))
    df = df.dropna(subset=RICH_FEATURES).reset_index(drop=True)
    features = drop_basin(RICH_FEATURES)

    t = df['sat_datetime']
    ev = df[t >= WINDOW[0]].reset_index(drop=True)
    in_win = (ev['sat_datetime'] < WINDOW[1]).to_numpy()
    subsets = {'window': in_win, 'later': ~in_win, 'all': np.ones(len(ev), dtype=bool)}

    train_a, val_a, _ = chronological_split(df)
    train_b = df[t <= '2025-09-30'].reset_index(drop=True)
    val_b = df[(t > '2025-09-30') & (t < WINDOW[0])].reset_index(drop=True)
    print(f"A: train={len(train_a)} val={len(val_a)} | B: train={len(train_b)} val={len(val_b)} | "
          f"eval: window={in_win.sum()} later={(~in_win).sum()}  features={len(features)}", flush=True)

    y = ev[TARGET_COLUMN].to_numpy(dtype=np.float64)
    for name, m in subsets.items():
        r = compute_metrics(ev['sat_sss'].to_numpy()[m], y[m])
        print(f"raw satellite, {name}: rmse={r['rmse']:.4f} bias={r['bias']:+.4f}", flush=True)

    preds = {'A': [], 'B': []}
    rows = []
    for seed in range(args.seeds):
        for name, (tr, va) in {'A': (train_a, val_a), 'B': (train_b, val_b)}.items():
            p = fit_predict(features, tr, va, ev, seed)
            preds[name].append(p)
            out = {'model': name, 'seed': seed}
            for sub, m in subsets.items():
                r = compute_metrics(p[m], y[m])
                out[f'rmse_{sub}'], out[f'bias_{sub}'] = r['rmse'], r['bias']
            rows.append(out)
            print(f"  model {name} seed={seed}  rmse window={out['rmse_window']:.4f} "
                  f"later={out['rmse_later']:.4f} all={out['rmse_all']:.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_parquet(f'{OUT_DIR}/recency_training_per_seed.parquet', index=False)

    ens = {k: np.mean(v, axis=0) for k, v in preds.items()}
    pd.DataFrame({
        'sat_lat': ev['sat_lat'], 'sat_lon': ev['sat_lon'], 'sat_datetime': ev['sat_datetime'],
        'in_window': in_win, 'argo_salinity': y, 'sat_sss': ev['sat_sss'],
        'pred_A_ens': ens['A'], 'pred_B_ens': ens['B'],
    }).to_parquet(f'{OUT_DIR}/recency_training_eval_predictions.parquet', index=False)

    print(f"\n=== RMSE over {args.seeds} seeds, mean +/- std (min-max) ===")
    for sub in subsets:
        print(f"[{sub}  n={subsets[sub].sum()}]")
        for name in ('A', 'B'):
            v = res.loc[res['model'] == name, f'rmse_{sub}']
            print(f"   model {name}: {v.mean():.4f} +/- {v.std():.4f}  ({v.min():.4f}-{v.max():.4f})")

    rmse = lambda e: np.sqrt((e ** 2).mean())
    print("\nseed-averaged predictions, paired row-bootstrap of RMSE(A) - RMSE(B) (positive = B better):")
    rng = np.random.default_rng(0)
    for sub, m in subsets.items():
        ea, eb = (ens['A'] - y)[m], (ens['B'] - y)[m]
        d = np.array([rmse(ea[i]) - rmse(eb[i]) for i in (rng.integers(0, m.sum(), m.sum()) for _ in range(5000))])
        print(f"   {sub:<7} A={rmse(ea):.4f} B={rmse(eb):.4f}  diff mean {d.mean():+.4f}, "
              f"95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")


if __name__ == '__main__':
    main()
