#!/usr/bin/env python3
"""
Is the benefit of the 'recent' training split (DESIGN.md 41, 44) more data, or more
recent data? Three models, same inputs (35, no basins), same target (Argo - sat_sss),
same seeds, scored on the same untrained rows as 41.1 (sat_datetime >= 2025-12-10):

  A  old split:      train <= 2023-12-31                          (n = 63,620; early-stop val: Jan-Feb 2024)
  B  recent split:   train <= 2025-09-30                          (n = 96,495; early-stop val: Oct 1 - Dec 9 2025)
  C  equal-size:     the most recent len(A) rows with sat_datetime <= 2025-09-30
                     (a subset of B's training set, same size as A's)

C vs. A: same size, different period  -> isolates recency.
C vs. B: same end date, different size -> isolates volume.
C and B share the same early-stopping validation rows; A's differ (its own adjacent
validation period) -- unavoidable if each model is validated just before its training
end, and a minor confound for the A comparison.
"""

import argparse

import numpy as np
import pandas as pd

from compare_recency_training import fit_predict
from train_baseline import compute_metrics
from train_rich_features_poc import RICH_FEATURES, TARGET_COLUMN, add_features, chronological_split

DEFAULT_MATCHUPS = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups_gdacdirect.parquet'
OUT_DIR = '/Users/afeman/Desktop/work/sss-bias/data/matchups'
WINDOW_END = pd.Timestamp('2026-01-20')
HELDOUT_START = pd.Timestamp('2025-12-10')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-path', default=DEFAULT_MATCHUPS)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    df = add_features(pd.read_parquet(args.matchups_path)).dropna(subset=RICH_FEATURES).reset_index(drop=True)
    features = list(RICH_FEATURES)
    t = df['sat_datetime']

    train_a, val_a, _ = chronological_split(df)
    train_b = df[t <= '2025-09-30'].reset_index(drop=True)
    val_b = df[(t > '2025-09-30') & (t < HELDOUT_START)].reset_index(drop=True)
    train_c = train_b.sort_values('sat_datetime').tail(len(train_a)).reset_index(drop=True)
    ev = df[t >= HELDOUT_START].reset_index(drop=True)
    in_win = (ev['sat_datetime'] < WINDOW_END).to_numpy()
    subsets = {'window': in_win, 'later': ~in_win, 'all': np.ones(len(ev), dtype=bool)}
    y = ev[TARGET_COLUMN].to_numpy(dtype=np.float64)

    span = lambda d: f"{d['sat_datetime'].min().date()} .. {d['sat_datetime'].max().date()}"
    print(f"A: train n={len(train_a)} [{span(train_a)}], val n={len(val_a)}", flush=True)
    print(f"B: train n={len(train_b)} [{span(train_b)}], val n={len(val_b)}", flush=True)
    print(f"C: train n={len(train_c)} [{span(train_c)}], val n={len(val_b)} (same as B)", flush=True)
    print(f"eval: window={in_win.sum()} later={(~in_win).sum()}", flush=True)

    models = {'A': (train_a, val_a), 'B': (train_b, val_b), 'C': (train_c, val_b)}
    preds = {k: [] for k in models}
    rows = []
    for seed in range(args.seeds):
        for name, (tr, va) in models.items():
            p = fit_predict(features, tr, va, ev, seed)
            preds[name].append(p)
            out = {'model': name, 'seed': seed}
            for sub, m in subsets.items():
                out[f'rmse_{sub}'] = compute_metrics(p[m], y[m])['rmse']
            rows.append(out)
            print(f"  model {name} seed={seed}  window={out['rmse_window']:.4f} later={out['rmse_later']:.4f} "
                  f"all={out['rmse_all']:.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_parquet(f'{OUT_DIR}/equal_size_recency_per_seed.parquet', index=False)
    ens = {k: np.mean(v, axis=0) for k, v in preds.items()}
    pd.DataFrame({'sat_datetime': ev['sat_datetime'], 'in_window': in_win, 'argo_salinity': y,
                  **{f'pred_{k}_ens': v for k, v in ens.items()}}
                 ).to_parquet(f'{OUT_DIR}/equal_size_recency_eval_predictions.parquet', index=False)

    print(f"\n=== single networks over {args.seeds} seeds: mean +/- std (min-max) ===")
    for sub in subsets:
        print(f"[{sub}  n={subsets[sub].sum()}]")
        for name in models:
            v = res.loc[res['model'] == name, f'rmse_{sub}']
            print(f"   model {name}: {v.mean():.4f} +/- {v.std():.4f}  ({v.min():.4f}-{v.max():.4f})")

    rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))
    print("\n=== seed-averaged ensembles; paired row-bootstrap, positive = second is better ===")
    rng = np.random.default_rng(0)
    for sub, m in subsets.items():
        n = int(m.sum())
        idx = [rng.integers(0, n, n) for _ in range(2000)]
        errs = {k: (v - y)[m] for k, v in ens.items()}
        print(f"[{sub}]  ensemble RMSE: " + " | ".join(f"{k} {rmse(e):.4f}" for k, e in errs.items()))
        for a, b in [('A', 'C'), ('C', 'B'), ('A', 'B')]:
            d = np.array([rmse(errs[a][i]) - rmse(errs[b][i]) for i in idx])
            print(f"      {a} - {b}: {d.mean():+.4f}  95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")


if __name__ == '__main__':
    main()
