#!/usr/bin/env python3
"""
Combines the three improvements found so far (DESIGN.md 43): recent training
data (41), a train-years climatology as an input (42), and seed-averaging (41).

All on the 'recent' split of 41.1 (train <= 2025-09-30, validate 2025-10-01 up
to the day before the window), scored on the same untrained rows as 41.1
(everything from 2025-12-10: 'window' to 2026-01-19, 'later' from 2026-01-20),
no basin inputs, 5 seeds each:

  B0  recent split, existing form:  35 inputs,        target = Argo - sat_sss   (41's model B)
  B1  + climatology input:          35 + clim,        target = Argo - sat_sss
  B2  + climatology input + target: 35 + clim,        target = Argo - clim

The climatology is built from this split's TRAIN rows only (leave-one-out for
train rows), as in ladder_tb_retrieval.py. Single-seed means show per-network
quality; the seed-averaged ('ensemble') predictions are what a deployed
ensemble would deliver.
"""

import argparse

import numpy as np
import pandas as pd

from climatology_input_test import fit_predict
from compare_basin_ablation import drop_basin
from ladder_tb_retrieval import climatology
from train_baseline import compute_metrics
from train_rich_features_poc import RICH_FEATURES, TARGET_COLUMN, add_features

DEFAULT_MATCHUPS = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups_gdacdirect.parquet'
OUT_DIR = '/Users/afeman/Desktop/work/sss-bias/data/matchups'
WINDOW = (pd.Timestamp('2025-12-10'), pd.Timestamp('2026-01-20'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-path', default=DEFAULT_MATCHUPS)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    df = add_features(pd.read_parquet(args.matchups_path)).dropna(subset=RICH_FEATURES).reset_index(drop=True)
    t = df['sat_datetime']
    train = df[t <= '2025-09-30'].reset_index(drop=True)
    val = df[(t > '2025-09-30') & (t < WINDOW[0])].reset_index(drop=True)
    ev = df[t >= WINDOW[0]].reset_index(drop=True)
    train, val, ev = (d.assign(clim=climatology(train, d, d is train)) for d in (train, val, ev))
    in_win = (ev['sat_datetime'] < WINDOW[1]).to_numpy()
    subsets = {'window': in_win, 'later': ~in_win, 'all': np.ones(len(ev), dtype=bool)}
    y = ev[TARGET_COLUMN].to_numpy(dtype=np.float64)
    rm = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))
    print(f"train={len(train)} val={len(val)} eval: window={in_win.sum()} later={(~in_win).sum()}", flush=True)
    print("references (RMSE vs Argo): " + " | ".join(
        f"{n} {rm(ev[c].to_numpy() - y):.4f}" for n, c in [('raw sat_sss', 'sat_sss'), ('anc_sss', 'sat_anc_sss'), ('climatology', 'clim')]),
        flush=True)

    base = drop_basin(RICH_FEATURES)
    configs = {
        'B0 recent, existing form': (base, 'sat_sss'),
        'B1 + clim input': (['clim'] + base, 'sat_sss'),
        'B2 + clim input + target-clim': (['clim'] + base, 'clim'),
    }
    preds = {k: [] for k in configs}
    rows = []
    for name, (feats, baseline) in configs.items():
        for seed in range(args.seeds):
            p = fit_predict(feats, baseline, train, val, ev, seed)
            preds[name].append(p)
            out = {'config': name, 'seed': seed}
            for sub, m in subsets.items():
                out[f'rmse_{sub}'] = compute_metrics(p[m], y[m])['rmse']
            rows.append(out)
            print(f"  {name:<30} seed={seed}  window={out['rmse_window']:.4f} later={out['rmse_later']:.4f} "
                  f"all={out['rmse_all']:.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_parquet(f'{OUT_DIR}/recent_climatology_ensemble_per_seed.parquet', index=False)
    ens = {k: np.mean(v, axis=0) for k, v in preds.items()}
    pd.DataFrame({'sat_datetime': ev['sat_datetime'], 'in_window': in_win, 'argo_salinity': y,
                  **{f'pred_{k[:2]}_ens': v for k, v in ens.items()}}
                 ).to_parquet(f'{OUT_DIR}/recent_climatology_ensemble_eval_predictions.parquet', index=False)

    print(f"\n=== single networks over {args.seeds} seeds: mean +/- std (min-max) ===")
    for sub in subsets:
        print(f"[{sub}  n={subsets[sub].sum()}]")
        for name in configs:
            v = res.loc[res['config'] == name, f'rmse_{sub}']
            print(f"   {name:<30} {v.mean():.4f} +/- {v.std():.4f}  ({v.min():.4f}-{v.max():.4f})")

    print(f"\n=== seed-averaged ('ensemble') predictions; paired row-bootstrap, positive = second is better ===")
    rng = np.random.default_rng(0)
    for sub, m in subsets.items():
        n = int(m.sum())
        idx = [rng.integers(0, n, n) for _ in range(2000)]
        errs = {k[:2]: (v - y)[m] for k, v in ens.items()}
        line = " | ".join(f"{k} {rm(e):.4f}" for k, e in errs.items())
        print(f"[{sub}]  ensemble RMSE: {line}")
        for a, b in [('B0', 'B1'), ('B1', 'B2'), ('B0', 'B2')]:
            d = np.array([rm(errs[a][i]) - rm(errs[b][i]) for i in idx])
            print(f"      {a} - {b}: {d.mean():+.4f}  95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")


if __name__ == '__main__':
    main()
