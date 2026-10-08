#!/usr/bin/env python3
"""
Does the 'Argo - climatology' target (B2) buy its lower RMSE by pulling estimates
toward the climatology, i.e. by damping real anomalies? (DESIGN.md 46)

Uses the saved seed-averaged held-out predictions from recent_climatology_ensemble.py
(B0 existing form, B1 + climatology input, B2 + climatology input + target-clim; 5 seeds
each; 12,413 untrained rows from 2025-12-10 on). No retraining: the climatology is rebuilt
from the same split's train rows (<= 2025-09-30, leave-one-out not needed for held-out
rows) and the saved predictions are checked to line up row for row.

Strata of the held-out rows by how far things sit from the climatology:
  - by the true anomaly (Argo - clim): selects on the outcome, so every model shows
    regression toward the mean there; the model-to-model DIFFERENCE is what matters
  - by the input anomaly (|anc_sss - clim|, |sat_sss - clim|): selects on inputs only,
    so there is no selection bias
Plus a calibration slope: regress the true anomaly on each model's predicted anomaly.
Slope 1 = calibrated; > 1 = predictions damped toward the climatology; < 1 = overspread.
"""

import numpy as np
import pandas as pd

from ladder_tb_retrieval import climatology
from train_rich_features_poc import RICH_FEATURES, TARGET_COLUMN, add_features

M = '/Users/afeman/Desktop/work/sss-bias/data/matchups/'
HELDOUT_START = pd.Timestamp('2025-12-10')
MODELS = ['B0', 'B1', 'B2']
rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def main():
    df = add_features(pd.read_parquet(M + 'smap_cap_argo_matchups_gdacdirect.parquet'))
    df = df.dropna(subset=RICH_FEATURES).reset_index(drop=True)
    t = df['sat_datetime']
    train = df[t <= '2025-09-30'].reset_index(drop=True)
    ev = df[t >= HELDOUT_START].reset_index(drop=True)
    ev['clim'] = climatology(train, ev, False)

    saved = pd.read_parquet(M + 'recent_climatology_ensemble_eval_predictions.parquet')
    assert len(saved) == len(ev), (len(saved), len(ev))
    assert np.allclose(saved['argo_salinity'].to_numpy(), ev[TARGET_COLUMN].to_numpy()), "row misalignment"
    assert (saved['sat_datetime'].to_numpy() == ev['sat_datetime'].to_numpy()).all(), "time misalignment"
    print(f"saved predictions line up with the rebuilt held-out rows (n={len(ev)})")

    y = ev[TARGET_COLUMN].to_numpy(dtype=np.float64)
    clim = ev['clim'].to_numpy(dtype=np.float64)
    pred = {m: saved[f'pred_{m}_ens'].to_numpy(dtype=np.float64) for m in MODELS}
    print("ensemble RMSE, all rows (should match 43): " + " | ".join(f"{m} {rmse(pred[m] - y):.4f}" for m in MODELS))
    print(f"climatology alone: {rmse(clim - y):.4f}   raw sat_sss: {rmse(ev['sat_sss'].to_numpy() - y):.4f}\n")

    true_an = y - clim
    hycom_an = (ev['sat_anc_sss'].to_numpy() - clim)
    smap_an = (ev['sat_sss'].to_numpy() - clim)
    lo5, hi5 = np.percentile(true_an, [5, 95])
    ab = np.abs(true_an)
    strata = {
        'all rows': np.ones(len(y), bool),
        'true |anom| lowest 50%': ab <= np.percentile(ab, 50),
        'true |anom| 50-90%': (ab > np.percentile(ab, 50)) & (ab <= np.percentile(ab, 90)),
        'true |anom| top 10%': ab > np.percentile(ab, 90),
        'true anom lowest 5% (fresher)': true_an <= lo5,
        'true anom highest 5% (saltier)': true_an >= hi5,
        '|HYCOM anom| top 10%': np.abs(hycom_an) > np.percentile(np.abs(hycom_an), 90),
        '|HYCOM anom| bottom 90%': np.abs(hycom_an) <= np.percentile(np.abs(hycom_an), 90),
        '|SMAP anom| top 10%': np.abs(smap_an) > np.percentile(np.abs(smap_an), 90),
        '|SMAP anom| bottom 90%': np.abs(smap_an) <= np.percentile(np.abs(smap_an), 90),
    }

    rng = np.random.default_rng(0)
    rows = []
    pairs = [('B0', 'B1'), ('B0', 'B2'), ('B1', 'B2')]
    print(f"{'stratum':<32}{'n':>6} | {'RMSE B0':>8}{'B1':>8}{'B2':>8} | {'bias B0':>8}{'B1':>8}{'B2':>8}")
    ci_lines = []
    for name, m in strata.items():
        n = int(m.sum())
        e = {k: (pred[k] - y)[m] for k in MODELS}
        idx = [rng.integers(0, n, n) for _ in range(2000)]
        print(f"{name:<32}{n:6d} | {rmse(e['B0']):8.4f}{rmse(e['B1']):8.4f}{rmse(e['B2']):8.4f} | "
              f"{e['B0'].mean():+8.4f}{e['B1'].mean():+8.4f}{e['B2'].mean():+8.4f}")
        row = {'stratum': name, 'n': n, **{f'rmse_{k}': rmse(e[k]) for k in MODELS},
               **{f'bias_{k}': float(e[k].mean()) for k in MODELS}}
        parts = []
        for a, b in pairs:
            d = np.array([rmse(e[a][i]) - rmse(e[b][i]) for i in idx])
            lo, hi = np.percentile(d, [2.5, 97.5])
            row.update({f'{a}_minus_{b}': float(d.mean()), f'{a}_minus_{b}_lo': float(lo), f'{a}_minus_{b}_hi': float(hi)})
            parts.append(f"{a}-{b} {d.mean():+.4f} ({lo:+.4f},{hi:+.4f})")
        ci_lines.append(f"{name:<32}" + "  ".join(parts))
        rows.append(row)
    print("\npaired row-bootstrap RMSE differences, first - second (+ = second is better), 95% CI:")
    print("\n".join(ci_lines))
    pd.DataFrame(rows).to_parquet(M + 'anomaly_preservation_strata.parquet', index=False)

    print("\ncalibration: OLS slope of true anomaly (Argo - clim) on predicted anomaly (pred - clim); 1 = calibrated, >1 = damped")
    for m in MODELS:
        pa = pred[m] - clim
        for label, sel in [('all rows', np.ones(len(y), bool)), ('|pred anom| top 10% of that model', np.abs(pa) > np.percentile(np.abs(pa), 90))]:
            slope = np.polyfit(pa[sel], true_an[sel], 1)[0]
            print(f"   {m}  {label:<36} slope {slope:.3f}  (std of predicted anomaly {pa[sel].std():.3f}, of true {true_an[sel].std():.3f})")
    print("\nstd of the predicted anomaly vs. the true anomaly, all rows: "
          + " | ".join(f"{m} {(pred[m] - clim).std():.3f}" for m in MODELS) + f" | true {true_an.std():.3f}")


if __name__ == '__main__':
    main()
