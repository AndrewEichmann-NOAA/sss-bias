#!/usr/bin/env python3
"""
Can salinity be derived from the brightness temperatures instead of correcting
the SMAP SSS product? (DESIGN.md 42) An input-group "ladder" with a climatology
baseline:

    prediction = climatology(lat, lon, month)  +  network(inputs)

so the network only learns the departure from a seasonal salinity map built
from TRAINING years only, not the global pattern. The climatology value is ALSO
given to the network as an input at every rung: brightness temperatures and
anc_sss carry absolute salinity information, so predicting a departure from the
climatology requires comparing them against it. (A first version omitted this;
it left R4 at 0.4045 vs. 0.3295 for anc_sss used directly -- the network could
not subtract a climatology it was never shown.) Each rung adds one group of
inputs; the question at each step is how much test RMSE drops.

Climatology: hierarchical cell means of Argo salinity over the train split
(5deg lat x 10deg lon x month; falls back to 10x20xmonth, then 10deg-lat x
month, then the global mean, when a cell has < MIN_COUNT observations). For
TRAIN rows it is leave-one-out -- otherwise each row's own Argo value would be
inside its baseline, shrinking training residuals relative to validation/test
rows (where the baseline never saw the row).

Rungs (cumulative, no basin inputs, 35-input set at the top):
  R0  climatology only (no network)
  R1  + position/time      lat, lon sin/cos, day-of-year sin/cos
  R2  + environment        anc_sst, anc_spd, anc_dir (NCEP/NOAA, not salinity)
  R3  + brightness temps   TBs, TB bias adj, NEDT, geometry, land/ice, ascending
                           -- the first rung with no salinity field at all
  R4  + anc_sss            HYCOM salinity (itself a salinity field)
  R5  + product byproducts smap_spd, smap_high_spd/dir/dir_smooth (from the
                           joint SSS/wind solve or fixed at anc_sss)
  R6  + the product        sat_sss, sat_smap_sss_uncertainty (= the full 35 inputs)
"""

import argparse

import numpy as np
import pandas as pd
import torch

from train_baseline import compute_metrics, train_ffann
from train_rich_features_poc import (
    RICH_FEATURES, TARGET_COLUMN, Standardizer, add_features, chronological_split,
)

DEFAULT_MATCHUPS = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups_gdacdirect.parquet'
OUT = '/Users/afeman/Desktop/work/sss-bias/data/matchups/ladder_tb_retrieval_results_v2.parquet'
MIN_COUNT = 5

G_POS = ['sat_lat', 'lon_sin', 'lon_cos', 'doy_sin', 'doy_cos']
G_ENV = ['sat_anc_sst', 'sat_anc_spd', 'sat_anc_dir']
G_TB = ['sat_tb_h_fore', 'sat_tb_h_aft', 'sat_tb_v_fore', 'sat_tb_v_aft', 'sat_tb_h_bias_adj', 'sat_tb_v_bias_adj',
        'sat_nedt_h_fore', 'sat_nedt_h_aft', 'sat_nedt_v_fore', 'sat_nedt_v_aft',
        'sat_inc_fore', 'sat_inc_aft', 'sat_azi_fore', 'sat_azi_aft', 'sat_antazi_fore', 'sat_antazi_aft',
        'sat_land_fraction_fore', 'sat_land_fraction_aft', 'sat_ice_concentration', 'sat_ascending']
G_ANC = ['sat_anc_sss']
G_BYPROD = ['sat_smap_spd', 'sat_smap_high_spd', 'sat_smap_high_dir', 'sat_smap_high_dir_smooth']
G_PROD = ['sat_sss', 'sat_smap_sss_uncertainty']

RUNGS = {
    'R1 +position/time': G_POS,
    'R2 +SST/wind': G_POS + G_ENV,
    'R3 +brightness temps': G_POS + G_ENV + G_TB,
    'R4 +anc_sss (HYCOM)': G_POS + G_ENV + G_TB + G_ANC,
    'R5 +product byproducts': G_POS + G_ENV + G_TB + G_ANC + G_BYPROD,
    'R6 +sat_sss, uncertainty': G_POS + G_ENV + G_TB + G_ANC + G_BYPROD + G_PROD,
}


def make_key(d, level):
    lat, lon, mon = d['sat_lat'], d['sat_lon'], d['sat_datetime'].dt.month.astype(str)
    cat = lambda *parts: parts[0].astype(str).str.cat([p.astype(str) for p in parts[1:]], sep='_')
    if level == 1:
        return cat(np.floor(lat / 5).astype(int), np.floor(lon / 10).astype(int), mon)
    if level == 2:
        return cat(np.floor(lat / 10).astype(int), np.floor(lon / 20).astype(int), mon)
    if level == 3:
        return cat(np.floor(lat / 10).astype(int), mon)
    return pd.Series('all', index=d.index)


def climatology(train, d, is_train):
    out = np.full(len(d), np.nan)
    y = d[TARGET_COLUMN].to_numpy(dtype=np.float64)
    for level in (1, 2, 3, 4):
        g = train.groupby(make_key(train, level))[TARGET_COLUMN].agg(['sum', 'count'])
        k = make_key(d, level)
        s = k.map(g['sum']).fillna(0).to_numpy(dtype=np.float64)
        n = k.map(g['count']).fillna(0).to_numpy(dtype=np.float64)
        if is_train:
            s, n = s - y, n - 1
        ok = np.isnan(out) & (n >= (MIN_COUNT if level < 4 else 1))
        out[ok] = s[ok] / n[ok]
    return out


def fit_predict(features, train, val, test, clim_train, clim_val, clim_test, seed):
    scaler = Standardizer()
    X_train = scaler.fit_transform(train[features].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val[features].to_numpy(dtype=np.float64))
    X_test = scaler.transform(test[features].to_numpy(dtype=np.float64))
    r_train = train[TARGET_COLUMN].to_numpy(dtype=np.float64) - clim_train
    r_val = val[TARGET_COLUMN].to_numpy(dtype=np.float64) - clim_val
    torch.manual_seed(seed)
    model = train_ffann(X_train, r_train, X_val, r_val, n_features=X_train.shape[1])
    model.eval()
    with torch.no_grad():
        r = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    return clim_test + r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-path', default=DEFAULT_MATCHUPS)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    df = add_features(pd.read_parquet(args.matchups_path)).dropna(subset=RICH_FEATURES).reset_index(drop=True)
    train, val, test = chronological_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)

    clim_train = climatology(train, train, True)
    clim_val = climatology(train, val, False)
    clim_test = climatology(train, test, False)
    train, val, test = (d.assign(clim=c) for d, c in ((train, clim_train), (val, clim_val), (test, clim_test)))
    y = test[TARGET_COLUMN].to_numpy(dtype=np.float64)
    rm = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))
    print(f"references on test (RMSE vs Argo): constant {rm(y - train[TARGET_COLUMN].mean()):.4f} | "
          f"raw sat_sss {rm(test['sat_sss'] - y):.4f} | anc_sss alone {rm(test['sat_anc_sss'] - y):.4f}", flush=True)
    print(f"R0 climatology alone: test {rm(clim_test - y):.4f} (bias {np.mean(clim_test - y):+.4f}) | "
          f"train (leave-one-out) {rm(clim_train - train[TARGET_COLUMN].to_numpy()):.4f} | "
          f"val {rm(clim_val - val[TARGET_COLUMN].to_numpy()):.4f}", flush=True)

    rows = []
    for name, feats in RUNGS.items():
        feats = ['clim'] + feats
        for seed in range(args.seeds):
            p = fit_predict(feats, train, val, test, clim_train, clim_val, clim_test, seed)
            m = compute_metrics(p, y)
            rows.append({'rung': name, 'n_features': len(feats), 'seed': seed, **m})
            print(f"  {name:<26} ({len(feats):2d} inputs) seed={seed}  rmse={m['rmse']:.4f}  bias={m['bias']:+.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_parquet(OUT, index=False)
    summ = res.groupby('rung', sort=False).agg(n_inputs=('n_features', 'first'), rmse_mean=('rmse', 'mean'),
                                              rmse_std=('rmse', 'std'), rmse_min=('rmse', 'min'),
                                              rmse_max=('rmse', 'max'), bias_mean=('bias', 'mean'))
    print(f"\n=== ladder summary over {args.seeds} seeds (test n={len(test)}) ===")
    print(summ.to_string(float_format=lambda v: f'{v:.4f}'))
    print(f"\nSaved per-run results to {OUT}")


if __name__ == '__main__':
    main()
