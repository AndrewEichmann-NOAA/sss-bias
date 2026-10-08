#!/usr/bin/env python3
"""
Generates bias-corrected SMAP SSS IODA observation files, for assimilation,
as drop-in replacements for the ones in common_obsForge -- WITHOUT touching
the originals (written to a separate output tree with the same internal
directory layout: <out-base>/gdas.YYYYMMDD/HH/ocean/sss/gdas.tHHz.sss_smap_l2_bc.nc -- '_bc' suffix
distinguishes it from the original filename it's a corrected copy of).

Approach validated in DESIGN.md 39 before writing this: for a given cycle's
original IODA file, its `obs_source_files` global attribute lists the exact
raw JPL CAP swath files obsForge ingested. Loading those same raw files with
build_raw_smap_matchups.load_raw_smap_file, filtering to sss > 0 (obsForge's
own filter -- a handful of degenerate 0.0 retrievals, e.g. over Hudson Bay,
that pass obsForge's fill-value check but not this one), and concatenating
in the attribute's listed order reproduces the original file's
lat/lon/ObsValue/PreQC/ObsError arrays exactly, row for row (dateTime is
reproduced to within +/-64s, a minor rounding difference not chased further
-- negligible against a 6h DA window). That exact positional correspondence
is what lets this script safely graft model-corrected values onto a copy of
the original file: every other field (MetaData, PreQC, ObsError, and
critically oceanBasin, which has no raw-file equivalent) is copied byte for
byte from the original, and only ObsValue/seaSurfaceSalinity is replaced.

Model inputs use the SAME rich feature set as train_rich_features_poc.py's
RICH_FEATURES, computed here directly from the raw per-pixel fields
(load_raw_smap_file already extracts every field RICH_EXTRA_FEATURES needs)
plus the original file's own oceanBasin for the basin_0..5 one-hot (more
accurate than re-deriving it with classify_ocean_basin.py's ~89% approximate
classifier, since the real obsForge-computed value is already sitting right
there in the file being replaced).
"""

import argparse
import shutil
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
import pandas as pd
import torch

from build_raw_smap_matchups import load_raw_smap_file
from train_baseline import FFANN
from train_rich_features_poc import BASIN_CODES, BASELINE_FEATURES, RICH_EXTRA_FEATURES, RICH_FEATURES

RAW_SMAP_DIR = Path('/Users/afeman/Desktop/work/sss-bias/data/raw_smap_cap')
IODA_BASE = Path('/Users/afeman/Desktop/work/sss-bias/data/common_obsForge')

# RICH_EXTRA_FEATURES/BASELINE_FEATURES names are 'sat_<raw_field>' -- load_raw_smap_file's
# columns are the raw field names directly (see build_raw_smap_matchups.RICH_FIELDS),
# plus 'sss'/'lat'/'lon'/'ascending'. Strips the 'sat_' prefix to look each one up.
DIRECT_FEATURES = [f for f in RICH_EXTRA_FEATURES if f not in ('sat_ascending',)]


def load_model(checkpoint_path):
    """Returns (list of models, checkpoint). Handles both a single-model checkpoint
    ('model_state_dict') and an ensemble ('model_state_dicts', train_and_save_ensemble.py);
    process_cycle averages the members' predicted corrections.
    """
    ckpt = torch.load(checkpoint_path, weights_only=False)
    states = ckpt['model_state_dicts'] if 'model_state_dicts' in ckpt else [ckpt['model_state_dict']]
    models = []
    for sd in states:
        m = FFANN(ckpt['n_features'], hidden=ckpt['hidden'])
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, ckpt


def build_feature_matrix(combined, ocean_basin, rich_features, scaler_mean, scaler_std):
    n = len(combined)
    feat = pd.DataFrame(index=combined.index)
    feat['sat_sss'] = combined['sss']
    feat['sat_lat'] = combined['lat']
    lon_rad = np.radians(combined['lon'].astype(float))
    feat['lon_sin'] = np.sin(lon_rad)
    feat['lon_cos'] = np.cos(lon_rad)
    day_of_year = pd.to_datetime(combined['datetime']).dt.dayofyear.astype(float)
    feat['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    feat['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    for code in BASIN_CODES:
        feat[f'basin_{code}'] = (ocean_basin == code).astype(float)
    for f in DIRECT_FEATURES:
        raw_col = f[len('sat_'):]
        feat[f] = combined[raw_col]
    feat['sat_ascending'] = combined['ascending']

    X = feat[rich_features].to_numpy(dtype=np.float64)
    return X, feat


def process_cycle(date_str, hour, models, ckpt, out_base, verbose=True):
    cycle_dir = f'gdas.{date_str}/{hour}/ocean/sss'
    ioda_rel = Path(f'{cycle_dir}/gdas.t{hour}z.sss_smap_l2.nc')
    out_rel = Path(f'{cycle_dir}/gdas.t{hour}z.sss_smap_l2_bc.nc')
    ioda_path = IODA_BASE / ioda_rel
    if not ioda_path.exists():
        if verbose:
            print(f"  [{date_str} {hour}Z] no original IODA file -- skipping")
        return None

    src = nc.Dataset(ioda_path)
    src_files = src.obs_source_files
    if isinstance(src_files, str):
        src_files = [src_files]
    ocean_basin = np.asarray(src['MetaData/oceanBasin'][:])
    n_ioda = len(ocean_basin)
    src.close()

    dfs = []
    for sf in src_files:
        fname = Path(sf).name
        fpath = RAW_SMAP_DIR / fname
        if not fpath.exists():
            print(f"  [{date_str} {hour}Z] MISSING raw file {fname} -- skipping cycle")
            return None
        df = load_raw_smap_file(str(fpath))
        df = df[df['sss'] > 0].reset_index(drop=True)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if len(combined) != n_ioda:
        print(f"  [{date_str} {hour}Z] row-count mismatch: raw={len(combined)} ioda={n_ioda} -- skipping cycle")
        return None

    X, _ = build_feature_matrix(combined, ocean_basin, ckpt['rich_features'], ckpt['scaler_mean'], ckpt['scaler_std'])
    Xs = (X - ckpt['scaler_mean']) / ckpt['scaler_std']

    # Only correct QC-pass (quality_flag == 0) obs: the model was trained
    # exclusively on QC-pass satellite-Argo matches (build_raw_smap_dir's own
    # filter), so applying it to QC-fail rows is out-of-distribution -- those
    # keep their original SSS value, same as the DA system will downweight/
    # reject them via PreQC regardless. A handful of QC-pass rows can still
    # have a NaN ancillary field (rare, seen before at ~1e-4 rate in training)
    # -- also left uncorrected rather than fed a NaN into the model.
    qc_pass = combined['quality_flag'].to_numpy() == 0
    valid = qc_pass & ~np.isnan(Xs).any(axis=1)
    n_qc_fail = (~qc_pass).sum()
    n_nan_qc_pass = (qc_pass & np.isnan(Xs).any(axis=1)).sum()
    if n_qc_fail or n_nan_qc_pass:
        print(f"  [{date_str} {hour}Z] {n_qc_fail}/{len(X)} QC-fail rows and "
              f"{n_nan_qc_pass} QC-pass-but-NaN-feature rows left uncorrected")

    with torch.no_grad():
        resid = np.zeros(len(Xs))
        Xt = torch.tensor(Xs[valid], dtype=torch.float32)
        resid[valid] = np.mean([m(Xt).numpy() for m in models], axis=0)
    corrected_sss = combined['sss'].to_numpy(dtype=np.float64) + resid

    out_path = out_base / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ioda_path, out_path)

    # netCDF4-python (this environment's build) can't reopen these particular
    # HDF5-backed NETCDF4 files in write mode ("Can't write file", regardless
    # of permissions/locking) -- h5py opens and writes the same files fine,
    # since a NETCDF4 file is just HDF5 underneath with netCDF4's group model.
    with h5py.File(out_path, 'r+') as out_ds:
        out_ds['ObsValue/seaSurfaceSalinity'][:] = corrected_sss.astype(np.float32)
        out_ds.attrs['bias_correction_model'] = str(Path(ckpt.get('_checkpoint_path', 'rich_correction_model.pt')).name)

    n_corrected = int(valid.sum())
    mean_corr = resid[valid].mean() if n_corrected else float('nan')
    std_corr = resid[valid].std() if n_corrected else float('nan')
    if verbose:
        print(f"  [{date_str} {hour}Z] wrote {out_path} ({len(combined)} obs, {n_corrected} corrected, "
              f"mean correction {mean_corr:+.4f}, std {std_corr:.4f})")
    return dict(date=date_str, hour=hour, n=len(combined), n_corrected=n_corrected,
                mean_correction=mean_corr, std_correction=std_corr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2025-12-10')
    parser.add_argument('--end-date', default='2026-01-19')
    parser.add_argument('--checkpoint', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/rich_correction_model.pt')
    parser.add_argument('--out-base', default='/Users/afeman/Desktop/work/sss-bias/data/bias_corrected_obsForge')
    args = parser.parse_args()

    models, ckpt = load_model(args.checkpoint)
    ckpt['_checkpoint_path'] = args.checkpoint
    out_base = Path(args.out_base)

    dates = pd.date_range(args.start_date, args.end_date, freq='D')
    hours = ['00', '06', '12', '18']

    results = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        for hour in hours:
            r = process_cycle(date_str, hour, models, ckpt, out_base)
            if r is not None:
                results.append(r)

    summary = pd.DataFrame(results)
    print(f"\nProcessed {len(summary)} cycles successfully out of {len(dates) * len(hours)} requested.")
    if not summary.empty:
        summary.to_parquet('/Users/afeman/Desktop/work/sss-bias/data/matchups/bias_correction_summary.parquet', index=False)
        print(summary[['mean_correction', 'std_correction', 'n']].describe())


if __name__ == '__main__':
    main()
