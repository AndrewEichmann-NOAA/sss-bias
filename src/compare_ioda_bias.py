#!/usr/bin/env python3
"""
Compares original-vs-bias-corrected IODA SSS files (DESIGN.md 39) against
Argo, using the same nearest-neighbor match_to_argo logic as
build_raw_smap_matchups.py, restricted to what GDAC-direct Argo coverage
currently allows for this period (2025-12-10 to 2026-01-19).

Restricted to QC-pass (PreQC == 0) obs -- the only ones the correction model
touched; QC-fail obs are identical between original and corrected files, so
including them would dilute the comparison rather than inform it, and most
would fail match_to_argo's own gross-error check regardless.
"""

import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

from build_raw_smap_matchups import match_to_argo
from gdac_qc_filter import load_gdac_direct_argo

IODA_ORIG_BASE = Path('/Users/afeman/Desktop/work/sss-bias/data/common_obsForge')


def load_cycle_as_file_df(ioda_path):
    f = nc.Dataset(ioda_path)
    lat = np.asarray(f['MetaData/latitude'][:], dtype=np.float64)
    lon = np.asarray(f['MetaData/longitude'][:], dtype=np.float64)
    dt_sec = np.asarray(f['MetaData/dateTime'][:])
    sss = np.asarray(f['ObsValue/seaSurfaceSalinity'][:], dtype=np.float64)
    preqc = np.asarray(f['PreQC/seaSurfaceSalinity'][:])
    f.close()

    qc_pass = preqc == 0
    datetime = pd.to_datetime(dt_sec[qc_pass], unit='s')
    return pd.DataFrame({
        'lat': lat[qc_pass], 'lon': lon[qc_pass], 'sss': sss[qc_pass], 'datetime': datetime,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2025-12-10')
    parser.add_argument('--end-date', default='2026-01-19')
    parser.add_argument('--corrected-base', default='/Users/afeman/Desktop/work/sss-bias/data/bias_corrected_obsForge')
    parser.add_argument('--gdac-direct-dir', default='/Users/afeman/Desktop/work/sss-bias/data/gdac_argo_direct')
    parser.add_argument('--max-dist-km', type=float, default=50.0)
    parser.add_argument('--max-time-delta-hours', type=float, default=3.0)
    parser.add_argument('--max-abs-diff', type=float, default=10.0)
    parser.add_argument('--out-suffix', default='',
                         help="Appended to the saved matchup filenames (e.g. '_nobasin') so a run against a "
                              "different corrected-file set doesn't overwrite earlier results.")
    args = parser.parse_args()

    corrected_base = Path(args.corrected_base)
    dates = pd.date_range(args.start_date, args.end_date, freq='D')
    hours = ['00', '06', '12', '18']

    orig_files, corr_files = [], []
    n_cycles = 0
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        for hour in hours:
            corr_path = corrected_base / f'gdas.{date_str}/{hour}/ocean/sss/gdas.t{hour}z.sss_smap_l2_bc.nc'
            if not corr_path.exists():
                continue
            orig_path = IODA_ORIG_BASE / f'gdas.{date_str}/{hour}/ocean/sss/gdas.t{hour}z.sss_smap_l2.nc'
            orig_files.append(load_cycle_as_file_df(orig_path))
            corr_files.append(load_cycle_as_file_df(corr_path))
            n_cycles += 1
    print(f"Loaded {n_cycles} cycles (QC-pass obs only)")

    start_pad = pd.Timestamp(args.start_date) - pd.Timedelta(hours=args.max_time_delta_hours)
    end_pad = pd.Timestamp(args.end_date) + pd.Timedelta(days=1, hours=args.max_time_delta_hours)
    argo_df = load_gdac_direct_argo(args.gdac_direct_dir, start_pad, end_pad)
    print(f"Loaded {len(argo_df)} QC-good Argo profiles in range (+/- match padding)")

    max_time_delta = pd.Timedelta(hours=args.max_time_delta_hours)

    print("\nMatching against ORIGINAL (raw) SSS...")
    result_orig = match_to_argo(argo_df, orig_files, args.max_dist_km, max_time_delta, args.max_abs_diff)
    print(f"  {len(result_orig)} matches")

    print("Matching against BIAS-CORRECTED SSS...")
    result_corr = match_to_argo(argo_df, corr_files, args.max_dist_km, max_time_delta, args.max_abs_diff)
    print(f"  {len(result_corr)} matches")

    for label, result in [('ORIGINAL', result_orig), ('CORRECTED', result_corr)]:
        diff = result['sat_sss'] - result['argo_salinity']
        print(f"\n=== {label} vs. Argo ===")
        print(f"  n={len(result)}  bias={diff.mean():+.4f}  std={diff.std():.4f}  "
              f"rmse={np.sqrt((diff**2).mean()):.4f}")

    out_dir = Path('/Users/afeman/Desktop/work/sss-bias/data/matchups')
    result_orig.to_parquet(out_dir / f'ioda_compare_original{args.out_suffix}.parquet', index=False)
    result_corr.to_parquet(out_dir / f'ioda_compare_corrected{args.out_suffix}.parquet', index=False)
    print(f"\nSaved matchup tables to {out_dir}")


if __name__ == '__main__':
    main()
