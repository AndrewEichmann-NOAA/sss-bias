#!/usr/bin/env python3
"""
3D plot (lon, lat, salinity) of a single raw SMAP CAP swath file -- shows the
along-track/cross-track (812 x 76) structure as a surface with retrieved
salinity as height, rather than flattening it into a 2D map.

QC-fail and fill-value pixels are masked to NaN so they leave gaps in the
surface rather than plotting as spurious low spikes (the -9999.0 sentinel
would otherwise dominate the z-scale).
"""

import argparse
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

RAW_DIR = '/Users/afeman/Desktop/work/sss-bias/data/raw_smap_cap'


def load_swath(path):
    ds = nc.Dataset(path)
    ds.set_auto_mask(False)
    try:
        lat = np.asarray(ds['lat'][:], dtype=np.float64)
        lon = np.asarray(ds['lon'][:], dtype=np.float64)
        sss = np.asarray(ds['smap_sss'][:], dtype=np.float64)
        qc = np.asarray(ds['quality_flag'][:], dtype=np.float64)
    finally:
        ds.close()

    bad = (lat == -9999.0) | (lon == -9999.0) | (sss == -9999.0) | (qc != 0)
    lat[bad] = np.nan
    lon[bad] = np.nan
    sss[bad] = np.nan
    return lat, lon, sss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=None, help='Path to a raw SMAP CAP .h5 file.')
    parser.add_argument('--date', default='20240101', help='YYYYMMDD -- picks the first swath from this date '
                                                             'if --file is not given.')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    if args.file:
        path = args.file
    else:
        candidates = sorted(glob.glob(f'{RAW_DIR}/*_{args.date}T*.h5'))
        if not candidates:
            print(f"No swath files found for {args.date}")
            return
        path = candidates[0]
    print(f"Plotting {path}")

    lat, lon, sss = load_swath(path)
    n_valid = np.isfinite(sss).sum()
    print(f"{n_valid}/{sss.size} QC-pass ocean pixels")

    stem = path.split('/')[-1].rsplit('.', 1)[0]
    out_path = args.out or f'/Users/afeman/Desktop/work/sss-bias/data/matchups/swath_3d_{stem}.png'

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    vmin, vmax = np.nanpercentile(sss, [2, 98])
    surf = ax.plot_surface(lon, lat, sss, cmap='viridis', vmin=vmin, vmax=vmax,
                            rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('SSS (PSU)')
    ax.set_title(f'SMAP CAP swath (3D): {path.split("/")[-1]}\nn={n_valid:,} QC-pass pixels')
    fig.colorbar(surf, ax=ax, label='PSU', fraction=0.03, pad=0.08, shrink=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
