#!/usr/bin/env python3
"""
Fetch raw Argo near-surface profiles from the public GDAC via argopy, to
recover the WMO float ID and real per-obs QC that obsForge's IODA conversion
strips out (see DESIGN.md 14, 17).

Scoped to the period already present locally (2021-01-01 to 2025-11-30, the
span of data/common_obsForge) and to near-surface depths (0-10m, a small
buffer past the 5m near-surface cutoff used elsewhere in this project).

Uses argopy's 'standard' user mode, which already implements the
delayed-mode-preferred / real-time-fallback logic internally (PSAL_ADJUSTED
when available, else raw PSAL) -- no need to hand-roll that merge. DATA_MODE
is kept in the output so provenance (D/A/R) is still visible per row.

Requires: pip install argopy 'erddapy==3.2.1' (argopy 1.4.0 is incompatible
with erddapy's latest release -- see DESIGN.md 17 for the import error this
produces if erddapy is left at its default-installed version).
"""

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

COLUMNS = ['PLATFORM_NUMBER', 'LATITUDE', 'LONGITUDE', 'TIME', 'PRES',
           'PSAL', 'PSAL_QC', 'TEMP', 'TEMP_QC', 'DATA_MODE']


def month_chunks(start, end, chunk_months):
    chunks = []
    cur = pd.Timestamp(start).replace(day=1)
    end = pd.Timestamp(end)
    while cur <= end:
        nxt = cur + pd.DateOffset(months=chunk_months)
        chunks.append((cur, min(nxt, end + pd.Timedelta(days=1))))
        cur = nxt
    return chunks


def fetch_chunk(t0, t1, max_depth, retries=3):
    from argopy import DataFetcher

    for attempt in range(retries):
        try:
            f = DataFetcher(src='gdac', mode='standard')
            box = [-180, 180, -90, 90, 0, max_depth,
                   t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d')]
            ds = f.region(box).load().data
            df = ds.to_dataframe().reset_index()
            return df[[c for c in COLUMNS if c in df.columns]]
        except Exception as e:
            print(f"    attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2021-01-01')
    parser.add_argument('--end-date', default='2025-11-30')
    parser.add_argument('--chunk-months', type=int, default=3)
    parser.add_argument('--max-depth', type=float, default=10.0)
    parser.add_argument('--out-dir', default='/Users/afeman/Desktop/work/sss-bias/data/raw_argo')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = month_chunks(args.start_date, args.end_date, args.chunk_months)
    print(f"{len(chunks)} chunks of ~{args.chunk_months} month(s) each, "
          f"{args.start_date} to {args.end_date}, depth 0-{args.max_depth}m, global")

    total_rows = 0
    for i, (t0, t1) in enumerate(chunks):
        out_path = out_dir / f"raw_argo_{t0.strftime('%Y%m')}.parquet"
        if out_path.exists():
            print(f"  [{i+1}/{len(chunks)}] {t0.date()} to {t1.date()}: already fetched, skipping")
            continue

        print(f"  [{i+1}/{len(chunks)}] {t0.date()} to {t1.date()}: fetching...")
        df = fetch_chunk(t0, t1, args.max_depth)
        if df is None:
            print(f"    FAILED after retries, skipping this chunk")
            continue
        if df.empty:
            print(f"    0 rows")
            continue

        df.to_parquet(out_path, index=False)
        total_rows += len(df)
        print(f"    {len(df)} rows, {df['PLATFORM_NUMBER'].nunique()} unique floats -> {out_path}")

    print(f"\nDone. Total rows fetched this run: {total_rows}")


if __name__ == '__main__':
    main()
