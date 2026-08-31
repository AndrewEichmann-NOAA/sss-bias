#!/usr/bin/env python3
"""
Empirical recalibration-regime check (DESIGN.md 26): plot the JPL CAP
algorithm's own applied brightness-temperature bias-adjustment fields
(tb_h_bias_adj, tb_v_bias_adj) over time, to look for a step-change that
would indicate an instrument recalibration -- rather than relying only on
public announcements, which don't always exist (DESIGN.md 26.1's Dec-2023
gap had none).

Uses the already-built Argo-matchup table (not a fresh raw-file scan): its
sat_tb_h_bias_adj/sat_tb_v_bias_adj columns already span the full archive
continuously (~1300+ matches/month), and since this bias-adjustment term is
an instrument/geometry property rather than a geophysical one, the Argo-
matched subsample should show the same regime structure as the full raw
archive would, at much lower cost than re-scanning every swath file.

Flags a week as a candidate regime-change point if its mean bias-adjustment
jumps by more than FLAG_SIGMAS times the pooled weekly std from the previous
week -- a simple heuristic, not a rigorous changepoint test, meant to draw
the eye to candidates for closer inspection rather than to be definitive.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MATCHUPS_PATH = '/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups.parquet'
OUT_PATH = '/Users/afeman/Desktop/work/sss-bias/data/matchups/tb_calibration_timeseries.png'
FLAG_SIGMAS = 3.0


def weekly_stats(df, col):
    g = df.set_index('sat_datetime')[col].resample('W').agg(['mean', 'std', 'count'])
    return g[g['count'] >= 5]


def flag_jumps(weekly, sigmas=FLAG_SIGMAS):
    mean_diff = weekly['mean'].diff()
    pooled_std = np.sqrt(weekly['std'] ** 2 + weekly['std'].shift(1) ** 2).replace(0, np.nan)
    z = (mean_diff / pooled_std).abs()
    return weekly.index[z > sigmas]


def main():
    df = pd.read_parquet(MATCHUPS_PATH, columns=['sat_datetime', 'sat_tb_h_bias_adj', 'sat_tb_v_bias_adj'])
    print(f"{len(df)} matches, {df['sat_datetime'].min()} to {df['sat_datetime'].max()}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    for ax, col, label in zip(axes, ['sat_tb_h_bias_adj', 'sat_tb_v_bias_adj'], ['H-pol', 'V-pol']):
        weekly = weekly_stats(df, col)
        ax.plot(weekly.index, weekly['mean'], color='tab:blue', linewidth=1)
        ax.fill_between(weekly.index, weekly['mean'] - weekly['std'], weekly['mean'] + weekly['std'],
                         color='tab:blue', alpha=0.2, label='+/-1 std (weekly)')

        flags = flag_jumps(weekly)
        print(f"\n{label} ({col}): {len(flags)} candidate jump(s) (|z| > {FLAG_SIGMAS})")
        for f in flags:
            print(f"  {f.date()}: mean {weekly.loc[f, 'mean']:+.4f} "
                  f"(prev week {weekly['mean'].shift(1).loc[f]:+.4f})")
        if len(flags):
            ax.scatter(flags, weekly.loc[flags, 'mean'], color='red', zorder=5,
                       label=f'candidate jump (|z|>{FLAG_SIGMAS:.0f})')

        ax.set_ylabel(f'{label} tb_bias_adj (K)')
        ax.set_title(f'{label} brightness-temperature bias adjustment, weekly mean +/- std '
                      f'(n={len(df):,} Argo-matched obs, min 5/week)', fontsize=10)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('Date')
    fig.suptitle('Empirical recalibration-regime check: JPL CAP TB bias-adjustment over time (DESIGN.md 26)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved {OUT_PATH}")


if __name__ == '__main__':
    main()
