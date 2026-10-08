# ML Bias Correction of SMAP SSS for MOM6/WCDA Assimilation — Design Doc

Status: draft for review. No code changes made yet — this documents the plan we agreed on before implementation starts.

## 1. Objective

Learn a correction/mapping from SMAP satellite sea surface salinity (SSS, effectively a "skin"/near-surface
retrieval) to Argo-observed **bulk salinity** (top ~5 m), so the corrected product can be assimilated into
MOM6's surface layer in the Weakly Coupled Data Assimilation (WCDA) configuration of the coupled GFS. This
follows the approach of Vernieres et al. (2014), who used a Feed-Forward ANN (FFANN) to map Aquarius SSS to
bulk salinity for GEOS-5, and incorporates ideas from Trossman & Bayler (2022) on Arctic SMAP bias correction
(wind stress, air-sea fluxes as candidate predictors).

The deliverable is a trained model (and the pipeline to reproduce it) that takes SMAP SSS + auxiliary fields
and outputs a bias-corrected bulk salinity estimate, plus an evaluation of which inputs actually reduce
error against Argo.

## 2. Data inventory (verified against what's in this repo)

`data/common_obsForge/gdas.YYYYMMDD/HH/ocean/` — 1,787 cycle directories, 4 synoptic cycles/day (00/06/12/18Z),
spanning 2021–2025 (SMOS) / 2021–2023 (SMAP tarballs) with the live `common_obsForge` tree extending further.
Each cycle has:

- `sss/gdas.tHHz.sss_smap_l2.nc` — IODA-formatted SMAP L2 obs. Verified contents (via `h5py`, one sample cycle):
  - `ObsValue/seaSurfaceSalinity`, `ObsError/seaSurfaceSalinity`, `PreQC/seaSurfaceSalinity`
  - `MetaData/latitude`, `MetaData/longitude`, `MetaData/dateTime` (seconds since 1970-01-01), `MetaData/oceanBasin`
  - ~112,000 obs per 6h cycle, all valid (no fill values) in the sample checked.
  - **`sss_smos_l2.nc` also present alongside SMAP** with the same schema — SMOS could serve as an independent
    validation source or a second sensor to fold in later, but is out of scope for phase 1.
- `insitu/gdas.tHHz.insitu_salt_profile_argo.nc` — Argo profile obs, same IODA style:
  - `ObsValue/salinity`, `MetaData/depth`, `latitude`, `longitude`, `dateTime`, `originalDateTime`, `oceanBasin`
  - ~522,000 obs per cycle (full profiles), but only **~2,100 within depth ≤ 5 m** in the sample cycle — this
    is the near-surface subset `process_netcdf_cycles.py` already filters for, and it's the scarce resource
    that gates how much training data we ultimately have.

### Important gap found

**The obsForge files do not carry SST, sensor beam ID, ascending/descending flag, or surface roughness** —
only SSS/salinity, lat/lon, time, and a coarse `oceanBasin` code. Those richer fields exist in the original
SMAP L2 swath granules but were stripped during IODA-ization for this repo. Per your direction, **phase 1
proceeds SSS-only** (plus derived lat/lon/time features); SST, roughness, wind stress, and air-sea flux
inputs are deferred to a later phase once a source (e.g. GDAS atmosphere/ocean background fields, or raw
SMAP L2 granules) is identified and collocated in.

## 3. Target definition

Target = mean Argo `salinity` over `depth ≤ 5 m`, per (lat, lon, datetime) group, exactly as already
implemented in `process_netcdf_cycles.py::process_insitu`. This is "bulk salinity" in the Vernieres sense —
a shallow-averaged in-situ value, not a single-depth point sample.

## 4. Matchup / collocation methodology

**Note**: this section's matchup counts/stats predate the `originalDateTime` fix in 15 -- the actual
timestamp used for Argo time-matching was wrong for ~86-91% of obs until then. See 15 for the corrected
numbers; the methodology description below (space/time window choice, QC layers) is otherwise still accurate.

SMAP and Argo obs are not on the same grid — they must be paired by proximity in space and time. Proposed
default window, tunable later:

- **Space**: SMAP footprint is ~40 km; match each qualifying Argo near-surface group to the nearest SMAP
  obs within a search radius (start at 50 km, using haversine distance — flat lat/lon distance is invalid at
  high latitude, and this dataset spans to ±78° lat).
  Convert lat/lon to radians and use e.g. `sklearn.neighbors.BallTree` with `metric='haversine'` for
  efficient nearest-neighbor search (this is the standard tool for this and is worth adding to the venv
  regardless of the modeling framework choice, since it also brings in `scikit-learn` for baselines/metrics).
- **Time**: match within the same synoptic window ± some tolerance (start at ±3h, i.e. roughly one cycle
  either side) since both obs streams are already organized into 6h cycles.
- **Multiplicity**: if more than one SMAP obs falls in an Argo group's window, take the nearest in space;
  document this choice since it affects sample independence.
- Output of this step: a flat matchup table (parquet/CSV), one row per Argo-SMAP pair, with columns for
  SMAP SSS, Argo bulk salinity, lat, lon, datetime, oceanBasin, and the match distance/time-delta (kept as
  QC columns, not model inputs).

This matchup step is the piece most worth getting right before any modeling — it's shared by all later
phases, and errors here (e.g. a window too wide) directly leak into training data quality.

**Implemented in `src/build_matchups.py`** (verified on 2022-06-06 to 2022-06-08 and June 2022). Two QC
issues found during validation, both now handled:

- **SMAP `PreQC` is a real, usable bitmask** — verified ~81% of obs pass at `PreQC == 0` in a sample cycle,
  remainder split across gross-check/RFI/land-contamination flag values. Filtered to `PreQC == 0`.
- **Argo `PreQC` is *not* usable** — verified uniformly `0` (pass) across all ~522k obs in a sample cycle,
  including a profile reporting ~0.118 PSU across all depths (a stuck/fouled conductivity sensor). Real Argo
  delayed-mode QC flags were not carried through the obsForge/IODA conversion. Two QC layers substitute for
  this: (1) a physical valid-range filter on salinity (default `[20, 42]` PSU — raised from an initial `[2,
  42]`, which still let through open-ocean profiles reading 5-10 PSU that were clearly bad sensors, not real
  low-salinity water); (2) a gross-error/background-check-style filter on the *matched pair* itself
  (`max_abs_diff`, default 10 PSU) — some bad-sensor values still land inside a lenient physical range but
  produce an implausible ~30 PSU discrepancy against the collocated SMAP value, which is a QC failure to
  reject, not bias signal to train on.

**Measured rejection rates** (full archive, via `src/qc_diagnostics.py`):

| Stage | Rejected | Rate |
|---|---|---|
| SMAP `PreQC != 0` (of 345,473,226 non-fill obs) | 73,224,082 | 21.20% |
| SMAP out-of-range (of PreQC-pass) | 483,423 | 0.14% |
| Argo out-of-range (of 18,503,586 near-surface obs) | 823,783 | 4.45% |
| Matched pairs: gross-mismatch >10 PSU (of 167,202 space/time candidates) | 1,053 | 0.63% |

SMAP's PreQC rejection rate (21%) is the dominant filter and is expected — swath-edge, RFI, and land/ice
contamination flags are a normal fraction of any satellite retrieval product. Argo's 4.45% out-of-range rate
is the one QC layer that's actually catching real data problems this pipeline introduced its own filter for
(§above); the very low gross-mismatch rate at the final stage (0.63%) shows the two upstream per-obs filters
already catch nearly all the bad data before matching, which is reassuring for the pipeline's soundness.

On the June 2022 test month: raw (uncorrected) SMAP-vs-Argo RMSE ≈ 1.36 PSU, bias ≈ +0.26 PSU — consistent
in magnitude with the raw Aquarius-vs-Argo discrepancies reported in Vernieres et al. (2014), a reassuring
sanity check that the matchup logic itself is sound.

## 5. Feature set (phased)

| Phase | Inputs | Source | Status |
|---|---|---|---|
| 1 (baseline) | SMAP SSS | obsForge, in repo | ready |
| 1 | latitude, longitude | obsForge, in repo | ready |
| 1 | Julian day (sin/cos encoded, for seasonality) | derived from `dateTime` | ready |
| 1 | oceanBasin (categorical) | obsForge, in repo | ready |
| 2 | SST | not in repo — needs GDAS ocean background or raw SMAP L2 | blocked |
| 2 | ascending/descending orbit flag | not in repo — needs raw SMAP L2 | blocked |
| 2 | sensor beam / incidence angle | not in repo — needs raw SMAP L2 | blocked |
| 3 | surface roughness | not in repo | blocked |
| 3 | wind stress, air-sea fluxes (Trossman & Bayler) | not in repo — needs GDAS atmosphere background | blocked |

Cyclical encoding note: Julian day and any angle-like features (day-of-year, longitude if treated
cyclically) should be encoded as `(sin(2π·x/period), cos(2π·x/period))` pairs rather than raw integers, so
the model sees Dec 31 and Jan 1 as adjacent.

## 6. Train / validation / test split

Naive random splitting will leak information, via three mechanisms:

1. **Argo float drift** — floats drift slowly (~a few km/day) and re-profile every ~10 days, so a random
   split can put two profiles from the *same float*, days apart, on opposite sides of the split.
2. **Spatial autocorrelation** — SSS is smooth over a correlation length of tens to ~100 km, so even
   different floats near each other in space/time are correlated.
3. **Temporal autocorrelation** — ocean state evolves slowly relative to the 6h cycle spacing.

**Constraint found**: the Argo `MetaData` group has no WMO float ID/platform number (checked: only
`dateTime`, `depth`, `latitude`, `longitude`, `oceanBasin`, `originalDateTime`) — so we can't cleanly hold
out whole floats by ID without extra engineering (clustering profiles into inferred trajectories, or joining
the external Argo GDAC index). Deferred; a good temporal split captures most of the benefit anyway.

**Revised after building the matchup table**: SMAP L2 obs are only present in this repo for **2021-2023**
(verified: 713/968/1257 cycle-files with a `sss_smap_l2.nc` for 2021/2022/2023 respectively, zero for
2024-2025 — matches the source tarballs, which only include `sss_smap_l2_2021/2022/2023.tgz`; SMOS extends
through 2025 but SMAP does not). The original train-2021–2023/validate-2024/test-2025 plan isn't buildable
from this data. Revised scheme, chosen to maximize training data over having a full clean test year:

- **Train**: 2021-01-01 – 2022-12-31 (two full annual cycles)
- **Embargo**: 2023-01-01 – 2023-01-14
- **Validate**: 2023-01-15 – 2023-06-30
- **Embargo**: 2023-07-01 – 2023-07-14
- **Test**: 2023-07-15 – 2023-12-31 (touched once, at the end)

Embargo gaps (standard in time-series ML, e.g. López de Prado's purged CV) absorb float-drift leakage across
each boundary that the missing float ID (see above) prevents blocking directly.

- **Blocked/rolling validation within the training period** remains useful for hyperparameter choices during
  development — e.g. expanding-window CV within 2021-2022, each fold with its own embargo gap, analogous to
  `sklearn.model_selection.TimeSeriesSplit` — distinct from the final held-out validate/test blocks above.
- Both training years still give 3 full seasonal cycles combined; the test window (H2 2023) covers one full
  Northern Hemisphere autumn/winter but not a full annual cycle on its own — worth keeping in mind when
  interpreting seasonal error breakdowns on the test set specifically.
- Matchup density is not stable across years in this data (~90 matches/cycle-file in 2021, ~38 in 2022, ~52
  in 2023 — not simply proportional to file counts) — flagged as an open item in §11, not yet explained.
- Track performance by `oceanBasin` and by latitude band separately — polar/high-latitude behavior is a
  known problem area for SMAP salinity (per Trossman & Bayler's Arctic-specific focus) and a global metric
  can hide regional failure.

## 7. Model architecture

PyTorch FFANN, mirroring Vernieres et al. structurally:

- Input layer sized to the active feature set (4–7 features in phase 1).
- 1–2 hidden layers, modest width (e.g. 16–32 units) — this is a small-input tabular regression problem,
  not a place that needs a deep network. Start small and only grow if validation error plateaus and
  underfits.
- ReLU or tanh activations, linear output (single bulk-salinity value).
- Standardize/normalize all inputs (z-score) before training; salinity target can be modeled directly or as
  a residual/correction relative to raw SMAP SSS (i.e. `output = SMAP_SSS + Δ`) — the residual formulation
  is likely to train faster and is closer to what "bias correction" implies. Worth comparing both.
- Loss: MSE. Report RMSE and mean bias (signed) in PSU as the headline metrics, plus correlation coefficient
  against Argo, consistent with how Vernieres et al. reported skill.

## 8. Baselines (to contextualize the ANN's value)

Report these alongside the FFANN so "does ML help" is answerable, not assumed:

1. **Raw SMAP SSS vs. Argo** (no correction at all) — the number we're trying to beat.
2. **Global constant bias correction** (mean offset only).
3. **Linear regression** on the same feature set as the FFANN.
4. **FFANN** (the actual proposal).

## 9. Proposed repo structure

Kept flat and script-based given the project's current size — no need for a package/library layer yet:

```
src/
  process_netcdf_cycles.py   # existing cycle-walking utility (extend, don't fork)
  build_matchups.py          # SMAP/SMOS-Argo collocation -> matchup table (parquet), --sensor smap|smos
  qc_diagnostics.py          # per-stage QC rejection rate diagnostics, --sensor smap|smos
  features.py                # shared feature-engineering functions (cyclical time, split boundaries, etc.)
  train_baseline.py          # load matchup table, train + evaluate FFANN vs. baselines, --sensor smap|smos
  plot_geographic_errors.py  # global RMSE/bias maps, raw vs. FFANN-corrected
data/
  matchups/                  # output of build_matchups.py / train_baseline.py, gitignored (derived data)
```

**Update**: project directory renamed from `sss` to `sss-bias` (matching the conda env name) partway through
this session; a git repository was initialized at the new root and pushed to
`github.com/AndrewEichmann-NOAA/sss-bias` (private). `.gitignore` covers `data/`, `.DS_Store`, `src/.venv/`,
`__pycache__/`. All hardcoded absolute paths in the scripts above were updated to match.

## 10. Roadmap

1. ~~**Matchup pipeline**~~ **Done**: `src/build_matchups.py` produces the SMAP-Argo matchup table across
   the full archive (SMAP obs only exist for 2021-2023, see §6). 166,149 matchups written to
   `data/matchups/smap_argo_matchups.parquet`. Distance/time distributions validated (median 10.7 km, median
   time delta 1h28m, both within the configured 50 km / 3h bounds as expected).
2. **Baseline eval**: compute raw-SMAP-vs-Argo error stats (no model) as the reference number. Partially
   done as a sanity check during matchup validation (overall raw RMSE 1.51 PSU, bias +0.30 PSU across all
   166,149 matchups) — still need this computed properly *on the test split only* as the actual baseline
   number to beat.
3. **Phase 1 FFANN**: train PyTorch model on SSS + lat/lon + Julian day + oceanBasin, compare to baselines
   from step 2, break down by basin/latitude band.
4. **Source SST/wind/roughness**: identify and collocate a source for phase-2/3 features (needs your input
   on where GDAS background fields or raw SMAP L2 granules are accessible).
5. **Ablation**: re-run phase 1 training with each phase-2/3 feature added incrementally, to see which
   actually reduces held-out error (this directly answers the project's stated question).

## 11. Open questions / risks

- ~~Matchup sparsity...~~ Resolved by the full-archive run: 166,149 total matchups from 4,583,040 near-surface
  Argo obs and ~272M SMAP obs scanned. With the train/validate/test split in §6, that's roughly ~110k train /
  ~28k validate / ~28k test rows (exact split counts not yet computed) — enough for a small FFANN, though
  still worth watching for overfitting given only a handful of input features.
- matchup density per SMAP cycle-file is not stable across years (~90/file in 2021, ~38/file in 2022,
  ~52/file in 2023) and doesn't track the number of available cycle-files proportionally. Not yet explained —
  could be genuine (Argo float density changes, seasonal coverage) or a pipeline artifact worth
  double-checking (e.g. was 2022 missing some months' worth of Argo data in the source tarball?) before
  trusting per-year comparisons too far. **Update**: §14 found the real Argo GDAC is publicly accessible —
  could cross-check obsForge's 2022 Argo coverage against the authoritative GDAC index directly if this
  becomes worth resolving.
- ~~PyTorch is not yet installed...~~ Resolved: project now uses the `sss-bias` conda environment
  (`/opt/miniconda3/envs/sss-bias`, Python 3.14) instead of `src/.venv`. Installed and verified: `torch`
  2.11.0, `scikit-learn` 1.9.0, `pyarrow` 25.0.0, `matplotlib` 3.11.0, on top of the env's existing `numpy`,
  `pandas`, `scipy`, `xarray`, `netcdf4`. Confirmed `netcdf4` engine reads the grouped IODA files correctly
  (`xr.open_dataset(f, group='ObsValue', engine='netcdf4')`), so scripts should use `engine='netcdf4'`
  instead of the `h5netcdf` default. Run scripts with `/opt/miniconda3/envs/sss-bias/bin/python` — `conda
  run -n sss-bias` inside a nested non-interactive shell was unreliable in this environment (silently
  produced no output) and should be avoided in favor of the direct interpreter path or `conda activate`.

## 12. Phase 1 results (`src/train_baseline.py`)

**Note**: superseded by the `originalDateTime` fix in 15 -- see 15.3 for corrected numbers (SMAP improved,
SMOS roughly a wash). Kept here for the record of how results evolved; the qualitative conclusions (FFANN
beats all baselines, basin-0 instability, etc.) still hold.

Split sizes: train 100,599 / validate 30,722 / test 30,996. Test-set metrics (2023-07-15 to 2023-12-31):

| method | RMSE (PSU) | bias (PSU) | corr |
|---|---|---|---|
| raw SMAP (uncorrected) | 1.643 | +0.365 | 0.536 |
| constant bias correction | 1.604 | +0.082 | 0.536 |
| linear regression | 1.460 | +0.045 | 0.573 |
| **FFANN** | **1.348** | **+0.050** | **0.653** |

The FFANN beats every baseline: ~18% RMSE reduction vs. raw SMAP, and removes most of the systematic bias
(+0.365 -> +0.050 PSU) that a constant correction alone only partially addresses — confirming the input
features (lat, lon, day-of-year, basin) carry real information about the SMAP-Argo discrepancy beyond a
single global offset, not just noise.

Two things to watch, not yet acted on:

- **Training had not converged at 300 epochs** — validation loss was still improving and early stopping
  (patience 20) never triggered. The reported numbers are a lower bound on what this architecture can do;
  worth rerunning with more epochs / a learning-rate schedule before treating this as the final phase-1
  number.
- **`oceanBasin` code 0 is small-sample and unstable** (45 test rows, ~170 total in the full matchup table)
  — both linear regression (bias -1.95) and the FFANN (bias -1.19) do *worse* than the raw/constant-bias
  baselines there, most likely overfitting to a handful of points rather than a real regional failure. Not
  worth tuning around until there's more data in that basin; flag rather than fix.
- Need confirmation on where SST/wind/roughness will come from (blocks phases 2–3). **Update**: investigated
  in §14 — raw SMAP/SMOS L2 granules that carry these fields are both credential-gated (NASA Earthdata,
  CATDS), not self-serve. Still unresolved; blocks phase 2 until either credentials are obtained or another
  source (e.g. GDAS atmosphere/ocean background fields) is identified.

### 12.1 SMOS comparison

Ran the identical pipeline against SMOS instead of SMAP (`--sensor smos` on all four scripts; pipeline
generalized to a sensor-agnostic `sat_*` matchup-table schema so no code duplication was needed —
`SENSOR_CONFIG` in `build_matchups.py`). Same train/val/test date windows as SMAP (2021-2022 / H1 2023 / H2
2023) for a direct comparison, even though SMOS obs actually extend through 2025 in this repo (opportunity
noted below, not yet acted on).

**QC differs by sensor and required its own investigation, not a copy of SMAP's rule**: SMOS's `PreQC` is a
continuous quality/uncertainty index (`ObsError` and SSS variance both scale up with it), not a bitmask, plus
a distinct high-uncertainty catch-all bucket at exactly `999` (21.6% of obs, 20x the out-of-range rate of the
well-behaved bins). Threshold set at `PreQC < 600` (excludes the small high-error `[600,900)` bin and the
`999` bucket) — see `build_matchups.py` docstring for the full reasoning. SMOS's overall PreQC rejection rate
(32.85%) is markedly higher than SMAP's (21.20%), consistent with SMOS's L-band radiometer being more
RFI-prone, a known characteristic of the sensor rather than a pipeline issue.

| stage | SMAP | SMOS |
|---|---|---|
| PreQC rejected | 21.20% | 32.85% |
| final matchups (full archive) | 166,149 | 298,115 |

Test-set metrics (2023-07-15 to 2023-12-31), train 2021-2022:

| method | SMAP RMSE | SMOS RMSE | SMAP bias | SMOS bias | SMAP corr | SMOS corr |
|---|---|---|---|---|---|---|
| raw (uncorrected) | 1.643 | 2.318 | +0.365 | +0.025 | 0.536 | 0.407 |
| constant bias | 1.604 | 2.322 | +0.082 | +0.147 | 0.536 | 0.407 |
| linear regression | 1.460 | 1.485 | +0.045 | +0.061 | 0.573 | 0.480 |
| **FFANN** | **1.348** | **1.429** | **+0.050** | **+0.056** | **0.653** | **0.540** |

Two things stand out:

- **SMOS starts noisier (raw RMSE 2.318 vs. SMAP's 1.643) but the FFANN closes most of the gap** (1.429 vs.
  1.348) — a much larger relative improvement (~38% RMSE reduction vs. SMAP's ~18%). Consistent with the
  premise of Trossman & Bayler (2022): SMOS's bias is more structured/correctable than SMAP's, not just
  larger noise that ML can't help with.
- **SMOS's constant-bias baseline makes RMSE slightly *worse*, not better** (2.318 -> 2.322), unlike SMAP
  where it helped a little. The train-set (2021-2022) mean offset doesn't transfer to the test period (H2
  2023) for SMOS — its systematic bias is less temporally stable than SMAP's, which the FFANN's
  latitude/season/basin-conditioned correction handles but a single global constant cannot. Worth watching
  if the phase-2 split is later extended into 2024-2025.
- Same small-sample instability pattern as SMAP: basins 0 and 4 (n=249, n=162 in the SMOS test set) show the
  FFANN doing *worse* than raw (bias +1.68 and +1.32 respectively) — same overfitting-to-few-points issue,
  not sensor-specific.
- **Opportunity not yet acted on**: SMOS obs extend through 2025 in this repo (unlike SMAP's 2021-2023 cutoff)
  — a SMOS-only run using the fuller date range (e.g. train 2021-2023, validate 2024, test 2025, the
  originally-envisioned split) would use ~3x more data than the SMAP-matched window above and give a real
  test of generalization across more calendar time. Not done here to keep this comparison apples-to-apples
  with SMAP on identical dates.

Results/models saved to `data/matchups/phase1_results_smos.json` / `phase1_ffann_smos.pt`.

### 12.2 Geographic error maps (`src/plot_geographic_errors.py`)

5deg-binned global maps of RMSE and bias (satellite - Argo), raw vs. FFANN-corrected, for both sensors.
Raw panels use the full matchup table (all years -- no fitting involved, so no leakage risk); FFANN panels
use only the held-out test-set predictions (`phase1_test_predictions_<sensor>.parquet`), to keep the
correction's spatial performance honestly out-of-sample. No coastline basemap needed -- land shows up
naturally as empty cells since these are ocean-only observations. Saved to `data/matchups/geo_errors_smap.png`
and `geo_errors_smos.png`.

Findings:

- **Raw RMSE is dominated by the Southern Ocean** (south of ~40S) for both sensors -- SMOS especially, >3.5
  PSU -- consistent with known satellite salinity retrieval difficulty in cold, high-wind, high-roughness
  conditions there.
- **SMAP and SMOS disagree in the *sign* of high-latitude bias**, not just magnitude: SMAP shows a strong
  positive bias near ~70-80N (with a matching RMSE hotspot there), while SMOS shows a strong negative bias in
  roughly the same region. Directly relevant to the Trossman & Bayler Arctic-focused correction that partly
  motivated this project.
- The FFANN's RMSE improvement is spatially broad, not a fluke of the aggregate number -- the Southern Ocean
  band visibly cools in both sensors' corrected panels.
- **New concern**: the FFANN-corrected bias map shows a systematic negative-bias band across the
  tropics/subtropics (~30S-30N) for both sensors that is much weaker in the raw data. The aggregate test-set
  bias looked near-zero (+0.05 PSU) because this negative band is being canceled out by opposite-signed error
  elsewhere -- the aggregate metric was masking a real regional pattern. Root-caused in 12.3.

### 12.3 Root cause of the tropical bias artifact: train/test straddles an ENSO transition

Checked whether the tropical negative-bias band in 12.2 is overfitting/noise or a real signal, by comparing
*raw* (uncorrected, no fitting involved) satellite-Argo bias in the tropics (|lat|<30) between the train
window (2021-2022) and the test window (H2 2023):

| sensor | train raw bias | test raw bias | shift |
|---|---|---|---|
| SMAP | +0.2522 | +0.2079 | -0.044 PSU |
| SMOS | +0.0770 | +0.0496 | -0.027 PSU |

The domain-average shift is tiny for both sensors -- nowhere near large enough to explain the ~0.5-1+ PSU
swings seen in the FFANN's geographic bias map. That rules out a simple "the mean bias changed" explanation
and points instead at a **spatial reorganization of the bias pattern that cancels out in the domain average**
but shows up strongly once binned geographically. The coarse lat-band breakdown printed by
`train_baseline.py` (e.g. SMAP FFANN bias for lat[-30,30) = -0.009, essentially zero) already hinted at this:
it's near-zero in aggregate specifically because it's not uniform -- some basins/longitudes within the band
run strongly negative, others near-neutral or positive, and a plain latitude-band average hides that.

**Working hypothesis**: the train window (2021-2022) was almost entirely inside a prolonged "triple-dip" La
Nina; the test window (H2 2023) falls inside the subsequent El Nino onset/strengthening (transition ~May-June
2023). ENSO phase is known to reorganize tropical Pacific (and connected-basin) precipitation/freshwater
patterns *spatially* without necessarily shifting the domain-wide mean much -- consistent with what's
measured above. This also explains why validation-based early stopping never caught it: the validation window
(H1 2023) still contains months of lingering pre-transition conditions, so it looked fine while the test
window, entirely past the transition, didn't.

This is a general climate-knowledge-based hypothesis, not something verified against an actual ENSO index in
this session -- worth confirming against NOAA's ONI series before treating it as settled. Practical
implication either way: **this isn't an overfitting bug fixable with more epochs or regularization** -- the
model has no training examples of El Nino conditions at all (2021-2022 never saw one), so applying it to H2
2023 is extrapolation to an unseen regime, not interpolation. Fixes need either (a) an ENSO-state input
feature (e.g. ONI) so the model can condition on large-scale ocean state, and/or (b) training data spanning
multiple ENSO phases -- not available for SMAP (capped at 2023 in this repo) but possible for SMOS (extends
to 2025, which includes the El Nino peak/decay). See 13.3 for a related, deeper QC finding that also affects
which observations should even be in the training set to begin with.

## 13. Operational QC investigation (obsForge / GDASApp source review)

The data used throughout this project came from internal NOAA EMC/OMD sources reflecting the QC and IODA
processing applied in obsForge/global-workflow as of Jan 2026. To check how closely this project's own QC
choices (4) matched what the operational system actually does, read the public source directly rather than
continuing to infer thresholds empirically:

- [`NOAA-EMC/obsForge`](https://github.com/NOAA-EMC/obsForge) -- the obs-to-IODA conversion code
- [`NOAA-EMC/GDASApp`](https://github.com/NOAA-EMC/GDASApp) -- the actual DA-cycle QC filter configs
- Both are public, no credentials needed, no bulk download required (shallow-cloned to inspect source only)

### 13.1 What obsForge's converter actually does (`utils/preproc/Smap2Ioda.h`, `Smos2Ioda.h`)

Both converters read the sensor's *own* official quality field and copy it through **completely
unfiltered** -- no threshold is applied at conversion time, beyond a trivial `obsVal_ > 0.0` sanity mask:

- SMAP's `PreQC` in the IODA file *is* NASA's own `quality_flag` field from the L2 product, verbatim.
- SMOS's `PreQC` in the IODA file *is* ESA/CATDS's own `Dg_quality_SSS_corr` field, verbatim (the source
  even cites the official ESA SMOS L2 Aux Data Product Specification for this field).

This retroactively validates the empirical approach in 4 -- SMAP's `PreQC == 0` pass convention matches the
standard NASA quality-bitmask convention (0 = no flags raised) for the exact field it turns out to be, and
SMOS's non-bitmask, continuous-index behavior is explained by it being a genuinely different kind of field
(a quality index, not a flag) from a different agency's product. The `PreQC < 600` threshold derived in 4 was
inferred correctly as *a* reasonable data-driven split, but see 13.2 -- it turns out not to be what the
operational system actually uses for QC at all.

### 13.2 What GDASApp's assimilation QC actually does (`parm/jcb-gdas/observations/marine/sss_{smap,smos}_l2.yaml.j2`)

Identical filter chain for both sensors, and it **does not reference `PreQC`/`quality_flag`/`Dg_quality_SSS_corr`
anywhere**:

1. `Domain Check`: `GeoVaLs/sea_area_fraction >= 0.9` (ocean mask, from model background)
2. `Bounds Check`: SSS in `[0.1, 40.0]` PSU -- notably wider than this project's `[20, 42]`
3. `Background Check`, threshold 5.0 -- gross-error check against the **model's own background field**,
   not against Argo (this project's `max_abs_diff` check against Argo is a reasonable stand-in given no
   background field is available here, but is conceptually different from the real filter)
4. `Domain Check` with `passivate` action: `GeoVaLs/sea_surface_temperature < -4.0`C -- near-freezing/
   ice-covered water is excluded from active assimilation (kept in the file, downweighted to zero impact)
5. `Gaussian_Thinning` -- currently commented out/disabled in the live config (LETKF compatibility issue
   noted in a comment), so not actually active despite being present in the file
6. `Domain Check`: `GeoVaLs/distance_from_coast >= 100e3` (100 km) -- all near-coastal obs excluded entirely

Filters 1, 3, and 4 require **GeoVaLs** -- the model's own background state (MOM6/GFS) interpolated to each
observation location during a live DA cycle. This is fundamentally not present in the obsForge-derived obs
files this project works with; it isn't something strippable-but-recoverable from raw satellite data either,
it only exists as an output of running the actual coupled model. Filter 6 (distance-from-coast) is
recoverable without model output, from a public coastline dataset -- not yet implemented here.

### 13.3 Implication: this project's QC diverges from operational QC, and the divergence is informative

- Filter 4 (SST < -4C passivation) exists *because* near-ice retrievals are known-bad -- this lines up
  directly with the high-latitude Arctic-adjacent RMSE/bias hotspot found in 12.2. The operational system
  doesn't try to correct those observations at all; it excludes them from assimilation. This project's
  training data currently includes them, uncorrected, which likely inflates the high-latitude error metrics
  and may be teaching the FFANN to "correct" a regime the operational system simply throws out.
- Filter 6 (distance-from-coast) is also entirely absent from this project's pipeline -- near-coastal
  contamination is a known satellite SSS problem and could be contributing to some of the noisier coastal
  cells seen in 12.2's maps.
- The `PreQC`-based filtering implemented in `build_matchups.py` is not wrong on its own terms (it removes
  genuinely low-confidence retrievals per each sensor's own quality field) but is **not what the operational
  system relies on** -- worth being explicit about this whenever comparing this project's results to
  operational assimilation behavior.
- None of this explains the ENSO-related tropical bias in 12.3 -- that's a train/test regime issue,
  orthogonal to which observations get admitted in the first place.

**Not yet acted on**: implementing the distance-from-coast filter (no blockers, public coastline data);
deciding whether/how to approximate the SST-passivation filter given SST itself is still an unresolved
missing-input problem for phase 2 (5).

## 14. Data source access summary (for extending date ranges / recovering stripped fields)

Investigated whether raw/fuller source data could be obtained to extend the SMAP/SMOS/Argo date ranges
beyond what's in this repo, and to recover fields IODA processing strips out (SST, sensor beam, ascending/
descending flag, roughness -- see 2's "Important gap found").

| source | access | notes |
|---|---|---|
| Argo GDAC (raw profiles) | **Public, no login** (`ftp.ifremer.fr/ifremer/argo` or US-GODAE) | Recovers real per-obs QC and WMO float ID -- would resolve the "no float ID, can't block by platform" limitation noted in 6 |
| obsForge / GDASApp source | **Public GitHub**, no credentials | Used directly in 13; no bulk data, just source code |
| SMAP L2 raw swaths (PO.DAAC/JPL) | Requires NASA Earthdata Login (increasingly S3-credentialed) | Blocked -- account creation is out of scope for this assistant; needs user-provided credentials or user-downloaded files |
| SMOS L2 raw swaths (CATDS) | "Free access by FTP upon email request" -- manual registration with a human | Blocked -- same reason |

Practical caveat not yet weighed: full L2 swath archives with all original fields, across multiple years,
are substantially larger than the already-thinned obsForge tarballs this project started from -- a real
bandwidth/storage question even where access isn't blocked (Argo).

## 15. Major correction: Argo timestamps were wrong (`dateTime` vs. `originalDateTime`)

**Everything in 4, 6, 12, 12.1, 12.2, 12.3 above used the wrong Argo timestamp.** `build_matchups.py` read
Argo's `dateTime` field for both the +/-3h time-window match filter and the `time_delta`/gross-error QC. Per
domain input: Argo obs are assimilated on a wider +/-4-cycle window than satellite obs (which use +/-3h), and
during IODA processing `dateTime` gets snapped to a nearby synoptic cycle slot for DA-window bookkeeping,
while the true measurement time is preserved separately in `originalDateTime`.

### 15.1 Verifying the bug

Checked directly against the data (three sampled cycles across different years):

| cycle | frac. `dateTime == originalDateTime` | `dateTime - originalDateTime` range |
|---|---|---|
| 2021-10-01 12Z | 13.7% | -25h to +24h |
| 2022-06-07 00Z | 11.1% | -25h to +24h |
| 2023-03-15 06Z | 9.1% | -23h to +24h |

Only ~9-14% of Argo obs had a `dateTime` matching their true observation time; the rest were offset by up to
a full day, spread across nearly the entire range rather than clustered near zero. This means `build_matchups.py`
was silently pairing satellite retrievals with Argo profiles up to ~24h apart while its own QC believed them
to be within the 3h match window -- pure temporal-mismatch noise with no relationship to actual satellite
bias, on top of everything else already in the matchup table. (Also checked: the `(lat, lon, dateTime)`
profile-grouping key stays valid despite this -- only 3 of 988 groups in the sample had ambiguous
`originalDateTime`, i.e. two genuinely different casts sharing a group key. Not a real concern.)

### 15.2 Fix and impact on matchup counts

`build_matchups.py` and `qc_diagnostics.py` now use `originalDateTime` (converted from raw epoch-seconds
float, since like `salinity` it carries no `_FillValue`/`units` attributes -- a plausible-range guard
[2000, 2030] is applied defensively, though a full-archive sample found zero implausible values). Full-archive
rebuild:

| sensor | matchups before fix | matchups after fix | retained |
|---|---|---|---|
| SMAP | 166,149 | **32,615** | 19.6% |
| SMOS | 298,115 | **61,402** | 20.6% |

The ~80% drop is expected and correct -- it's removing spurious matches that were never really within 3
hours of each other, not losing good data (with one caveat, see 15.5).

### 15.3 Impact on phase-1 results

Retrained both sensors on the corrected matchup tables (same 2021-2022 train / H1 2023 val / H2 2023 test
windows). Test-set metrics, before -> after:

| sensor | method | RMSE before | RMSE after | bias before | bias after | corr before | corr after |
|---|---|---|---|---|---|---|---|
| SMAP | raw | 1.643 | 1.599 | +0.365 | +0.342 | 0.536 | 0.552 |
| SMAP | FFANN | 1.348 | **1.282** | +0.050 | +0.044 | 0.653 | **0.676** |
| SMOS | raw | 2.318 | 2.325 | +0.025 | -0.005 | 0.407 | 0.413 |
| SMOS | FFANN | 1.429 | 1.434 | +0.056 | +0.131 | 0.540 | 0.521 |

SMAP improved modestly across the board after the fix (lower RMSE, higher correlation) -- consistent with
removing pure noise from the training/test data. SMOS is roughly a wash (FFANN RMSE flat, correlation and
bias slightly worse) -- plausibly just sampling noise given the much smaller test set now (5,178 vs. 27,213
rows), though not confirmed. Test sets shrank a lot (SMAP 30,996 -> 6,173; SMOS 27,213 -> 5,178) -- still
workable for this small model, but worth keeping in mind for how much to trust fine-grained (e.g.
per-basin) breakdowns going forward.

### 15.4 The tropical ENSO bias artifact (12.3) survives the fix

Regenerated the geographic error maps (`geo_errors_smap.png`, `geo_errors_smos.png`) on the corrected data.
**The tropical/subtropical negative-bias band in the FFANN-corrected panels is still there for both sensors**,
sparser now (far fewer test points per cell) but the same basic pattern. This is useful negative evidence:
it rules out the datetime bug as the explanation for that artifact (a plausible alternative hypothesis before
this fix) and strengthens the ENSO train/test regime-shift diagnosis in 12.3, since the artifact persists
after removing an entirely unrelated source of noise.

### 15.5 Known residual limitation: cross-cycle-boundary matches are not recovered

`build_matchups.py` processes one cycle directory at a time, loading only the satellite and Argo files
physically present in that directory. An Argo obs whose *true* time falls within 3h of a satellite obs in a
*different* cycle's directory (up to +/-4 cycles away, per the wider Argo DA window) will never be matched to
it, even if that would be a valid match -- because that Argo obs and that satellite obs are never loaded into
memory at the same time. The 15.2 counts are therefore a **lower bound**: some real matches are being missed,
not just spurious ones being removed. Fixing this properly would mean scanning a +/-4-cycle neighborhood
of Argo files against each cycle's satellite file, rather than one cycle at a time -- **implemented in 16.**

## 16. Cross-cycle-boundary matching (`match_windowed` in `build_matchups.py`)

Implemented the fix flagged in 15.5: for each cycle's Argo obs, search satellite candidates from a window of
+/-`cycle_window` cycles (default 4 = +/-24h, matching Argo's wider DA assimilation window), not just the one
cycle sharing Argo's own directory. Each candidate cycle's satellite file is loaded and its BallTree built at
most once (cached across the sliding window), so this costs extra tree *queries* per cycle, not extra file
I/O. Deliberately does NOT pool all candidate cycles into one combined BallTree and take the single nearest
point -- a spatially-nearer-but-wrong-time match from a neighboring cycle could otherwise mask a valid,
slightly-farther, correct-time match. Instead each candidate cycle is queried independently and the best
*valid* (passes distance/time/gross-error filters) match across all of them is kept.

### 16.1 An unexpected discovery: Argo profiles are replicated across cycle files

Initial testing (2-week sample, `--cycle-window 4` vs `--cycle-window 0`) showed a startling ~8x increase in
raw match count (SMAP 725 -> 6099, SMOS 958 -> 8002). Investigating *why* before trusting it turned up a real
duplication bug: **the same physical Argo profile appears in multiple cycle files**. Traced one profile
concretely (lat -53.53, lon -126.65, true `originalDateTime` 2022-06-01 06:10) through the output -- it showed
up as a candidate in six different cycle files, spanning `2022-06-01 00Z` through `2022-06-02 06Z` (nearly 24h
after its true measurement time), each copy carrying the same lat/lon/salinity/`originalDateTime` but a
different (re-snapped) `dateTime`.

This makes sense in hindsight: obsForge must replicate each Argo profile into every cycle file within its own
+/-4-cycle assimilation window, so that whichever cycle's DA run uses it, the obs is physically present in
that cycle's own file. Since `build_matchups.py` processes each cycle's Argo file independently, the *same*
real profile was being found and matched once per cycle-file appearance -- producing byte-identical duplicate
rows. Fixed with `drop_duplicates(subset=['argo_lat','argo_lon','argo_datetime'])` on the final result.

**A quieter version of this same bug already existed in the pre-windowing (15) matchup tables** -- checked
directly: 116/32,615 SMAP rows and 122/61,402 SMOS rows were exact duplicates by that same key (a profile
happening to find a valid single-cycle match in more than one of its replica cycle files). Small (~0.2-0.36%),
not enough to have meaningfully affected the 15.3 results, but a real pre-existing data-quality issue that
predates this session's windowing work -- now fixed as a side effect.

### 16.2 The honest result: windowing recovers almost nothing, once deduplicated

After fixing the duplication bug, `--cycle-window 4` vs `--cycle-window 0` on the same 2-week sample gave
**724 vs 725 matches (SMAP)** and **953 vs 958 (SMOS)** -- essentially identical, not the ~8x suggested by the
buggy version. The reason: Argo's own replication across ~9 cycle files (16.1) already gave single-cycle
matching multiple independent implicit tries at finding a valid same-cycle satellite match, since the same
profile would be re-attempted against each of its ~9 different home files' own satellite data. Deliberate
windowing turned out to be the *more correct and principled* way to search (one consolidated search per
profile, not luck-dependent on which specific replica's own file happens to contain a nearby satellite pass),
but not a source of meaningfully more matches -- the ground had already been implicitly covered.

Full-archive rebuild confirms this at scale:

| sensor | matches (15, pre-windowing, w/ dup bug) | matches (16, windowed + deduplicated) |
|---|---|---|
| SMAP | 32,615 | 32,557 |
| SMOS | 61,402 | 61,897 |

Both essentially unchanged (SMAP very slightly down after removing duplicates; SMOS very slightly up, likely
a few genuine boundary-case recoveries netting against duplicate removal). Retrained both sensors on the new
tables -- results also essentially unchanged from 15.3:

| sensor | method | RMSE (15.3) | RMSE (16) | corr (15.3) | corr (16) |
|---|---|---|---|---|---|
| SMAP | raw | 1.599 | 1.601 | 0.552 | 0.552 |
| SMAP | FFANN | 1.282 | 1.284 | 0.676 | 0.676 |
| SMOS | raw | 2.325 | 2.321 | 0.413 | 0.411 |
| SMOS | FFANN | 1.434 | 1.430 | 0.521 | 0.523 |

Geographic error maps (`geo_errors_smap.png`, `geo_errors_smos.png`) regenerated and visually unchanged from
12.2/15.4 -- the tropical ENSO bias artifact is still present, as expected (16 doesn't touch anything related
to that diagnosis).

**Net assessment**: this was worth doing for correctness and rigor (removes a luck-dependent matching
mechanism and a real, if small, duplication bug) even though it didn't move the headline numbers. `--cycle-window`
is exposed as a CLI flag on `build_matchups.py` for anyone who wants to experiment with wider/narrower search
windows later.

## 17. Raw Argo retrieval: recovering WMO float ID and real QC

Motivated by 14: obsForge's Argo `PreQC` is unusable (§4) and there's no float ID, blocking clean
float-based train/test splitting (§6). Investigated pulling from the raw public GDAC directly.

### 17.1 Why delayed-mode Argo for the training/eval target, even though SMAP/SMOS must stay real-time

The project's purpose is real-time operational bias correction, so SMAP/SMOS *inputs* must always be
real-time -- there's no delayed-mode reprocessing of the satellite side to fall back on operationally. But
Argo here is the training *label* (ground truth), not an input: the model's job is "given a biased real-time
satellite retrieval, predict the true near-surface salinity," and "true near-surface salinity" is a fixed
physical fact that doesn't change based on when Argo's own QC/calibration happened. Delayed-mode Argo is
simply a more accurate estimate of that same fixed quantity; training against noisier real-time Argo would
inject Argo's own sensor-drift error into the target for no benefit. Not a leakage concern -- standard
practice is to use the best available labels even when the deployed model sees noisier real-world inputs.
Practical rule: prefer delayed-mode (D), fall back to real-time (R/A) only where D isn't yet available
(delayed-mode processing lags real-time by ~6-12 months; largely moot for our 2021-2023 window since it's
several years old by now).

### 17.2 Setup and a real dependency bug

`pip install argopy` pulled `erddapy==3.3.0`, which is incompatible with `argopy` 1.4.0 (`ImportError:
cannot import name '_quote_string_constraints'`) -- broke *all* of argopy's data fetchers, including the
GDAC one needed here, at import time. Fixed by pinning `erddapy==3.2.1`. Worth remembering if this env is
rebuilt: `pip install argopy 'erddapy==3.2.1'`, not just `pip install argopy`.

Validated against a real matchup row (SMAP, lat 4.98976, lon -168.84845, 2021-07-02 05:57): argopy's `region()`
fetcher (mode='standard', which auto-applies delayed-mode-preferred/real-time-fallback -- no need to hand-roll
that merge) returned the *exact* matching profile -- same lat/lon, true time off by 33s, near-surface salinity
average 34.7317 vs. our obsForge-derived 34.7318. Confirms our existing near-surface-averaging logic is
correct, and recovers `PLATFORM_NUMBER` (WMO float ID, 5906681) and real QC (`PSAL_QC=1`, `DATA_MODE='D'`)
that obsForge's version has neither of.

### 17.3 `region()` doesn't scale to a global bounding box -- pivoted to `ArgoIndex`

A single **1-day, global** near-surface `region()` fetch took ~20 minutes and ~21GB RAM before completing.
A second attempt (testing whether caching would help) was killed after 13+ minutes with no output at all --
though it's worth being honest that this was inferred from resource/timing similarity to the first call, not
confirmed certain it would never have returned (a fair challenge raised mid-session). Either way, scaling this
to 5 years of global chunks was clearly impractical.

Pivoted to `argopy.ArgoIndex`, which downloads the GDAC's global profile index file *once* rather than
querying the server per time/region window. Loading the full index (3,371,859 records, all-time, all
profile types) took **2.88 seconds**. Filtering it locally (no network) to our 2021-2025 window: **853,543
matching profiles**, in under 5 seconds. Dramatically more scalable regardless of whether the killed
`region()` call was broken or just slow -- this is now the right tool for bulk index lookups.

Revised plan (not yet implemented): rather than bulk-fetching all 853,543 profiles globally (a separate,
redundant dataset), nearest-neighbor match our *existing* 32,557 (SMAP) + 61,897 (SMOS) matchup rows against
the loaded index by lat/lon/date (same collocation technique already built for SMAP/SMOS-vs-Argo) to recover
WMO ID + file path per row, then fetch only those specific profile files (likely well under 94,454 given
overlap between the two sensors' shared underlying Argo profiles) -- targeted enrichment of what we have,
not a bulk pull.

### 17.4 What QC does obsForge vs. the operational DA system actually apply to Argo?

Read the actual source (`NOAA-EMC/obsForge` b2i converter and `NOAA-EMC/GDASApp` DA filter config) rather
than continue inferring from data alone.

**obsForge's own QC (`utils/b2i/b2iconverter/ioda_variables.py`) is exactly as crude as suspected**: global
bounds only -- salinity `[0, 45]` PSU, temperature `[-10, 50]`C -- plus basic lat/lon/depth NaN cleaning and
an ID-pattern filter (`stationID` second digit `== 9`) to separate Argo from other profiling-float types in
the same raw BUFR "subpfl" tank. `[0,45]` PSU would not have caught the 0.118 PSU stuck-sensor profile found
in an earlier session (§4) -- confirms `PreQC` being unusable isn't a processing bug, it's simply that no
real QC is computed at this stage at all.

**Structural finding**: the Argo b2i converter (`bufr2ioda_insitu_profile_argo.py`) reads from WMO GTS BUFR
messages ("subpfl" tank) -- our entire local Argo archive is **real-time GTS data**, not the GDAC's delayed-
mode archive. Anything that doesn't transmit via GTS promptly (recovered-after-the-fact data, non-GTS DACs,
transmission gaps) is simply absent from our local files regardless of QC -- this is the actual source of
"delayed-mode might have more profiles than we have locally," distinct from any QC question. Also notable:
raw BUFR *does* carry a WMO platform ID (`stationID` from descriptor `WMOP`) and obsForge's converter reads
it (uses it for the Argo-vs-other-floats filter above) but does not carry it through into the final IODA
`MetaData` group -- the float ID is available upstream and simply not surfaced in the files we've been using.

**GDASApp's actual Argo salinity DA filter chain** (`parm/jcb-gdas/observations/marine/insitu_salt_profile_argo.yaml.j2`)
is far richer than the satellite chain (13), and directly informative for QC we should adopt:
- **Region-specific salinity bounds**, not one global range: global `[2,41]` PSU, but separately tuned for
  Red Sea `[2,41]`, Mediterranean `[2,40]` (two sub-boxes), Northwestern European shelves `[0,37]`,
  Southwestern shelves `[0,38]`, Arctic (lat>=60) `[2,40]`. Real brackish/marginal-sea water is
  accommodated regionally -- our single global `[20,42]` filter (§4) doesn't do this, and may have been
  rejecting legitimate low-salinity obs in exactly the shelf/high-latitude regions where basin-level
  instability already showed up (§12).
- **A "Spike and Step Check"** (tolerance 0.05 PSU) purpose-built to catch rounded/stair-stepped depth
  profiles -- precisely the signature of the stuck-sensor artifact found manually in an earlier session.
  The operational system catches this class of error systematically; our pipeline doesn't.
- A bathymetry consistency check (reject if reported depth exceeds the model's own seafloor depth there)
  and background checks (need live GeoVaLs -- same "not available to us" limitation as 13).

**Implication for QC if the profile set is expanded**: use Argo's own native per-obs QC flags (`PSAL_QC`,
confirmed populated in the raw GDAC data -- our validated test profile had `PSAL_QC=1`) instead of the
current ad hoc `[20,42]` range + gross-mismatch filter. This is the actual scientific QC assessment Argo
performs, strictly more principled than inferring thresholds from data. Plan: filter to `PSAL_QC == 1` (good),
optionally allow `2` (probably good); adopt GDASApp's region-specific bounds as a cheap complementary sanity
layer; keep the gross-mismatch-vs-satellite check since it serves a different purpose (bad collocation, not
bad individual obs).

## 18. Float-ID-aware train/val/test split

Decided to do the "narrower thing" first (incorporate the recovered WMO float ID into matching/splitting)
with the broader profile-set expansion deferred to later -- see the discussion in this session about expected
payoff: float ID mainly buys evaluation *rigor* (proper no-leakage grouping), not better model performance
per se, whereas real `PSAL_QC` and (especially) expanding the profile set were judged more likely to move
actual metrics. This section covers the rigor step.

### 18.1 Implementation

`src/attach_wmo_to_matchups.py`: merges `data/matchups/argo_wmo_lookup.parquet` (17.3's index-matched WMO
IDs) onto both matchup tables by `(argo_lat, argo_lon, argo_datetime)`, adding a `wmo` column (NaN where
unmatched). Coverage: **94.9% of SMAP rows, 78.5% of SMOS rows** got a float ID (SMOS lower, consistent with
its lower index-match rate for 2024-2025 noted in 17.3).

`src/features.py::split_data()` redesigned to be float-aware: compute the naive date-only partition as
before, then for every row with a known `wmo`, reassign the *entire* float to whichever of train/val/test
contains its **earliest** in-window observation -- guaranteeing no float ever appears in more than one
partition. Rows with no recovered float ID (~5% SMAP, ~21% SMOS) keep the naive date-only assignment, relying
on the existing embargo gaps as their only leakage protection, same as before this change. Embargo-period
rows are untouched either way -- this reassignment only moves rows *between* train/val/test, never pulls an
embargo-dropped row back in. Verified directly: zero floats appear in more than one partition, for both
sensors, after the change.

### 18.2 The cost of doing this properly: val/test collapse in size

| sensor | split | date-only n | float-aware n |
|---|---|---|---|
| SMAP | test | 6,124 | **1,035** |
| SMAP | val | ~5,000s | 1,885 |
| SMOS | val | ~5,000s | **480** |
| SMOS | test | 5,212 | 10,592 |

Test-set metrics moved differently per sensor: SMAP's FFANN RMSE barely changed (1.284 -> 1.288) but bias got
notably *worse* (+0.042 -> +0.339) and correlation improved (0.676 -> 0.768); SMOS's numbers improved across
the board (RMSE 1.430 -> 1.211, bias 0.135 -> -0.019, corr 0.523 -> 0.562). Given how much smaller and
differently-composed these test sets now are (SMAP down to just 1,035 rows), **these movements are more
likely sample-composition noise than genuine model-quality change** -- not a conclusion to lean on either way
without a larger, more representative evaluation set.

**Why this happens, and why it's not a fixable artifact of the specific rule chosen**: Argo floats have a
4-5 year typical lifespan. Any float active during the H1/H2 2023 val/test windows was almost certainly
*already* active back in 2021-2022, so strict no-leakage float grouping pulls nearly every float into train
regardless of which "earliest partition" rule is used -- val/test end up containing only floats that
happened to be newly deployed during that specific ~5.5-month window, a small and possibly non-representative
population, not a random sample of the ocean. This is inherent to the data's true structure (long-lived
floats, short observation window), not a bug in the reassignment logic.

**This is the direct link to "expanding the data selection later"**: a wider date range would mean more
calendar time for new floats to appear within each window, growing val/test back toward a usable, more
representative size. Deferred for now per the plan agreed this session (float ID/rigor first, profile-set
expansion later).

**Resolved in 18.3-18.5 below**: rather than debate the open decision above, tested empirically whether
float-level leakage is actually material for this model. It isn't -- the float-aware split has been
reverted to the plain date-based one as the default.

### 18.3 Testing whether float leakage is actually material, rather than assuming it

Challenged directly: bulk salinity is a physical property of the ocean, not of the instrument measuring it.
If Argo floats are well-calibrated and QC-good data is truly interchangeable across instruments, float
identity shouldn't matter, and the whole float-aware partitioning in 18.1/18.2 would be solving a problem
that doesn't exist for this model (which never takes float ID as an input in the first place).

Built `src/float_leakage_diagnostic.py` to test this directly rather than continue reasoning about it in the
abstract: train on the plain date-based split (`split_data_naive`, full-size test set), then label each test
row by whether its float *also* has rows in train ("seen"), has no rows in train at all ("unseen", i.e. a
genuinely novel instrument), or has no recovered float ID ("unknown"). If "seen" test rows show better error
than "unseen" ones, that's evidence of real instrument-level leakage; if not, the concern is unfounded for
this model.

**Result: no evidence of leakage, and if anything the opposite pattern.**

| sensor | group | n | raw RMSE | FFANN RMSE | improvement |
|---|---|---|---|---|
| SMAP | seen | 4,312 | 1.652 | 1.353 | 18.1% |
| SMAP | unseen | 1,364 | 1.233 | 0.790 | 35.9% |
| SMOS | seen | 3,000 | 2.340 | 1.416 | 39.5% |
| SMOS | unseen | 1,831 | 2.271 | 1.245 | 45.2% |

"Seen" floats show *worse* RMSE than "unseen" in both sensors, both before and after correction -- the
opposite of what leakage would predict. Critically, **the gap is already present in the raw, uncorrected
satellite data**, which the model never touches -- so it can't be a model artifact. It's a population
difference between long-lived floats (in train because they've been active since 2021-2022) and newly-
deployed ones (first appearing during the test window), not evidence of instrument-identity memorization.
The model's relative improvement (raw -> corrected) is also slightly *larger* for unseen floats in both
sensors, again the opposite of the leakage prediction.

Side finding from the same run: the "unknown" (no recovered float ID) group has the worst error of the three
in both sensors and both raw/corrected -- confirms these rows aren't missing at random (14 speculated this;
now confirmed).

**Decision: reverted to `split_data_naive` as the function actually used by `train_baseline.py`.**
`split_data()` (float-aware) is kept in `features.py` for reference/future use -- e.g. if 17's profile-set
expansion ever grows val/test enough to make strict float grouping affordable again -- but is not the
current default. Recovers the full-size test sets (SMAP 6,124, SMOS 5,212) rather than the collapsed
1,035/10,592 from 18.2.

### 18.4 Chasing the seen/unseen gap: two hypotheses, both weakened by evidence

Investigated *why* seen floats show worse error than unseen ones (a real, if now-understood-as-harmless-to-
leakage, pattern worth understanding). Two mechanistic hypotheses, both checked against data rather than left
as speculation:

- **Regional/drift hypothesis**: long-lived floats have had years to drift into dynamically active regions
  (fronts, boundary currents) that are harder for both satellite retrieval and Argo profiling geometry;
  newly-deployed floats sit near their deliberately-chosen, more "typical" deployment sites. Checked: mean
  |latitude| and ocean-basin distribution for seen vs. unseen test rows, both sensors. Result: nearly
  identical between groups (mean |lat| within ~2 degrees; similar basin fractions) -- weakens this as the
  primary driver, though a finer-grained check (specific frontal proximity, not just basin/lat) wasn't done.
- **Sensor-aging hypothesis**: real-time (non-delayed-mode) data doesn't get drift/calibration correction;
  older floats have had more time to accumulate it. Tested directly (`src/test_sensor_aging_hypothesis.py`):
  fetched the actual GDAC profile file for all 9,267 unique seen/unseen test profiles (via direct HTTPS,
  bypassing argopy's `profile()` fetcher which errors on this dataset's filename pattern -- see script
  docstring), compared each profile's real-time (obsForge-derived) near-surface salinity against its own
  delayed-mode-preferred value. **Result: does not support the hypothesis.** Median discrepancy is tiny and
  nearly identical between groups (~0.0001-0.0002 PSU for both seen and unseen, both sensors) -- most
  profiles in both groups already agree closely between real-time and delayed-mode, contradicting an
  "accumulated drift" story. Mean discrepancy is actually slightly *higher* for unseen floats in both
  sensors, opposite the hypothesis's predicted direction, though driven by heavier-tailed outliers (much
  larger std for unseen) rather than a systematic shift.

**Net: both leading hypotheses are weakened by direct evidence, and the true cause of the seen/unseen gap is
an open question.** The gap itself is real (too large to dismiss as pure sampling noise) but its explanation
isn't pinned down by what's been checked -- left unresolved rather than reaching for a third untested story.
Not blocking anything (18.3 already established the gap doesn't threaten model validity), just an
interesting loose end if revisited later.

## 19. Are the white gaps in the corrected maps a deployment-confidence problem?

Prompted by a direct question: if this model were used operationally for SMAP/SMOS bias correction feeding
an ocean model, do the white (masked, <5 obs) gaps in 12.2's corrected panels mark where the correction
shouldn't be trusted? And is it sound to deploy a correction that isn't verified everywhere?

### 19.1 Test-coverage gaps are not the same thing as model confidence

The FFANN is a plain deterministic network with no self-aware uncertainty signal -- it produces a point
estimate for any SMAP/SMOS pixel, including places with zero Argo verification, with no internal "I'm
extrapolating" flag. So the white gaps mark where *we* can't verify performance, not where the model itself
expresses low confidence. The two are related but distinct: low test-window density is a reasonable *proxy*
for low training-window density (Argo's spatial sampling pattern is fairly stable year to year), and low
training density is a real reason to distrust a learned local correction -- but proxy isn't identity.

**Concrete evidence this isn't hypothetical**: ocean basin 0, the smallest-sample basin in this project
(11-45 test rows), is exactly the case where the FFANN did *worse* than the raw or constant-bias baseline
(12) -- a real, already-observed instance of extra model flexibility hurting in a sparse region rather than
generalizing.

**Structural parallel already in the operational system**: GDASApp's own QC (13, 18) uses domain checks
(sea_area_fraction, distance-from-coast, SST passivation) to exclude assimilation in specific
known-unreliable conditions. The same principle applies to gating a *correction*, not just raw observations:
apply the learned correction where locally well-supported, fall back to something safer (raw value, or the
simpler constant-bias correction, far less prone to this failure mode) where it isn't.

**Answer to the soundness question**: applying the same learned, spatially-varying correction everywhere
with no gating is not sound -- it risks introducing a new, unvalidated bias exactly where there's the least
ability to catch it. A gated/tiered deployment (ML correction where supported, safer fallback elsewhere) is
the standard, defensible approach, not a special case invented for this project.

### 19.2 Training density vs. test-masked cells: quantified overlap (`src/plot_training_density.py`)

Checked how much of the test-set masking is genuine low-confidence (also sparse in training) vs. just a
short-test-window validation blind spot over otherwise-adequate training data:

| sensor | masked test cells | also low train density (real concern) | zero training data (pure extrapolation) | train was fine (validation-only gap) |
|---|---|---|---|---|
| SMAP | 552 | 97 (17.6%) | 27 (4.9%) | 428 (77.5%) |
| SMOS | 700 | 309 (44.1%) | 129 (18.4%) | 262 (37.4%) |

**SMAP**: the large majority (77.5%) of its masked test cells were actually well-sampled during training --
mostly a validation-window artifact, not a real confidence problem. Only ~22.5% combined represent genuine
low-training-support regions. **SMOS is meaningfully worse**: 62.5% combined are real low-confidence zones
(44.1% compounding sparse train+test, 18.4% literally zero training support). This matters directly for 19.1
-- SMOS would need confidence-gating before any operational use much more urgently than SMAP.

### 19.3 Is Argo itself sparse in the gap regions? No -- confirms the user's suspicion directly

Built `src/compute_argo_coverage.py` (full-archive scan, globally deduplicated across the ~9-cycle-file
replication per 16.1 -- **542,304** unique near-surface Argo profiles found locally, 2021-2025) and
`src/plot_argo_coverage.py` to compare against the satellite-matched subsets. Saved to
`data/matchups/argo_coverage.png`.

**Result: Argo coverage is dense and close to globally uniform** (hundreds to 1000+ profiles per 5deg cell
over the period), including in regions that look sparse in the SMAP-matched (n=32,557) and SMOS-matched
(n=61,897) panels -- only the most extreme polar margins near ice edges are genuinely Argo-thin. The
satellite-matched subsets are visibly patchier and lower-density *everywhere*, even in open ocean where raw
Argo is abundant.

**This reframes 19.1's gaps**: they are not regions where ground truth is fundamentally unavailable -- they
are regions where the current satellite-matching pipeline (tight 50km/3h window, ~78-81% satellite QC pass
rate, a few years of study period) hasn't accumulated enough *verified pairs* yet, despite abundant
underlying Argo data. A wider match window or a longer accumulation period could likely recover much better
validation density in these regions specifically because Argo itself isn't the bottleneck -- a more
optimistic framing than "permanently unverifiable."

Side observation: a distinct, unusually dense hotspot around 30-40N, 50-70W (western North Atlantic) in the
raw Argo coverage -- likely a dedicated research array, not representative of typical open-ocean density.

## 20. Identifying the actual raw SMAP source obsForge ingests

Investigated a specific candidate raw source (PO.DAAC's `SMAP_RSS_L2_SSS_NRT_V6`, produced by Remote Sensing
Systems) to recover the fields IODA stripped out (14, 17). Found a real schema mismatch: that product's
variables (`cellat`, `cellon`, `sss_smap`, `iqc_flag`, `time`) don't match anything `Smap2Ioda.h` reads
(`lat`, `lon`, `smap_sss`, `smap_sss_uncertainty`, `quality_flag`, `row_time`, plus `REV_START_YEAR`/
`REV_START_DAY_OF_YEAR` attributes) -- and it's also titled "Level 2C", while the converter's `row_time`/
`REV_START_YEAR` naming is characteristic of swath-level ("Level 2B") data, not a resampled/gridded product
(RSS's "L2C" is smoothed to ~70km on a fixed per-orbit Earth grid, per its own variable descriptions).

**Resolved definitively, not by inference** -- checked the actual local IODA files' global attributes
(`obs_source_files`, e.g. `SMAP_L2B_SSS_NRT_39245_A_20220606T230121.h5`), confirmed consistent across cycles
from 2021, 2022, and 2023. This is genuinely the real raw filename pattern feeding obsForge, not a stale
converter or a wrong guess. Searching for that exact filename convention (rather than the RSS product) found
the real source: **`SMAP_JPL_L2B_NRT_SSS_CAP_V5`** -- produced by **JPL itself**, using their own "CAP"
(Combined Active-Passive) retrieval algorithm, a completely different processing center/algorithm than RSS.
Confirmed via its Variables tab: exact match to `Smap2Ioda.h`'s expected fields, including `row_time`
described word-for-word as "Approximate observation time for each row, UTC seconds of day" -- identical to
the C++ code's own comment.

This product is substantially richer than what IODA retains, and covers essentially all of the phase-2/3
candidate inputs from 5 and the original project brief:
- `inc_fore`/`inc_aft`, `azi_fore`/`azi_aft`, `antazi_fore`/`antazi_aft` -- fore/aft look geometry, SMAP's
  equivalent to Aquarius's multi-beam/horn inputs in Vernieres et al. (SMAP conically scans with fore/aft
  looks rather than Aquarius's 3 physical horns)
- `anc_sst` -- ancillary SST (NOAA OI), the SST input from Vernieres et al.
- `anc_spd`/`anc_dir` -- ancillary wind speed/direction (NCEP), the wind-stress input Trossman & Bayler
  suggest
- `smap_spd`, `smap_high_spd`, `smap_high_dir` -- SMAP's own retrieved wind speed/direction, independent of
  the ancillary field
- `ice_concentration` -- directly relevant to the high-latitude/ice-edge error hotspot found in 12.2
- per-polarization, per-look brightness temperatures (`tb_h_fore`/`tb_h_aft`/`tb_v_fore`/`tb_v_aft`)

Access still requires NASA Earthdata Login (14's blocker, unchanged). No SMOS-side equivalent investigation
done yet -- worth doing the same exercise for whatever raw source feeds `Smos2Ioda.h` before assuming the
same access situation applies.

## 21. Open architectural question: where does the correction actually run?

Raised directly: is it feasible to train on the rich JPL CAP inputs (20) -- SST, wind, roughness proxies,
fore/aft look geometry, ascending/descending, ice concentration -- while still being able to apply the
correction using only the fields already present in the stripped-down SMAP IODA files? Not resolved, and the
answer depends entirely on an architecture decision that hasn't been made (or at least hasn't been surfaced
to this project):

- **If the correction runs at/near the raw-to-IODA conversion step** (e.g. folded into `Smap2Ioda.h` or a
  processing stage sharing its input), all the rich CAP fields are available at exactly the moment the
  model would need them -- the model's output (corrected SSS) is the only thing that needs to survive into
  the final IODA file. Fully feasible, no proxies needed.
- **If the correction is meant to run downstream, as an independent step consuming only the already-produced
  IODA files** (e.g. a standalone Python tool deliberately decoupled from obsForge's C++ codebase), the rich
  fields are genuinely gone by that point -- but this is a much smaller problem than it first appears (21.1).

### 21.1 Refinement: retaining the raw fields in IODA is a minor code change, not a rearchitecture

Checked directly against `Smap2Ioda.h`'s actual code rather than assuming: every field it currently extracts
uses the identical trivial pattern `ncFile.getVar("<name>").getVar(buffer)` -- `lat`, `smap_sss`,
`quality_flag`, etc. are not special; `anc_sst`, `anc_spd`, `anc_dir`, `inc_fore`/`inc_aft`, `ice_concentration`
are just other variables in the same raw file, currently simply not read. The base `IodaVars` class already
has a `floatMetadataNames` vector for exactly this purpose -- SMAP's converter currently sets it to `{}`, i.e.
the infrastructure for carrying extra metadata into the IODA output already exists and is unused, not absent.

One exception worth being precise about: ascending/descending is not a netCDF variable at all -- it's
embedded in the raw filename itself (`_A_`/`_D_` in `SMAP_L2B_SSS_NRT_..._A_...`). Retaining it needs a small
filename-parsing addition, a different (still easy) code path than the rest.

**This adds a third, likely-best option to 21's fork**: keep the correction model downstream and decoupled
from obsForge's C++ codebase (simpler to iterate on as ML code, no need to embed model inference in the
conversion pipeline), while making a one-time, additive change to `Smap2Ioda.h` (and presumably `Smos2Ioda.h`)
to carry the needed raw fields into the IODA output as new metadata variables. This gets full data richness
at the correction step without either embedding model inference inside the C++ pipeline, or falling back to
climatology approximations for genuinely observation-specific fields. The climatology-proxy fallback noted
above is still worth keeping in mind if extending the converter turns out to be impractical for institutional
reasons, but it's no longer the default answer.

**Not blocking**: regardless of which architecture turns out to be real, training the full rich-feature model
as a research upper bound (once Earthdata access is sorted out) is valuable on its own -- it quantifies how
much accuracy is actually at stake, which is exactly the number needed to justify pursuing either the
converter change or embedding the model in conversion, versus staying with the current lat/lon/season/basin-
only model.

## 22. Confirmed the raw JPL CAP source end-to-end with real data

Earthdata Login set up (`~/.netrc`, `machine urs.earthdata.nasa.gov`, `chmod 600` -- credentials never passed
through this assistant). Installed `podaac-data-subscriber`/`podaac-data-downloader` (needed
`brew install geos` first -- a `shapely` build dependency wasn't present; downgraded `packaging` to 23.2 as a
side effect, verified this doesn't break xarray/torch/sklearn/pyarrow/matplotlib despite the version-mismatch
warning). Downloaded two real granules from `SMAP_JPL_L2B_NRT_SSS_CAP_V5` for 2022-06-07.

**Every open question from 20 is now confirmed directly against real data, not inferred:**
- Variable names, `phony_dim_0`/`phony_dim_1` dimensions: exact match to `Smap2Ioda.h`.
- `REV_START_YEAR`/`REV_START_DAY_OF_YEAR` global attributes (unverifiable from the PO.DAAC web page, since
  it only lists variables, not attributes): present and correct (2022, day-of-year 158).
- `row_time` = 20112s for the first row exactly equals `REV_START_TIME` (05:35:12 UTC) converted to seconds
  since midnight -- confirms the "UTC seconds of day" documentation is accurate.
- `quality_flag` values (`0, 2, 64, 66, 529, 641, 643, 705, 707, ...`) overlap almost entirely with the
  `PreQC` values found in local IODA files early in this project -- closes the loop: local `PreQC` really is
  this exact raw bitmask, passed through unmodified, exactly as the converter code shows.
- `anc_sst` (273-310K, physically sensible), `anc_spd`/`anc_dir`, `ice_concentration` (0-6.1% in this sample),
  `inc_fore`/`azi_fore`: all genuinely populated with real values, not placeholders. Note: like Argo's
  `salinity` and `originalDateTime` (4, 15), some of these fields have a real `_FillValue` (`-9999.0`) that
  isn't always caught by default masked-array handling -- checked with `np.ma.filled() + np.isfinite()`
  rather than trusting the mask alone, same lesson as those earlier fields.

This fully closes out 20's open item and validates 21/21.1's premise (the rich fields genuinely exist in the
raw file obsForge already reads) with real downloaded data rather than documentation alone. Next step, not
yet done: pull a large enough sample to train the rich-feature research upper-bound model discussed in 21.

## 23. Rich-feature matchup table from raw JPL CAP swaths (`src/build_raw_smap_matchups.py`)

Pulled a full week (2022-06-01 to 2022-06-08, 207 orbit files, 2.0GB) of real `SMAP_JPL_L2B_NRT_SSS_CAP_V5`
granules into `data/raw_smap_cap/` (kept outside git via `data/`'s existing `.gitignore` entry, but persisted
on disk across sessions rather than left in the scratchpad). Built a matchup table against near-surface Argo
carrying all the rich per-pixel fields the operational IODA converter currently drops (incidence/azimuth
angles, brightness temperatures, ancillary wind/SST, ice concentration, retrieval uncertainty -- see 21.1's
`RICH_FIELDS` list), written to `data/matchups/smap_cap_argo_matchups.parquet`.

Two implementation details that don't apply to `build_matchups.py`'s IODA-based pipeline:

- **Timestamp reconstruction**: raw files have no per-obs absolute timestamp field. `row_time` (UTC seconds
  of day, varying only along-track) is combined with the file's `REV_START_YEAR`/`REV_START_DAY_OF_YEAR`
  global attributes to get an absolute datetime per pixel. Verified safe to assume no day-rollover within a
  single file (each file's `row_time` span is ~1-2h, far short of 86400s).
- **Fill-value handling**: bypasses netCDF4's auto-masking entirely (`ds.set_auto_mask(False)`) and manually
  replaces each variable's declared `_FillValue` with NaN, rather than trusting the mask -- continuing the
  same lesson as 22 and 4/15's Argo fields. One rich field, `anc_swh` (ancillary wave height), turned out to
  be `_FillValue` for 100% of pixels in this product -- confirmed genuine (not a reading bug) by checking a
  raw file directly; it survives in the table as an all-NaN column.

**Bug caught before trusting the output**: satellite files aren't chunked into DA cycles the way IODA sss
files are -- there's no natural "search a window of nearby cycles" structure. A first attempt pooled all
206k+ QC-pass pixels across all 207 files into one global BallTree and took each Argo obs's single nearest
neighbor. This found only 51 matches, vs. 310 for the IODA-based `build_matchups.py` pipeline over the
identical week/settings (50km/3h/QC-pass). Diagnosis: pooling across days means a spatially-closer pixel from
a *different day's* orbit pass can mask the true same-pass match, which is farther in pure distance but
correct in time -- exactly the failure mode `build_matchups.py`'s `match_windowed` docstring already warns
about for cycle-windowed search (15.5-adjacent). Fixed by keeping each orbit file as a separate candidate
(mirroring `match_windowed`'s per-cycle-candidate design, generalized to carry arbitrary rich columns through
via a (file, row) index rather than per-column `np.where`): querying every file's own tree per Argo obs and
keeping the best result that *also* passes the time/gross-error filters, not just the single spatially
nearest pixel overall. This produced exactly 310 matches -- matching the IODA-based reference count for the
same window, and specifically the 259 obs that were being silently dropped are the ones whose truly-matching
pass wasn't the day's spatially closest pixel.

Not yet done: training the rich-feature model on this table and comparing it against the lat/lon/season/basin
baseline (21's "research upper bound" question) -- this table only builds the inputs.

## 24. Rich-feature POC results: 1 week vs. 12 weeks -- the 1-week result was mostly noise

`src/train_rich_features_poc.py` trains two FFANNs on an identical random 70/15/15 split of the rich matchup
table: one restricted to the operational baseline's feature set (`sat_sss`, lat, lon, season, basin), one with
the rich JPL CAP fields added, so any RMSE gap is attributable to the extra fields rather than to different
data or splits.

**1 week** (310 matches, test n=47): rich RMSE 0.595 vs. baseline-features RMSE 0.628 -- looked like a ~5%
improvement.

**12 weeks** (2022-06-01 to 2022-08-24, 1882 raw files, 3550 matches, test n=533, after extending the download
-- see below): rich RMSE 1.349 vs. baseline-features RMSE 1.371 -- still an improvement, same direction, but
only ~1.6%, an order of magnitude smaller than the 1-week number suggested. (Absolute RMSE is also much higher
than the 1-week run's -- not a regression, just a much more representative, harder-to-fit sample; the 1-week
window's 47-row test set was too small to trust either its RMSE level or the size of the gap between models.)
Confirms the concern raised when the 1-week result first came back: a 47-row test set can't resolve an effect
this small, and it didn't -- the true gap is real but modest, not the ~5% first glimpsed.

Caveat that still applies at 12 weeks: the whole window is boreal summer 2022, so this doesn't yet test
whether the gain holds across seasons the way the operational baseline's train/val/test split (spanning
2021-2023) is designed to. That requires the full-year pull discussed when 12 weeks was chosen as a cheaper
checkpoint.

**Bug caught along the way**: `sat_ice_concentration` is ~constant 0 at these (mostly ice-free, mid-latitude)
float locations -- not bit-exactly constant, so its train-set std came out to `5.6e-19` rather than exactly 0,
and the `Standardizer`'s `std == 0` guard didn't catch it. Dividing by that near-zero std blew any val/test row
differing by float noise up to nonsense (RMSE ~129,000 on the first run). Fixed by thresholding the guard
(`std < 1e-8`) instead of checking exact equality.

**Download note**: the 11 additional weeks (1677 files) had to be resumed once -- the download stalled
mid-run (TCP connection stayed `ESTABLISHED` but no bytes moved for 10+ minutes) after ~260 files. Killed and
re-ran the identical command; `podaac-data-downloader` skips files already present on disk, so this picked up
exactly where it left off with no lost progress or re-downloaded data.

## 25. Full year (2022-06-01 to 2023-06-01): a real season-held-out answer, and a genuine outage

Extended the download to a full year: 8,937 raw files, 88GB, 14,705 Argo matches -- a 4.15x increase over the
12-week table's 3,550, roughly tracking the ~4.35x longer time span (a bit less, because of the gap below).

**Real SMAP outage found, not a bug**: the Aug-Sep 2022 chunk returned only 80 files instead of the expected
~850 for a normal month. Checked several other months (Sep-Oct 2022, Jan 2023, May 2023) for comparison -- all
normal (~830-890 files) -- so this was isolated, not a retention-policy artifact. Root cause confirmed via web
search: **SMAP entered spacecraft safe mode on 2022-08-08** (cause never publicly detailed beyond "the Project
team is investigating the cause of the anomaly"), and NASA's own dataset-release announcement states "No SMAP
salinity data were available from 08/09/2022 - 10/06/2022, as SMAP was not in science mode or flying in its
nominal orbit during that time." This matches what the archive actually shows almost exactly (a handful of
files Sept 21-23, likely early recovery downlinks, then a return to normal cadence). The eventual matchup
table carries this as a real ~2-month-thin patch, which is correct behavior, not something to fix.

**36 silently-corrupted downloads found and fixed**: after the full download finished cleanly (0 reported
failures across all 10 chunks), a full-file validation pass (opening every file and reading `smap_sss`) found
36 files (0.4%) that were truncated -- file sizes from 16KB up to just under the normal ~10.3MB, scattered
across Oct 2022 through May 2023, most likely transient connection resets that completed the HTTP request but
not the full byte stream, which `podaac-data-downloader` doesn't check (no checksum/size verification, only
"did the request not raise"). Fixed the same way as the earlier stall: delete the corrupt files, re-run the
full-range download command, and it re-fetched exactly the missing 36 (plus 8 boundary files the monthly
chunking had skipped) while skipping the ~8,893 good ones. Re-validated after: 0 bad files. Lesson for any
future large `podaac-data-downloader` pull: always validate file integrity after "0 failures" is reported --
that count is necessary but not sufficient.

**Quadratic runtime discovered in `build_raw_smap_matchups.py`'s matching step**: the full-year matchup build
took ~2h47m, far more than linear scaling from the 12-week run's runtime would suggest. Cause: `match_to_argo`
checks every Argo observation against every raw orbit file's BallTree, regardless of whether that file's date
could possibly be within the 3h match window -- there's no date-based candidate pruning the way
`build_matchups.py`'s cycle-windowed search has. Since both the number of files and the number of Argo obs
grow linearly with the requested date span, total cost is `O(files x argo) = O(span^2)`: a ~4.35x longer span
cost ~19x more compute, which is what was observed. Output is still correct (validated against the IODA
baseline for the 12-week window, DESIGN.md 23), just not efficient -- worth adding a date-proximity pre-filter
before extending this further (e.g. to multiple years).

### 25.1 Fixed: sort Argo obs by time, binary-search each file's relevant window

Each raw file spans only ~1-2h, so an Argo obs more than `max_time_delta` outside a file's own `[min, max]`
datetime range can never pass the time filter regardless of distance -- querying the tree for it is pure
waste. Fix: sort all Argo obs by datetime once, then for each file, `np.searchsorted` the sorted array for
the small window `[file.min - max_time_delta, file.max + max_time_delta]` and only query/update that subset.
This drops each file's relevant-candidate count to roughly constant regardless of total span (rather than
scaling with the full Argo table), turning the match step from `O(span^2)` into `O(span)`.

One bug on the way to the fix: `file_datetime.min() - max_time_delta` (a `datetime64[s]` minus a
`pd.Timedelta`) silently produces a pandas `Timestamp`, not a numpy `datetime64` -- `np.searchsorted` can't
compare that scalar type against the sorted `datetime64[ns]` Argo array (`TypeError: '<' not supported between
instances of 'int' and 'Timestamp'`). Fixed by wrapping the window bounds in `np.datetime64(...)` explicitly
rather than relying on numpy to coerce a pandas type.

Verified correct, not just faster: re-ran both the 12-week and full-year builds after the fix and got exactly
the same match counts (3550 and 14705) with byte-identical distance/time-delta summary statistics as the
pre-fix runs. Real-world speedup: the full-year build dropped from ~2h47m to **3m51s** -- about 43x, and now
dominated by file-loading I/O rather than the matching step, consistent with the fix turning quadratic cost
into linear.

**The season-held-out result** (`train_rich_features_poc.py --split chronological`, now the default once a
table spans multiple seasons): train on 2022-06 through 2023-01 (summer through early winter), test on 2023-04
onward (spring -- a season absent from training), n=2441 test rows:

| method | rmse | bias | corr |
|---|---|---|---|
| raw | 1.536 | +0.272 | 0.524 |
| constant bias | 1.512 | -0.023 | 0.524 |
| linear regression | 1.405 | +0.086 | 0.554 |
| FFANN, baseline features | 1.345 | +0.067 | 0.603 |
| FFANN, rich features | **1.299** | +0.022 | **0.637** |

Rich features beat the baseline-feature FFANN by ~3.4% RMSE and a clearer correlation gap (0.637 vs. 0.603) --
bigger than the 12-week random-split gain (1.6%) and, more importantly, this is the first result where the
gain is measured on a season the model never saw in training. This is a real (if still not final -- one
year, one train/test boundary, no float-ID leakage check unlike the operational baseline's split) answer to
21's research-upper-bound question: the rich JPL CAP fields carry real, generalizing signal beyond
lat/lon/season/basin, supporting the case for the one-time additive change to `Smap2Ioda.h` discussed in 21.1.

## 26. Operational deployment question: how much history, and does it go stale?

Raised directly: for a real-time DA bias-correction model, is "use all available data up to today" actually
the right training strategy, or does secular drift (instrument recalibration, algorithm-version changes,
ENSO-scale ocean variability, the Aug-2022 safe-mode discontinuity already found in 22/25) argue for a
recency-weighted or rolling window instead? Conclusion reached by discussion, not yet implemented:

- The rigorous version of "try multiple window lengths" needs to separate two axes that are easy to
  conflate: training-window **length** (does more history help, holding recency fixed?) and **recency gap**
  to the test period (does old data actively hurt, holding length fixed?). A single train/test split (like
  25's) can't distinguish these.
- A proper test needs **walk-forward validation with multiple training origins** (train ending at several
  different dates, not just one), not a single boundary -- otherwise "the trailing-year choice works" and
  "we got lucky with this particular season boundary" are indistinguishable.
- **Recalibration-regime segmentation** (training only within a period of stable instrument calibration,
  detected empirically from `tb_h_bias_adj`/`tb_v_bias_adj` step-changes rather than relying solely on public
  announcements, which don't always exist -- see below) is a complementary idea to a time-window, not a
  replacement: it addresses instrument-level drift specifically, not ocean-state/climate-scale drift (e.g.
  ENSO phase), which a calendar window still helps with even within one calibration regime.
- This kind of test needs more than one year: multiple training origins each need their own trailing history
  plus a subsequent test period, and testing across genuinely different ocean-climate states (not just
  calendar seasons) needs data spanning more than one ENSO phase -- 2022-2023 was a La Nina-to-El Nino
  transition, so a single year sits mostly within one phase.

This motivated pulling a second year (below). The actual window-length/recency/regime-detection experiments
are not yet built -- this section records the reasoning, not results.

### 26.1 Second year pulled (2023-06-01 to 2024-06-01): two more real gaps, one undocumented

Total archive is now 2 years (2022-06-01 to 2024-06-01), 18,353 files, 192GB, 31,601 Argo matches (rebuilt
with the now-linear-time matcher from 25.1 in 8 minutes -- roughly 2x the 1-year build's time for 2x the
data, confirming the fix scales linearly rather than quadratically, not just working on the original case).

Two more real, confirmed gaps found (in addition to the Aug-Oct 2022 safe-mode gap from 22/25):

- **Dec 9-27, 2023** (~19 days): confirmed via direct file-date inspection (only 308 files found for the
  whole month vs. the normal ~830-890). Unlike the 2022 safe-mode event, **no public documentation found** --
  checked PO.DAAC's own announcements page directly for Nov-Dec 2023 and found nothing about an outage
  (only an unrelated Dec 21 dataset release). Logged as "gap confirmed, cause unknown," not force-fit to a
  cause that isn't there.
- **April 2024**: smaller, scattered gaps (Apr 4-5, Apr 16-20, Apr 23) rather than one clean event -- not
  individually investigated (diminishing returns on searching for an explanation of every few-day gap), but
  worth remembering that gaps in this archive aren't a single one-off event, they recur at a few-gaps-per-year
  rate.

**Operational issues hit during this pull, distinct from the data-content findings above:**

- The stall-detecting wrapper script (`download_smap_cap_year.sh`'s design, reused for year 2) has a real bug:
  its own polling loop can take far longer than its configured `STALL_SECS` to actually detect a stall (one
  case took ~30 minutes against a configured 180s threshold) -- the detection logic is correct in principle
  but something makes it far slower in practice than intended. Root cause not fully diagnosed; worked around
  by abandoning the wrapper for a manually-supervised one-chunk-at-a-time approach (start a chunk, wait on it
  directly, check its own completion marker, move to the next) for the remainder of the pull.
- **`podaac-data-downloader`'s skip-already-downloaded behavior is not reliable for a very wide date range in
  a single invocation.** Re-running the full 2022-06-01 to 2024-06-01 range to backfill 11 known-corrupt files
  started re-downloading files that already existed and were valid (confirmed: fresh timestamps on files
  whose content hadn't changed, for dates far from the 11 actually-missing ones) instead of skipping them --
  the same skip logic that worked correctly for the smaller 1-year backfill in 22. Worked around by using
  narrow, per-incident date ranges (a day or two around each known-missing file) instead of one wide range --
  even there, skip-detection was imperfect (a 2-day window re-fetched 58 files to recover 3 actually-missing
  ones), but the blast radius stayed small. Lesson: never re-run this tool over a wide range expecting it to
  cheaply no-op on existing files -- scope any backfill request as tightly as possible to the specific
  missing dates.
- One additional silent corruption (1 file) was introduced by killing the runaway wide-range backfill
  mid-transfer -- caught by the same full-archive validation pass used throughout this project, then fixed
  with one more narrow-range re-fetch. Final state: 18,353 files, 0 bad, confirmed by validating every file.

### 26.2 First window-length/recency result: more history helped at every origin tested, no sign of staleness yet

`src/test_training_window_sweep.py` runs the first real version of 26's proposed test: for each of 3 origin
dates (2023-09-01, 2023-12-01, 2024-03-01), trains the operational baseline-feature FFANN on several trailing
window lengths (3mo, 6mo, 12mo, all-available-history) all ending at that origin, and evaluates every one of
them on the SAME fixed 2-month test period immediately after -- holding the test period fixed isolates window
length/recency from test-period difficulty, which a single train/test split can't do.

RMSE by window length x origin (from the run committed alongside this entry):

| window | 2023-09-01 origin | 2023-12-01 origin | 2024-03-01 origin |
|---|---|---|---|
| 3mo | 1.533 | 1.263 | 1.684 |
| 6mo | 1.449 | 1.095 | 1.274 |
| 12mo | 1.408 | 1.059 | 1.171 |
| all-history | **1.382** | **1.044** | 1.173 |

At all 3 origins, RMSE improves monotonically as the window grows -- going all the way back to the start of
the archive (up to 21 months at the last origin) never hurt, and mostly helped substantially over shorter
windows. The only near-tie is 12mo vs. all-history at the last origin (1.171 vs. 1.173), suggesting diminishing
returns past ~12 months rather than active harm from older data. Bias is even more telling: the 3-month window
at the last origin has bias -0.96 PSU vs. -0.11 for the 12-month window at the same origin -- too little
training data isn't just noisier, it's poorly calibrated.

This is a real, if preliminary, answer to 26's question: across a 2-year span that already contains the
Aug-2022 safe-mode gap, the Dec-2023 gap, and a La Nina-to-El Nino transition, there is **no evidence yet**
that a rolling/recency-weighted window beats using all available history -- "use everything downloaded so
far" looks like the right default, not a risk, at least at this scale.

Caveats before trusting this for the operational decision: only 3 origins (not enough to rule out this being
specific to these particular test windows), only the baseline feature set (not yet repeated with the rich
JPL CAP fields from 21-25), and the longest window tested (21 months) still doesn't reach a second full ENSO
phase transition or a detected recalibration-regime boundary (`tb_h_bias_adj`/`tb_v_bias_adj` step-change
detection from 26 is still not done). Natural next step once the archive extends further (26.1's overnight
pull, in progress): add more origins spanning further into 2024-2026 and re-run.

### 26.3 Recalibration-regime check: no step-change found over the 2-year archive

`src/plot_tb_calibration_timeseries.py` runs the empirical check proposed in 26: weekly mean +/- std of
`sat_tb_h_bias_adj`/`sat_tb_v_bias_adj` (the JPL CAP algorithm's own applied TB bias-adjustment terms) across
the full Argo-matchup table (31,601 obs, continuous weekly coverage), flagging any week whose mean jumps by
more than 3 pooled-std from the previous week as a step-change candidate.

**Result: zero candidates flagged, in either polarization, across the full 2022-06 to 2024-05 span.** The
time series instead shows a smooth, roughly annual oscillation (troughs around Sep-Oct, peaks around
May-Jun, in both 2022-2023 and 2023-2024) -- consistent with the bias-adjustment term compensating for
something seasonal (SST- or wind-correlated, most likely), not a discrete recalibration event. No visible
discontinuity coincides with either the Aug-Oct 2022 safe-mode gap or the Dec 2023 gap (25/26.1) -- the wider
error bands right after each gap just reflect fewer/sparser matches in those weeks, not a level shift.

Caveats: this is a coarse check (weekly aggregation, a 3-sigma heuristic rather than a rigorous changepoint
test, and the Argo-matched subsample rather than the full raw archive) -- a subtle or short-lived
recalibration could still be masked by this level of aggregation. But at this resolution, over 2 years,
there's no evidence of the instrument-level drift that would argue against 26.2's "use all available
history" conclusion.

## 26.4 Archive extended to 2025-04-30 (35 months); 26.2's conclusion needs revising -- not universal

Extended the raw archive from 2 years to 2022-06-01 through 2025-04-30 (35 months). Two more corruption
rounds along the way (38 files this time, mostly clustered June 9-16 2024, scattered elsewhere in
2024-2025), same delete-and-narrow-range-backfill fix as 22/25. One new operational lesson: re-running
`podaac-data-downloader` over the *full* 2022-06-01 to 2024-06-01 range to backfill 11 known-missing files
started **re-downloading files that already existed and were valid** -- confirms 25.1's warning that skip-
detection is unreliable over wide ranges wasn't a one-off; scoped backfills to narrow per-incident date
ranges instead, as that section recommends. Rebuilt the matchup table (54,580 matches, up from 31,601) in
13 minutes -- consistent with the linear-time fix from 25.1 (roughly 1.46x the data took roughly 1.7x the
time of the 2-year build).

**Re-ran `test_training_window_sweep.py` with 2 more origins (2024-09-01, 2025-02-01) and a 24-month window
option**, addressing 26.2's caveat that only 3 origins (all within a few months of each other) was too thin
a basis for "use all available history." Result: **4 of 5 origins still show monotonic or near-monotonic
improvement with more history, but the newest origin does not**:

| window | 2023-09-01 | 2023-12-01 | 2024-03-01 | 2024-09-01 | 2025-02-01 |
|---|---|---|---|---|---|
| 3mo | 1.603 | 1.760 | 2.284 | 1.174 | 1.455 |
| 6mo | 1.442 | 1.095 | 1.312 | 1.154 | **1.161** |
| 12mo | 1.407 | 1.054 | 1.183 | 1.057 | 1.228 |
| 24mo | 1.383 | 1.037 | 1.186 | 1.056 | 1.206 |
| all-history | 1.408 | 1.056 | 1.172 | **1.048** | 1.216 |

At the 2025-02-01 origin (test = Feb-Mar 2025), the 6-month window (RMSE 1.161) beats both the 12-month
(1.228) and all-history (1.216) windows -- the first non-monotonic result in the whole sweep, and exactly
the kind of signal the sweep was designed to catch. **26.2's "no evidence yet that a rolling window beats
using all available history" conclusion needs to be revised to: mostly true, but not universal** -- one
origin out of five (the most recent one, nearest the edge of the currently-downloaded archive) shows a
medium window winning outright.

Not yet resolved: whether this is a genuine recency effect specific to conditions actually changing near
the end of the archive (worth checking against the recalibration-timeseries plot from 26.3 and against
ENSO state for that period), or an idiosyncrasy of this one origin/test window -- one anomalous origin out
of five is not enough to conclude either way. More origins, especially as the archive extends further past
April 2025 (26.1's overnight pull is still in progress toward 2026), would help settle whether this is a
recurring pattern near "the present" specifically (which would matter a great deal for the operational
choice) or a one-off.

## 27. Ascending/descending added as a feature; wider match-window tables built and modeled

Added `sat_ascending` (parsed from the raw filename's `_A_`/`_D_` flag, not a netCDF field -- see 21.1) to
`RICH_EXTRA_FEATURES`. Permutation importance ranks it 29th of 40 (Delta-RMSE +0.0014, negligible) -- its
likely signal (AM/PM diurnal SST/wind cycle) is apparently already captured more directly by the continuous
`sat_anc_sst`/`sat_anc_spd`/`sat_smap_spd` values themselves.

Built two additional matchup tables at wider time windows -- `smap_cap_argo_matchups_12h.parquet` and
`_24h.parquet` -- motivated by the earlier Vernieres et al. (2014) discussion (his Aquarius-based correction
used 3 degrees/24h, much looser than this project's 50km/3h, itself resolution-matched to SMAP's finer ~40km
footprint). Same 50km spatial radius, same 2022-06-01 to 2025-05-01 span, only `--max-time-delta-hours`
changed. Match counts: 54,787 (3h) / 173,711 (12h) / 243,041 (24h) -- a wider window recovers substantially
more matches, and mean match distance actually *drops* slightly as the window widens (13.4km to 10.7km),
since more candidate passes give the nearest-neighbor search more options to find a closer one.

`train_rich_features_poc.py` now takes `--window {3h,12h,24h}`. Trained the same baseline/rich FFANN pair on
all three (same chronological split convention, same feature sets):

| window | n_test | raw RMSE | baseline-FFANN | rich-FFANN | rich improvement |
|---|---|---|---|---|---|
| 3h | 28,094 | 1.477 | 1.243 | 1.127 | 9.3% |
| 12h | 87,160 | 1.475 | 1.284 | 1.115 | 13.2% |
| 24h | 121,925 | 1.507 | 1.315 | **1.123** | 14.6% |

Notable pattern: rich-FFANN RMSE stays essentially flat (1.115-1.127) across all three windows despite ~4.3x
more data at 24h, but baseline-FFANN RMSE steadily *worsens* as the window widens (1.243 -> 1.284 -> 1.315).
The relative rich-feature advantage grows from 9% at 3h to 15% at 24h. Plausible explanation: a wider time
window pairs satellite obs with less-contemporaneous Argo profiles -- pure temporal-mismatch noise the
baseline model (lat/lon/season/basin/sat_sss only) has no way to compensate for, whereas the rich model's
extra fields (actual per-pass SST/wind, not a stale proxy) and/or the added training volume appear to offset
that noise. Not yet isolated which of those two explanations (better features vs. more data) is doing the
work -- would need a controlled test (same n_train across windows) to separate them.

Also built `plot_geographic_errors_window.py` (generalized from a 12h-specific script) for raw bias/RMSE maps
at either window and any bin size. At 5deg bins, 12h and 24h look nearly identical to the original 3h map
(48-49% cell coverage either way) -- 5deg bins already had enough matches at 3h that widening the window
mostly doesn't change the coarse picture. At 2deg bins the gap widens a bit (12h: 38.9%, 24h: 40.7% cell
coverage) and finer structure becomes visible (streaky patterns suggestive of western boundary currents),
though 1deg bins (tried on 12h only so far, 18.7% coverage) get sparse enough that isolated extreme cells
should be treated with more suspicion than the coherent streaky features.

## 28. A fourth window (+/-3.5 days) and the Schanze et al. (2020) matchup-definition paper

Built a fourth matchup table at +/-84h (3.5 days), same 50km/2022-06-01 to 2025-05-01 span: 269,188 matches
(vs. 243,041 at 24h -- only ~11% more for 3.5x the window, diminishing returns setting in). Mean match
distance keeps dropping with wider windows (13.4km at 3h -> 7.3km at 84h), same reason as before: more
candidate passes per Argo obs gives the nearest-neighbor search more options to find a spatially closer one.

Trained the baseline/rich FFANN pair on all four windows now:

| window | n_test | raw RMSE | baseline-FFANN | rich-FFANN | rich improvement |
|---|---|---|---|---|---|
| 3h | 28,094 | 1.477 | 1.243 | 1.127 | 9.3% |
| 12h | 87,160 | 1.475 | 1.284 | 1.115 | 13.2% |
| 24h | 121,925 | 1.507 | 1.315 | 1.123 | 14.6% |
| 84h | 138,156 | 1.543 | 1.339 | **1.159** | 13.4% |

At 84h, for the first time, *both* models degrade relative to the previous window (27's pattern was
baseline-only degradation, rich staying flat). This lines up with a specific finding in Schanze, Le Vine,
Dinnat & Kao (2020, "Comparing Satellite Salinity Retrievals with In Situ Measurements: A Recommendation for
Aquarius and SMAP," the paper the 12h/24h/84h windows were prompted by -- their SSDT analysis, Fig. 5, shows
RMSD getting *worse* as the window widens for nearest-single-sample-in-time matching, because the temporally
closest match in a wide window is no longer guaranteed to be spatially close. Our matching (all four windows)
is still nearest-neighbor, not their recommended box-average -- so 27/28's results are likely the same
effect: widening the window without averaging just admits noisier matches past some point, rather than
usefully expanding the sample.

**The paper's actual recommendation is different from what 26-28 built**: average *every* satellite sample
within the 50km/+/-3.5day box per Argo report (their Appendix B.1), not take the single nearest one at a
wider window. Asked directly whether this validation-designed method is sound to use for *training* the
deployed per-observation correction model: no, for two reasons -- (1) the box is centered on the in situ
report, so roughly half the averaged samples come from after it, which doesn't exist yet at the moment a
real-time DA correction needs to run on a single incoming observation; (2) even ignoring the future-data
issue, training on an averaged (lower-noise) input while deploying on raw single-pixel retrievals is a
train/deploy distribution mismatch -- the same class of concern as the float-ID leakage work in 17/18, just
at the input-noise level rather than the sample-identity level. It remains legitimate as a validation/
diagnostic tool (its designed purpose: a cleaner estimate of true systematic bias, by construction lower-
noise than any single-window matchup table) and a causal (backward-only) version of the averaging could be
made real-time-deployable in principle, though the input-distribution-mismatch concern would still need a
separate answer. Not yet implemented -- box-averaging would require reworking `match_to_argo`'s per-file
accumulation from "track single best match" (`best_dist`/`best_file_idx`/`best_local_idx`) to a running
sum+count per Argo obs across all files, reusing the existing time-window pruning infrastructure.

Also added `plot_raw_smap_snapshot.py`: a geographic snapshot of the raw retrieved SSS field itself (no Argo
comparison), either gridded/averaged or as a full-resolution rasterized scatter of every QC-pass pixel. The
scatter view (2.55M points, one week) shows real mesoscale structure (filaments, eddies) and coastal RFI/
land-contamination artifacts near Japan/Southeast Asia and off South America that 5deg gridding smooths away
entirely.

## 29. Implemented and ran the Schanze et al. box-average validation -- found and fixed a real bug along the way

Implemented `box_average_match_to_argo` (28's proposed fix): for each Argo report, average every raw
satellite sample within 50km/+/-3.5 days rather than picking the single nearest one, matching the paper's
literal Appendix B.1 recipe. Reuses `match_to_argo`'s per-file time-window pruning, swapping the
single-nearest-neighbor `tree.query` for a `tree.query_radius` and accumulating a running sum/count per Argo
obs instead of tracking one best match. Wired in as `--box-average` on `build_raw_smap_matchups.py`.

**First run** (12-week window, full 34275-file archive as candidates): 3,567 matches, RMSD 1.52 PSU, only
~11.2 samples averaged per matchup on average -- surprisingly low given SMAP's near-daily global revisit and
a 7-day-wide window. Investigated directly: manually verified a specific "unmatched" Argo profile actually
*does* have a valid QC-pass sample 1.73km away within the window (confirmed via three separate raw files).
Traced this through several rounds of debugging (two of which turned out to be bugs in the *verification*
script itself -- a `round(x,4) == unrounded_x` comparison that can never be true, and initially suspecting a
timezone/dtype mismatch that wasn't the actual cause) before isolating the real cause: `load_raw_smap_dir`
loads every file in the archive unconditionally regardless of the requested Argo date range (already flagged
as a performance problem in 26.4/28's introduction), and at the archive's current size (35k+ files spanning
years), processing that many files was silently causing many true matches to be missed -- confirmed by
re-running with only the ~89 files actually relevant to one specific profile, which correctly found it, while
the same profile was absent from the full-34275-file production output.

**Fix**: added `start_date`/`end_date`/`max_time_delta` parameters to `load_raw_smap_dir` so it only loads
files whose filename-embedded date falls within the requested Argo window (padded by `max_time_delta` on each
side) -- the same fix already applied to `plot_raw_smap_snapshot.py`'s file globbing. This is a real
correctness fix, not just a performance one: full-archive loading was silently dropping the large majority of
true matches, not just running slowly.

**Re-run with the fix** (same 12-week window, now loading only 1,882 relevant files instead of 35,511):
**18,213 matches** (5.1x more than the buggy run), **77% of all Argo profiles matched** (vs. 15% before) --
now consistent with SMAP's actual near-daily coverage -- and **67 samples averaged per matchup on average**
(vs. 11.2 before). Total runtime: 53 seconds (vs. ~11 minutes), confirming the fix helps both correctness and
performance simultaneously.

**The RMSD result holds up under the fix**: Bias +0.36 PSU, Std 1.55, **RMSD 1.59 PSU** -- essentially
unchanged from the buggy run's 1.52 (if anything slightly higher), despite 5x more matches and 6x more
samples per box. This rules out "the 12-week sample was just too small" as an explanation for the gap with
Schanze et al.'s reported SMAP RSS box-averaged RMSD (~0.25 PSU, their Fig. 4) -- the gap is robust to a much
larger, bug-fixed sample. The most likely remaining explanation, per 28's discussion: their SMAP validation
used the RSS product, whose L2 field is itself a Backus-Gilbert-interpolated 9-point spatial average before
the user ever touches it (their own paper's Section 1.3), while JPL CAP's `smap_sss` (used throughout this
project) is a raw, unsmoothed per-pixel swath retrieval -- comparing "box-average of already-smoothed RSS
pixels" to "box-average of raw CAP pixels" isn't apples-to-apples, and CAP likely has a genuinely higher
per-sample noise floor to begin with.

Not yet done: running this on the full multi-year archive (not just 12 weeks) for a fully representative
number, and a geographic breakdown to check whether the RMSD gap is uniform or concentrated in specific
regions (the coarse 12-week geographic map in this section's initial pass didn't show an obvious pattern, but
had only 10.8% cell coverage at 5deg -- worth revisiting now that the match count is 5x larger; re-run at
37.2% coverage shows one contiguous (not isolated-spike) high-RMSD streak along the equator near 80-100E,
plausibly the Bay of Bengal/equatorial Indian Ocean freshening plumes documented in Tang et al. (20) --
genuine fast-moving salinity fronts are exactly what a 3.5-day averaging window would smear rather than
denoise).

**Checked whether this bug affects any previously-built matchup table: it does not.** Re-ran the original
foundational 12-week/+/-3h nearest-neighbor test (`match_to_argo`, the one validated against the IODA pipeline
very early in this project) with the same date-filtering fix applied, and got exactly 3,550 matches with
byte-identical distance/time-delta summary statistics to the pre-fix result. Every matchup table this project
has trained models on -- all four match-window tables (3h/12h/24h/84h), the 2-year and 35-month builds, the
ascending-feature rebuild -- used `match_to_argo`, not `box_average_match_to_argo`, so none of them were
affected. The likely reason the bug is specific to the new function: `match_to_argo` uses `tree.query(k=1)`,
a single fully-vectorized call per file with no per-point Python loop, while `box_average_match_to_argo` uses
`tree.query_radius()`, which returns a ragged/variable-length array that must be processed with a per-point
Python loop repeated across tens of thousands of files -- something about that combination at the archive's
current scale silently dropped matches in a way the simpler vectorized path didn't.

With the fix confirmed safe, the box-average validation was re-run over the full 35-month validated archive
(2022-06-01 to 2025-05-01), the same span used for the project's main 2-year+ training tables:

```
Filtered 35938 total archive files down to 26863 within [20220528, 20250504]
310329 unique near-surface profiles
Matching...
  270368 matches
Bias: 0.3106 PSU   Std: 1.5463 PSU   RMSD: 1.5772 PSU
Samples averaged per matchup: mean 62.5, median 64, min 1, max 500
```

87% of all Argo profiles found a box match (270,368 / 310,329), consistent with SMAP's near-global daily
coverage. Compared against the earlier 12-week test:

| | 12-week (n=18,213) | 35-month (n=270,368) |
|---|---|---|
| Bias | +0.36 | +0.31 |
| Std | 1.55 | 1.55 |
| RMSD | 1.59 | 1.58 |
| Mean samples/matchup | 67.3 | 62.5 |

The statistics are essentially unchanged across a ~15x larger, fully independent multi-year sample -- strong
evidence that the ~1.5-1.6 PSU RMSD is a stable, genuine property of this validation method applied to JPL CAP
data, not a small-sample artifact. It remains far above Schanze et al.'s reported ~0.25 PSU for SMAP, which is
consistent with their validation using the RSS product (already 9-point Backus-Gilbert pre-averaged) rather
than JPL CAP's raw per-pixel retrievals -- box-averaging raw single-footprint noise over 50km/3.5 days doesn't
recover the same noise reduction as averaging an already-smoothed product.

### 30. Argo QC gap: stuck-sensor profiles inflating the box-average RMSD

Geographic RMSD maps of the 35-month box-average table (5deg and 2deg bins, `plot_boxavg_geographic_rmsd.py`)
show a handful of 2deg cells with RMSD >10 PSU, far above the ~0.85-1.6 PSU typical elsewhere. Per-match
inspection of the three worst cells (20N/156W near Hawaii, 8N/62E in the Arabian Sea, 16S/10E in the South
Atlantic) found the same signature in each: a subset of matches with `argo_salinity` in the 20-27 PSU range --
physically implausible bulk salinity for any of these open-ocean regions (no river mouth, no ice melt) --
paired against a completely normal, mutually-consistent satellite reading of 34.5-37 PSU across dozens of
independent samples. The satellite side is fine; a handful of malfunctioning/stuck Argo sensors are not. The
repeating ~10-day date spacing within a cluster (e.g. the South Atlantic cell's matches land on 6/5, 6/15,
6/25, 7/5, 7/15...) is the signature of a single drifting float reporting a bad value on every cycle, not
independent random failures.

This isn't a new problem -- `build_matchups.py`'s `load_argo_near_surface` docstring already documented
(section on PreQC) that Argo's own QC flag is useless here (verified uniformly 0/pass even for a stuck sensor
reading 0.06 PSU), which is why a physically-motivated `min_salinity` valid-range filter was used instead.
The gap was that `min_salinity` defaulted to 20.0, shared between the satellite and Argo sides of the loader --
loose enough that these 20-27 PSU stuck-sensor profiles still passed. `match_to_argo`'s nearest-neighbor
tables have a second line of defense (`max_abs_diff`, default 10 PSU, rejects the matched pair outright if
|satellite - Argo| exceeds it) that `box_average_match_to_argo` never had, which is why this leaked through so
visibly in the box-average table specifically.

Quantified impact on the full 35-month box-average table: only 9,221 of 270,368 matches (3.4%) have
`argo_salinity` < 30, but removing them alone roughly halves the whole-dataset error:

| | All matches (n=270,368) | argo_salinity >= 30 (n=261,147) |
|---|---|---|
| Bias | +0.31 | +0.11 |
| Std | 1.55 | 0.85 |
| RMSD | 1.58 | 0.85 |

0.85 PSU is much closer to Schanze et al.'s ~0.25 PSU than the original 1.58 -- most (not all) of the earlier
"unaveraged JPL CAP vs. pre-averaged RSS" explanation in section 29 was actually this QC gap. The residual
~0.85 vs. ~0.25 gap is likely where that pre-averaging explanation still applies.

Fix: added a separate `--argo-min-salinity` CLI arg (default 30.0) to both `build_matchups.py` and
`build_raw_smap_matchups.py`, decoupled from the satellite-side `--min-salinity` (still 20.0, left alone since
real satellite obs can legitimately read fresher near major river plumes at low latitude, unlike a bulk Argo
profile average at these mid-ocean locations). `load_argo_near_surface`/`load_argo_for_window` now receive the
stricter Argo-specific floor. This affects every matchup table's Argo loading, not just the box-average table.
Rebuilt the 35-month box-average table (261,357 matches, Bias +0.10, Std 0.84, RMSD 0.84 -- down from 1.58)
and the 24h nearest-neighbor table (236,245 matches, down from 243,041) with the fix applied.

### 31. Testing whether raw GDAC Argo data (real PSAL_QC) offers a real improvement over the range-filter heuristic

Motivated directly by 30: our fix works by picking a numeric floor, which is inherently blunt -- it can only
catch bad profiles by how extreme their value looks, not by whether the value is actually correct. The real
Argo GDAC archive carries actual per-obs QC (`PSAL_QC`, delayed-mode-reviewed) that doesn't have this
limitation. First segment-scale test of whether switching to it is worth doing, following up 14/17's
investigation and reusing `enrich_argo_metadata.py`'s GDAC index matching and `test_sensor_aging_hypothesis.py`'s
direct-HTTPS profile fetch (argopy's own `profile()` fetcher errors on this dataset's filename pattern).

`src/test_gdac_qc_recovery.py`: loads one month (June 2022) of obsForge near-surface Argo obs with NO
valid-range filter applied (min_salinity=0, max_salinity=45 -- obsForge's own crude bound) so the filter's
passes and misses are both visible, matches each to the GDAC index by (lat, lon, datetime) (89.5% matched
within 1km/10min), and fetches each matched profile's real `PSAL_QC`/`PSAL_ADJUSTED_QC` and delayed-mode
salinity directly from the archive. (One re-run needed: the argopy index cache directory from an earlier
session's `enrich_argo_metadata.py` run had gone stale/incompatible -- FileNotFoundError loading from cache --
worked around by pointing to a fresh cache directory rather than debugging the old one.)

**Result: real QC agrees strongly with the new 30 PSU floor, and reveals a substantially bigger problem the
floor can't see at all.**

| obsForge salinity range | n | worst near-surface PSAL_QC: good (1) | bad (3/4) |
|---|---|---|---|
| <20 (old filter rejected) | 267 | 47 (18%) | 172 (64%) |
| 20-30 (this session's just-fixed gap) | 258 | 3 (1%) | 250 (97%) |
| >=30 (passes both filters, old and new) | 8,056 | 6,165 (77%) | 1,383 (17%) |

The 20-30 PSU fix is strongly validated -- 97% of what it now excludes is confirmed bad by real QC, essentially
nothing legitimate is being thrown out. But 1,383 profiles (17%) *inside* the range both filters accept are
still QC-bad -- over 5x the volume of the problem just fixed. These are corrupted-but-plausible values (a
slowly drifting sensor reading 32 instead of 35, say) that no numeric floor/ceiling can ever catch, because
there's no threshold to draw. Every matchup table in this project, even post-fix, still contains this
population uncorrected.

Side finding: of the <20 PSU profiles the range filter already rejects, 47 (18%) are actually real, QC-good
low-salinity water (river plume/shelf/Arctic regions) -- a single global floor also discards some legitimate
data, consistent with GDASApp's own region-specific bounds noted in 17.4 (Northwestern European shelves down
to 0, Arctic down to 2, etc., rather than one global number).

Overall |obsForge - delayed-mode-preferred| discrepancy: median ~0.0001 PSU (most profiles already agree
closely), but heavy-tailed (std 0.23, max 12.0), consistent with the QC-bad population above driving the tail.
DATA_MODE for this 2022 window is already mostly delayed-mode (7,702 D / 481 R / 398 A of 8,581), so most of
these profiles' best-available QC is already final, not provisional.

**Conclusion: yes, switching (at least the QC layer) to raw GDAC data is worth it, not marginal** -- it would
catch roughly 5x more bad data than range-filtering alone, for the same already-available profiles, and would
also stop discarding legitimate shelf/Arctic/river-plume data. Not yet implemented: this was a diagnostic
segment test (one month, matched via lat/lon/datetime proximity, not yet wired into the matchup-building
pipeline itself). Next step, if pursued: extend `enrich_argo_metadata.py`'s WMO/file lookup + this session's
QC-fetch pattern to the full matchup date range, then filter matchup tables by real `PSAL_QC` instead of (or
in addition to) the range heuristic.

## 32. Scaled 31 to the full matchup range and retrained a model -- the biggest result this project has produced

Followed through on 31's conclusion: scaled the GDAC QC fetch from one month to the project's full matchup
range, wired it into the matchup-building pipeline as a real option (not just a diagnostic), and retrained.

**Infrastructure**: `src/fetch_gdac_argo_qc.py` generalizes 31's one-off script into a chunked, resumable
fetcher (same pattern as `fetch_raw_argo.py`) -- one month at a time, skipping months already fetched, so an
interrupted run costs nothing. Fetched 2022-06-01 to 2025-05-01 (the project's main 35-month span, 264,090
profiles, ~47 min) and then extended to 2025-11-30 (the actual end of the local obsForge Argo archive -- the
raw SMAP archive now extends to 2026-08-29 following that overnight download, but obsForge's Argo side stops
at 2025-11-30, so that's the true limit for now; +62,359 more profiles, ~15 min). Combined QC-good (1/2) rate:
69.8% of 264,090 (first pass); the extension added a comparable proportion.

`src/gdac_qc_filter.py` merges an Argo obs DataFrame against the fetched QC lookup by an EXACT (lat, lon,
datetime) match -- not fuzzy nearest-neighbor -- since both sides come from the identical deterministic NetCDF
parse of the same source files, so coordinates are bit-identical whichever call site loaded them. Keeps only
QC 1/2 profiles and replaces the obsForge value with the delayed-mode-preferred one. Wired into
`build_raw_smap_matchups.py` as `--gdac-qc` (works for both nearest-neighbor and `--box-average` modes, since
it only changes how `argo_df` is loaded before matching starts).

**One hiccup worth recording**: the first attempt to extend the fetch to 2025-05-01 onward silently would have
skipped most of May 2025 -- the original 35-month fetch's last chunk was a 1-day sliver (`2025-05-01` to
`2025-05-02`, a `month_chunks` boundary artifact) that had already produced a real `gdac_argo_qc_202505.parquet`
file, which the resumable "skip if exists" logic then treated as a complete month. Caught before it mattered,
fixed by deleting the stale partial file and re-running.

### 32.1 Rebuilt matchup table, ±3h window, full range

`smap_cap_argo_matchups_gdacqc.parquet`: 402,047 obsForge near-surface profiles (no range filter) -> 223,351
kept after the GDAC QC filter (75,932 no GDAC match, 102,764 failed QC) -> 40,609 matches against raw SMAP CAP
(one corrupted archive file skipped gracefully: `SMAP_L2B_SSS_NRT_57323_D_20251025T011752.h5`, HDF read error,
not investigated further -- a single file out of 32,511 scanned). Raw (uncorrected) test-period RMSE: **0.897
PSU** -- consistent with 31's one-month finding, and roughly half of every prior raw-RMSE number in this
project (~1.5 PSU pre-fix), confirming most of what looked like satellite retrieval error throughout this
project's history was actually corrupted Argo ground truth.

### 32.2 A training-budget artifact, caught and fixed

First training attempt (on the pre-extension 32,735-row table) showed the rich-feature FFANN doing *worse*
than raw (RMSE 0.973 vs. raw 0.901) and far worse than the baseline-feature FFANN (0.498) -- a first for this
project; every previous test had rich features matching or beating baseline. Diagnosed rather than reported
at face value: the rich model's val_loss was still decreasing at the `train_ffann` epoch cap (300) with no
early stop triggered, while the baseline model (fewer parameters, presumably an easier loss surface) had
already converged in fewer epochs. Confirmed by direct test: rerunning with `max_epochs=1500, patience=60`
(vs. the shared default 300/20) let the rich model early-stop at epoch 1116, dropping its RMSE from 0.973 to
0.543 -- much more in line with expectations, though still trailing baseline features (0.498) at that smaller
(pre-extension) sample size.

Root cause: `train_ffann`'s `max_epochs=300, patience=20` defaults (in `train_baseline.py`, shared by every
FFANN caller in this project) were tuned against the larger range-filtered tables. The GDAC-QC-filtered table
has ~40% fewer rows, and with 41 rich features vs. baseline's 12, the rich model needs more iterations to
converge on less data -- the fixed epoch cap silently truncated it without any warning or error, just a
quietly worse number. Fixed by raising the shared defaults to `max_epochs=2000, patience=50`: early stopping
already protects every existing caller (`analyze_rich_feature_importance.py`, `float_leakage_diagnostic.py`,
`test_training_window_sweep.py`, `train_baseline.py`, `train_rich_features_poc.py`) from wasted compute once
they converge, so raising the ceiling only helps the cases that were being cut off early, and changes nothing
for the cases that already converged well inside it.

### 32.3 Final result, full extended range (2022-06-01 to 2025-11-30), fixed epoch budget

| method | RMSE | bias | corr |
|---|---|---|---|
| raw (no correction) | 0.897 | -0.011 | 0.729 |
| constant bias | 0.904 | -0.115 | 0.729 |
| linear regression | 0.642 | -0.016 | 0.799 |
| FFANN, baseline features | 0.338 | -0.040 | 0.949 |
| **FFANN, rich features** | **0.302** | -0.048 | 0.960 |

Test set n=22,292 (post-2024-02-29 chronological split). Rich features are back to beating baseline features
(0.302 vs. 0.338), confirming 32.2's diagnosis -- the earlier reversal was the epoch-budget artifact on a
smaller sample, not a real property of GDAC-QC-filtered data. **This is the best result this project has
produced by a wide margin**: every previous rich-feature FFANN test, across every match-window table built on
the range-filtered obsForge Argo, landed around RMSE 1.1-1.3 PSU. Recovering real Argo QC and delayed-mode
labels roughly quadruples the apparent skill of the same model architecture on the same satellite data --
strong, converging evidence (alongside 30's and 31's findings) that a large fraction of this project's
error budget, throughout its whole history, has been corrupted-label noise rather than genuine satellite-
retrieval-vs-bulk-salinity physical mismatch.

## 33. Backfilling to 2021-01-01, and a newly-discovered local archive gap

Decided to shift primary reliance to GDAC Argo (real QC, delayed-mode salinity) and raw SMAP CAP going
forward, and to backfill both back to 2021-01-01 -- the earliest date the local obsForge Argo archive goes
back to (raw SMAP CAP locally only went back to 2022-05-31 until now). Two backfills launched:

- Raw SMAP CAP, 2021-01-01 to 2022-06-01: same monthly-chunked, stall-detecting downloader as the earlier
  overnight backfill (DESIGN.md 25/26.4), run unattended in the background.
- GDAC Argo QC, 2021-01-01 to 2022-06-01: `fetch_gdac_argo_qc.py`, same as 31/32's fetches. Completed in
  ~25 minutes, 133,192 rows.

**Found a new, previously-unknown local data gap while sanity-checking the GDAC fetch's own output**: Jan 2022
returned only 834 rows (vs. ~10,000-11,000/month elsewhere in 2021) and Feb-Apr 2022 returned exactly 0. Traced
directly to disk rather than assumed to be a script bug: `data/common_obsForge/gdas.YYYYMMDD/00/ocean/insitu/
gdas.t00z.insitu_salt_profile_argo.nc` is simply **missing from the local archive for every cycle-day from
2022-01-05 through 2022-04-30** (confirmed by checking file presence directly) -- the cycle-day directories
themselves exist, just without this one file inside them. 2022-01-01 through 01-04 are present; 2022-05-01
onward resumes normally. This is a different, previously-undiscovered gap from the known Aug-Sep 2022 SMAP
spacecraft-safe-mode outage (25) -- nobody had looked at obsForge data this early (pre-2022-06) before this
session, since every prior matchup table in this project started at 2022-06-01. Whether this reflects a real
upstream Argo/obsForge production gap or just an incomplete local sync was not investigated further -- worth
revisiting if this ~4-month hole in the 2021-2022 backfill period turns out to matter for any test run on it.

## 34. obsForge-independent GDAC fetch: obsForge undercounts by ~50% even where it has no local gap

Raised directly: why does a *local obsForge* gap block fetching *GDAC* data for that period at all? Answer:
`fetch_gdac_argo_qc.py`'s design enriches obsForge's existing profile list with real QC -- it never queries
GDAC independently. If obsForge's local list for a period is empty, there's nothing to enrich, regardless of
what GDAC itself holds for that place and time.

Checked whether this matters even in the already-covered 2022-06-01 to 2025-11-30 window (used to train every
model in this project so far, no known local gap there): loaded the GDAC profile index directly (`argopy.ArgoIndex`,
global, date-filtered only, no obsForge cross-reference) and compared counts.

| source | profile count, 2022-06-01 to 2025-11-30 |
|---|---|
| GDAC index (independent) | 601,705 |
| obsForge-derived (what every model so far has used) | 402,047 |

**obsForge undercounts the true Argo record by ~50%, even in a period with no known local gap.** Consistent
with obsForge's Argo ingestion path being a real-time GTS/BUFR feed (17.4) rather than the GDAC's complete
archival record -- not every float's report makes it through GTS relay promptly or at all.

Built `src/fetch_gdac_argo_direct.py`: queries the GDAC index directly for a date range (global, no obsForge
involvement) and fetches every profile's real QC/delayed-mode salinity, same per-profile fetch logic as
`fetch_gdac_argo_qc.py`. `src/gdac_qc_filter.py::load_gdac_direct_argo` reshapes this into the same
(lat, lon, datetime, oceanBasin, depth, salinity) schema `match_to_argo`/`box_average_match_to_argo` expect;
`build_raw_smap_matchups.py --gdac-direct` wires it in as a third Argo source alongside `--gdac-qc` and the
plain range-heuristic default.

**Fetch scope and reprioritization**: per direction to prioritize testability, fetched the already-covered
2022-06-2025-11 range first (529,024 profiles, ~9.5h -- much slower than estimated, likely competing for
network/CPU with the concurrent SMAP 2021 backfill overnight), ahead of the earlier 2021-01-2022-06 backfill
(6 months done before reprioritizing, remainder deferred). QC-good (1/2) yield: 354,961/529,024 (67.1%),
consistent with earlier segment-test rates.

Rebuilt the main 3h matchup table using this fully independent source (`smap_cap_argo_matchups_gdacdirect.parquet`):
**63,923 matches**, up from 40,609 with the obsForge-gated `--gdac-qc` table (+57%, tracking the ~50% index
undercount).

### 34.1 Characterizing the ~57% more data: geographic redistribution, not a quality difference

Retrained on the larger table and got slightly *worse* aggregate numbers than the obsForge-gated table despite
more data (raw RMSE 0.919 vs. 0.897; rich-feature FFANN 0.318 vs. 0.302) -- investigated rather than accepted
at face value, since more data making things worse needed an explanation.

Fuzzy-matched (5km/30min, not naive exact-key matching -- an initial naive attempt wrongly suggested 99.9% of
the new set was unseen, an artifact of obsForge and GDAC-index metadata reporting slightly different position/
time for the same physical profile) the new table's Argo obs against the old table's. Result: 64.3% genuinely
overlap with what obsForge already had; 35.7% (22,845) are genuinely new profiles GDAC has that obsForge never
surfaced.

Comparing raw RMSE by latitude band, old set vs. the genuinely-new subset, shows **near-identical per-band
performance** (e.g. -60/-30: 1.13 vs. 1.12; 0/30: 0.67 vs. 0.71; 60/90: 1.50 vs. 1.45) -- the new profiles
aren't individually noisier. What differs is the **mix**: the new profiles skew away from the well-covered,
low-RMSE equatorial bands (42% of the new-only set vs. 49% of the old set) and toward higher latitudes/the
Southern Ocean (58% vs. 51%), bands that have always had intrinsically higher raw RMSE in this project. Likely
explanation: obsForge's real-time GTS feed systematically under-samples remote/high-latitude floats relative
to their true share of the network (plausibly weaker/less prompt relay coverage away from well-traveled mid-
latitude shipping lanes). The earlier "best result" number was evaluated on a population that quietly
over-weighted the easier regions; the GDAC-direct number is more representative, not worse-quality.

Checked mid/low-latitude bands specifically for whether the extra profiles improve anything there: no --
per-band RMSE for new-only profiles in the -30/0 and 0/30 bands is if anything marginally *worse* than the old
set's (0.64 vs. 0.60, 0.71 vs. 0.67). The benefit in those bands is in volume, not quality: +48-63% more
matched profiles per band, useful for training density/statistical power, not evidence the existing low/mid-
latitude data was previously under-measured in quality.

## 35. Lat/lon ocean-basin classifier, to restore a feature GDAC-direct data can't otherwise carry

`load_gdac_direct_argo` had left `oceanBasin` as NaN (GDAC has no equivalent field), silently zeroing out all
six `basin_0..5` one-hot features (`add_features` computes them as `argo_oceanBasin == code`, always False for
NaN) -- not a crash, but a quiet loss of a feature previously found to matter for some codes (permutation
importance: basin_1 ranked 6th of 41 features, basin_2 12th, basin_5 13th, basin_3 21st; basin_0 and basin_4
were negligible, near-zero ΔRMSE).

Built `src/classify_ocean_basin.py`: a lat/lon rule (Southern Ocean by latitude, then Atlantic/Indian/Pacific
by longitude), with its 3 threshold values empirically derived and validated against ~40,000 real (lat, lon,
oceanBasin) triples from the obsForge-derived table, not assumed from a textbook definition. Final rule:
southern if lat < -40; else Atlantic if -70 <= lon < 20; else Indian if 20 <= lon < 125; else Pacific.
**89.3% overall agreement** with real obsForge codes (basins 1/2/3/5 combined; basin_0's marginal/enclosed seas
are geographically scattered and not capturable by a simple rule, basin_4/Arctic is indistinguishable from
high-latitude Atlantic by threshold alone -- both fine to drop given their established negligible importance).
Widening the Atlantic's western boundary to swallow the Gulf of Mexico/Caribbean (lon >= -100) was tried and
made overall accuracy *worse* (87.8% vs. 89.3%) by pulling in genuine Pacific points -- reverted.

Wired into `load_gdac_direct_argo` for future builds, and used to patch `argo_oceanBasin` in the already-built
`smap_cap_argo_matchups_gdacdirect.parquet` in place (no need to re-run the raw-SMAP scan or Argo matching --
oceanBasin is carried-through metadata, not a matching input). Retrained:

| method | no basin (n=32,470 test) | with classifier (n=32,470 test) |
|---|---|---|
| linear regression | 0.717 | **0.670** |
| FFANN, baseline features | 0.376 | 0.373 |
| FFANN, rich features | 0.318 | 0.314 |

Linear regression recovered ~6.6% of its RMSE, exactly as expected (no nonlinear layers to route around a
missing categorical signal); the FFANNs improved only marginally, having already partially compensated via
lat/lon directly. The remaining gap vs. the obsForge-gated table's rich-feature RMSE (0.314 vs. 0.302) is now
almost entirely 34.1's population-composition effect (more representative high-latitude coverage), not a
feature-engineering or data-quality gap.

## 36. Full 2021-2025 archive: an OOM discovered and worked around by chunking the build

With both archives finally aligned -- raw SMAP CAP backfilled to 2021-01-01 (35, including a fixed March 2021
that had given up after 5 stall-retries during the original backfill), and GDAC-direct Argo fetched for the
same full range (34, run in parallel with the SMAP backfill specifically to test whether concurrent local jobs
were the stall cause -- they weren't: the SMAP downloader's last chunk completed cleanly *during* that overlap,
and stalls recurred even on a later night with nothing else running, pointing to intermittent time-of-day-
correlated server load on PO.DAAC's end rather than local contention) -- rebuilt the main matchup table over
the complete 2021-01-01 to 2025-11-30 span.

**First two attempts silently died**: real CPU usage (14:50 wall time, matching the expected cost), RSS
climbing steadily past 47GB, then the process vanished -- zero output, destination file untouched, no
traceback. Confirmed as a genuine OS-level Jetsam (memory pressure) kill, not a script bug, by finding the
actual JetsamEvent report in `/Library/Logs/DiagnosticReports/` and matching the killed PID's cpuTime against
the failed run's own reported CPU time. `load_raw_smap_dir` keeps every orbit file's full per-pixel DataFrame
in memory simultaneously for the whole requested date range (`file_dfs`) -- fine at the ~42-month scale used
throughout this session, but 59 months of raw JPL CAP swaths (2021-01 to 2025-11) pushed past what this
machine's memory (96GB, already under real pressure -- swap sitting at ~90%+ used from the session's
accumulated activity) could sustain, even before accounting for that other jobs might be running.

**Worked around by chunking, not by fixing the memory-heavy loader**: split the build into three ~20-month
sub-ranges (2021-01/2022-09, 2022-09/2024-05, 2024-05/2025-11), each run separately (peaking around 30-44GB,
comfortably under the failure threshold), then concatenated and deduplicated on
`(argo_lat, argo_lon, argo_datetime)` -- zero boundary duplicates found, confirming the ~3h match window's
boundary-adjacent edge effect (an Argo obs within 3h of a chunk cutoff could in principle miss a same-side
match) didn't materialize in practice here. Final table: **94,214 matches**, 2021-01-01 to 2025-11-29 --
+30,291 over the 2022-06-2025-11-only table (63,923), from the newly backfilled 2021-2022 period.

Not fixed: `load_raw_smap_dir`'s memory scaling with date-range length. A real fix (streaming/incremental
matching instead of holding every file in memory, or capping per-chunk memory automatically) would remove the
need for this manual chunking if the archive keeps growing -- deferred, since manual chunking is a working,
low-effort solution at the current archive size.

### 36.1 Retrained on the full range: new best-ever result

| method | 2022-06/2025-11 only (n=32,470 test) | full 2021-2025 (n=32,470 test) |
|---|---|---|
| raw | 0.919 | 0.919 (identical -- same test set) |
| linear regression | 0.670 | 0.681 |
| FFANN, baseline features | 0.373 | 0.353 |
| FFANN, rich features | 0.314 | **0.299** |

Test set is byte-identical between the two runs (n=32,470, everything after the 2024-02-29 chronological split
cutoff) since all of the newly-backfilled 2021-2022 data lands in the training period -- a clean isolation of
"more training data, same eval set." Both FFANNs improved (baseline -5.4%, rich -4.8%); linear regression
degraded marginally, plausibly noise given it has little capacity to exploit the extra data. **RMSE 0.299 is
the best result this project has produced**, finally surpassing the original obsForge-gated table's 0.302 --
on real QC, delayed-mode salinity, the complete independent GDAC record (not obsForge's ~50%-undercounted
version of it), and a validated basin feature, across the full 5-year span this project's local archives cover.

## 37. Extended to the true archive limits: 2020-09-23 to 2026-09-12

Extended both archives further, in both directions from the 2021-2025 span: forward to the present (raw SMAP
CAP to 2026-09-12, GDAC-direct Argo to 2026-09-13) and backward toward 2020. Forward and backward GDAC Argo
fetches ran fine at the established per-month rate. The SMAP backward fetch found a real product boundary:
**`SMAP_JPL_L2B_NRT_SSS_CAP_V5` has zero files before 2020-09-23** (Jan-Aug 2020 chunks correctly reported
"completed cleanly" -- nothing to download, not a failure -- and September itself is a partial month starting
the 23rd) -- an actual dataset-availability limit, not a download gap, confirmed by direct per-month file
counts. GDAC Argo data for Jan-Aug 2020 was still fetched (real Argo obs exist for that period) but has no
SMAP counterpart to match against, so it contributes nothing to the matchup table.

Rebuilding the matchup table over this ~72-month span (vs. the 59 months that already needed 3-way chunking in
36) was chunked into four ~18-month pieces from the start, same reasoning: `load_raw_smap_dir`'s memory scales
with date-range length, so a range this size would predictably repeat 36's OOM. Each chunk peaked at 25-44GB
(RSS monitored live via a polling loop throughout every chunk of both this and the prior rebuild) and completed
cleanly; concatenated and deduplicated with zero boundary matches, same as before. Final table: **112,597
matches**, 2020-09-23 to 2026-09-12.

Retrained: test set grew to n=45,387 (vs. 32,470 previously) since the chronological split's fixed 2024-02-29
val/test cutoff now leaves a longer test tail (through 2026-09 instead of 2025-11) -- no longer a pure
same-test-set comparison, but a genuinely larger and more temporally demanding one.

| method | 2021-2025 (n=32,470 test) | full 2020-2026 (n=45,387 test) |
|---|---|---|
| raw | 0.919 | 0.916 |
| linear regression | 0.681 | 0.673 |
| FFANN, baseline features | 0.353 | 0.342 |
| FFANN, rich features | 0.299 | **0.298** |

Every metric improved slightly despite the harder, 40%-larger, 2.5-year test window -- a good sign of genuine
robustness rather than a favorable test-set draw. **[Correction, see 40.3: single runs vary by about 0.005 PSU
RMSE (std across random initializations), so the 0.299 -> 0.298 change here, and similar sub-0.005 differences
elsewhere in 36/37, are within noise -- not evidence of improvement.]** **RMSE 0.298 is the new best result for this project**, on
the fullest dataset assembled to date and the true limits of what's locally available (bounded by the SMAP CAP
product's actual 2020-09-23 start and the current date on the recent end).

## 38. Why not go back to 2015 (SMAP's launch)? NRT vs. non-NRT are different products, not just different latency

SMAP itself launched in 2015, so 37's 2020-09-23 floor raised an obvious question: is that a real limit, or an
artifact of only having looked at one specific PO.DAAC collection? Checked directly against NASA's CMR
(Common Metadata Repository) rather than assuming.

**Two separate SMAP CAP L2B collections exist, with very different histories**:

| collection | actual granule coverage |
|---|---|
| `SMAP_JPL_L2B_NRT_SSS_CAP_V5` (what this project has used throughout) | a brief pilot in June-July 2016 (896 granules total), then nothing until continuous operation began 2020-09-23 |
| `SMAP_JPL_L2B_SSS_CAP_V5` (non-NRT / delayed, science-reprocessed) | continuous full-mission coverage, ~3,700-5,300 granules/year, every year from April 2015 onward including through 2025 |

The collection's own catalog metadata (`time_start`) claims 2015-04-01 for *both* -- misleading for the NRT
one, since that's inherited from the mission's overall record rather than reflecting when NRT granules actually
exist. Confirmed via direct per-year and per-month CMR granule counts (`cmr-hits` header), not just the
collection-level metadata, given the metadata's own claim didn't match what `podaac-data-downloader` actually
returned for 2020-01 through 2020-08 (zero files, correctly).

**So SMAP CAP data back to 2015 exists -- just not as NRT.** Tested whether that distinction actually matters
(same skepticism-before-acting pattern as 17's real-time-vs-delayed-mode Argo test) rather than assuming NRT
and non-NRT are the same retrieval at two latencies. Downloaded both products' files for the same 16 orbit
revolutions (2022-05-31/06-01, REVs 39157-39172 -- a day already in the local archive) and compared
`smap_sss` pixel-by-pixel. One format wrinkle: NRT splits each revolution into separate ascending/descending
files (812 along-track rows each), while non-NRT bundles both into one file (1624 rows) -- confirmed via
`row_time` continuity that the first 812 rows are the ascending half and the last 812 the descending half,
matching NRT's own split, before comparing.

**Result: NRT and non-NRT are meaningfully different retrievals, not the same algorithm at two latencies.**
Over 514,806 co-located pixels: mean diff (non-NRT minus NRT) +0.043 PSU, but std 0.75 PSU and median
|diff| **0.116 PSU** -- only 0.02% of pixels are bit-identical, 94.4% differ by more than 0.01 PSU, 54.7% by
more than 0.1 PSU, and 6.9% by more than a full PSU (max 29.2 PSU). The median discrepancy alone is roughly a
third of this project's best model RMSE (0.298, 37) -- comparable in size to the actual signal being modeled,
not a rounding-level difference.

**Decision: not backfilling with non-NRT data.** The deployed use case is real-time operational bias
correction, so the model's actual input is always NRT SMAP -- unlike Argo (17.1), where the target label can
safely use the best available (delayed-mode) data since Argo is never a live model input. Training years of
2015-2020 on non-NRT retrievals while the model only ever sees NRT operationally would reintroduce exactly the
kind of train/deploy satellite-side mismatch this project has otherwise been careful to avoid. 2020-09-23
remains the practical, correct start of the usable local archive for this project's actual purpose -- not an
artifact of an incomplete download, and not worth working around given what the comparison found.

## 39. Generating bias-corrected IODA files for assimilation (2025-12-10 to 2026-01-19)

First real deliverable aimed at actual use, not just validation: bias-corrected SMAP SSS observation files in
the same IODA format as `common_obsForge/gdas.YYYYMMDD/HH/ocean/sss/gdas.tHHz.sss_smap_l2.nc`, meant to be fed
to an assimilation run in place of the originals. Three-phase plan, each phase gated on confirming the
previous one actually worked rather than assumed.

### 39.1 Confirmed the existing IODA files can be replicated from raw SMAP files on hand

Inspected one file's schema directly: root dimension `Location`, groups `MetaData` (dateTime int64 seconds-
since-1970, latitude/longitude float32, oceanBasin int32), `ObsValue`/`ObsError`/`PreQC` each holding
`seaSurfaceSalinity`. Critically, the root attribute **`obs_source_files`** lists the exact raw JPL CAP swath
filenames obsForge ingested for that cycle -- removing any guesswork about which raw files feed which cycle.

Loading those exact files with `build_raw_smap_matchups.load_raw_smap_file` and concatenating in the
attribute's listed order reproduced latitude/longitude/ObsValue/PreQC exactly... after finding one extra
filter: obsForge drops a handful of pixels with `sss` exactly `0.0` (a degenerate retrieval near Hudson Bay in
the test cycle -- passes the fill-value check but is not a real salinity) that our own `load_raw_smap_file`
keeps. With `sss > 0` added, every field matched exactly (confirmed on the first cycle and three more spot
checks across the range, including an 18Z cycle and one near the far end of the requested range) except
`dateTime`, off by up to +/-64 seconds -- not chased further, negligible against a 6h DA cycle window.
`ObsError` also confirmed to equal the raw file's `smap_sss_uncertainty` field directly.

### 39.2 Generated the bias-corrected files

`src/train_and_save_correction_model.py`: retrained the rich-feature FFANN (same architecture/split as
`train_rich_features_poc.py`) on the full GDAC-direct matchup table and saved model weights + the feature
Standardizer's mean/std + the `RICH_FEATURES` column order to `rich_correction_model.pt`, since no checkpoint
had existed before (train_rich_features_poc.py only ever trained in-memory).

`src/generate_bias_corrected_ioda.py`: for each cycle, reproduces the exact obs population per 39.1, computes
the same rich-feature set directly from the raw per-pixel fields (`load_raw_smap_file` already extracts
everything `RICH_EXTRA_FEATURES` needs) -- using the **original file's own `oceanBasin`** for the basin_0..5
one-hot rather than `classify_ocean_basin.py`'s ~89%-accurate approximation, since the real obsForge-computed
value is sitting right there in the file being replaced. **[Correction, see 40.2: this was a mistake. The model
was trained on the classifier's basins, so feeding it obsForge's real ones was out-of-distribution for ~10.7% of
observations. Superseded by a model with no basin inputs, 40.]** Copies the original file byte-for-byte and overwrites
only `ObsValue/seaSurfaceSalinity`, and only for **QC-pass (`PreQC==0`) rows** -- the model was trained
exclusively on QC-pass satellite-Argo matches, so applying it to QC-fail rows would be out-of-distribution;
those keep their original value; the DA system's own PreQC-based downweighting handles them either way.

One implementation snag: netCDF4-python in this environment couldn't reopen these particular HDF5-backed
NETCDF4 files in write mode (`"Can't write file"`, independent of permissions or file locking) -- root-caused
by testing raw h5py against the same file, which opened and wrote it fine (a NETCDF4 file is HDF5 underneath,
same group model), so the script uses h5py for the copy-and-modify step instead of fighting the netCDF4
library's write path.

Result: **144 of 164 requested cycles generated** (2025-12-10 00Z to 2026-01-19 18Z), written to a separate
`data/bias_corrected_obsForge/` tree with the same layout, none of the originals touched. 20 cycles skipped.
**[Correction, see 40.4: this originally said 11 missing + 1 mismatch, all in Jan 2026 -- that came from a
run whose output I had truncated. The full breakdown is 18 cycles with no original IODA file, spread across
Dec 2025 and Jan 2026 (e.g. 3 on 2025-12-12, 3 on 2026-01-01, 3 on 2026-01-15), plus 2 single-row mismatches
(2025-12-22 12Z and 2026-01-17 06Z, raw = ioda + 1 in both, the near-zero-salinity edge case).]** Verified on the first cycle that every
field except `ObsValue/seaSurfaceSalinity` is byte-identical to the original, and that the changed rows are
exactly the QC-pass ones.

### 39.3 Bias comparison against real Argo

`src/compare_ioda_bias.py`: loads each generated cycle's QC-pass obs from both the original and corrected
files, matches each independently against GDAC-direct Argo (same `match_to_argo` nearest-neighbor logic as the
rest of this project, 50km/+/-3h) for the 2025-12-10 to 2026-01-19 window -- a check on the actual files
intended for assimilation. This window is after the 2024-02-29 split cutoff, so the model never trained on it,
but it uses the same matching and Argo source as the held-out test set and very likely overlaps heavily with it
(overlap measured in 41.3: 588 of the 590); it is not a separate, independent validation sample.

| | n | bias | std | RMSE |
|---|---|---|---|---|
| Original (raw) SSS | 590 | -0.080 | 0.983 | 0.985 |
| Bias-corrected SSS | 590 | -0.036 | 0.302 | **0.304** |

Same 590 matched obs both ways (only the SSS value differs). **69% RMSE reduction**, and the corrected RMSE
(0.304) lines up closely with the model's own held-out test performance (0.293-0.298, 37) -- as expected if
these matches are largely a subset of that test set. This shows the correction was applied correctly to the
generated files and holds on data the model did not train on; it is not evidence of generalization to a
separate sample.

## 40. Do the basin inputs earn their place? Feature importance, a train/inference mismatch, and a no-basin model

Started as a question about which SMAP inputs matter, and turned into finding and fixing a mistake in 39.

### 40.1 Permutation importance on the current table

Re-ran `analyze_rich_feature_importance.py` (permutation importance, 10 shuffles per feature) on the full
2020-2026 GDAC-direct table (45,387 test points); added `--matchups-path`/`--out` so it no longer silently
overwrites the older results (still in `rich_feature_importance.parquet`; new run in
`rich_feature_importance_gdacdirect.parquet`). ΔRMSE in PSU when the column is shuffled in the test set:

| rank | input | ΔRMSE | rank | input | ΔRMSE |
|---|---|---|---|---|---|
| 1 | sat_sss | 1.243 | 7 | sat_smap_sss_uncertainty | 0.137 |
| 2 | sat_anc_sss | 0.529 | 8-9 | lon_cos / lon_sin | 0.126 / 0.118 |
| 3 | sat_anc_sst | 0.477 | 10-11 | basin_5 / basin_1 | 0.088 / 0.074 |
| 4 | sat_lat | 0.276 | 12 | doy_sin | 0.053 |
| 5-6 | basin_3 / basin_2 | 0.225 / 0.209 | 13-14 | sat_inc_aft / sat_inc_fore | 0.045 / 0.027 |

Everything else is <= 0.023: wind speeds, land fractions, brightness temperatures, noise-equivalent
temperatures; the azimuth/antenna-azimuth/wind-direction fields, ice concentration and ascending flag are
~0 (<= 0.002). `basin_0` and `basin_4` are exactly 0 *by construction* (the classifier never emits those
codes), not a finding.

What is shuffled: only the one column of the standardized test matrix, replaced by a random permutation of its
own values (distribution unchanged, row pairing broken); the Argo target and the satellite salinity added back
to the model's output are not shuffled, so the `sat_sss` score measures the model's *correction* depending on
its input, not removal of the retrieval itself. Row order is irrelevant (the FFANN scores rows independently).

Caveat corrected during this discussion: I had said correlated inputs are *understated*; that holds when the
model can fall back on a redundant partner, but shuffling one of several derived features also creates
impossible combinations (e.g. a tropical latitude with a Southern Ocean basin flag) the model never saw, which
can *inflate* the score. Lat/lon/basin are mutually derived here, so their ranks are ambiguous in both
directions. `sat_lat` moving from ~0 (earlier run) to rank 4 is plausibly this, but untested. Grouped
shuffling (lat, lon, basin together) would separate real location signal from the artifact; not run.

### 40.2 What basin adds, and the mismatch it caused in 39

In the GDAC-direct table the basin flags are a deterministic function of lat/lon
(`classify_ocean_basin.py`: south of 40S, else three longitude bands), so they carry no information beyond what
the model already receives (latitude directly, longitude as sin/cos); at most they give hard step boundaries
the network would otherwise have to build. In the original obsForge data `oceanBasin` was a richer
coastline-following mask (marginal seas, Arctic), also a function of lat/lon but too intricate for a small
network to learn from raw coordinates -- the classifier matches it only 89% of the time.

That is the problem with 39: the model is trained on classifier-derived basins (`basin_0`/`basin_4` always 0),
but the generated IODA files fed it the file's real obsForge `oceanBasin`. Measured over 12 cycles (463,246
QC-pass observations): obsForge's basin differs from the classifier's on **10.7%** of observations (code 0:
0.07%, code 4: absent -- so mostly ordinary boundary disagreement, not the untrained inputs); the correction
shifts by std **0.129 PSU** between the two (mean +0.001), >0.2 PSU on 6.0% of observations, max 2.35 PSU.
39's reasoning ("more accurate, sitting right there in the file") ignored that the model never saw that
distribution.

### 40.3 Ablation: with vs. without basin inputs, 5 random initializations each

`src/compare_basin_ablation.py` (same split and test set; torch seeded per run -- full-batch training is
otherwise deterministic; per-run results in `basin_ablation_results.parquet`):

| model | inputs | test RMSE mean +/- std | range |
|---|---|---|---|
| rich, with basins | 41 | 0.2939 +/- 0.0049 | 0.290-0.302 |
| rich, no basins | 35 | 0.2965 +/- 0.0045 | 0.291-0.302 |
| baseline, with basins | 12 | 0.3455 +/- 0.0047 | 0.340-0.351 |
| baseline, no basins | 6 | 0.3732 +/- 0.0110 | 0.363-0.391 |

Rich model: the 0.0026 gap is smaller than the standard error of the difference (~0.003) and the ranges nearly
coincide -- no detectable effect (effects up to ~0.006 not excluded at 5 seeds). Baseline model: removing the
basins clearly hurts (+0.028, non-overlapping ranges) -- with only 6 inputs nothing else carries regional
information, whereas the rich model's ancillary SST/salinity and uncertainty fields stand in for it. Consistent
with 35's finding that basins helped linear regression (no nonlinearity to compensate) but barely helped the
FFANNs. Decision: drop the basin inputs from the deployed rich model.

**Noise caveat for earlier results**: seed-to-seed std is ~0.005 PSU, so differences of that size or smaller
reported elsewhere (36.1/37's 0.299 -> 0.298; likely part of 35's 0.314 vs. 0.302) are not evidence of
improvement. Conclusions in 29-34 about differences of ~0.01+ are less affected.

### 40.4 No-basin model, regenerated IODA files, and the Argo comparison

- `train_and_save_correction_model.py` gained `--no-basin` and `--seed`; the no-basin default output is
  `rich_correction_model_nobasin.pt` so the original checkpoint is never overwritten. Seed 0, fixed in advance
  (not chosen by test score): test RMSE 0.2939, 35 inputs. Re-running the ablation's seed 0 reproduced it exactly.
- Regenerated all 144 cycles with it into **`data/bias_corrected_obsForge_nobasin/`** (the earlier with-basin
  set and the originals untouched; `bias_correction_model` attribute in each file names the checkpoint). Same
  20 cycles skipped. **Correction to 39.2's count**: its "11 missing + 1 mismatch, all Jan 2026" came from a
  run whose output I had truncated. Full breakdown, from an untruncated run: **18** cycles with no original
  IODA file (Dec 2025 and Jan 2026 both) and **2** single-row mismatches (2025-12-22 12Z, 2026-01-17 06Z;
  raw = ioda + 1, the near-zero-salinity edge case).
- `compare_ioda_bias.py` gained `--out-suffix` (so a second run doesn't overwrite the first's matchup files).
  Same 590 Argo matches for all three:

| | n | bias | std | RMSE |
|---|---|---|---|---|
| Original SSS | 590 | -0.080 | 0.983 | 0.985 |
| Corrected, with basin inputs (39) | 590 | -0.036 | 0.302 | 0.304 |
| Corrected, no basin inputs | 590 | -0.042 | 0.262 | **0.266** |

Read this cautiously. A paired bootstrap (5,000 resamples) on RMSE(with) - RMSE(without) gives +0.038 with a
95% CI of [+0.002, +0.091] -- barely excludes zero. Mechanism check: 85 of the 590 matched pixels (14.4%) have
an obsForge basin differing from the classifier's; the no-basin model is better on those (0.469 -> 0.410) *and*
on the 505 where the basins agree and the with-basin model saw exactly the inputs it was trained on (0.267 ->
0.233). So most of the gap is not the basin mismatch; it is more plausibly ordinary model-to-model variation
(different initialization) on a small sample -- on the 45k held-out test set the two models are
indistinguishable (40.3). Not checked: how much other seeds vary on these same 590 points. The no-basin set is
still the one to use, because it removes a known inconsistency at no measured accuracy cost, but "RMSE 0.266"
should not be quoted as a demonstrated improvement over 0.304.

Scripts touched: `analyze_rich_feature_importance.py`, `train_and_save_correction_model.py`,
`compare_ioda_bias.py` (options added), `compare_basin_ablation.py` (new). Nothing committed.

## 41. Does more recent training data help? And an ensemble-corrected IODA set

Follows 40: with the basin inputs dropped, the remaining question for the files meant for assimilation was
whether the model should be trained on more recent data. The model behind the IODA files (A) was trained only
through 2023-12-31 (63,620 rows; 3,316 for validation), so it never saw the 32,953 matches from March 2024
to 9 December 2025 -- about half again as much data, and the closest in time to the period being corrected.

### 41.1 Recency comparison (`src/compare_recency_training.py`)

Both models use the 35-input no-basin set, 5 seeds each, scored on the same untrained rows (all
`sat_datetime` >= 2025-12-10, straight from the raw-SMAP/GDAC-Argo matchup table -- no IODA files):
- **A (old split)**: train <= 2023-12-31, validate to 2024-02-29.
- **B (recent split)**: train <= 2025-09-30 (96,495 rows), validate 2025-10-01 to 2025-12-09 (3,415 rows).

Scoring only the IODA window (2025-12-10 to 2026-01-19, 1,961 rows) gives little statistical power, and B's
training now overlaps most of the old 45k test set, so B can't be scored there. The first run was stopped and
restarted with the evaluation widened to also include everything from 2026-01-20 on ("later", 10,452 rows),
reported separately and combined. Raw satellite RMSE: window 0.961, later 0.900, all 0.910.

| held-out rows | A mean +/- std (min-max) | B mean +/- std (min-max) | A - B, seed-averaged (95% CI) |
|---|---|---|---|
| window, n=1,961 | 0.2704 +/- 0.0029 (0.265-0.273) | 0.2657 +/- 0.0033 (0.261-0.270) | +0.002 (-0.003, +0.007) |
| later, n=10,452 | 0.2758 +/- 0.0091 (0.268-0.292) | 0.2594 +/- 0.0036 (0.255-0.265) | +0.012 (+0.007, +0.016) |
| all, n=12,413 | 0.2749 +/- 0.0078 (0.269-0.289) | 0.2604 +/- 0.0035 (0.256-0.266) | +0.010 (+0.006, +0.014) |

(Positive = B better; the last column is a paired 5,000-resample row bootstrap on seed-averaged predictions.)

- **Window**: B is better or tied in all 5 seeds, but the difference is inside the uncertainty at ~2k rows --
  the window alone cannot show an effect.
- **Later rows**: B wins every seed and the interval excludes zero (about 6% lower RMSE). Model A is also less
  stable there (seed 3: 0.2916; std 0.009 vs. 0.004 for B).
- Both models score ~0.26-0.28 on these rows, lower than the ~0.29-0.30 on the old 45k test set, while raw RMSE
  is about the same (0.91 vs. 0.92). Not investigated.
- Caveat: the "later" rows run through September 2026, so B's advantage there mixes "more data" with "less
  staleness" -- the two are not separated by this design. [Resolved in 45: recency, not volume.]

**Seed-averaging is as large an effect as recency.** Averaging the five seeds' predictions lowers RMSE on the
same rows by 0.011-0.020: A window 0.2704 -> 0.2572, B window 0.2657 -> 0.2549; A later 0.2758 -> 0.2560, B later
0.2594 -> 0.2441. The IODA files so far came from a single network.

### 41.2 Ensemble-corrected IODA files

- `src/train_and_save_ensemble.py`: trains the five B members (seeds 0-4, same train set so one shared
  Standardizer) and saves them in one checkpoint, `rich_correction_ensemble_recent_nobasin.pt`. Members score
  0.256-0.266 on the 12,413 untrained rows; their mean scores **0.2458** (bias +0.003).
- `generate_bias_corrected_ioda.py` now accepts single-model or ensemble checkpoints and averages the members'
  predicted corrections; behavior for single-model checkpoints is unchanged.
- Output: **`data/bias_corrected_obsForge_ensemble/`**, 144 of 164 cycles, same 20 skipped as before (18 with no
  original file, 2 one-row mismatches). The original files, `bias_corrected_obsForge/` (with basins) and
  `bias_corrected_obsForge_nobasin/` were not touched. Verified on one cycle: every field except
  `ObsValue/seaSurfaceSalinity` is byte-identical to the original; only QC-pass rows changed (49,135 of 49,148 --
  the 13 with a missing input field stay uncorrected); the `bias_correction_model` attribute names the checkpoint.

Argo comparison, same 590 matches for every row (`compare_ioda_bias.py`, now with `--out-suffix`):

| satellite SSS | bias | std | RMSE |
|---|---|---|---|
| Original (obsForge) | -0.080 | 0.983 | 0.985 |
| Corrected, with basins, trained to 2023 (39) | -0.036 | 0.302 | 0.304 |
| Corrected, no basins, trained to 2023 (40.4) | -0.042 | 0.262 | 0.266 |
| Corrected, ensemble of 5, trained to Sep 2025 | -0.012 | 0.258 | **0.258** |

The 0.258 vs. 0.266 gap is not distinguishable at 590 points (the earlier bootstrap on 0.304 vs. 0.266 had a
95% interval of roughly +/-0.04), and 41.1 found no detectable difference between A and B on this window. The
ensemble's measurable advantage is on the larger untrained set (41.1). The window is untrained for this model, so
0.258 is a fair held-out number -- but see 41.3 for why it is not an independent one.

### 41.3 How much do the 590 overlap the test set, and why only 590?

- Of the 590 IODA-comparison matches, **588 (99.7%)** are rows of the held-out test split with the same Argo
  profile *and* the same satellite pixel; none are in train/validation; 2 are not in the matchup table at all
  (cause not checked). So the 590 are essentially a subset of the test set, not an independent sample.
- Conversely they are only about 30% of the test rows in the window (590 of 1,966 -- 1,961 by the
  `sat_datetime` criterion used in 41.1; the two counts differ because 41.1 selects on `sat_datetime` and
  this on the Argo profile time). The reason: the original IODA cycles draw on only about a third of the raw
  swath files -- 99 distinct source files referenced for 2025-12-10..19 against 273 raw files in the archive
  for those days (36%). It is not an ascending-only effect (sources listed: 58 ascending, 41 descending; ascending
  share 52% in the 590 vs. 49% in the rest). Why obsForge ingests only about a third of the files was not
  determined, and is not something this project changes.

Side note: `generate_bias_corrected_ioda.py` writes `bias_correction_summary.parquet` to a fixed path, so each run
replaces the previous run's summary (earlier runs' summaries were already gone). Scripts new or changed:
`compare_recency_training.py` (new), `train_and_save_ensemble.py` (new), `generate_bias_corrected_ioda.py`.
Nothing committed.

### 41.4 Where `sat_anc_sss` and `sat_anc_sst` come from (documentation findings)

Asked how the two ancillary inputs are generated -- they rank 2nd and 3rd in permutation importance (40.1).
Sources: the raw files' own variable/global attributes (V5 NRT, a 2025-12-10 file) and JPL's *SMAP Salinity
and Wind Speed Data User's Guide, Version 4.2* (Fore et al., Jan 2019; read via a copy hosted at
gmao.gsfc.nasa.gov). **The guide is for V4.2, not the V5 product used here**; I could not load the V5 PO.DAAC page
or locate a V5 guide (searches returned the dataset pages but not the ATBD/guide text).

- **Neither is a SMAP measurement.** JPL's pre-processing collocates outside datasets with the L1B brightness
  temperatures and with the L2B swath cells: HYCOM SSS, NCEP GDAS wind speed/direction, NOAA optimum
  interpolation SST, and NOAA WaveWatch III significant wave height (the last always fill in these files, 23).
  Each is matched "at the approximate time of the SMAP observations". The files record their ancillary source
  paths per revolution (`ANC_SSS_FILE`, `ANC_SST_FILE`, ... under an NRT-pipeline directory), i.e. they are built
  per orbit in the NRT chain.
- **`anc_sst`**: V5 file long_name "NOAA Optimum Interpolation sea surface temperature" (K); V4.2 guide: NOAA OI
  SST collocated to the swath cell. Neither names the OI version, resolution, or which analysis is used in NRT.
  In the retrieval it is an input to the brightness-temperature forward model (the geophysical model function
  takes SST, wind direction and wave height); it is not solved for.
- **`anc_sss`**: V4.2 guide: "the HYCOM ancillary SSS collocated to the particular [swath cell]". V5 file
  long_name: "Ancillary salinity used for high-winds processing, **either SMAP or HYCOM**" -- what "SMAP" means
  there, and when each is used, is **undocumented in anything I found; unresolved**.
- **How it enters (V4.2)**: the main retrieval (eq. 3.7) leaves salinity unconstrained and only puts a
  +/-1.5 m/s prior on wind speed around the NCEP value, so `anc_sss` does not enter `smap_sss`. It does enter the
  separate high-wind retrieval (eq. 3.8), which fixes salinity at `anc_sss` and fits wind speed/direction --
  `smap_high_spd`, `smap_high_dir`, `smap_high_dir_smooth` are therefore functions of it. The guide warns that errors in
  the ancillary salinity map into those wind speeds (erroneously high winds near the Amazon and other major river
  outflows). The guide also uses HYCOM as its comparison reference for `smap_sss` and for its uncertainty field.
- **Other fields from the same metadata**: `anc_spd` is NCEP 10 m wind speed scaled by 1.03; `anc_dir` the NCEP
  direction (oceanographic convention); the ice map is NCEP sea ice.

Two cautions for interpreting `sat_anc_sss` as a model input:
1. It is a model salinity field independent of the retrieval's own physics (in V4.2), which is plausibly why it is
   so informative about the retrieval's error. If V5's "SMAP" branch is a satellite-derived field, that
   independence may not hold in places -- unresolved.
2. *General knowledge, not from the documentation read here*: operational HYCOM salinity analyses assimilate in-situ
   profiles, which include Argo. If that applies to the HYCOM product used, `anc_sss` carries Argo information, and
   part of what the model learns is HYCOM's analysis rather than SMAP physics. Legitimate as an operational
   input (the field is in the NRT product) but worth knowing before reading its importance as a SMAP-error signal;
   not verified, including which HYCOM product/latency JPL uses in NRT.

Also noted while collecting the input descriptions: the direction/angle inputs (azimuths, antenna azimuths, wind
direction) enter the model as raw degrees rather than sine/cosine, so the 0/360 wrap is not handled; this may
partly explain their near-zero importance (untested).

## 42. Deriving salinity from the brightness temperatures instead of correcting the SMAP product

The question: could the model bypass `sat_sss` (and `sat_smap_sss_uncertainty`) and derive salinity from the
brightness temperatures, using the ancillary fields SMAP supplies? Tested with an input-group "ladder"
(`src/ladder_tb_retrieval.py`) and a follow-up isolating one confound (`src/climatology_input_test.py`).

### 42.1 Why a baseline is needed, and how big the real bar is

With no `sat_sss`, the target is Argo salinity itself (std 1.06 PSU) and a small network would first have to
rebuild the global salinity pattern before any brightness-temperature signal could show. So the network
predicts Argo minus a baseline. RMSE vs. Argo on the standard test split (45,387 rows), for context:

| predictor | RMSE |
|---|---|
| constant (training mean) | 1.072 |
| raw SMAP `sat_sss` | 0.916 |
| crude climatology, only on the 96% of rows whose 5x10deg-month cell was populated | 0.382 |
| hierarchical climatology, all rows (below) | 0.433 |
| **`anc_sss` (HYCOM) used directly** | **0.330** |
| existing corrected model (40.3, 5 seeds) | 0.2965 +/- 0.0045 |

The bar is not the raw product (0.92): HYCOM alone is 0.33, and the corrected model is only ~10% better than
that (see 41.4's caveat that HYCOM may assimilate Argo, which would make it not independent of the targets).

Climatology: hierarchical cell means of Argo salinity over the **train split only** (5x10deg x month, falling
back to 10x20deg x month, 10deg-lat x month, then the global mean when a cell has < 5 observations). For train
rows it is **leave-one-out** -- otherwise each row's own Argo value sits inside its baseline: train RMSE 0.408
in-sample vs. 0.464 leave-one-out (val 0.399, test 0.433).

### 42.2 The ladder, and a design flaw caught mid-run

Cumulative rungs, no basin inputs, 5 seeds each: R1 position/month (lat, lon sin/cos, doy sin/cos); R2 + SST and
NCEP wind (`anc_sst`, `anc_spd`, `anc_dir`); R3 + brightness temperatures, TB bias adjustments, NEDT, geometry,
land/ice, ascending (**no salinity field of any kind**); R4 + `anc_sss`; R5 + product byproducts (`smap_spd`,
`smap_high_spd/dir/dir_smooth` -- from the joint SSS/wind solve, or fixed at `anc_sss`); R6 + `sat_sss` and
`sat_smap_sss_uncertainty` (= all 35 inputs).

**First version was flawed**: the network predicted Argo minus climatology but was never given the climatology as
an input. Inputs carrying absolute salinity (brightness temperatures, `anc_sss`) then could not be turned into a
departure from climatology without rebuilding the climatology from lat/month. Tell-tale: R4 scored 0.4045-0.4110
while `anc_sss` used directly scores 0.3295, and even R6 (with `sat_sss`) scored 0.412 vs. 0.29 for the existing
model on the same inputs. I stopped what looked like a stalled run to fix it; it had in fact already finished, so
its full results exist (`ladder_tb_retrieval_results.parquet`, `ladder_v1_flawed.log`: R1 0.433, R2 0.428,
R3 0.426, R4 0.408, R5 0.412, R6 0.412) and are **invalid**. Fix: the climatology value is an input at every rung.

### 42.3 Ladder results (corrected; `ladder_tb_retrieval_results_v2.parquet`)

| rung | inputs (incl. climatology) | test RMSE mean +/- std (min-max) |
|---|---|---|
| R0 | climatology alone, no network | 0.4327 |
| R1 +position/month | 6 | 0.3892 +/- 0.0203 (0.3625-0.4133) |
| R2 +SST/wind | 9 | 0.3626 +/- 0.0209 (0.3417-0.3913) |
| R3 +brightness temps | 29 | 0.3338 +/- 0.0134 (0.3190-0.3489) |
| R4 +`anc_sss` (HYCOM) | 30 | **0.2763 +/- 0.0052** (0.2729-0.2854) |
| R5 +product byproducts | 34 | 0.2759 +/- 0.0017 (0.2737-0.2780) |
| R6 +`sat_sss`, uncertainty | 36 | 0.2747 +/- 0.0037 (0.2717-0.2807) |

- **Bypassing the product costs nothing**: R4 (no `sat_sss`, no uncertainty) and R6 (with both) differ by 0.0016,
  inside the seed spread. Given brightness temperatures, ancillary fields and the climatology, the product's own
  retrieval adds nothing detectable.
- **The brightness temperatures carry modest signal**: R3 improves on R2 by ~0.029, but that is only ~1.5x the
  seed spread of R1-R3 (+/-0.013 to +/-0.021), so suggestive rather than conclusive. R3 (no salinity field) only
  reaches about HYCOM-alone level (0.334 vs. 0.330).
- R1's gain over R0 (0.433 -> 0.389) is the network refining the coarse cell-mean climatology from position and
  month, not new information; the real baseline for later rungs is R1.
- `anc_sss` is the largest single step (R3 -> R4: -0.058).

### 42.4 What actually made R6 beat the existing model (`climatology_input_test.py`)

R6 beat the existing model by ~0.02, but differed in two ways at once (the climatology input, and the target
Argo - climatology instead of Argo - `sat_sss`). Isolated, same split and seeds:

| config | inputs | target | RMSE mean +/- std |
|---|---|---|---|
| C0 existing | 35 | Argo - `sat_sss` | 0.2965 +/- 0.0045 |
| C1 + climatology input | 36 | Argo - `sat_sss` | **0.2838 +/- 0.0007** |
| C2 (= ladder R6) | 36 | Argo - climatology | 0.2767 +/- 0.0063 |

Seed-averaged predictions, paired row-bootstrap (positive = second is better): C0 - C1 +0.0100 (95% CI +0.0058 to
+0.0145); C1 - C2 +0.0067 (+0.0055 to +0.0080); C0 - C2 +0.0167 (+0.0120 to +0.0216). So roughly **60% of the gain
is the climatology input and 40% the target definition**; the input also cuts seed-to-seed spread from 0.0045 to
0.0007. C2 here (0.2767) differs from the ladder's R6 (0.2747) by 0.002 -- the same configuration with inputs in a
different column order -- a feel for run-to-run noise at this level.

### 42.5 Conclusions and caveats

- Deriving salinity without the product is **feasible and not worse** (R4 ~ R6), provided `anc_sss` is available.
  Without any salinity field (R3) it is about as good as HYCOM alone and clearly worse than with it.
- The better numbers over the existing model come mainly from adding a train-years climatology to the model's
  inputs, not from dropping `sat_sss`. This is a separate, cheap improvement (~0.01-0.02 PSU) for the deployed
  model: the climatology is a small lookup table shipped with it (rebuildable from more years of Argo), and training
  needs the leave-one-out version.
- Caveats: 5 seeds; same split (train to 2023-12-31, test from 2024-03) as 40.3; R1-R3 seed spread is large;
  HYCOM/Argo non-independence (41.4); single-pixel matchups, not footprint-averaged.
- Not done here: a longer climatology (e.g. WOA) from outside the project's Argo. (The recent-data split and an
  ensemble of the best configuration were tested afterwards, in 43.)

Scripts new: `ladder_tb_retrieval.py`, `climatology_input_test.py`. Nothing committed.

## 43. Combining the three improvements: recent data, a climatology input, and an ensemble

41 found that more recent training data and seed-averaging help; 42 found that a train-years climatology (as an
input, and as the baseline of the target) helps. This tests them together
(`src/recent_climatology_ensemble.py`; per-seed results in `recent_climatology_ensemble_per_seed.parquet`,
seed-averaged held-out predictions in `recent_climatology_ensemble_eval_predictions.parquet`).

Setup: the 'recent' split of 41.1 (train <= 2025-09-30, 96,495 rows; validate 2025-10-01 to 2025-12-09, 3,415 rows),
no basin inputs, 5 seeds per configuration, scored on the same untrained rows as 41.1 -- everything from
2025-12-10: 'window' (to 2026-01-19, 1,961 rows) and 'later' (2026-01-20 on, 10,452 rows). The climatology is
built from this split's train rows only (leave-one-out for train rows, as in 42.1).

- **B0**: existing form (35 inputs, target Argo - `sat_sss`) -- 41's model B; reproduces 41 exactly (e.g. seed 4: 0.2614).
- **B1**: + climatology as an input (36 inputs), same target.
- **B2**: + climatology input, and target = Argo - climatology (36 inputs).

References on the 12,413 rows (RMSE vs. Argo): raw `sat_sss` 0.9095; climatology alone 0.3610; **`anc_sss` (HYCOM)
alone 0.2521**.

### 43.1 Results (test RMSE, PSU; all untrained rows, n=12,413)

| config | single network, mean +/- std (min-max), 5 seeds | seed-averaged ensemble |
|---|---|---|
| B0 existing form | 0.2604 +/- 0.0035 (0.2561-0.2656) | 0.2458 |
| B1 + climatology input | 0.2536 +/- 0.0026 (0.2494-0.2567) | 0.2437 |
| **B2 + climatology input + target-clim** | **0.2451 +/- 0.0011** (0.2438-0.2467) | **0.2365** |

By subset (single networks mean; ensemble): window B0 0.2657 / 0.2549, B1 0.2595 / 0.2513, B2 0.2499 / 0.2437;
later B0 0.2594 / 0.2441, B1 0.2524 / 0.2422, B2 0.2442 / 0.2351.

Paired row-bootstrap on the ensembles (positive = second is better; 2,000 resamples), all rows:
B0 - B1 +0.0021 (95% CI -0.0022 to +0.0070); B1 - B2 **+0.0072** (+0.0053 to +0.0091); B0 - B2 **+0.0094**
(+0.0052 to +0.0144). On the window alone the intervals are too wide to separate any pair (B0 - B2: -0.0002 to
+0.0238), as in 41.1; the evidence for B2 comes from the later rows and the combined set.

- **B2 is the best configuration on every time subset** (not on the rows with the largest anomalies; see 46). A single B2 network (0.2451) matches the old five-network
  average of B0 (0.2458).
- **The target definition is the dependable gain**: B1 -> B2 is significant in every subset, ensembled or not. The
  climatology input alone is clear for single networks (-0.007) but not significant once seeds are averaged.
- **The gains overlap.** Seed-averaging improves B0 by 0.0146 but B2 by only 0.0086, because the climatology
  changes already remove much of the seed-to-seed variance (std 0.0035 -> 0.0011).
- Consistent with 42.4's split (climatology input vs. target) on the older split, though the proportions differ
  (there ~60/40 input/target; here single networks gain 0.0068 from the input and 0.0085 from the target).

### 43.2 The HYCOM comparison -- and a caution for the assimilation use

On these rows `anc_sss` alone scores 0.2521 -- better than B0's single networks (0.2604), only 0.006 above its
ensemble (0.2458), and 0.016 above B2's ensemble (0.2365). Almost all of the skill is in the HYCOM,
climatology and brightness-temperature inputs; the product's own salinity adds little (42.3: R4 ~ R6).

That bears on the purpose of the corrected files (assimilation into MOM6/the coupled model). If the corrected
observations are largely HYCOM-informed, assimilating them feeds a model analysis back in as if it were an
independent measurement, and HYCOM may itself assimilate Argo (general knowledge; not verified, 41.4). The
observation-error values written into the IODA files would need to account for that. Raised as a design question to
settle before using the files, not a conclusion.

### 43.3 Not done

- The IODA files have **not** been regenerated with B2. That needs real generator changes: computing the
  climatology per observation (shipping the lookup table with the model) and writing climatology + network output
  rather than `sat_sss` + a correction.
- No checkpoint of the B2 ensemble was saved (the script trains and scores, and stores predictions only).
- The climatology is from the project's own Argo; an external one (e.g. WOA) was not tried.

Scripts new: `recent_climatology_ensemble.py`. Nothing committed.

## 44. Like-for-like comparison of the three saved models; basin inputs retired

The three saved checkpoints (`rich_correction_model.pt`, `rich_correction_model_nobasin.pt`,
`rich_correction_ensemble_recent_nobasin.pt`) had been quoted with test numbers from **different row sets**: 0.293
for the first (45,387 rows, 2024-03 onward) vs. 0.2458 for the ensemble (12,413 rows, 2025-12-10 onward), so the two
were not comparable as quoted. They also differ in several ways at once (basin inputs, training end date, one network vs.
five). All were therefore re-scored on the same untrained rows as 41.1 -- everything with `sat_datetime` >=
2025-12-10 in the GDAC-direct matchup table (n=12,413: 'window' 1,961 to 2026-01-19, 'later' 10,452 after). None of
the models saw these rows in training or validation. Raw `sat_sss` on them: 0.9095. Test RMSE in PSU:

| model | training data | inputs | window | later | all |
|---|---|---|---|---|---|
| `rich_correction_model.pt` (single net, seed not fixed) | to 2023-12-31 | 41 (with basins) | 0.2859 | 0.2732 | **0.2753** |
| `rich_correction_model_nobasin.pt` (single net, seed 0) | to 2023-12-31 | 35 | 0.2712 | 0.2745 | 0.2740 |
| ensemble member, seed 0 | to 2025-09-30 | 35 | 0.2652 | 0.2577 | 0.2589 |
| ensemble member, seed 1 | to 2025-09-30 | 35 | 0.2700 | 0.2648 | 0.2656 |
| ensemble member, seed 2 | to 2025-09-30 | 35 | 0.2671 | 0.2587 | 0.2600 |
| ensemble member, seed 3 | to 2025-09-30 | 35 | 0.2609 | 0.2551 | 0.2561 |
| ensemble member, seed 4 | to 2025-09-30 | 35 | 0.2655 | 0.2607 | 0.2614 |
| **ensemble mean of the 5** | to 2025-09-30 | 35 | 0.2549 | 0.2441 | **0.2458** |

Mean of the five members as single networks: 0.2604 (all rows).

- **With-basin vs. ensemble: 0.0294** (paired row-bootstrap, 2,000 resamples, 95% CI 0.0251 to 0.0339). Two parts of
  about equal size: the step from the with-basin single net (0.2753) to a typical ensemble member (0.2604) is
  ~0.015, and averaging the five members (0.2604 -> 0.2458) adds ~0.015. The first step mixes **more recent
  training data** (to Sep 2025 vs. to Dec 2023) with dropping the basins; they are not separated by this table. [Update: 45 finds this step is almost entirely recency, not volume.] Against
  member 0 alone (0.2589) the paired difference is 0.0164 (CI 0.0113 to 0.0217). (An earlier message split the
  total as ~0.016 / ~0.013 using member 0 rather than the member mean; the member-mean split above is the fairer one.)
- **Basin inputs at fixed training data**: 0.2753 (with basins) vs. 0.2740 (without) -- both trained to 2023, scored
  identically. The 0.0013 gap is well inside the ~0.005 seed-to-seed spread (40.3), and one network had no fixed seed:
  no detectable basin effect, consistent with 40.
- **Caveat on the first IODA file set**: this table scores the with-basin model using basins from the matchup table
  (the lat/lon classifier's, as trained). The files in `bias_corrected_obsForge/` were generated by feeding it
  obsForge's own basin codes (40.2), which differ on ~10.7% of observations, so that file set's real performance may
  be slightly worse than 0.2753. Not measured.

### 44.1 Decision: no basin inputs in any future model

The basin flags are not a native SMAP or Argo quantity. They are derived from lat/lon (`classify_ocean_basin.py`,
~89% agreement with obsForge's codes) and were introduced after the fact to aid analysis of the early IODA-based
results, not as model information (35, 40.2). 40.3 found no detectable benefit for the rich model and a train/inference
mismatch risk, and this comparison again finds none at fixed training data. **From here on, models use the 35-input
set without `basin_0..5`.** Results quoted in earlier sections that include basins are left as recorded.
[Update: the script defaults that still included basins were flipped afterwards, see 44.2.]

### 44.2 Script defaults flipped to no basins

- `train_rich_features_poc.py`: `BASELINE_FEATURES` (now 6 inputs) and `RICH_FEATURES` (now 35) contain no basin
  inputs. New `BASELINE_FEATURES_WITH_BASIN` (12) and `RICH_FEATURES_WITH_BASIN` (41) exist only for the basin
  ablation and for reading checkpoints from before this rule. `add_features` builds the `basin_*` columns only when
  `argo_oceanBasin` is present, and nothing in the default feature sets requires them.
- `compare_basin_ablation.py`: the "with basin" configurations now use the explicit `*_WITH_BASIN` lists (it would
  otherwise have silently become a no-basin vs. no-basin comparison); `drop_basin()` stays because other scripts
  import it, and is now a no-op on the default lists.
- `train_and_save_correction_model.py`: the `--no-basin` option is removed (always no-basin); the default output is now
  `rich_correction_model_nobasin.pt`, so a default run can no longer overwrite the older with-basin
  `rich_correction_model.pt`.
- Affected by the flip without edits: `analyze_rich_feature_importance.py` (now 35 inputs) and
  `test_training_window_sweep.py` (now the 6-input baseline). Anything that rereads their earlier results should
  treat them as the basin-inclusive versions. `generate_bias_corrected_ioda.py` is unchanged: it still builds basin
  columns from the original file's `oceanBasin` so it can read the older with-basin checkpoint; a no-basin checkpoint
  simply ignores them. `train_baseline.py` and `features.py` belong to the older IODA-based pipeline and were not touched.

Verification: `RICH_FEATURES_WITH_BASIN` equals the with-basin checkpoint's stored input list in order, and
`RICH_FEATURES` equals the no-basin checkpoint's; row counts after the NaN drop are unchanged (112,323); all eleven
dependent scripts import. Regression: the basin ablation rerun at seed 0 reproduces the earlier results exactly (rich
with basins 0.2930, rich without 0.2939, baseline with 0.3416, baseline without 0.3631), written to `/tmp` so the saved
ablation results were not overwritten. Nothing committed.

## 45. More data, or more recent data? An equal-size test

44 and 41.1 left a confound: the 'recent' model B differed from the old model A in both training-set size
(96,495 vs. 63,620 rows) and training end date (2025-09-30 vs. 2023-12-31), so the gain from B could be volume,
recency, or both. This separates them (`src/equal_size_recency_test.py`; per-seed results in
`equal_size_recency_per_seed.parquet`, seed-averaged held-out predictions in
`equal_size_recency_eval_predictions.parquet`).

Three models, same 35 inputs (no basins), same target (Argo - `sat_sss`), same seeds 0-4, scored on the same
untrained rows as 41.1 (everything with `sat_datetime` >= 2025-12-10: 12,413 rows; 'window' 1,961 to 2026-01-19,
'later' 10,452 after):

| model | training rows | training span | early-stopping validation |
|---|---|---|---|
| A (old split) | 63,620 | 2020-09-23 to 2023-12-30 | 3,316 rows, Dec 2023 - Feb 2024 |
| B (recent split) | 96,495 | 2020-09-23 to 2025-09-29 | 3,415 rows, Oct - Dec 9 2025 |
| **C (equal-size)** | **63,620** | **2022-04-11 to 2025-09-29** | 3,415 rows (same as B) |

C is the most recent 63,620 rows before October 2025, a subset of B's training set with A's size. C vs. A holds size
fixed and varies the period (isolates recency); C vs. B holds the end date fixed and varies size (isolates volume).
A and B reproduced their 41.1 results exactly.

### 45.1 Results (test RMSE, PSU)

Single networks, mean +/- std (min-max) over 5 seeds:

| held-out rows | A | B | C |
|---|---|---|---|
| window, n=1,961 | 0.2704 +/- 0.0029 (0.2654-0.2729) | 0.2657 +/- 0.0033 (0.2609-0.2700) | 0.2672 +/- 0.0043 (0.2622-0.2735) |
| later, n=10,452 | 0.2758 +/- 0.0091 (0.2683-0.2916) | 0.2594 +/- 0.0036 (0.2551-0.2648) | 0.2591 +/- 0.0032 (0.2543-0.2627) |
| all, n=12,413 | 0.2749 +/- 0.0078 (0.2690-0.2885) | 0.2604 +/- 0.0035 (0.2561-0.2656) | 0.2604 +/- 0.0033 (0.2556-0.2644) |

Seed-averaged ensembles (RMSE): window A 0.2572 / B 0.2549 / C 0.2560; later A 0.2560 / B 0.2441 / C 0.2452;
all A 0.2562 / B 0.2458 / C 0.2469. Paired row-bootstrap (2,000 resamples), positive = second is better:

| subset | A - C (recency, size fixed) | C - B (size, end date fixed) | A - B (total) |
|---|---|---|---|
| window | +0.0013 (-0.0035, +0.0064) | +0.0011 (-0.0036, +0.0054) | +0.0025 (-0.0034, +0.0075) |
| later | **+0.0110** (+0.0067, +0.0153) | +0.0011 (-0.0028, +0.0041) | +0.0121 (+0.0070, +0.0164) |
| all | **+0.0093** (+0.0054, +0.0132) | +0.0010 (-0.0024, +0.0039) | +0.0104 (+0.0059, +0.0141) |

- **Recency, not volume.** C matches B as single networks (0.2604 vs. 0.2604) despite 32,875 fewer training rows,
  and C - B on the ensembles is +0.0010 with an interval spanning zero. C beats A at equal size by 0.0093
  (interval +0.0054 to +0.0132). Of the 0.0104 total A - B gap on the ensembles, about 89% is the C - A step.
- **Where it shows.** The effect is concentrated in the later rows (A - C: 0.0110) and not detectable in the
  1,961-row window (0.0013, interval spanning zero) -- consistent with A going stale as the held-out period moves
  away from its end date, though the window's small sample cannot confirm that. A is also less stable across seeds
  (range 0.269-0.289 vs. about 0.256-0.265 for B and C).
- This refines 44: its decomposition of the with-basin-single-to-ensemble gap attributed about half to the
  recent training data; that half is now attributed to recency specifically.

### 45.2 Caveats

- **Early-stopping validation sets differ.** A stops on Dec 2023 - Feb 2024 rows; B and C stop on Oct - Dec 2025 rows,
  just before the held-out period. Some of the "recency" gain could come from better-timed early stopping rather than
  the training period. Not separated (giving A the recent validation set would peek past its training end).
- Holds at 64k to 96k rows; it does not say more data is worthless beyond that range, only that it did not help here.
- One training end date and one test period; the existing form of the model (target Argo - `sat_sss`), not the
  climatology form of 43 (B2), where the recency effect might differ.
- In direction this is consistent with 26.4's one non-monotonic origin (a 6-month window beating longer ones at the
  newest origin), though that used the older range-filtered Argo labels and the baseline-feature model, so it is only
  suggestive.

### 45.3 Implication and not done

For the deployed model the evidence favors retraining on a rolling recent window rather than accumulating older data.
Not tested: other training end dates or window lengths (e.g. 12 vs. 24 months), recency weighting, and whether the
recency effect holds for the climatology form (43). Scripts new: `equal_size_recency_test.py`. Nothing committed.

## 46. Does the 'Argo - climatology' target buy its gain by damping anomalies?

Raised against 43: for a bias correction (target Argo - `sat_sss`) the B2 form (target Argo - climatology) changes what the
product is, and a lower RMSE could be bought by pulling estimates toward the climatology, at the expense of real anomalies.
`src/anomaly_preservation_check.py` (new; strata in `anomaly_preservation_strata.parquet`) tests this on the saved
seed-averaged held-out predictions from 43 -- B0 (existing form), B1 (+ climatology input), B2 (+ input and target =
Argo - climatology), 5 seeds each, on the 12,413 untrained rows from 2025-12-10. No retraining; the climatology is rebuilt
from the same split's train rows and the saved predictions are checked to line up row for row (they do; ensemble RMSEs
0.2458 / 0.2437 / 0.2365 reproduce 43).

Held-out rows are stratified two ways. By the **true anomaly** (Argo - climatology): selects on the outcome, so any model that
shrinks toward the mean looks worse in the extreme strata, partly by construction -- read the model-to-model differences, not
the levels. By the **input anomaly** (`|anc_sss - clim|`, `|sat_sss - clim|`): selects only on inputs, so no selection bias.

### 46.1 Results (ensemble RMSE, PSU; paired row-bootstrap, 2,000 resamples, 95% CI, "+" = second is better)

| stratum | n | B0 | B1 | B2 | B0 - B2 | B1 - B2 |
|---|---|---|---|---|---|---|
| all rows | 12,413 | 0.2458 | 0.2437 | 0.2365 | +0.0093 (+0.0051, +0.0144) | +0.0072 (+0.0054, +0.0091) |
| true \|anom\| lowest 50% | 6,207 | 0.1635 | 0.1276 | 0.1216 | +0.0419 (+0.0387, +0.0453) | +0.0061 (+0.0040, +0.0082) |
| true \|anom\| 50-90% | 4,964 | 0.2230 | 0.2219 | 0.2148 | +0.0083 (+0.0035, +0.0137) | +0.0071 (+0.0050, +0.0092) |
| true \|anom\| top 10% | 1,242 | **0.5212** | 0.5616 | 0.5484 | **-0.0273** (-0.0439, -0.0055) | +0.0132 (+0.0066, +0.0204) |
| true anom lowest 5% (fresher) | 621 | 0.6357 | 0.6612 | 0.6495 | -0.0138 (-0.0404, +0.0200) | +0.0121 (+0.0020, +0.0227) |
| true anom highest 5% (saltier) | 621 | **0.3704** | 0.4394 | 0.4219 | **-0.0517** (-0.0643, -0.0383) | +0.0176 (+0.0090, +0.0260) |
| \|HYCOM anom\| top 10% | 1,242 | 0.4113 | 0.4260 | 0.4154 | -0.0046 (-0.0259, +0.0213) | +0.0108 (+0.0032, +0.0185) |
| \|HYCOM anom\| bottom 90% | 11,171 | 0.2199 | 0.2140 | 0.2073 | +0.0126 (+0.0098, +0.0155) | +0.0068 (+0.0053, +0.0083) |
| \|SMAP anom\| top 10% | 1,242 | 0.3339 | 0.3187 | **0.3019** | +0.0319 (+0.0181, +0.0474) | +0.0169 (+0.0090, +0.0244) |
| \|SMAP anom\| bottom 90% | 11,171 | 0.2340 | 0.2339 | 0.2281 | +0.0059 (+0.0015, +0.0110) | +0.0058 (+0.0039, +0.0077) |

B0 - B1 (not in the table): all rows +0.0021 (-0.0025, +0.0071); lowest 50% +0.0358; top 10% **-0.0405** (-0.0586, -0.0183);
saltier 5% **-0.0692** (-0.0859, -0.0520); \|SMAP anom\| top 10% +0.0150 (+0.0004, +0.0303).

Calibration -- OLS slope of the true anomaly on each model's predicted anomaly (1 = calibrated, > 1 = damped toward
the climatology, < 1 = overspread) and the spread of the predicted anomaly (true anomaly std 0.361):

| model | slope, all rows | slope, own top-10% of \|predicted anomaly\| | std of predicted anomaly |
|---|---|---|---|
| B0 | 0.892 | 0.911 | 0.298 |
| B1 | 1.192 | 1.204 | 0.226 |
| B2 | 1.156 | 1.145 | 0.238 |

### 46.2 What this shows

- **B2's gain over B1 is not damping.** B2 beats B1 in all ten strata with every interval excluding zero, including the
  largest-anomaly rows (top 10%: +0.0132; both tails; large HYCOM and SMAP anomalies), and B2's predictions are *less*
  damped than B1's (spread 0.238 vs. 0.226; slope 1.156 vs. 1.192). The target change helps across the board.
- **The climatology *input* makes estimates more conservative, with a real cost at the extremes.** Both climatology-input models
  are clearly worse than B0 on the largest true anomalies (top 10%: B0 0.521 vs. B1 0.562 / B2 0.548, significant; saltier tail
  significant; fresher tail not) and have damped predictions (std 0.23-0.24 vs. 0.30; slopes > 1), while being much better on
  the ordinary rows (lowest 50%: 0.164 -> 0.128 / 0.122). The aggregate gain comes from the many ordinary rows. Caveat: the
  true-anomaly strata favor the less-shrunk B0 by construction.
- **Where inputs themselves signal a large anomaly there is no such loss.** On rows with a large SMAP-vs-climatology anomaly,
  B2 is the best (0.302 vs. B0 0.334); on rows with a large HYCOM anomaly B2 and B0 cannot be separated (-0.0046, interval
  spanning zero), and both beat B1.
- **Calibration**: B1 and B2 are under-dispersed (a 15-20% stretch of their predicted anomalies would lower MSE on these
  rows); B0 is slightly over-dispersed (0.89). Caveats: seed-averaging itself shrinks prediction spread, and the held-out
  rows are later than the training period, so a shift toward larger anomalies would also push the slope above 1. No
  recalibration was tried.

### 46.3 Consequence for the bias-correction form

My earlier recommendation (in conversation, after 43) was B1 -- target Argo - `sat_sss` with the climatology added as an
input -- as the form consistent with a bias correction. This test does not support it: B1's gain over B0 is not significant
in aggregate for ensembles (+0.0021), and it costs accuracy on the largest anomalies (-0.0405, significant) that matter
most for events. **For a strict bias correction, the B0 form (Argo - `sat_sss`, no climatology input) preserves large anomalies
best**, which is the form of the ensemble IODA files of 41.2 (`rich_correction_ensemble_recent_nobasin.pt`). B2 gives the
lowest overall RMSE and does not lose where the inputs themselves signal a large anomaly, but it is a different product
(a salinity estimate that uses SMAP), carries the conservative-estimate cost on the most anomalous rows, and shares the
HYCOM-dependence concern of 43.2.

### 46.4 Not done

- A recalibration/stretch of the damped models, and an extreme-weighted loss.
- Regional breakdown of the extreme rows; tests on any other held-out period.
- Comparing the forms on observations of known events (e.g. a river plume or ENSO excursion) rather than quantile strata.

Scripts new: `anomaly_preservation_check.py`. Nothing committed.
