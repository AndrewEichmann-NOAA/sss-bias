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

**Open decision, not yet resolved**: keep this stricter, leak-free split as the new default despite small/
noisy val/test, or find a compromise (e.g., blocked/rolling CV across multiple date windows per 6, to average
out the small-sample noise) before trusting comparisons against it.
