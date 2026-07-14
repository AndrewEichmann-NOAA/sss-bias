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
