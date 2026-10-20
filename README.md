# AI-NWP Precipitation-Type Evaluation

Code and analysis notebooks for diagnosing and verifying surface precipitation type (ptype) from AI-based numerical weather prediction (AI-NWP) models.

Four classes are diagnosed: **rain**, **snow**, **ice pellets** and **freezing rain**. Ptype is diagnosed from forecast temperature, dew point and wind profiles and verified against crowd-sourced mPING reports over the contiguous United States (CONUS) for the 2021–22 cold season.

## Forecast systems

Five AI-NWP models are run from ERA5 initial conditions, and GFS is used as an operational NWP baseline. Ptype is diagnosed with up to two methods per model, giving **12 forecast systems**:

| Model | Evidential deep neural network (EDNN) | Bourgouin (thermodynamic) | Native ptype |
|---|:---:|:---:|:---:|
| Aurora | ✓ | ✓ | |
| GraphCast (Operational) | ✓ | ✓ | |
| Pangu-Weather | ✓ | ✓ | |
| WXFormer (NCAR CREDIT) | ✓ | ✓ | |
| FuXi (NCAR CREDIT) | ✓ | ✓ | |
| GFS | ✓ | | ✓ |

The EDNN returns class probabilities plus aleatoric and epistemic uncertainty for each class. The Bourgouin method classifies from the warm (melting) and cold (refreezing) areas of the temperature profile.

## Experiment setup

| Item | Setting |
|---|---|
| Forecast initializations | Daily at 00 UTC, 24 Nov 2021 – 1 Apr 2022 |
| Forecast length | 168 h at 6-h steps |
| Verification period | 1 Dec 2021 – 31 Mar 2022 |
| Verification times | 00, 06, 12 and 18 UTC (the AI-NWP output times) |
| Forecast day *n* | Leads from 24(*n*−1)+6 to 24*n* h, days 1–7 |
| Domain | CONUS box, 24–50° N, 125–66° W |
| Profile grid | 0–5000 m above ground level, 250-m spacing (21 levels) |
| Observations | Quality-controlled mPING reports |
| Observed class | Most-reported ptype per record; ties broken freezing rain > ice pellets > snow > rain (`OBS_RULE = "modal"`) |
| Verification sample | Common-valid sample: only observation/day pairs where all 12 systems have a valid prediction |

## Repository structure

```text
ai-nwp-precipitation-type/
├── ai-weather-models/
│   ├── script/              # Aurora, GraphCast and Pangu-Weather forecast runners (Earth2Studio)
│   └── slurm_job_script/    # PBS job scripts for the forecast runners
├── ncar_credit/             # EDNN + Bourgouin ptype for WXFormer and FuXi output
├── gfs/                     # GFS download (AWS), EDNN ptype and native ptype
├── ptype/                   # ainwp_ptype_original package: EDNN + Bourgouin ptype for Aurora, GraphCast, Pangu-Weather
│   ├── ainwp_ptype_original/
│   ├── scripts/run_ptype.py
│   ├── tests/
│   └── run-*-ptype.pbs      # PBS array jobs, one initialization per array index
├── notebooks/               # Verification and figure notebooks (run in order, see below)
├── figures_modal/           # Manuscript figures produced by the notebooks (modal observed-class rule)
└── README.md
```

## Workflow

1. **Run the AI-NWP forecasts.** `ai-weather-models/script/run_{aurora,graphcast,pangu}_forecast.py` run each model from ERA5 through Earth2Studio and write one NetCDF file per initialization. The three runners share one ERA5 cache. WXFormer and FuXi forecasts come from NCAR CREDIT and are not run by this repository.
2. **Diagnose ptype.**
   - Aurora, GraphCast and Pangu-Weather: `ptype/scripts/run_ptype.py`. Each forecast is subset to CONUS, dew point and height above ground are computed, the profiles are interpolated to the 0–5 km grid, and the EDNN and Bourgouin ptypes are added.
   - WXFormer and FuXi: `ncar_credit/wxformer_bourgouin.py` and `ncar_credit/fuxi_bourgouin.py`.
   - GFS: `gfs/gfs_ptype.py` downloads the needed GRIB2 fields from the NOAA GFS bucket on AWS (`noaa-gfs-bdp-pds`), then adds the EDNN ptype alongside the native GFS ptype.
3. **Verify and make figures** with the notebooks below.

The `.pbs` files are PBS job scripts written for NCAR Derecho. They set the account, queue and conda environment paths for the original runs, and must be edited before use elsewhere.

## Notebooks

Run notebook 01 first. It builds `common_valid_multimodel_predictions.parquet`, which notebooks 02–04 read. Notebooks 05–07 and 09 share the forecast file settings (`MODEL_DIRS`, `FILE_GLOBS`, `PTYPE_VAR`) set in notebook 05.

| Notebook | What it does | Main outputs |
|---|---|---|
| `00_mping_qc.ipynb` | Observed-class assignment and distribution of mPING reports by class and reporter agreement | `fig_mping_report_distribution.pdf` |
| `01_multimodel_ptype_verification_skill.ipynb` | Collocates mPING reports with each model, builds the common-valid sample, and computes POD, FAR, CSI and frequency bias by system, forecast day and class | `multimodel_csi_bias_heatmap.pdf`, `multimodel_far_pod_heatmap.pdf`, `common_valid_multimodel_predictions.parquet`, `multimodel_skill_by_day.csv` |
| `02_ip_fzra_misclassification_by_leadtime.ipynb` | What each system predicts when mPING observes ice pellets or freezing rain, by forecast day | `ip_fzra_fate_by_leadtime.pdf` |
| `03_ednn_uncertainty_by_leadtime.ipynb` | EDNN total uncertainty (aleatoric + epistemic) by observed class and forecast day | `ednn_total_uncertainty_by_obsclass_leadtime.pdf`, `ednn_uncertainty_components_supp.pdf` |
| `041_ptype_skill_vs_near_surface_temperature.ipynb` | Skill binned by a reference RAP 2-m temperature at each report (RAP fields fetched with Herbie) | `accuracy_vs_reference_temperature.pdf`, `csi_pod_vs_reference_temperature_day1.pdf` |
| `05_case_study_ptype_maps.ipynb` | Case study, 1800 UTC 3 Feb 2022: ptype maps for all 12 systems, agreement maps, and point verification | `case_study_ptype_maps_*.pdf`, `case_study_agreement_*.pdf` |
| `06_case_study_uncertainty_maps.ipynb` | Case study: EDNN total-uncertainty maps for forecast days 1, 3 and 7 | `case_study_total_uncertainty_*.pdf` |
| `07_case_study_profiles.ipynb` | Case study: forecast temperature profiles at one representative mPING report per class | `case_study_profiles_*.pdf` |
| `08_synoptic_ivt_feb2022.ipynb` | Synoptic overview of the February 2022 storm from GFS 0.25° analyses: IVT, 500-hPa height, 850-hPa temperature and freezing lines | `synoptic_ivt_feb2022.pdf` |
| `09_profile_composites_vs_rap.ipynb` | Median forecast temperature and dew point profiles at ice pellet and freezing rain reports, and their difference from RAP | `composite_skewt_mixedphase.pdf`, `composite_bias_T_mixedphase.pdf` |

Each notebook starts with a description of its method and outputs, followed by a **Configuration** cell. Paths in the configuration cells point to NCAR GLADE storage (`/glade/...`) and must be changed to your own data locations.

Notebook 01 also supports a `"consensus"` observed-class rule (most-reported ptype with ≥ 70 % reporter agreement). Its outputs go to `verification_multimodel/` and `figures/` instead of the `_modal` folders.

## Installation

There is no single environment file yet. The original runs used separate conda environments for each AI model and one for ptype diagnosis.

**Ptype package** (Python ≥ 3.10):

```bash
cd ptype
python -m pip install -e .
```

The EDNN step also needs `keras`, `mlguess` and `bridgescaler`.

**Notebooks:** `numpy`, `pandas`, `xarray`, `matplotlib`, `scipy`, `pyarrow`, `cartopy`, `shapely`, `metpy`, `herbie-data` (notebook 041) and `cfgrib` (notebook 08).

**GFS processing:** additionally `boto3` and `numba`.

**Forecast runners:** `earth2studio` with the Aurora, GraphCast and Pangu-Weather model dependencies. GraphCast runs on JAX and uses the GPU only when both PyTorch and JAX can see one.

## Usage

Run ptype diagnosis on one forecast file:

```bash
python ptype/scripts/run_ptype.py \
  --input /path/to/forecast.nc \
  --model-name graphcast \
  --surface-geopotential /path/to/surface_geopotential.nc \
  --ednn-model /path/to/model.keras \
  --scaler /path/to/scaler.json \
  --output-dir /path/to/output
```

Options: `--overwrite` replaces existing output, and `--no-uncertainties` skips the aleatoric and epistemic fields. Output is written as `<output-dir>/<init>/<model>_<init>_<lead>.nc`.

Run a forecast model over the full period:

```bash
python ai-weather-models/script/run_pangu_forecast.py --start 2021-11-24 --end 2022-04-01
```

Run GFS processing:

```bash
python gfs/gfs_ptype.py --start-date 2021-11-24 --end-date 2022-04-01 \
  --model-path /path/to/model.keras --scaler-path /path/to/scaler.json
```

Run the unit tests for the Bourgouin method and vertical interpolation:

```bash
cd ptype
python -m pytest tests
```

## Output variables

Each ptype output file holds, on the 0–5 km height grid and CONUS lat/lon grid:

- Profiles: `T_HEIGHT`, `DPT_HEIGHT`, `U_HEIGHT`, `V_HEIGHT`
- EDNN class probabilities: `ML_rain`, `ML_snow`, `ML_icep`, `ML_frzr`
- EDNN uncertainty per class: `ML_<class>_ale` (aleatoric) and `ML_<class>_epi` (epistemic)
- Bourgouin ptype: `BOURG_ptype` (−1 invalid, 0 rain, 1 snow, 2 ice pellets, 3 freezing rain), with the melting and refreezing energies `BOURG_PA` and `BOURG_NA`

## Data

Large datasets are not stored in this repository. NetCDF files, caches, logs and PBS job output are excluded by `.gitignore`. To reproduce the results you need:

- **ERA5** reanalysis, used as initial conditions (fetched through Earth2Studio)
- **AI-NWP forecasts**, produced with the runners above; WXFormer and FuXi forecasts from NCAR CREDIT
- **GFS** 0.25° forecasts and analyses (NOAA GFS bucket on AWS)
- **RAP** 13-km analyses (via Herbie, notebook 041)
- **mPING** reports, quality-controlled and interpolated (`mPING_hourafter_interpolated_QC3.parquet`)
- The trained **EDNN** model and its scaler

## Citation

Citation information will be added when the associated manuscript is published.

## Author

Stanley Akor, Boise State University