# AI-NWP Precipitation-Type Evaluation

Code and analysis notebooks for diagnosing and verifying surface precipitation type (ptype) from AI-based numerical weather prediction (AI-NWP) models.

Four classes are diagnosed: **rain**, **snow**, **ice pellets** and **freezing rain**. Ptype is diagnosed from forecast temperature, dew point and wind profiles and verified against crowd-sourced mPING reports over the contiguous United States (CONUS) for the 2021–22 cold season.

## Forecast systems

Five AI-NWP models are run from ERA5 initial conditions, and GFS is used as an operational NWP baseline. Ptype is diagnosed with up to two methods per model, giving **12 forecast systems**:

| Model | Run with | EDNN | Bourgouin | Native ptype |
|---|---|:---:|:---:|:---:|
| Aurora | [Earth2Studio](https://github.com/NVIDIA/earth2studio) | ✓ | ✓ | |
| GraphCast (Operational) | [Earth2Studio](https://github.com/NVIDIA/earth2studio) | ✓ | ✓ | |
| Pangu-Weather | [Earth2Studio](https://github.com/NVIDIA/earth2studio) | ✓ | ✓ | |
| WXFormer | [NCAR CREDIT](https://github.com/NCAR/miles-credit) | ✓ | ✓ | |
| FuXi | [NCAR CREDIT](https://github.com/NCAR/miles-credit) | ✓ | ✓ | |
| GFS | Operational ([AWS archive](https://registry.opendata.aws/noaa-gfs-bdp-pds/)) | ✓ | | ✓ |

- **EDNN:** the evidential deep neural network ptype classifier of [Becker et al. (2026)](#references), obtained from [miles-guess](https://github.com/ai2es/miles-guess). It returns class probabilities plus aleatoric and epistemic uncertainty for each class.
- **Bourgouin:** a thermodynamic method that classifies from the warm (melting) and cold (refreezing) areas of the temperature profile. The implementation follows [Becker et al. (2026)](#references) and is in [`ptype/ainwp_ptype_original/bourgouin.py`](ptype/ainwp_ptype_original/bourgouin.py).

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
| Observed class | Most-reported ptype per mPING record; ties broken freezing rain > ice pellets > snow > rain |
| Verification sample | Only observation/day pairs where all 12 systems have a valid prediction |

## Repository structure

```text
ai-nwp-precipitation-type/
├── ai-weather-models/
│   ├── script/              # Aurora, GraphCast and Pangu-Weather forecast runners (Earth2Studio)
│   └── slurm_job_script/    # PBS job scripts for the forecast runners
├── ncar_credit/             # Profile extraction + EDNN/Bourgouin ptype from WXFormer and FuXi output
├── gfs/                     # GFS download (AWS), EDNN ptype and native ptype
├── ptype/                   # ainwp_ptype_original package: EDNN + Bourgouin ptype for Aurora, GraphCast, Pangu-Weather
│   ├── ainwp_ptype_original/
│   ├── scripts/run_ptype.py
│   ├── tests/
│   └── run-*-ptype.pbs      # PBS array jobs, one initialization per array index
├── notebooks/               # Verification and figure notebooks
├── figures_modal/           # Manuscript figures produced by the notebooks
└── README.md
```

## Workflow

1. **Run the AI-NWP forecasts.**
   - Aurora, GraphCast and Pangu-Weather: `ai-weather-models/script/run_{aurora,graphcast,pangu}_forecast.py` run each model from ERA5 with [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio) (version 0.17.0) and write one NetCDF file per initialization. The three runners share one ERA5 cache.
   - WXFormer and FuXi: these models are **not run by this repository**. They were run with the [NCAR CREDIT framework](https://github.com/NCAR/miles-credit); see the [CREDIT documentation](https://miles-credit.readthedocs.io/en/latest/) and [quickstart](https://miles-credit.readthedocs.io/en/latest/quickstart.html) to install it and run the models. Pretrained 6-hour WXFormer and FuXi weights are linked from the CREDIT README.
2. **Diagnose ptype.**
   - Aurora, GraphCast and Pangu-Weather: `ptype/scripts/run_ptype.py`. Each forecast is subset to CONUS, dew point and height above ground are computed, the profiles are interpolated to the 0–5 km grid, and the EDNN and Bourgouin ptypes are added.
   - WXFormer and FuXi: `ncar_credit/wxformer_bourgouin.py` and `ncar_credit/fuxi_bourgouin.py` read the CREDIT forecast output, extract the 0–5 km profiles and add the EDNN and Bourgouin ptypes.
   - GFS: `gfs/gfs_ptype.py` downloads the needed GRIB2 fields from the NOAA GFS bucket on AWS, then adds the EDNN ptype alongside the native GFS ptype.
3. **Verify and make figures** with the notebooks below.

The `.pbs` files are PBS job scripts written for NCAR Derecho. They set the account, queue and conda environment paths for the original runs, and must be edited before use elsewhere.

## Notebooks

| Notebook | Purpose |
|---|---|
| `00_mping_qc` | mPING verification sample and observed-class distribution |
| `01_multimodel_ptype_verification_skill` | Collocation, common-valid sample, and POD/FAR/CSI/bias by forecast day |
| `02_ip_fzra_misclassification_by_leadtime` | What each system predicts for observed ice pellets and freezing rain |
| `03_ednn_uncertainty_by_leadtime` | EDNN uncertainty by observed class and forecast day |
| `041_ptype_skill_vs_near_surface_temperature` | Skill as a function of RAP 2-m temperature |
| `05`–`07` | Case study, 1800 UTC 3 Feb 2022: ptype maps, uncertainty maps, profiles |
| `08_synoptic_ivt_feb2022` | Synoptic overview of the case-study storm |
| `09_profile_composites_vs_rap` | Forecast profiles at ice pellet and freezing rain reports vs RAP |

Run notebook 01 first: it writes `common_valid_multimodel_predictions.parquet`, which notebooks 02–04 read. Each notebook begins with a description of its method and outputs and a configuration cell. The data paths in those cells point to NCAR GLADE storage (`/glade/...`) and must be changed to your own locations. Figures are written to `figures_modal/`.

## Installation

There is no single environment file yet. The original runs used separate conda environments for each AI model and one for ptype diagnosis.

**Ptype package** (Python ≥ 3.10):

```bash
cd ptype
python -m pip install -e .
```

The EDNN step also needs `keras`, [`miles-guess`](https://github.com/ai2es/miles-guess) and `bridgescaler`.

**Notebooks:** `numpy`, `pandas`, `xarray`, `matplotlib`, `scipy`, `pyarrow`, `cartopy`, `shapely`, `metpy`, `herbie-data` (notebook 041) and `cfgrib` (notebook 08).

**GFS processing:** additionally `boto3` and `numba`.

**Forecast runners:** [`earth2studio`](https://github.com/NVIDIA/earth2studio) 0.17.0 with the Aurora, GraphCast and Pangu-Weather model dependencies. GraphCast runs on JAX and uses the GPU only when both PyTorch and JAX can see one.

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

Large datasets are not stored in this repository. NetCDF files, caches, logs and PBS job output are excluded by `.gitignore`.

| Data | Source |
|---|---|
| ERA5 initial conditions | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu) (fetched automatically by Earth2Studio) |
| GFS 0.25° forecasts and analyses | [NOAA GFS on the AWS Open Data Registry](https://registry.opendata.aws/noaa-gfs-bdp-pds/) |
| RAP analyses | [NOAA NCEI Rapid Refresh](https://www.ncei.noaa.gov/products/weather-climate-models/rapid-refresh-update) (notebook 041 downloads them with Herbie) |
| Quality-controlled mPING reports | [Becker et al. (2026)](#references) |
| EDNN ptype model | [miles-guess](https://github.com/ai2es/miles-guess) |

## References

Becker, C., Gagne, D. J., Demuth, J., Schreck, J. S., Radford, J., Gantos, G., Kim, E., Kimpara, D., Reiner, S., Willson, J., & Wirz, C. D. (2026). Winter precipitation-type diagnosis and uncertainty quantification with a physically consistent machine learning method. *Weather and Forecasting*, 41(10), 2051–2068. Preprint: [arXiv:2512.13899](https://arxiv.org/abs/2512.13899)

## Citation

Citation information will be added when the associated manuscript is published.

## Author

Stanley Akor, Boise State University