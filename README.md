
# AI-NWP Precipitation-Type Evaluation

This repository contains code and analysis for evaluating precipitation-type diagnosis from AI-based numerical weather prediction (AI-NWP) models.

The project focuses on diagnosing and evaluating four surface precipitation types:

- Rain
- Snow
- Freezing rain
- Ice pellets

Precipitation type is diagnosed from atmospheric profiles produced by several AI weather prediction models and evaluated against observations and operational numerical weather prediction guidance.

## Models

The current workflow includes:

- GraphCast
- Pangu-Weather
- Aurora
- WXFormer
- FuXi
- GFS

GraphCast, Pangu-Weather, and Aurora forecasts are initialized from ERA5 and evaluated over multiple forecast lead times. GFS provides an operational NWP baseline with directly diagnosed precipitation-type fields.

## Observations

Model predictions are evaluated using precipitation-type reports from the Meteorological Phenomena Identification Near the Ground (mPING) dataset.

Observation processing includes quality control and consensus filtering before comparison with model forecasts.

## Forecast Period

The primary evaluation period covers the 2021–2022 Northern Hemisphere cold season, with daily 00 UTC forecast initializations and forecast lead times extending to 168 hours.

## Repository Structure

```text
AI_NWP/
├── ai-weather-models/   # AI weather model workflows and utilities
├── gfs/                 # GFS download and precipitation-type processing
├── notebooks/           # Analysis and exploratory notebooks
├── figures/             # Figures generated for analysis and publication
├── ptype/               # Precipitation-type diagnosis and evaluation code
└── README.md
```

Large forecast datasets, NetCDF files, cache files, logs, and HPC job output files are excluded from version control.

## Research Workflow

The general workflow is:

1. Generate or obtain forecasts from each weather prediction model.
2. Subset forecasts to the study domain.
3. Standardize atmospheric variables and vertical coordinates.
4. Diagnose precipitation type from AI-NWP atmospheric profiles.
5. Process and quality-control mPING observations.
6. Match forecasts with observations in space and time.
7. Evaluate precipitation-type skill as a function of forecast lead time.
8. Compare AI-NWP performance with the GFS baseline.

## Status

This repository is under active development. Code organization, documentation, and evaluation workflows will continue to evolve as the project progresses.

## Data Availability

Large meteorological datasets are not stored in this repository.

Users are expected to obtain the required datasets separately, including:

- ERA5
- AI-NWP forecast outputs
- GFS forecasts
- mPING observations

Additional information on data acquisition and preprocessing will be added as the workflow is finalized.

## Citation

Citation information will be added upon publication of the associated study.

## Author

Stanley Akor  
Boise State University
