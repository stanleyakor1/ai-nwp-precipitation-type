import os
import sys
import time
import socket
import shutil
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc

from botocore import UNSIGNED
from botocore.config import Config
from metpy.units import units
from numba import njit, prange


# ============================================================
# CONFIG
# ============================================================

LEADS = np.arange(0, 169, 6)

PLEVELS = [1000, 975, 950, 925, 900, 850, 800, 750, 700,
           650, 600, 550, 500, 450, 400, 350, 300, 250]

PL_VARS = {"TMP", "RH", "HGT", "UGRD", "VGRD"}
PTYPE_VARS = {"CRAIN", "CSNOW", "CFRZR", "CICEP"}

TARGET_HEIGHTS = np.arange(0, 5001, 250, dtype=np.float32)
PROFILE_VARS = ("T_HEIGHT", "DPT_HEIGHT", "U_HEIGHT", "V_HEIGHT")
PTYPE_CLASSES = ("rain", "snow", "ice pellets", "freezing rain")
PTYPE_SHORT_NAMES = ("rain", "snow", "icep", "frzr")
N_HEIGHTS = len(TARGET_HEIGHTS)
N_FEATURES = len(PROFILE_VARS) * N_HEIGHTS

LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = -125.0, -66.0

BUCKET = "noaa-gfs-bdp-pds"
RESOLUTION = "0p25"
G = 9.80665

DEFAULT_BASE_DIR = Path("/glade/derecho/scratch/sakor/gfs")
DEFAULT_MODEL_PATH = Path("/glade/derecho/scratch/cbecker/ptype_model_20240909.keras")
DEFAULT_SCALER_PATH = Path("/glade/derecho/scratch/cbecker/ptype_scaler_20240909.json")

OUTPUT_UNCERTAINTIES = True
RUN_BOURGOUIN = True
KEEP_GRIB = False
MAX_WORKERS = 12


def make_s3_client(max_workers=12):
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            retries={"max_attempts": 10, "mode": "standard"},
            max_pool_connections=max_workers,
        ),
    )


s3 = make_s3_client(MAX_WORKERS)


def setup_logger(log_dir, start_date, end_date, cycle, level="INFO"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    start_tag = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_tag = pd.Timestamp(end_date).strftime("%Y%m%d")
    log_file = log_dir / f"gfs_ptype_{start_tag}_{end_tag}_{cycle}z.log"

    logger = logging.getLogger("gfs_ptype")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger, log_file


def initialization_context(init_date, cycle, base_dir):
    date = pd.Timestamp(init_date)
    init_date_compact = date.strftime("%Y%m%d")
    init_date_iso = date.strftime("%Y-%m-%d")

    init_time = np.datetime64(f"{init_date_iso}T{cycle}:00")
    init_string = np.datetime_as_string(
        init_time.astype("datetime64[h]"), unit="h"
    ) + "Z"

    base_dir = Path(base_dir)
    work_dir = base_dir / "tmp" / f"{init_date_compact}_{cycle}z"
    output_dir = base_dir / "processed" / init_string

    return {
        "init_date": init_date_compact,
        "init_time": init_time,
        "init_string": init_string,
        "work_dir": work_dir,
        "output_dir": output_dir,
    }


def output_path(output_dir, init_string, lead):
    return Path(output_dir) / f"gfs_{init_string}_{lead:03d}.nc"


# ============================================================
# EDNN ROUTINES
# ============================================================

def load_saved_model(model_path, scaler_path):
    from keras.models import load_model
    from mlguess.keras.models import CategoricalDNN
    from mlguess.keras.losses import evidential_cat_loss
    from bridgescaler.backend import load_scaler

    model = load_model(
        model_path,
        custom_objects={
            "CategoricalDNN": CategoricalDNN,
            "evidential_cat_loss": evidential_cat_loss,
        },
        compile=False,
    )

    scaler = load_scaler(scaler_path)
    input_features = [feature for group in scaler.groups_ for feature in group]

    if len(input_features) != N_FEATURES:
        raise ValueError(
            f"Scaler expects {len(input_features)} features, "
            f"but this workflow expects {N_FEATURES}."
        )

    return model, scaler, input_features


def extract_variable_levels(dataset):
    missing = [var for var in PROFILE_VARS if var not in dataset]
    if missing:
        raise KeyError("Missing profile variables: " + ", ".join(missing))

    dataset = dataset.sortby("height")
    heights = dataset["height"].values.astype(np.float32)

    if len(heights) != N_HEIGHTS:
        raise ValueError(f"Expected {N_HEIGHTS} heights; found {len(heights)}.")
    if not np.allclose(heights, TARGET_HEIGHTS):
        raise ValueError("Height coordinate does not match 0-5000 m at 250-m increments.")

    nlat, nlon = dataset.sizes["latitude"], dataset.sizes["longitude"]
    data_array = np.empty((N_FEATURES, nlat, nlon), dtype=np.float32)

    k = 0
    for var_name in PROFILE_VARS:
        values = (
            dataset[var_name]
            .isel(time=0)
            .transpose("height", "latitude", "longitude")
            .values
        )
        data_array[k:k + N_HEIGHTS, ...] = values
        k += N_HEIGHTS

    return data_array.reshape(N_FEATURES, -1).T


def transform_data(input_data, transformer, input_features):
    if input_data.shape[1] != len(input_features):
        raise ValueError(
            f"Input contains {input_data.shape[1]} features, "
            f"but scaler expects {len(input_features)}."
        )
    return transformer.transform(pd.DataFrame(input_data, columns=input_features))


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def run_model_prediction(model, input_data):
    if hasattr(model, "predict"):
        return model.predict(input_data, verbose=0)
    return model(input_data, training=False)


def grid_predictions(data, predictions, valid_mask, output_uncertainties=True):
    nlat, nlon = data.sizes["latitude"], data.sizes["longitude"]
    npoints = nlat * nlon

    if output_uncertainties:
        probabilities_valid = to_numpy(predictions[0])
        uncertainty_valid = to_numpy(predictions[1]).reshape(-1)
        ale_valid = to_numpy(predictions[2])
        epi_valid = to_numpy(predictions[3])
    else:
        probabilities_valid = to_numpy(
            predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        )

    if probabilities_valid.ndim != 2 or probabilities_valid.shape[1] != 4:
        raise ValueError(
            f"Expected probabilities with shape (n_samples, 4); "
            f"got {probabilities_valid.shape}."
        )

    probabilities = np.full((npoints, 4), np.nan, dtype=np.float32)
    probabilities[valid_mask, :] = probabilities_valid.astype(np.float32)

    ptype = np.full(npoints, -1, dtype=np.int8)
    ptype[valid_mask] = np.argmax(probabilities_valid, axis=1).astype(np.int8)

    result = xr.Dataset(
        coords={"latitude": data["latitude"], "longitude": data["longitude"]}
    ).expand_dims(time=data["time"].values)

    for class_id, (long_name, short_name) in enumerate(
        zip(PTYPE_CLASSES, PTYPE_SHORT_NAMES)
    ):
        probability = probabilities[:, class_id].reshape(nlat, nlon)
        categorical = (ptype == class_id).reshape(nlat, nlon).astype(np.uint8)

        result[f"ML_{short_name}"] = (
            ("time", "latitude", "longitude"),
            probability[None, ...],
        )
        result[f"ML_c{short_name}"] = (
            ("time", "latitude", "longitude"),
            categorical[None, ...],
        )

        result[f"ML_{short_name}"].attrs = {
            "long_name": f"machine learned probability of {long_name}",
            "units": "1",
        }
        result[f"ML_c{short_name}"].attrs = {
            "long_name": f"machine learned categorical {long_name}",
            "units": "1",
        }

    result["ML_ptype"] = (
        ("time", "latitude", "longitude"),
        ptype.reshape(1, nlat, nlon),
    )
    result["ML_ptype"].attrs = {
        "long_name": "machine learned precipitation type",
        "flag_values": np.array([-1, 0, 1, 2, 3], dtype=np.int8),
        "flag_meanings": "invalid rain snow ice_pellets freezing_rain",
    }

    if output_uncertainties:
        uncertainty = np.full(npoints, np.nan, dtype=np.float32)
        uncertainty[valid_mask] = uncertainty_valid.astype(np.float32)
        result["ML_u"] = (
            ("time", "latitude", "longitude"),
            uncertainty.reshape(1, nlat, nlon),
        )
        result["ML_u"].attrs = {
            "long_name": "evidential uncertainty",
            "units": "1",
        }

        ale = np.full((npoints, 4), np.nan, dtype=np.float32)
        epi = np.full((npoints, 4), np.nan, dtype=np.float32)
        ale[valid_mask, :] = ale_valid.astype(np.float32)
        epi[valid_mask, :] = epi_valid.astype(np.float32)

        for class_id, short_name in enumerate(PTYPE_SHORT_NAMES):
            result[f"ML_{short_name}_ale"] = (
                ("time", "latitude", "longitude"),
                ale[:, class_id].reshape(1, nlat, nlon),
            )
            result[f"ML_{short_name}_epi"] = (
                ("time", "latitude", "longitude"),
                epi[:, class_id].reshape(1, nlat, nlon),
            )

    return result


def ptype_classification(ds_interp, model, scaler, input_features,
                         output_uncertainties=True):
    input_data = extract_variable_levels(ds_interp)
    valid_mask = np.all(np.isfinite(input_data), axis=1)

    if int(valid_mask.sum()) == 0:
        raise ValueError("No valid profiles available for precipitation-type classification.")

    scaled_input = transform_data(input_data[valid_mask], scaler, input_features)
    predictions = run_model_prediction(model, scaled_input)

    return grid_predictions(
        ds_interp, predictions, valid_mask, output_uncertainties
    )


# ============================================================
# THERMODYNAMICS
# ============================================================

def dewpoint_dataset(ds):
    ds = ds.copy()

    if "t" not in ds:
        raise KeyError("Dataset must contain pressure-level temperature 't'.")

    ds["t_c"] = (ds["t"] - 273.15).astype(np.float32)
    ds["t_c"].attrs = {"long_name": "air temperature", "units": "degC"}

    dpt_values = np.empty(ds["t"].shape, dtype=np.float32)

    for t_idx in range(len(ds["time"].values)):
        if "q" in ds:
            pressure = ds["level"].values * 100.0
            pressure = np.broadcast_to(
                pressure[:, None, None],
                ds["q"].isel(time=t_idx).shape,
            ) * units.pascal
            q = ds["q"].isel(time=t_idx).values * units.dimensionless
            td = mpcalc.dewpoint_from_specific_humidity(
                pressure, q
            ).to("degC")

        elif "r" in ds:
            temperature = ds["t"].isel(time=t_idx).values * units.kelvin
            rh = ds["r"].isel(time=t_idx).values

            if np.nanmax(rh) > 1.5:
                rh = rh / 100.0

            rh = np.clip(rh, 1e-5, 1.0) * units.dimensionless
            td = mpcalc.dewpoint_from_relative_humidity(
                temperature, rh
            ).to("degC")

        else:
            raise KeyError("Dataset must contain either 'q' or 'r'.")

        dpt_values[t_idx, ...] = td.magnitude.astype(np.float32)

    ds["dpt"] = xr.DataArray(
        dpt_values,
        dims=ds["t"].dims,
        coords=ds["t"].coords,
        attrs={"long_name": "dew point temperature", "units": "degC"},
    )

    return ds


def prepare_surface_geopotential(surface_geopotential, ds):
    surf = surface_geopotential.copy()

    rename = {}
    if "lat" in surf.dims or "lat" in surf.coords:
        rename["lat"] = "latitude"
    if "lon" in surf.dims or "lon" in surf.coords:
        rename["lon"] = "longitude"
    if rename:
        surf = surf.rename(rename)

    if "time" in surf.dims and surf.sizes["time"] == 1:
        surf = surf.isel(time=0, drop=True)

    if surf["longitude"].max() > 180:
        surf = surf.assign_coords(
            longitude=((surf["longitude"] + 180) % 360) - 180
        ).sortby("longitude")

    if surf["latitude"][0] > surf["latitude"][-1]:
        surf = surf.sortby("latitude")

    return surf.sel(
        latitude=ds["latitude"],
        longitude=ds["longitude"],
    )


def compute_agl(ds, surface_geopotential, z_var="z", g=9.80665):
    if z_var not in ds:
        raise KeyError(f"Dataset does not contain '{z_var}'.")

    surface_geopotential = prepare_surface_geopotential(
        surface_geopotential, ds
    )

    z_geo = ds[z_var] / g
    z_surface = surface_geopotential / g

    if "time" not in z_surface.dims:
        z_surface = z_surface.expand_dims(time=ds["time"])

    z_agl = (
        (z_geo - z_surface)
        .astype(np.float32)
        .transpose("time", "level", "latitude", "longitude")
    )

    z_agl.name = "height_agl"
    z_agl.attrs = {
        "long_name": "height above ground level",
        "units": "m",
        "positive": "up",
    }

    return z_agl


# ============================================================
# VERTICAL INTERPOLATION -- ORIGINAL NOTEBOOK NUMERICS
# ============================================================

@njit(parallel=True)
def interp_to_heights_numba(var, z_agl, target_heights):
    time, level, lat, lon = var.shape
    out = np.full(
        (time, len(target_heights), lat, lon),
        np.nan,
        dtype=np.float32,
    )

    for t in prange(time):
        for i in prange(lat):
            for j in range(lon):
                heights = z_agl[t, :, i, j]
                values = var[t, :, i, j]

                if heights[0] > heights[-1]:
                    heights = heights[::-1]
                    values = values[::-1]

                out[t, :, i, j] = np.interp(
                    target_heights,
                    heights,
                    values,
                )

    return out


def interpolate_to_heights(ds, z_agl, target_heights, vars_to_interp):
    target_heights = np.asarray(target_heights, dtype=np.float32)
    z_values = (
        z_agl
        .transpose("time", "level", "latitude", "longitude")
        .values
        .astype(np.float32)
    )

    interpolated = {}

    for var_name in vars_to_interp:
        da = ds[var_name].transpose(
            "time", "level", "latitude", "longitude"
        )

        output = interp_to_heights_numba(
            da.values.astype(np.float32),
            z_values,
            target_heights,
        )

        interpolated[var_name] = xr.DataArray(
            output,
            dims=("time", "height", "latitude", "longitude"),
            coords={
                "time": ds["time"],
                "height": target_heights,
                "latitude": ds["latitude"],
                "longitude": ds["longitude"],
            },
            attrs=da.attrs,
        )

    result = xr.Dataset(interpolated)
    result["height"].attrs = {
        "long_name": "height above ground level",
        "units": "m",
        "positive": "up",
    }

    return result


# ============================================================
# BOURGOUIN
# ============================================================

def add_zero_crossings(profile, heights):
    profile = np.asarray(profile, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)

    new_profile = [profile[0]]
    new_heights = [heights[0]]

    for i in range(len(profile) - 1):
        t0, t1 = profile[i], profile[i + 1]
        z0, z1 = heights[i], heights[i + 1]

        if np.isfinite(t0) and np.isfinite(t1) and t0 * t1 < 0:
            if t0 < t1:
                z_cross = np.interp(0.0, [t0, t1], [z0, z1])
            else:
                z_cross = np.interp(0.0, [t1, t0], [z1, z0])

            new_profile.append(0.0)
            new_heights.append(z_cross)

        new_profile.append(t1)
        new_heights.append(z1)

    return np.asarray(new_profile), np.asarray(new_heights)


def calc_sounding_stats(profile, heights):
    freezing_k = 273.15
    gravity = 9.81

    profile, heights = add_zero_crossings(profile, heights)
    surface = profile[0]
    zero_indices = np.where(profile == 0)[0]

    metrics = {
        "cold_area": 0.0,
        "warm_area": 0.0,
        "cold_areas": [],
        "warm_areas": [],
        "cold_thickness": 0.0,
        "warm_thickness": 0.0,
        "min_cold": np.nan,
        "max_warm": np.nan,
        "highest_crossing": np.nan,
        "lowest_crossing": np.nan,
        "surface": surface,
        "n_crossings": len(zero_indices),
    }

    if not len(zero_indices):
        return metrics

    metrics["lowest_crossing"] = heights[zero_indices[0]]
    metrics["highest_crossing"] = heights[zero_indices[-1]]

    upper_bound = zero_indices[-1] + 1
    metrics["min_cold"] = np.nanmin(profile[:upper_bound])
    metrics["max_warm"] = np.nanmax(profile[:upper_bound])

    cold_areas, warm_areas = [], []
    cold_thickness = warm_thickness = 0.0
    low_i = 0

    for crossing_idx in zero_indices:
        if crossing_idx <= low_i:
            low_i = crossing_idx
            continue

        layer_t = profile[low_i:crossing_idx + 1]
        layer_z = heights[low_i:crossing_idx + 1]
        energy = np.trapezoid(gravity * (layer_t / freezing_k), layer_z)
        thickness = heights[crossing_idx] - heights[low_i]

        if energy < 0:
            cold_areas.append(abs(energy))
            cold_thickness += thickness
        elif energy > 0:
            warm_areas.append(energy)
            warm_thickness += thickness

        low_i = crossing_idx

    metrics.update(
        cold_area=float(np.sum(cold_areas)),
        warm_area=float(np.sum(warm_areas)),
        cold_areas=cold_areas,
        warm_areas=warm_areas,
        cold_thickness=cold_thickness,
        warm_thickness=warm_thickness,
    )

    return metrics


def bourgouin_ip_frzr(cold_area, warm_area):
    return 3 if cold_area <= 56.0 + (0.66 * warm_area) else 2


def calc_bourgouin_ptype(t_profile, h_profile):
    metrics = calc_sounding_stats(t_profile, h_profile)

    negative = metrics["cold_area"]
    positive = metrics["warm_area"]
    warm_areas = metrics["warm_areas"]
    surface = metrics["surface"]
    crossings = metrics["n_crossings"]

    if crossings == 0:
        return (0 if surface >= 0 else 1), metrics

    if crossings == 1 and surface > 0:
        return (0 if positive > 13.2 else 1), metrics

    if crossings == 2 and surface <= 0:
        if positive < 2.0:
            return 1, metrics
        return bourgouin_ip_frzr(negative, positive), metrics

    if crossings >= 3:
        last_warm_area = warm_areas[-1] if warm_areas else 0.0
        if last_warm_area < 2.0:
            return (0 if surface >= 0 else 1), metrics
        return bourgouin_ip_frzr(negative, positive), metrics

    return -1, metrics


def bourgouin_classification(ds_interp):
    temp = (
        ds_interp["T_HEIGHT"]
        .isel(time=0)
        .transpose("height", "latitude", "longitude")
        .values
    )

    heights = ds_interp["height"].values
    nlat = ds_interp.sizes["latitude"]
    nlon = ds_interp.sizes["longitude"]

    ptype = np.full((nlat, nlon), -1, dtype=np.int8)
    positive_energy = np.full((nlat, nlon), np.nan, dtype=np.float32)
    negative_energy = np.full((nlat, nlon), np.nan, dtype=np.float32)

    for i in range(nlat):
        for j in range(nlon):
            profile = temp[:, i, j]

            if not np.all(np.isfinite(profile)):
                continue

            cls, metrics = calc_bourgouin_ptype(profile, heights)
            ptype[i, j] = cls
            positive_energy[i, j] = metrics["warm_area"]
            negative_energy[i, j] = metrics["cold_area"]

    coords = {
        "time": ds_interp["time"],
        "latitude": ds_interp["latitude"],
        "longitude": ds_interp["longitude"],
    }

    out = xr.Dataset(coords=coords)
    dims = ("time", "latitude", "longitude")

    out["BOURG_ptype"] = (dims, ptype[None, ...])
    out["BOURG_PA"] = (dims, positive_energy[None, ...])
    out["BOURG_NA"] = (dims, negative_energy[None, ...])

    for class_id, name in enumerate(("rain", "snow", "icep", "frzr")):
        out[f"BOURG_c{name}"] = (
            dims,
            (ptype == class_id)[None, ...].astype(np.uint8),
        )

    out["BOURG_ptype"].attrs = {
        "long_name": "Bourgouin precipitation type",
        "flag_values": np.array([-1, 0, 1, 2, 3], dtype=np.int8),
        "flag_meanings": "invalid rain snow ice_pellets freezing_rain",
    }
    out["BOURG_PA"].attrs = {
        "long_name": "Bourgouin positive melting energy",
        "units": "J kg-1",
    }
    out["BOURG_NA"].attrs = {
        "long_name": "Bourgouin negative refreezing energy magnitude",
        "units": "J kg-1",
    }

    return out


# ============================================================
# GFS INVENTORY / BYTE-RANGE DOWNLOAD
# ============================================================

def gfs_key(init_date, cycle, lead):
    return (
        f"gfs.{init_date}/{cycle}/atmos/"
        f"gfs.t{cycle}z.pgrb2.{RESOLUTION}.f{lead:03d}"
    )


def read_inventory(grib_key):
    text = (
        s3.get_object(Bucket=BUCKET, Key=grib_key + ".idx")["Body"]
        .read()
        .decode()
    )
    size = s3.head_object(
        Bucket=BUCKET,
        Key=grib_key,
    )["ContentLength"]

    records = []

    for line in text.splitlines():
        parts = line.split(":")

        if len(parts) >= 5:
            records.append({
                "start": int(parts[1]),
                "var": parts[3].strip(),
                "level": parts[4].strip(),
                "desc": ":".join(parts[5:]).strip(),
            })

    for i, record in enumerate(records):
        record["end"] = (
            records[i + 1]["start"] - 1
            if i < len(records) - 1
            else size - 1
        )

    return records


def select_pressure(records):
    levels = {f"{p} mb" for p in PLEVELS}

    selected = [
        r for r in records
        if r["var"] in PL_VARS
        and r["level"] in levels
    ]

    found = {(r["var"], r["level"]) for r in selected}

    missing = [
        (var, f"{p} mb")
        for var in PL_VARS
        for p in PLEVELS
        if (var, f"{p} mb") not in found
    ]

    if missing:
        raise RuntimeError(f"Missing GFS pressure fields: {missing}")

    return sorted(selected, key=lambda x: x["start"])


def select_ptype(records):
    selected = []

    for var in PTYPE_VARS:
        candidates = [
            r for r in records
            if r["var"] == var
            and r["level"] == "surface"
            and "ave" not in r["desc"].lower()
        ]

        if not candidates:
            raise RuntimeError(
                f"Could not find instantaneous {var}:surface"
            )

        selected.append(candidates[0])

    return sorted(selected, key=lambda x: x["start"])


def select_surface_height(records):
    candidates = [
        r for r in records
        if r["var"] == "HGT"
        and r["level"] == "surface"
    ]

    if not candidates:
        raise RuntimeError("Could not find HGT:surface")

    return [candidates[0]]


def download_records(grib_key, records, path):
    def fetch(record):
        byte_range = f"bytes={record['start']}-{record['end']}"

        return s3.get_object(
            Bucket=BUCKET,
            Key=grib_key,
            Range=byte_range,
        )["Body"].read()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        chunks = list(pool.map(fetch, records))

    with open(path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)


# ============================================================
# GRID / CFGRIB HELPERS
# ============================================================

def subset_conus(ds):
    lon = ((ds.longitude + 180.0) % 360.0) - 180.0
    ds = ds.assign_coords(longitude=lon).sortby("longitude")

    if float(ds.latitude[0]) > float(ds.latitude[-1]):
        ds = ds.sortby("latitude")

    return ds.sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN, LON_MAX),
    )


def drop_scalar_forecast_coords(ds):
    names = [
        name for name in ("time", "step", "valid_time")
        if name in ds.coords
    ]

    return ds.drop_vars(names) if names else ds


def variable_from_shortname(ds, shortname):
    for name, da in ds.data_vars.items():
        if da.attrs.get("GRIB_shortName", name) == shortname:
            return da

    raise KeyError(
        f"{shortname!r} not found. "
        f"Available variables: {list(ds.data_vars)}"
    )


# ============================================================
# READ GFS FIELDS
# ============================================================

def read_pressure_file(path, init_time):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
        },
    )

    ds = subset_conus(ds)

    if "isobaricInhPa" in ds.dims:
        ds = ds.rename({"isobaricInhPa": "level"})

    t = variable_from_shortname(ds, "t")
    r = variable_from_shortname(ds, "r")
    u = variable_from_shortname(ds, "u")
    v = variable_from_shortname(ds, "v")
    gh = variable_from_shortname(ds, "gh")

    out = xr.Dataset({"t": t, "r": r, "u": u, "v": v})

    # GFS HGT/gh is geopotential height. Convert to geopotential so the
    # original compute_agl routine can be used unchanged.
    out["z"] = (gh * G).astype(np.float32)

    out = out.sortby("level", ascending=False)
    out = drop_scalar_forecast_coords(out)
    out = out.expand_dims(time=[init_time])
    out = out.transpose(
        "time", "level", "latitude", "longitude"
    ).load()

    ds.close()
    return out


def read_surface_height(path):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )

    ds = subset_conus(ds)
    hgt = next(iter(ds.data_vars.values())).load()

    # Convert GFS surface geopotential height to geopotential.
    surface_geopotential = (hgt * G).astype(np.float32)
    surface_geopotential = (
        surface_geopotential
        .reset_coords(drop=True)
        .load()
    )

    ds.close()
    return surface_geopotential


def read_native_ptype(path, init_time):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )

    ds = subset_conus(ds)

    out = xr.Dataset({
        "GFS_crain": variable_from_shortname(ds, "crain").astype(np.uint8),
        "GFS_csnow": variable_from_shortname(ds, "csnow").astype(np.uint8),
        "GFS_cfrzr": variable_from_shortname(ds, "cfrzr").astype(np.uint8),
        "GFS_cicep": variable_from_shortname(ds, "cicep").astype(np.uint8),
    })

    out = drop_scalar_forecast_coords(out)
    out = out.expand_dims(time=[init_time]).load()

    ds.close()
    return add_native_gfs_ptype(out)


def add_native_gfs_ptype(ds):
    flags = (
        ds["GFS_crain"].astype(np.int8)
        + ds["GFS_csnow"].astype(np.int8)
        + ds["GFS_cicep"].astype(np.int8)
        + ds["GFS_cfrzr"].astype(np.int8)
    )

    ptype = xr.full_like(flags, -1, dtype=np.int8)

    # Only assign a 4-class ptype where exactly one native flag is active.
    ptype = xr.where((flags == 1) & (ds["GFS_crain"] == 1), 0, ptype)
    ptype = xr.where((flags == 1) & (ds["GFS_csnow"] == 1), 1, ptype)
    ptype = xr.where((flags == 1) & (ds["GFS_cicep"] == 1), 2, ptype)
    ptype = xr.where((flags == 1) & (ds["GFS_cfrzr"] == 1), 3, ptype)

    ds["GFS_ptype"] = ptype.astype(np.int8)
    ds["GFS_none"] = (flags == 0).astype(np.uint8)
    ds["GFS_overlap"] = (flags > 1).astype(np.uint8)

    ds["GFS_ptype"].attrs = {
        "long_name": "native GFS precipitation type",
        "flag_values": np.array([-1, 0, 1, 2, 3], dtype=np.int8),
        "flag_meanings": "none_or_nonunique rain snow ice_pellets freezing_rain",
    }
    ds["GFS_none"].attrs = {
        "long_name": "no active native GFS precipitation-type flag",
        "units": "1",
    }
    ds["GFS_overlap"].attrs = {
        "long_name": "more than one native GFS precipitation-type flag active",
        "units": "1",
    }

    return ds


# ============================================================
# ONE LEAD: PROFILES + EDNN + BOURGOUIN + NATIVE GFS
# ============================================================

def process_lead(init_date, cycle, init_time, work_dir, lead,
                 surface_geopotential, model, scaler, input_features, logger):
    lead_start = time.perf_counter()
    logger.info("[%s %sZ +%03d] Starting lead", init_date, cycle, lead)

    key = gfs_key(init_date, cycle, int(lead))
    logger.info("[%s %sZ +%03d] Reading GFS inventory", init_date, cycle, lead)
    inventory = read_inventory(key)

    pl_file = work_dir / f"pl_f{lead:03d}.grib2"
    pt_file = work_dir / f"ptype_f{lead:03d}.grib2"

    logger.info("[%s %sZ +%03d] Downloading pressure-level fields", init_date, cycle, lead)
    download_records(key, select_pressure(inventory), pl_file)

    logger.info("[%s %sZ +%03d] Downloading native ptype fields", init_date, cycle, lead)
    download_records(key, select_ptype(inventory), pt_file)

    logger.info("[%s %sZ +%03d] Reading/subsetting GRIB fields", init_date, cycle, lead)
    ds = read_pressure_file(pl_file, init_time)
    native = read_native_ptype(pt_file, init_time)

    logger.info("[%s %sZ +%03d] Computing dewpoint and AGL heights", init_date, cycle, lead)
    ds = dewpoint_dataset(ds)
    z_agl = compute_agl(ds, surface_geopotential, z_var="z")

    logger.info("[%s %sZ +%03d] Interpolating profiles to fixed AGL heights", init_date, cycle, lead)
    profiles = interpolate_to_heights(
        ds, z_agl, TARGET_HEIGHTS, ["t_c", "u", "v", "dpt"]
    ).rename({
        "t_c": "T_HEIGHT",
        "dpt": "DPT_HEIGHT",
        "u": "U_HEIGHT",
        "v": "V_HEIGHT",
    })

    units_map = {
        "T_HEIGHT": "degC",
        "DPT_HEIGHT": "degC",
        "U_HEIGHT": "m s-1",
        "V_HEIGHT": "m s-1",
    }

    for name in PROFILE_VARS:
        profiles[name].attrs.update({
            "long_name": f"{name} on fixed AGL levels",
            "units": units_map[name],
        })

    logger.info("[%s %sZ +%03d] Running EDNN classification", init_date, cycle, lead)
    ml = ptype_classification(
        profiles, model, scaler, input_features, OUTPUT_UNCERTAINTIES
    )

    pieces = [profiles[list(PROFILE_VARS)], ml, native]

    if RUN_BOURGOUIN:
        logger.info("[%s %sZ +%03d] Running Bourgouin classification", init_date, cycle, lead)
        pieces.append(bourgouin_classification(profiles))

    final = xr.merge(pieces, compat="override", join="exact")
    final = final.expand_dims(lead_time=[np.int32(lead)])

    if not KEEP_GRIB:
        pl_file.unlink(missing_ok=True)
        pt_file.unlink(missing_ok=True)

    final = final.load()
    logger.info(
        "[%s %sZ +%03d] Lead complete in %.1f s",
        init_date, cycle, lead, time.perf_counter() - lead_start
    )
    return final


# ============================================================
# ATOMIC FINAL WRITE
# ============================================================

def write_netcdf_atomic(dataset, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".partial")

    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }
        for name in dataset.data_vars
    }

    try:
        dataset.to_netcdf(
            temporary,
            engine="netcdf4",
            encoding=encoding,
        )
        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


# ============================================================
# ONE INITIALIZATION
# ============================================================

def process_gfs_initialization(init_date, cycle, base_dir, model, scaler,
                               input_features, logger, overwrite=False):
    ctx = initialization_context(init_date, cycle, base_dir)
    init_date = ctx["init_date"]
    init_time = ctx["init_time"]
    init_string = ctx["init_string"]
    work_dir = ctx["work_dir"]
    output_dir = ctx["output_dir"]

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_start = time.perf_counter()
    logger.info("=" * 78)
    logger.info("Initialization: %s", init_string)
    logger.info("Output directory: %s", output_dir)

    # Terrain is static for the initialization; retrieve HGT:surface once from f000.
    key0 = gfs_key(init_date, cycle, 0)
    logger.info("[%s] Reading f000 inventory for surface terrain", init_string)
    inventory0 = read_inventory(key0)

    surface_file = work_dir / "surface_hgt.grib2"
    logger.info("[%s] Downloading GFS HGT:surface", init_string)
    download_records(key0, select_surface_height(inventory0), surface_file)
    surface_geopotential = read_surface_height(surface_file)

    if not KEEP_GRIB:
        surface_file.unlink(missing_ok=True)

    saved = {}
    skipped = {}

    for lead in LEADS:
        lead = int(lead)

        # Match the AI-NWP per-lead output convention.
        if lead == 0:
            continue

        destination = output_path(output_dir, init_string, lead)

        if destination.exists() and not overwrite:
            logger.info("[%s +%03d] Output exists; skipping: %s",
                        init_string, lead, destination)
            skipped[lead] = destination
            continue

        try:
            one = process_lead(
                init_date=init_date,
                cycle=cycle,
                init_time=init_time,
                work_dir=work_dir,
                lead=lead,
                surface_geopotential=surface_geopotential,
                model=model,
                scaler=scaler,
                input_features=input_features,
                logger=logger,
            )

            valid_time = init_time + np.timedelta64(lead, "h")
            one = one.assign_coords(
                valid_time=(
                    ("time", "lead_time"),
                    np.array([[valid_time]], dtype="datetime64[ns]"),
                )
            )

            one["lead_time"].attrs = {
                "long_name": "forecast lead time",
                "units": "hours",
            }

            one.attrs.update({
                "model": "GFS",
                "initialization_time": init_string,
                "forecast_lead_hours": lead,
                "horizontal_resolution": "0.25 degree",
                "ptype_model": "EDNN",
                "native_ptype_source": "GFS categorical precipitation-type diagnostics",
                "vertical_profile": "0-5000 m AGL at 250-m intervals",
                "vertical_coordinate": "height above ground level",
                "numerical_variant": "original_notebook",
            })

            assert one.sizes["time"] == 1
            assert one.sizes["lead_time"] == 1
            assert one.sizes["height"] == len(TARGET_HEIGHTS)
            assert np.allclose(one["height"].values, TARGET_HEIGHTS)

            logger.info("[%s +%03d] Writing %s", init_string, lead, destination)
            write_netcdf_atomic(one, destination)
            saved[lead] = destination

        except Exception:
            logger.exception("[%s +%03d] FAILED", init_string, lead)
            raise

    if not KEEP_GRIB:
        shutil.rmtree(work_dir, ignore_errors=True)

    elapsed = time.perf_counter() - init_start
    logger.info(
        "[%s] Finished in %.1f min | written=%d skipped=%d",
        init_string, elapsed / 60.0, len(saved), len(skipped)
    )

    return {"saved": saved, "skipped": skipped}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download GFS forecasts and run EDNN/Bourgouin precipitation-type diagnostics."
    )

    parser.add_argument(
        "--start-date", required=True,
        help="First initialization date, e.g. 2021-11-24"
    )
    parser.add_argument(
        "--end-date", default=None,
        help="Last initialization date, inclusive. Defaults to --start-date."
    )
    parser.add_argument(
        "--cycle", default="00", choices=("00", "06", "12", "18"),
        help="GFS cycle (default: 00)"
    )
    parser.add_argument(
        "--base-dir", type=Path, default=DEFAULT_BASE_DIR,
        help=f"Base output directory (default: {DEFAULT_BASE_DIR})"
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH,
        help="Path to trained EDNN Keras model."
    )
    parser.add_argument(
        "--scaler-path", type=Path, default=DEFAULT_SCALER_PATH,
        help="Path to fitted scaler."
    )
    parser.add_argument(
        "--max-workers", type=int, default=12,
        help="Concurrent S3 byte-range downloads per GRIB file (default: 12)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing per-lead NetCDF files."
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop the entire date-range job after the first failed initialization."
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO)."
    )

    return parser.parse_args()


def main():
    global s3, MAX_WORKERS

    args = parse_args()
    end_date = args.end_date or args.start_date

    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(end_date)

    if end_ts < start_ts:
        raise ValueError("--end-date must be on or after --start-date.")

    dates = pd.date_range(start_ts, end_ts, freq="D")
    MAX_WORKERS = args.max_workers
    s3 = make_s3_client(MAX_WORKERS)

    log_dir = args.base_dir / "logs"
    logger, log_file = setup_logger(
        log_dir=log_dir,
        start_date=start_ts,
        end_date=end_ts,
        cycle=args.cycle,
        level=args.log_level,
    )

    logger.info("GFS precipitation-type batch job starting")
    logger.info("Host: %s", socket.gethostname())
    logger.info("PID: %d", os.getpid())
    logger.info("PBS_JOBID: %s", os.environ.get("PBS_JOBID", "not-running-under-PBS"))
    logger.info("Date range: %s through %s (%d initialization(s))",
                start_ts.strftime("%Y-%m-%d"),
                end_ts.strftime("%Y-%m-%d"),
                len(dates))
    logger.info("Cycle: %sZ", args.cycle)
    logger.info("Forecast leads: +006 through +168 h every 6 h")
    logger.info("Base directory: %s", args.base_dir)
    logger.info("Model: %s", args.model_path)
    logger.info("Scaler: %s", args.scaler_path)
    logger.info("S3 workers: %d", MAX_WORKERS)
    logger.info("Log file: %s", log_file)

    batch_start = time.perf_counter()

    logger.info("Loading EDNN model and scaler once for the full date range")
    model, scaler, input_features = load_saved_model(
        args.model_path, args.scaler_path
    )
    logger.info("EDNN/scaler loaded successfully; %d input features", len(input_features))

    completed = []
    failed = []

    for date in dates:
        date_string = date.strftime("%Y-%m-%d")

        try:
            process_gfs_initialization(
                init_date=date,
                cycle=args.cycle,
                base_dir=args.base_dir,
                model=model,
                scaler=scaler,
                input_features=input_features,
                logger=logger,
                overwrite=args.overwrite,
            )
            completed.append(date_string)

        except Exception:
            failed.append(date_string)
            logger.exception("Initialization %s FAILED", date_string)

            if args.fail_fast:
                logger.error("--fail-fast requested; stopping batch.")
                break

    elapsed = time.perf_counter() - batch_start

    logger.info("=" * 78)
    logger.info("BATCH SUMMARY")
    logger.info("Completed initializations: %d", len(completed))
    logger.info("Failed initializations: %d", len(failed))
    logger.info("Total wall time: %.2f h", elapsed / 3600.0)

    if failed:
        logger.error("Failed dates: %s", ", ".join(failed))
        return 1

    logger.info("All requested initializations completed successfully.")
    return 0


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    raise SystemExit(main())
