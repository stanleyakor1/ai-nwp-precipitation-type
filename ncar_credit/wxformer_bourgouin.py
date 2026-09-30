import os
import re
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# CONFIG
# ============================================================

INPUT_ROOT = Path("/glade/derecho/scratch/sakor/bulk/wxformer")
OUTPUT_ROOT = Path(
    "/glade/derecho/scratch/sakor/bulk/era5_init/ML_PTYPE/wx_ptype"
)

PROFILE_VARS = (
    "T_HEIGHT",
    "DPT_HEIGHT",
    "U_HEIGHT",
    "V_HEIGHT",
)

EXPECTED_HEIGHTS = np.arange(0, 5001, 250, dtype=np.float32)

LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = -125.0, -66.0


# ============================================================
# LOGGING
# ============================================================

def setup_logger():
    logger = logging.getLogger("wxformer_bourgouin")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    return logger


logger = setup_logger()


# ============================================================
# CONUS SUBSET
# ============================================================

def subset_conus(ds):
    # 0...360 -> -180...180
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(
            longitude=((ds.longitude + 180.0) % 360.0) - 180.0
        )

    ds = ds.sortby("longitude")

    # WXFormer latitude is north -> south
    if float(ds.latitude[0]) > float(ds.latitude[-1]):
        ds = ds.sortby("latitude")

    return ds.sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN, LON_MAX),
    )


# ============================================================
# HEIGHT COORDINATE
# ============================================================

def standardize_height(ds):
    if "height" in ds.coords:
        height_name = "height"

    elif "height_agl" in ds.coords:
        ds = ds.rename({"height_agl": "height"})
        height_name = "height"

    else:
        raise KeyError(
            "Could not find 'height' or 'height_agl' coordinate."
        )

    if not np.allclose(
        ds[height_name].values,
        EXPECTED_HEIGHTS,
    ):
        raise ValueError(
            "Unexpected height coordinate:\n"
            f"{ds[height_name].values}"
        )

    ds["height"].attrs = {
        "long_name": "height above ground level",
        "units": "m",
        "positive": "up",
    }

    return ds


def add_ml_ptype(ds):
    """
    Construct a single EDNN ptype field.

    Classes:
        -1 = invalid / non-unique
         0 = rain
         1 = snow
         2 = ice pellets
         3 = freezing rain
    """

    required = {
        "ML_crain",
        "ML_csnow",
        "ML_cicep",
        "ML_cfrzr",
    }

    missing = required - set(ds.data_vars)

    if missing:
        raise KeyError(
            f"Cannot construct ML_ptype. Missing: {sorted(missing)}"
        )

    flags = (
        ds["ML_crain"].astype(np.int8)
        + ds["ML_csnow"].astype(np.int8)
        + ds["ML_cicep"].astype(np.int8)
        + ds["ML_cfrzr"].astype(np.int8)
    )

    ptype = xr.full_like(flags, -1, dtype=np.int8)

    ptype = xr.where(
        (flags == 1) & (ds["ML_crain"] == 1), 0, ptype
    )
    ptype = xr.where(
        (flags == 1) & (ds["ML_csnow"] == 1), 1, ptype
    )
    ptype = xr.where(
        (flags == 1) & (ds["ML_cicep"] == 1), 2, ptype
    )
    ptype = xr.where(
        (flags == 1) & (ds["ML_cfrzr"] == 1), 3, ptype
    )

    ds["ML_ptype"] = ptype

    ds["ML_ptype"].attrs = {
        "long_name": "machine learned precipitation type",
        "flag_values": np.array([-1, 0, 1, 2, 3], dtype=np.int8),
        "flag_meanings":
            "invalid rain snow ice_pellets freezing_rain",
    }

    return ds
    
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
                z_cross = np.interp(
                    0.0, [t0, t1], [z0, z1]
                )
            else:
                z_cross = np.interp(
                    0.0, [t1, t0], [z1, z0]
                )

            new_profile.append(0.0)
            new_heights.append(z_cross)

        new_profile.append(t1)
        new_heights.append(z1)

    return (
        np.asarray(new_profile),
        np.asarray(new_heights),
    )


def calc_sounding_stats(profile, heights):
    freezing_k = 273.15
    gravity = 9.81

    profile, heights = add_zero_crossings(
        profile, heights
    )

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

    metrics["lowest_crossing"] = heights[
        zero_indices[0]
    ]
    metrics["highest_crossing"] = heights[
        zero_indices[-1]
    ]

    upper_bound = zero_indices[-1] + 1

    metrics["min_cold"] = np.nanmin(
        profile[:upper_bound]
    )
    metrics["max_warm"] = np.nanmax(
        profile[:upper_bound]
    )

    cold_areas = []
    warm_areas = []

    cold_thickness = 0.0
    warm_thickness = 0.0

    low_i = 0

    for crossing_idx in zero_indices:

        if crossing_idx <= low_i:
            low_i = crossing_idx
            continue

        layer_t = profile[
            low_i:crossing_idx + 1
        ]

        layer_z = heights[
            low_i:crossing_idx + 1
        ]

        energy = np.trapezoid(
            gravity * (layer_t / freezing_k),
            layer_z,
        )

        thickness = (
            heights[crossing_idx]
            - heights[low_i]
        )

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
    return (
        3
        if cold_area <= 56.0 + 0.66 * warm_area
        else 2
    )


def calc_bourgouin_ptype(t_profile, h_profile):
    metrics = calc_sounding_stats(
        t_profile, h_profile
    )

    negative = metrics["cold_area"]
    positive = metrics["warm_area"]
    warm_areas = metrics["warm_areas"]
    surface = metrics["surface"]
    crossings = metrics["n_crossings"]

    if crossings == 0:
        return (
            0 if surface >= 0 else 1
        ), metrics

    if crossings == 1 and surface > 0:
        return (
            0 if positive > 13.2 else 1
        ), metrics

    if crossings == 2 and surface <= 0:

        if positive < 2.0:
            return 1, metrics

        return (
            bourgouin_ip_frzr(
                negative, positive
            ),
            metrics,
        )

    if crossings >= 3:

        last_warm_area = (
            warm_areas[-1]
            if warm_areas
            else 0.0
        )

        if last_warm_area < 2.0:
            return (
                0 if surface >= 0 else 1
            ), metrics

        return (
            bourgouin_ip_frzr(
                negative, positive
            ),
            metrics,
        )

    return -1, metrics


def bourgouin_classification(ds):
    temp = (
        ds["T_HEIGHT"]
        .isel(time=0)
        .transpose(
            "height",
            "latitude",
            "longitude",
        )
        .values
    )

    heights = ds["height"].values

    nlat = ds.sizes["latitude"]
    nlon = ds.sizes["longitude"]

    ptype = np.full(
        (nlat, nlon),
        -1,
        dtype=np.int8,
    )

    positive_energy = np.full(
        (nlat, nlon),
        np.nan,
        dtype=np.float32,
    )

    negative_energy = np.full(
        (nlat, nlon),
        np.nan,
        dtype=np.float32,
    )

    for i in range(nlat):
        for j in range(nlon):

            profile = temp[:, i, j]

            if not np.all(
                np.isfinite(profile)
            ):
                continue

            cls, metrics = (
                calc_bourgouin_ptype(
                    profile,
                    heights,
                )
            )

            ptype[i, j] = cls

            positive_energy[i, j] = (
                metrics["warm_area"]
            )

            negative_energy[i, j] = (
                metrics["cold_area"]
            )

    coords = {
        "time": ds["time"],
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
    }

    dims = (
        "time",
        "latitude",
        "longitude",
    )

    out = xr.Dataset(coords=coords)

    out["BOURG_ptype"] = (
        dims,
        ptype[None, ...],
    )

    out["BOURG_PA"] = (
        dims,
        positive_energy[None, ...],
    )

    out["BOURG_NA"] = (
        dims,
        negative_energy[None, ...],
    )

    for class_id, name in enumerate(
        ("rain", "snow", "icep", "frzr")
    ):
        out[f"BOURG_c{name}"] = (
            dims,
            (
                ptype == class_id
            )[None, ...].astype(np.uint8),
        )

    out["BOURG_ptype"].attrs = {
        "long_name":
            "Bourgouin precipitation type",
        "flag_values":
            np.array(
                [-1, 0, 1, 2, 3],
                dtype=np.int8,
            ),
        "flag_meanings":
            "invalid rain snow "
            "ice_pellets freezing_rain",
    }

    out["BOURG_PA"].attrs = {
        "long_name":
            "Bourgouin positive melting energy",
        "units": "J kg-1",
    }

    out["BOURG_NA"].attrs = {
        "long_name":
            "Bourgouin negative refreezing energy magnitude",
        "units": "J kg-1",
    }

    return out


# ============================================================
# FILE HELPERS
# ============================================================

def get_lead_hour(path):
    match = re.search(
        r"_(\d{3})\.nc$",
        path.name,
    )

    if not match:
        raise ValueError(
            f"Cannot determine lead from {path}"
        )

    return int(match.group(1))


def output_path(init_string, lead):
    return (
        OUTPUT_ROOT
        / init_string
        / f"wxformer_{init_string}_{lead:03d}.nc"
    )


def is_complete(path):
    if not path.is_file():
        return False

    required = {
        "T_HEIGHT",
        "DPT_HEIGHT",
        "U_HEIGHT",
        "V_HEIGHT",
        "ML_ptype",
        "BOURG_ptype",
    }

    try:
        with xr.open_dataset(path) as ds:
            missing = required - set(ds.data_vars)
            return len(missing) == 0

    except (OSError, ValueError):
        return False


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    source_path,
    init_string,
    overwrite=False,
):
    lead = get_lead_hour(source_path)

    destination = output_path(
        init_string, lead
    )

    if not overwrite and is_complete(destination):
        logger.info(
            "%s +%03d already complete; skipping",
            init_string,
            lead,
        )
        return destination

    start = time.perf_counter()

    logger.info(
        "%s +%03d reading %s",
        init_string,
        lead,
        source_path,
    )

    # --------------------------------------------------------
    # Open lazily -- DO NOT load the 1-GB global dataset
    # --------------------------------------------------------

    with xr.open_dataset(source_path) as source:

        required = {
            "T_HEIGHT",
            "time",
            "latitude",
            "longitude",
        }

        missing = required - set(source.variables)

        if missing:
            raise KeyError(
                f"{source_path} missing: "
                f"{sorted(missing)}"
            )

        # Keep only the compact fields needed in ML_PTYPE.
        keep_vars = [
            var
            for var in PROFILE_VARS
            if var in source.data_vars
        ]

        keep_vars += [
            var
            for var in source.data_vars
            if var.startswith("ML_")
        ]

        ds = source[keep_vars]

        # height_agl -> height
        ds = standardize_height(ds)

        # Subset BEFORE .load()
        ds = subset_conus(ds)

        logger.info(
            "%s +%03d CONUS grid: %d x %d",
            init_string,
            lead,
            ds.sizes["latitude"],
            ds.sizes["longitude"],
        )

        ds = ds.load()
        ds = add_ml_ptype(ds)

    # --------------------------------------------------------
    # Bourgouin
    # --------------------------------------------------------

    logger.info(
        "%s +%03d running Bourgouin",
        init_string,
        lead,
    )

    bourg = bourgouin_classification(ds)

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    final = xr.merge(
        [ds, bourg],
        compat="override",
        join="exact",
    )

    # Add lead as scalar coordinate
    final = final.assign_coords(
        lead_time=np.int32(lead)
    )

    final["lead_time"].attrs = {
        "long_name": "forecast lead time",
        "units": "hours",
    }

    final.attrs.update({
        "model": "WXFormer",
        "initialization_time": init_string,
        "forecast_lead_hours": lead,
        "domain": "CONUS",
        "ptype_model": "EDNN",
        "bourgouin_diagnostic": "enabled",
        "vertical_profile":
            "0-5000 m AGL at 250-m intervals",
    })

    # --------------------------------------------------------
    # Atomic write
    # --------------------------------------------------------

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        ".nc.partial"
    )

    encoding = {
        var: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }
        for var in final.data_vars
    }

    try:
        final.to_netcdf(
            temporary,
            engine="netcdf4",
            encoding=encoding,
        )

        os.replace(
            temporary,
            destination,
        )

    finally:
        if temporary.exists():
            temporary.unlink()

    logger.info(
        "%s +%03d saved in %.1f s -> %s",
        init_string,
        lead,
        time.perf_counter() - start,
        destination,
    )

    return destination


# ============================================================
# PROCESS DATE RANGE
# ============================================================

def process_archive(
    start_date,
    end_date,
    overwrite=False,
):
    dates = pd.date_range(
        start_date,
        end_date,
        freq="D",
    )

    completed = 0
    skipped_missing = 0
    failed = 0

    for date in dates:

        init_string = (
            date.strftime("%Y-%m-%d")
            + "T00Z"
        )

        source_dir = (
            INPUT_ROOT
            / init_string
        )

        if not source_dir.is_dir():

            logger.warning(
                "Missing initialization: %s",
                source_dir,
            )

            skipped_missing += 1
            continue

        files = sorted(
            source_dir.glob("pred_*.nc")
        )

        logger.info(
            "%s: found %d files",
            init_string,
            len(files),
        )

        for source_path in files:

            try:
                process_file(
                    source_path,
                    init_string,
                    overwrite=overwrite,
                )

                completed += 1

            except Exception:

                failed += 1

                logger.exception(
                    "FAILED: %s",
                    source_path,
                )

    logger.info(
        "Finished | files=%d "
        "missing_init=%d failed=%d",
        completed,
        skipped_missing,
        failed,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-date",
        default="2021-11-24",
    )

    parser.add_argument(
        "--end-date",
        default="2022-04-01",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    logger.info("WXFormer Bourgouin processing started")
    logger.info("INPUT_ROOT  = %s", INPUT_ROOT)
    logger.info("OUTPUT_ROOT = %s", OUTPUT_ROOT)
    logger.info(
        "Date range   = %s through %s",
        args.start_date,
        args.end_date,
    )

    process_archive(
        start_date=args.start_date,
        end_date=args.end_date,
        overwrite=args.overwrite,
    )

    logger.info("WXFormer Bourgouin processing finished")
