from pathlib import Path
import numpy as np
import xarray as xr

from .bourgouin import bourgouin_classification
from .config import PROFILE_VARS, PipelineConfig
from .ednn import ptype_classification
from .output import is_complete, output_path, write_netcdf_atomic
from .thermodynamics import compute_agl, dewpoint_dataset
from .vertical_profiles import interpolate_to_heights


def lead_time_to_hours(lead):
    value = np.asarray(lead)
    if np.issubdtype(value.dtype, np.timedelta64):
        return int(value / np.timedelta64(1, "h"))
    return int(value)


def get_init_time_string(ds):
    init_time = np.asarray(ds["time"].values).reshape(-1)[0]
    return np.datetime_as_string(np.datetime64(init_time, "h"), unit="h") + "Z"


def load_surface_geopotential(path: Path):
    with xr.open_dataset(path) as source:
        if "Z" in source:
            return source["Z"].load()
        if "z" in source:
            return source["z"].load()
    raise KeyError("Surface geopotential file does not contain 'Z' or 'z'.")


def process_lead(ds_lead, surface_geopotential, model, scaler, input_features,
                 lead_hour, config: PipelineConfig):
    ds = dewpoint_dataset(ds_lead)
    z_agl = compute_agl(ds, surface_geopotential, z_var="z")
    profiles = interpolate_to_heights(
        ds, z_agl, config.target_heights, ["t_c", "u", "v", "dpt"]
    ).rename({"t_c": "T_HEIGHT", "dpt": "DPT_HEIGHT", "u": "U_HEIGHT", "v": "V_HEIGHT"})
    units = {"T_HEIGHT": "degC", "DPT_HEIGHT": "degC", "U_HEIGHT": "m s-1", "V_HEIGHT": "m s-1"}
    for name in PROFILE_VARS:
        profiles[name].attrs.update({"long_name": f"{name} on fixed AGL levels", "units": units[name]})
    profiles = profiles.assign_coords(lead_time=lead_hour)
    profiles["lead_time"].attrs = {"long_name": "forecast lead time", "units": "hours"}
    ml = ptype_classification(profiles, model, scaler, input_features, config.output_uncertainties)
    bourgouin = bourgouin_classification(profiles)
    final = xr.merge([profiles[list(PROFILE_VARS)], ml, bourgouin], compat="override", join="exact")
    final.attrs.update({
        "model": config.model_name,
        "forecast_lead_hours": lead_hour,
        "ptype_model": "EDNN",
        "vertical_profile": "0-5000 m AGL at 250-m intervals",
        "vertical_coordinate": "height above ground level",
        "numerical_variant": "original_notebook",
    })
    return final


def process_initialization(ds_processed, model, scaler, input_features, config):
    init_string = get_init_time_string(ds_processed)
    surface = load_surface_geopotential(config.surface_geopotential_path)
    saved = {}
    for lead in ds_processed["lead_time"].values:
        hour = lead_time_to_hours(lead)
        if hour == 0:
            continue
        destination = output_path(config.output_dir, config.model_name, init_string, hour)
        if not config.overwrite and is_complete(destination, config.output_uncertainties):
            saved[hour] = destination
            continue
        print(f"Processing original numerics: {config.model_name} {init_string} +{hour:03d} h")
        one_lead = ds_processed.sel(lead_time=lead, drop=True)
        final = process_lead(one_lead, surface, model, scaler, input_features, hour, config)
        final.attrs["initialization_time"] = init_string
        write_netcdf_atomic(final, destination)
        saved[hour] = destination
    return saved
