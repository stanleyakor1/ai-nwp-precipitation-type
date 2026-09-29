"""Output utilities for the original-numerics experiment."""

import os
from pathlib import Path
import xarray as xr

from .config import required_output_vars


def output_path(parent, model_name, init_string, lead_hour):
    return Path(parent) / init_string / f"{model_name}_{init_string}_{lead_hour:03d}.nc"


def is_complete(path, output_uncertainties=True):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path) as dataset:
            return not (required_output_vars(output_uncertainties) - set(dataset.data_vars))
    except (OSError, ValueError):
        return False


def write_netcdf_atomic(dataset, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        dataset.to_netcdf(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
