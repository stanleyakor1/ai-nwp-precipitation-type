"""Notebook EDNN inference, reorganized without changing feature numerics."""

import numpy as np
import pandas as pd
import xarray as xr

from .config import EXPECTED_HEIGHTS, N_FEATURES, N_HEIGHTS, PROFILE_VARS, PTYPE_CLASSES, PTYPE_SHORT_NAMES


def load_saved_model(model_path, scaler_path):
    from keras.models import load_model
    from mlguess.keras.models import CategoricalDNN
    from mlguess.keras.losses import evidential_cat_loss
    from bridgescaler.backend import load_scaler
    model = load_model(
        model_path,
        custom_objects={"CategoricalDNN": CategoricalDNN, "evidential_cat_loss": evidential_cat_loss},
        compile=False,
    )
    scaler = load_scaler(scaler_path)
    input_features = [feature for group in scaler.groups_ for feature in group]
    if len(input_features) != N_FEATURES:
        raise ValueError(f"Scaler expects {len(input_features)} features, but this workflow expects {N_FEATURES}.")
    return model, scaler, input_features


def extract_variable_levels(dataset: xr.Dataset) -> np.ndarray:
    missing = [var for var in PROFILE_VARS if var not in dataset]
    if missing:
        raise KeyError("Missing profile variables: " + ", ".join(missing))
    dataset = dataset.sortby("height")
    heights = dataset["height"].values.astype(np.float32)
    if len(heights) != N_HEIGHTS:
        raise ValueError(f"Expected {N_HEIGHTS} heights; found {len(heights)}.")
    if not np.allclose(heights, EXPECTED_HEIGHTS):
        raise ValueError("Height coordinate does not match 0-5000 m at 250-m increments.")
    nlat, nlon = dataset.sizes["latitude"], dataset.sizes["longitude"]
    data_array = np.empty((N_FEATURES, nlat, nlon), dtype=np.float32)
    k = 0
    for var_name in PROFILE_VARS:
        values = dataset[var_name].isel(time=0).transpose("height", "latitude", "longitude").values
        data_array[k:k + N_HEIGHTS, ...] = values
        k += N_HEIGHTS
    return data_array.reshape(N_FEATURES, -1).T


def transform_data(input_data, transformer, input_features):
    if input_data.shape[1] != len(input_features):
        raise ValueError(f"Input contains {input_data.shape[1]} features, but scaler expects {len(input_features)}.")
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
        probabilities_valid = to_numpy(predictions[0] if isinstance(predictions, (list, tuple)) else predictions)
    if probabilities_valid.ndim != 2 or probabilities_valid.shape[1] != 4:
        raise ValueError(f"Expected probabilities with shape (n_samples, 4); got {probabilities_valid.shape}.")
    probabilities = np.full((npoints, 4), np.nan, dtype=np.float32)
    probabilities[valid_mask, :] = probabilities_valid.astype(np.float32)
    ptype = np.full(npoints, -1, dtype=np.int8)
    ptype[valid_mask] = np.argmax(probabilities_valid, axis=1).astype(np.int8)
    result = xr.Dataset(coords={"latitude": data["latitude"], "longitude": data["longitude"]})
    result = result.expand_dims(time=data["time"].values)
    if "lead_time" in data.coords:
        result = result.assign_coords(lead_time=data["lead_time"])
    for class_id, (long_name, short_name) in enumerate(zip(PTYPE_CLASSES, PTYPE_SHORT_NAMES)):
        probability = probabilities[:, class_id].reshape(nlat, nlon)
        categorical = (ptype == class_id).reshape(nlat, nlon).astype(np.uint8)
        result[f"ML_{short_name}"] = (("time", "latitude", "longitude"), probability[None, ...])
        result[f"ML_c{short_name}"] = (("time", "latitude", "longitude"), categorical[None, ...])
        result[f"ML_{short_name}"].attrs = {"long_name": f"machine learned probability of {long_name}", "units": "1"}
        result[f"ML_c{short_name}"].attrs = {"long_name": f"machine learned categorical {long_name}", "units": "1"}
    result["ML_ptype"] = (("time", "latitude", "longitude"), ptype.reshape(1, nlat, nlon))
    result["ML_ptype"].attrs = {
        "long_name": "machine learned precipitation type",
        "flag_values": np.array([-1, 0, 1, 2, 3], dtype=np.int8),
        "flag_meanings": "invalid rain snow ice_pellets freezing_rain",
    }
    if output_uncertainties:
        uncertainty = np.full(npoints, np.nan, dtype=np.float32)
        uncertainty[valid_mask] = uncertainty_valid.astype(np.float32)
        result["ML_u"] = (("time", "latitude", "longitude"), uncertainty.reshape(1, nlat, nlon))
        result["ML_u"].attrs = {"long_name": "evidential uncertainty", "units": "1"}
        ale = np.full((npoints, 4), np.nan, dtype=np.float32)
        epi = np.full((npoints, 4), np.nan, dtype=np.float32)
        ale[valid_mask, :] = ale_valid.astype(np.float32)
        epi[valid_mask, :] = epi_valid.astype(np.float32)
        for class_id, short_name in enumerate(PTYPE_SHORT_NAMES):
            result[f"ML_{short_name}_ale"] = (("time", "latitude", "longitude"), ale[:, class_id].reshape(1, nlat, nlon))
            result[f"ML_{short_name}_epi"] = (("time", "latitude", "longitude"), epi[:, class_id].reshape(1, nlat, nlon))
    return result


def ptype_classification(ds_interp, model, scaler, input_features, output_uncertainties=True):
    input_data = extract_variable_levels(ds_interp)
    valid_mask = np.all(np.isfinite(input_data), axis=1)
    if int(valid_mask.sum()) == 0:
        raise ValueError("No valid profiles available for precipitation-type classification.")
    scaled_input = transform_data(input_data[valid_mask], scaler, input_features)
    predictions = run_model_prediction(model, scaled_input)
    return grid_predictions(ds_interp, predictions, valid_mask, output_uncertainties)
