import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units


def dewpoint_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    if "t" not in ds:
        raise KeyError("Dataset must contain pressure-level temperature 't'.")
    ds["t_c"] = (ds["t"] - 273.15).astype(np.float32)
    ds["t_c"].attrs = {"long_name": "air temperature", "units": "degC"}
    dpt_values = np.empty(ds["t"].shape, dtype=np.float32)
    for t_idx in range(len(ds["time"].values)):
        if "q" in ds:
            pressure = ds["level"].values * 100.0
            pressure = np.broadcast_to(pressure[:, None, None], ds["q"].isel(time=t_idx).shape) * units.pascal
            q = ds["q"].isel(time=t_idx).values * units.dimensionless
            td = mpcalc.dewpoint_from_specific_humidity(pressure, q).to("degC")
        elif "r" in ds:
            temperature = ds["t"].isel(time=t_idx).values * units.kelvin
            rh = ds["r"].isel(time=t_idx).values
            if np.nanmax(rh) > 1.5:
                rh = rh / 100.0
            rh = np.clip(rh, 1e-5, 1.0) * units.dimensionless
            td = mpcalc.dewpoint_from_relative_humidity(temperature, rh).to("degC")
        else:
            raise KeyError("Dataset must contain either 'q' or 'r'.")
        dpt_values[t_idx, ...] = td.magnitude.astype(np.float32)
    ds["dpt"] = xr.DataArray(
        dpt_values, dims=ds["t"].dims, coords=ds["t"].coords,
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
        surf = surf.assign_coords(longitude=((surf["longitude"] + 180) % 360) - 180).sortby("longitude")
    if surf["latitude"][0] > surf["latitude"][-1]:
        surf = surf.sortby("latitude")
    return surf.sel(latitude=ds["latitude"], longitude=ds["longitude"])


def compute_agl(ds, surface_geopotential, z_var="z", g=9.80665):
    if z_var not in ds:
        raise KeyError(f"Dataset does not contain '{z_var}'.")
    surface_geopotential = prepare_surface_geopotential(surface_geopotential, ds)
    z_geo = ds[z_var] / g
    z_surface = surface_geopotential / g
    if "time" not in z_surface.dims:
        z_surface = z_surface.expand_dims(time=ds["time"])
    z_agl = (z_geo - z_surface).astype(np.float32).transpose("time", "level", "latitude", "longitude")
    z_agl.name = "height_agl"
    z_agl.attrs = {"long_name": "height above ground level", "units": "m", "positive": "up"}
    return z_agl
