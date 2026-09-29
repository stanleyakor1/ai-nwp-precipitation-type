import numpy as np
import xarray as xr
from numba import njit, prange


@njit(parallel=True)
def interp_to_heights_numba(var, z_agl, target_heights):
    time, level, lat, lon = var.shape
    out = np.full((time, len(target_heights), lat, lon), np.nan, dtype=np.float32)
    for t in prange(time):
        for i in prange(lat):
            for j in range(lon):
                heights = z_agl[t, :, i, j]
                values = var[t, :, i, j]
                if heights[0] > heights[-1]:
                    heights = heights[::-1]
                    values = values[::-1]
                out[t, :, i, j] = np.interp(target_heights, heights, values)
    return out


def interpolate_to_heights(ds, z_agl, target_heights, vars_to_interp):
    target_heights = np.asarray(target_heights, dtype=np.float32)
    z_values = z_agl.transpose("time", "level", "latitude", "longitude").values.astype(np.float32)
    interpolated = {}
    for var_name in vars_to_interp:
        da = ds[var_name].transpose("time", "level", "latitude", "longitude")
        output = interp_to_heights_numba(da.values.astype(np.float32), z_values, target_heights)
        interpolated[var_name] = xr.DataArray(
            output, dims=("time", "height", "latitude", "longitude"),
            coords={"time": ds["time"], "height": target_heights, "latitude": ds["latitude"], "longitude": ds["longitude"]},
            attrs=da.attrs,
        )
    result = xr.Dataset(interpolated)
    result["height"].attrs = {"long_name": "height above ground level", "units": "m", "positive": "up"}
    return result
