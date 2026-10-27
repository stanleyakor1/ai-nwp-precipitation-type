import numpy as np
import xarray as xr


def add_zero_crossings(profile, heights):
    profile = np.asarray(profile, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    new_profile = [profile[0]]
    new_heights = [heights[0]]
    for i in range(len(profile) - 1):
        t0, t1 = profile[i], profile[i + 1]
        z0, z1 = heights[i], heights[i + 1]
        if np.isfinite(t0) and np.isfinite(t1) and t0 * t1 < 0:
            z_cross = np.interp(0.0, [t0, t1], [z0, z1]) if t0 < t1 else np.interp(0.0, [t1, t0], [z1, z0])
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
        "cold_area": 0.0, "warm_area": 0.0, "cold_areas": [], "warm_areas": [],
        "cold_thickness": 0.0, "warm_thickness": 0.0,
        "min_cold": np.nan, "max_warm": np.nan,
        "highest_crossing": np.nan, "lowest_crossing": np.nan,
        "surface": surface, "n_crossings": len(zero_indices),
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
        cold_area=float(np.sum(cold_areas)), warm_area=float(np.sum(warm_areas)),
        cold_areas=cold_areas, warm_areas=warm_areas,
        cold_thickness=cold_thickness, warm_thickness=warm_thickness,
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



def bourgouin_classification(ds_interp):
    temp = ds_interp["T_HEIGHT"].isel(time=0).transpose("height", "latitude", "longitude").values
    heights = ds_interp["height"].values
    nlat, nlon = ds_interp.sizes["latitude"], ds_interp.sizes["longitude"]
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
    coords = {"time": ds_interp["time"], "latitude": ds_interp["latitude"], "longitude": ds_interp["longitude"]}
    out = xr.Dataset(coords=coords)
    dims = ("time", "latitude", "longitude")
    out["BOURG_ptype"] = (dims, ptype[None, ...])
    out["BOURG_PA"] = (dims, positive_energy[None, ...])
    out["BOURG_NA"] = (dims, negative_energy[None, ...])
    for class_id, name in enumerate(("rain", "snow", "icep", "frzr")):
        out[f"BOURG_c{name}"] = (dims, (ptype == class_id)[None, ...].astype(np.uint8))
    out["BOURG_ptype"].attrs = {"long_name": "Bourgouin precipitation type", "flag_values": np.array([0, 1, 2, 3], dtype=np.int8), "flag_meanings": "rain snow ice_pellets freezing_rain"}
    out["BOURG_PA"].attrs = {"long_name": "Bourgouin positive melting energy", "units": "J kg-1"}
    out["BOURG_NA"].attrs = {"long_name": "Bourgouin negative refreezing energy magnitude", "units": "J kg-1"}
    if "lead_time" in ds_interp.coords:
        out = out.assign_coords(lead_time=ds_interp["lead_time"])
    return out
