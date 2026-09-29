import re
import xarray as xr


class AINWPPreprocessor:
    def __init__(self, pressure_vars=("t", "z", "u", "v", "w", "q"),
                 lat_bounds=(24.0, 50.0), lon_bounds=(-125.0, -66.0)):
        self.pressure_vars = pressure_vars
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.variable_metadata = {
            "t": {"long_name": "air temperature", "units": "K"},
            "z": {"long_name": "geopotential", "units": "m2 s-2"},
            "u": {"long_name": "eastward wind", "units": "m s-1"},
            "v": {"long_name": "northward wind", "units": "m s-1"},
            "w": {"long_name": "pressure vertical velocity", "units": "Pa s-1"},
            "q": {"long_name": "specific humidity", "units": "kg kg-1"},
            "t2m": {"long_name": "2 metre temperature", "units": "K"},
            "u10m": {"long_name": "10 metre eastward wind", "units": "m s-1"},
            "v10m": {"long_name": "10 metre northward wind", "units": "m s-1"},
            "msl": {"long_name": "mean sea level pressure", "units": "Pa"},
            "tp06": {"long_name": "6-hour accumulated total precipitation", "units": "m", "accumulation_period": "6 hours"},
        }

    def subset_conus(self, ds):
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        if ds[lon_name].max() > 180:
            ds = ds.assign_coords({lon_name: ((ds[lon_name] + 180) % 360) - 180})
            ds = ds.sortby(lon_name)
        if ds[lat_name][0] > ds[lat_name][-1]:
            ds = ds.sortby(lat_name)
        return ds.sel({lat_name: slice(*self.lat_bounds), lon_name: slice(*self.lon_bounds)})

    def get_levels(self, ds, var):
        pattern = re.compile(rf"^{re.escape(var)}(\d+)$")
        levels = [int(match.group(1)) for name in ds.data_vars if (match := pattern.match(name))]
        return sorted(levels, reverse=True)

    def stack_levels(self, ds, var):
        levels = self.get_levels(ds, var)
        if not levels:
            raise ValueError(f"No pressure levels found for '{var}'")
        da = xr.concat([ds[f"{var}{level}"].expand_dims(level=[level]) for level in levels], dim="level")
        da["level"].attrs = {"long_name": "pressure level", "standard_name": "air_pressure", "units": "hPa", "positive": "down", "axis": "Z"}
        return da

    def rename_spatial_dims(self, ds):
        rename = {}
        if "lat" in ds.dims or "lat" in ds.coords:
            rename["lat"] = "latitude"
        if "lon" in ds.dims or "lon" in ds.coords:
            rename["lon"] = "longitude"
        return ds.rename(rename) if rename else ds

    def add_metadata(self, ds):
        for var, attrs in self.variable_metadata.items():
            if var in ds.data_vars:
                ds[var].attrs.update(attrs)
        attrs = {
            "latitude": {"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north", "axis": "Y"},
            "longitude": {"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east", "axis": "X"},
            "level": {"standard_name": "air_pressure", "long_name": "pressure level", "units": "hPa", "positive": "down", "axis": "Z"},
            "lead_time": {"long_name": "forecast lead time"},
        }
        for name, values in attrs.items():
            if name in ds.coords:
                ds[name].attrs.update(values)
        return ds

    def process(self, ds):
        ds = self.rename_spatial_dims(self.subset_conus(ds))
        out = ds.copy()
        for var in self.pressure_vars:
            levels = self.get_levels(ds, var)
            if not levels:
                continue
            out[var] = self.stack_levels(ds, var).transpose("time", "lead_time", "level", "latitude", "longitude")
            out = out.drop_vars([f"{var}{level}" for level in levels])
        return self.add_metadata(out)
