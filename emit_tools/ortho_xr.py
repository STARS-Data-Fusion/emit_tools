import numpy as np
import xarray as xr

from .apply_glt import apply_glt
from .get_pixel_center_coords import get_pixel_center_coords

def ortho_xr(ds: xr.Dataset, GLT_NODATA_VALUE=0, fill_value=-9999):
    """
    This function uses `apply_glt` to create an orthorectified xarray dataset.

    Parameters:
    ds: an xarray dataset produced by emit_xarray
    GLT_NODATA_VALUE: no data value for the GLT tables, 0 by default
    fill_value: the fill value for EMIT datasets, -9999 by default

    Returns:
    ortho_ds: an orthocorrected xarray dataset.

    """
    # Build glt_ds

    glt_ds = np.nan_to_num(
        np.stack([ds["glt_x"].data, ds["glt_y"].data], axis=-1), nan=GLT_NODATA_VALUE
    ).astype(int)

    # List Variables
    var_list = list(ds.data_vars)

    # Remove flat field from data vars - the flat field is only useful with additional information before orthorectification
    if "flat_field_update" in var_list:
        var_list.remove("flat_field_update")

    # Create empty dictionary for orthocorrected data vars
    data_vars = {}

    # Extract Rawspace Dataset Variable Values (Typically Reflectance)
    for var in var_list:
        raw_ds = ds[var].data
        var_dims = ds[var].dims
        # Apply GLT to dataset
        out_ds = apply_glt(raw_ds, glt_ds, GLT_NODATA_VALUE=GLT_NODATA_VALUE)

        # Update variables - Only works for 2 or 3 dimensional arays
        if raw_ds.ndim == 2:
            out_ds = out_ds.squeeze()
            data_vars[var] = (["latitude", "longitude"], out_ds)
        else:
            data_vars[var] = (["latitude", "longitude", var_dims[-1]], out_ds)

        del raw_ds

    # Calculate Lat and Lon Vectors
    lon, lat = get_pixel_center_coords(
        ds
    )  # Reorder this function to make sense in case of multiple variables

    # Apply GLT to elevation
    elev_ds = apply_glt(ds["elev"].data, glt_ds)

    # Delete glt_ds - no longer needed
    del glt_ds

    # Create Coordinate Dictionary
    coords = {
        "latitude": (["latitude"], lat),
        "longitude": (["longitude"], lon),
        **ds.coords,
    }  # unpack to add appropriate coordinates

    # Remove Unnecessary Coords
    for key in ["downtrack", "crosstrack", "lat", "lon", "glt_x", "glt_y", "elev"]:
        del coords[key]

    # Add Orthocorrected Elevation
    coords["elev"] = (["latitude", "longitude"], np.squeeze(elev_ds))

    # Build Output xarray Dataset and assign data_vars array attributes
    out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)

    del out_ds
    # Assign Attributes from Original Datasets
    for var in var_list:
        out_xr[var].attrs = ds[var].attrs
    out_xr.coords["latitude"].attrs = ds["lat"].attrs
    out_xr.coords["longitude"].attrs = ds["lon"].attrs
    out_xr.coords["elev"].attrs = ds["elev"].attrs

    # Add Spatial Reference in recognizable format
    out_xr.rio.write_crs(ds.spatial_ref, inplace=True)

    return out_xr
