import numpy as np
import geopandas as gpd
import xarray as xr

from .get_pixel_center_coords import get_pixel_center_coords

def spatial_subset(ds: xr.Dataset, gdf: gpd.GeoDataFrame):
    """
    Uses a geodataframe containing polygon geometry to clip the GLT of an emit dataset read with emit_xarray, then uses the min/max downtrack and crosstrack
    indices to subset the extent of the dataset in rawspace, masking areas outside the provided spatial geometry. Uses rioxarray's clip function.

    Parameters:
    ds: an emit dataset read into xarray using the emit_xarray function.
    gdf: a geodataframe.

    Returns:
    clipped_ds: an xarray dataset clipped to the extent of the provided geodataframe that can be orthorectified with ortho_xr.
    """
    # Reformat the GLT
    lon, lat = get_pixel_center_coords(ds)
    data_vars = {
        "glt_x": (["latitude", "longitude"], ds.glt_x.data),
        "glt_y": (["latitude", "longitude"], ds.glt_y.data),
    }
    coords = {
        "latitude": (["latitude"], lat),
        "longitude": (["longitude"], lon),
        "ortho_y": (["latitude"], ds.ortho_y.data),
        "ortho_x": (["longitude"], ds.ortho_x.data),
    }
    glt_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
    glt_ds.rio.write_crs(glt_ds.spatial_ref, inplace=True)

    # Clip the emit glt
    clipped = glt_ds.rio.clip(gdf.geometry.values, gdf.crs, all_touched=True)
    # Get the clipped geotransform
    clipped_gt = np.array(
        [float(i) for i in clipped["spatial_ref"].GeoTransform.split(" ")]
    )

    valid_gltx = clipped.glt_x.data > 0
    valid_glty = clipped.glt_y.data > 0
    # Get the subset indices, -1 to convert to 0-based
    subset_down = [
        int(np.min(clipped.glt_y.data[valid_glty]) - 1),
        int(np.max(clipped.glt_y.data[valid_glty]) - 1),
    ]
    subset_cross = [
        int(np.min(clipped.glt_x.data[valid_gltx]) - 1),
        int(np.max(clipped.glt_x.data[valid_gltx]) - 1),
    ]

    # print(subset_down, subset_cross)

    crosstrack_mask = (ds.crosstrack >= subset_cross[0]) & (
        ds.crosstrack <= subset_cross[-1]
    )

    downtrack_mask = (ds.downtrack >= subset_down[0]) & (
        ds.downtrack <= subset_down[-1]
    )

    # Mask Areas outside of crosstrack and downtrack covered by the shape
    clipped_ds = ds.where((crosstrack_mask & downtrack_mask), drop=True)
    # Replace Full dataset geotransform with clipped geotransform
    clipped_ds.attrs["geotransform"] = clipped_gt

    # Drop unnecessary vars from dataset
    clipped_ds = clipped_ds.drop_vars(["glt_x", "glt_y", "downtrack", "crosstrack"])

    # Re-index the GLT to the new array
    glt_x_data = np.maximum(clipped.glt_x.data - subset_cross[0], 0)
    glt_y_data = np.maximum(clipped.glt_y.data - subset_down[0], 0)

    clipped_ds = clipped_ds.assign_coords(
        {
            "glt_x": (["ortho_y", "ortho_x"], np.nan_to_num(glt_x_data)),
            "glt_y": (["ortho_y", "ortho_x"], np.nan_to_num(glt_y_data)),
        }
    )
    clipped_ds = clipped_ds.assign_coords(
        {
            "downtrack": (
                ["downtrack"],
                np.arange(0, clipped_ds[list(ds.data_vars.keys())[0]].shape[0]),
            ),
            "crosstrack": (
                ["crosstrack"],
                np.arange(0, clipped_ds[list(ds.data_vars.keys())[0]].shape[1]),
            ),
        }
    )

    clipped_ds.attrs["subset_downtrack_range"] = subset_down
    clipped_ds.attrs["subset_crosstrack_range"] = subset_cross

    return clipped_ds
