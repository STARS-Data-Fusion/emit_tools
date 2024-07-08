from typing import Dict

import numpy as np
import geopandas as gpd
import xarray as xr

def merge_emit(datasets: Dict, gdf: gpd.GeoDataFrame):
    """
    A function to merge xarray datasets formatted using emit_xarray. This could probably be improved,
    lots of shuffling data around to keep in xarray and get it to merge properly. Note: GDF may only work with a
    single geometry.
    """
    nested_data_arrays = {}
    # loop over datasets
    for dataset in datasets:
        # create dictionary of arrays for each dataset

        # create dictionary of 1D variables, which should be consistent across datasets
        one_d_arrays = {}

        # Dictionary of variables to merge
        data_arrays = {}
        # Loop over variables in dataset including elevation
        for var in list(datasets[dataset].data_vars) + ["elev"]:
            # Get 1D for this variable and add to dictionary
            if not one_d_arrays:
                # These should be an array describing the others (wavelengths, mask_bands, etc.)
                one_dim = [
                    item
                    for item in list(datasets[dataset].coords)
                    if item not in ["latitude", "longitude", "spatial_ref"]
                    and len(datasets[dataset][item].dims) == 1
                ]
                # print(one_dim)
                for od in one_dim:
                    one_d_arrays[od] = datasets[dataset].coords[od].data

                # Update format for merging - This could probably be improved
            da = datasets[dataset][var].reset_coords("elev", drop=False)
            da = da.rename({"latitude": "y", "longitude": "x"})
            if len(da.dims) == 3:
                if any(item in list(da.coords) for item in one_dim):
                    da = da.drop_vars(one_dim)
                da = da.drop_vars("elev")
                da = da.to_array(name=var).squeeze("variable", drop=True)
                da = da.transpose(da.dims[-1], da.dims[0], da.dims[1])
                # print(da.dims)
            if var == "elev":
                da = da.to_array(name=var).squeeze("variable", drop=True)
            data_arrays[var] = da
            nested_data_arrays[dataset] = data_arrays

            # Transpose the nested arrays dict. This is horrible to read, but works to pair up variables (ie mask) from the different granules
    transposed_dict = {
        inner_key: {
            outer_key: inner_dict[inner_key]
            for outer_key, inner_dict in nested_data_arrays.items()
        }
        for inner_key in nested_data_arrays[next(iter(nested_data_arrays))]
    }

    # remove some unused data
    del nested_data_arrays, data_arrays, da

    # Merge the arrays using rioxarray.merge_arrays()
    merged = {}
    for _var in transposed_dict:
        merged[_var] = merge_arrays(
            list(transposed_dict[_var].values()),
            bounds=gdf.unary_union.bounds,
            nodata=-9999,
        )

    # Create a new xarray dataset from the merged arrays
    # Create Merged Dataset
    merged_ds = xr.Dataset(data_vars=merged, coords=one_d_arrays)
    # Rename x and y to longitude and latitude
    merged_ds = merged_ds.rename({"y": "latitude", "x": "longitude"})
    del transposed_dict, merged
    return merged_ds
