import netCDF4 as nc
import os
from spectral.io import envi
from osgeo import gdal
import numpy as np
import math
from skimage import io
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio as rio
import rioxarray as rxr
import s3fs
from rioxarray.merge import merge_arrays
from fsspec.implementations.http import HTTPFile

from os.path import join, expanduser

from .constants import *
from .ortho_xr import ortho_xr

def emit_xarray(filepath, ortho=False, qmask=None, unpacked_bmask=None, engine: str = ENGINE):
    """
    This function utilizes other functions in this module to streamline opening an EMIT dataset as an xarray.Dataset.

    Parameters:
    filepath: a filepath to an EMIT netCDF file
    ortho: True or False, whether to orthorectify the dataset or leave in crosstrack/downtrack coordinates.
    qmask: a numpy array output from the quality_mask function used to mask pixels based on quality flags selected in that function. Any non-orthorectified array with the proper crosstrack and downtrack dimensions can also be used.
    unpacked_bmask: a numpy array from  the band_mask function that can be used to mask band-specific pixels that have been interpolated.

    Returns:
    out_xr: an xarray.Dataset constructed based on the parameters provided.

    """
    # Grab granule filename to check product

    if type(filepath) == s3fs.core.S3File:
        granule_id = filepath.info()["name"].split("/", -1)[-1].split(".", -1)[0]
    elif type(filepath) == HTTPFile:
        granule_id = filepath.path.split("/", -1)[-1].split(".", -1)[0]
    else:
        granule_id = os.path.splitext(os.path.basename(filepath))[0]

    # Read in Data as Xarray Datasets
    wvl_group = None

    ds = xr.open_dataset(filepath, engine=engine)
    loc = xr.open_dataset(filepath, engine=engine, group="location")

    # Check if mineral dataset and read in groups (only ds/loc for minunc)

    if "L2B_MIN_" in granule_id:
        wvl_group = "mineral_metadata"
    elif "L2B_MINUNC" not in granule_id:
        wvl_group = "sensor_band_parameters"

    wvl = None

    if wvl_group:
        wvl = xr.open_dataset(filepath, engine=engine, group=wvl_group)

    # Building Flat Dataset from Components
    data_vars = {**ds.variables}

    # Format xarray coordinates based upon emit product (no wvl for mineral uncertainty)
    coords = {
        "downtrack": (["downtrack"], ds.downtrack.data),
        "crosstrack": (["crosstrack"], ds.crosstrack.data),
        **loc.variables,
    }

    product_band_map = {
        "L2B_MIN_": "name",
        "L2A_MASK_": "mask_bands",
        "L1B_OBS_": "observation_bands",
        "L2A_RFL_": "wavelengths",
        "L1B_RAD_": "wavelengths",
        "L2A_RFLUNCERT_": "wavelengths",
    }

    # if band := product_band_map.get(next((k for k in product_band_map.keys() if k in granule_id), 'unknown'), None):
    # coords['bands'] = wvl[band].data

    if wvl:
        coords = {**coords, **wvl.variables}

    out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
    out_xr.attrs["granule_id"] = granule_id

    if band := product_band_map.get(
        next((k for k in product_band_map.keys() if k in granule_id), "unknown"), None
    ):
        if "minerals" in list(out_xr.dims):
            out_xr = out_xr.swap_dims({"minerals": band})
            out_xr = out_xr.rename({band: "mineral_name"})
        else:
            out_xr = out_xr.swap_dims({"bands": band})

    # Apply Quality and Band Masks, set fill values to NaN
    for var in list(ds.data_vars):
        if qmask is not None:
            out_xr[var].data[qmask == 1] = -9999
        if unpacked_bmask is not None:
            out_xr[var].data[unpacked_bmask == 1] = -9999

    if ortho is True:
        out_xr = ortho_xr(out_xr)
        out_xr.attrs["Orthorectified"] = "True"

    return out_xr
