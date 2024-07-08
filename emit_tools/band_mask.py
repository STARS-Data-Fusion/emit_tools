import numpy as np
import xarray as xr

from .constants import *

def band_mask(filepath, engine: str = ENGINE):
    """
    This function unpacks the packed band mask to apply to the dataset. Can be used manually or as an input in the emit_xarray() function.

    Parameters:
    filepath: an EMIT L2A Mask netCDF file.
    packed_bands: the 'packed_bands' dataset from the EMIT L2A Mask file.

    Returns:
    band_mask: a numpy array that can be used with the emit_xarray function to apply a band mask.
    """
    # Open Dataset
    mask_ds = xr.open_dataset(filepath, engine=ENGINE)
    # Open band_mask and convert to uint8
    bmask = mask_ds.band_mask.data.astype("uint8")
    # Print Flags used
    unpacked_bmask = np.unpackbits(bmask, axis=-1)
    # Remove bands > 285
    unpacked_bmask = unpacked_bmask[:, :, 0:285]
    # Check for data bands and build mask
    return unpacked_bmask
