import numpy as np

from .constants import *

# Function to Apply the GLT to an array
def apply_GLT(swath_array: np.ndarray, GLT_array: np.ndarray, fill_value: int = FILL_VALUE, GLT_nodata_value: int = GLT_NODATA_VALUE) -> np.ndarray:
    """
    This function applies the GLT array to a numpy array of either 2 or 3 dimensions.

    Parameters:
    ds_array: numpy array of the desired variable
    glt_array: a GLT array constructed from EMIT GLT data

    Returns:
    out_ds: a numpy array of orthorectified data.
    """

    # Build Output Dataset
    if swath_array.ndim == 2:
        swath_array = swath_array[:, :, np.newaxis]

    # get the length of the latitude dimension from the first dimension of the geometry lookup table array
    latitude_length = GLT_array.shape[0]
    # get the length of the longitude dimension from the second dimension of the geometry lookup table array
    longitude_length = GLT_array.shape[1]
    # get the length of the band dimension from the last dimension of the swath array
    band_length = swath_array.shape[-1]
    # collect the shape of the orthorectified array
    ortho_array_shape = (latitude_length, longitude_length, band_length)

    # create an empty orthorectified array filled with the fill value
    ortho_array = np.full(
        ortho_array_shape,
        fill_value,
        dtype=np.float32,
    )

    # mask of where valid gometry lookup table values are
    valid_GLT = np.all(GLT_array != GLT_nodata_value, axis=-1)

    # convert geometry lookup table indices to zero-based indices for numpy
    zero_based_indices = GLT_array - 1  

    # sample the swath array at the zero-based indices and assign to the orthorectified array
    ortho_array[valid_GLT, :] = swath_array[zero_based_indices[valid_GLT, 1], zero_based_indices[valid_GLT, 0], :]

    return ortho_array
