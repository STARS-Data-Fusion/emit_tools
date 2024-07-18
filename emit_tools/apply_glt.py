import numpy as np

from .constants import *

# Function to Apply the GLT to an array
def apply_GLT(swath_array: np.ndarray, GLT_array: np.ndarray, fill_value: int = FILL_VALUE, GLT_nodata_value: int = GLT_NODATA_VALUE):
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
    out_ds = np.full(
        (GLT_array.shape[0], GLT_array.shape[1], swath_array.shape[-1]),
        fill_value,
        dtype=np.float32,
    )

    valid_GLT = np.all(GLT_array != GLT_nodata_value, axis=-1)

    # Adjust for One based Index - make a copy to prevent decrementing multiple times inside ortho_xr when applying the glt to elev
    glt_array_copy = GLT_array.copy()
    glt_array_copy[valid_GLT] -= 1
    out_ds[valid_GLT, :] = swath_array[
        glt_array_copy[valid_GLT, 1], glt_array_copy[valid_GLT, 0], :
    ]

    return out_ds
