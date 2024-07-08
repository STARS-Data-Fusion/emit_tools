import numpy as np

# Function to Apply the GLT to an array
def apply_glt(ds_array, glt_array, fill_value=-9999, GLT_NODATA_VALUE=0):
    """
    This function applies the GLT array to a numpy array of either 2 or 3 dimensions.

    Parameters:
    ds_array: numpy array of the desired variable
    glt_array: a GLT array constructed from EMIT GLT data

    Returns:
    out_ds: a numpy array of orthorectified data.
    """

    # Build Output Dataset
    if ds_array.ndim == 2:
        ds_array = ds_array[:, :, np.newaxis]
    out_ds = np.full(
        (glt_array.shape[0], glt_array.shape[1], ds_array.shape[-1]),
        fill_value,
        dtype=np.float32,
    )
    valid_glt = np.all(glt_array != GLT_NODATA_VALUE, axis=-1)

    # Adjust for One based Index - make a copy to prevent decrementing multiple times inside ortho_xr when applying the glt to elev
    glt_array_copy = glt_array.copy()
    glt_array_copy[valid_glt] -= 1
    out_ds[valid_glt, :] = ds_array[
        glt_array_copy[valid_glt, 1], glt_array_copy[valid_glt, 0], :
    ]
    return out_ds
