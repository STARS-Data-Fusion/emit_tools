import numpy as np
import xarray as xr

from skimage import io

from .apply_geometry_lookup_table import apply_GLT

def ortho_browse(url, glt, spatial_ref, geotransform, white_background=True):
    """
    Use an EMIT GLT, geotransform, and spatial ref to orthorectify a browse image. (browse images are in native resolution)
    """
    # Read Data
    data = io.imread(url)
    # Orthorectify using GLT and transpose so band is first dimension
    if white_background == True:
        fill = 255
    else:
        fill = 0
    ortho_data = apply_GLT(data, glt, fill_value=fill).transpose(2, 0, 1)
    coords = {
        "y": (
            ["y"],
            (geotransform[3] + 0.5 * geotransform[5])
            + np.arange(glt.shape[0]) * geotransform[5],
        ),
        "x": (
            ["x"],
            (geotransform[0] + 0.5 * geotransform[1])
            + np.arange(glt.shape[1]) * geotransform[1],
        ),
    }
    ortho_data = ortho_data.astype(int)
    ortho_data[ortho_data == -1] = 0
    # Place in xarray.datarray
    da = xr.DataArray(ortho_data, dims=["band", "y", "x"], coords=coords)
    da.rio.write_crs(spatial_ref, inplace=True)
    return da
