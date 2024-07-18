"""
This Module has the functions related to working with an EMIT dataset. This includes doing things
like opening and flattening the data to work in xarray, orthorectification, and extracting point and area samples.

Author: Erik Bolch, ebolch@contractor.usgs.gov 

Last Updated: 05/09/2024

TO DO: 
- Rework masking functions to be more flexible
- Update format to match AppEEARS outputs
"""

from .constants import *
from .load_EMIT_swath import load_EMIT_swath
from .extract_GLT import extract_GLT
from .apply_geometry_lookup_table import apply_GLT
from .ortho_xr import ortho_xr
from .get_pixel_center_coords import get_pixel_center_coords
from .quality_mask import quality_mask
from .band_mask import band_mask
from .envi_header import envi_header
from .write_envi import write_envi
from .spatial_subset import spatial_subset
from .is_adjacent import is_adjacent
from .merge_emit import merge_emit
from .ortho_browse import ortho_browse
from .emit_xarray import emit_xarray
