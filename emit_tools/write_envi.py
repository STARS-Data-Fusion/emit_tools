import os

import numpy as np
import xarray as xr

from spectral.io import envi

from .envi_header import envi_header

def write_envi(
    xr_ds: xr.Dataset,
    output_dir: str,
    overwrite: bool = False,
    extension: str = ".img",
    interleave: str = "BIL",
    glt_file: bool = False,
):
    """
    This function takes an EMIT dataset read into an xarray dataset using the emit_xarray function and then writes an ENVI file and header. Does not work for L2B MIN.

    Parameters:
    xr_ds: an EMIT dataset read into xarray using the emit_xarray function.
    output_dir: output directory
    overwrite: overwrite existing file if True
    extension: the file extension for the envi formatted file, .img by default.
    glt_file: also create a GLT ENVI file for later use to reproject

    Returns:
    envi_ds: file in the output directory
    glt_ds: file in the output directory

    """
    # Check if xr_ds has been orthorectified, raise exception if it has been but GLT is still requested
    if (
        "Orthorectified" in xr_ds.attrs.keys()
        and xr_ds.attrs["Orthorectified"] == "True"
        and glt_file == True
    ):
        raise Exception("Data is already orthorectified.")

    # Typemap dictionary for ENVI files
    envi_typemap = {
        "uint8": 1,
        "int16": 2,
        "int32": 3,
        "float32": 4,
        "float64": 5,
        "complex64": 6,
        "complex128": 9,
        "uint16": 12,
        "uint32": 13,
        "int64": 14,
        "uint64": 15,
    }

    # Get CRS/geotransform for creation of Orthorectified ENVI file or optional GLT file
    gt = xr_ds.attrs["geotransform"]
    mapinfo = (
        "{Geographic Lat/Lon, 1, 1, "
        + str(gt[0])
        + ", "
        + str(gt[3])
        + ", "
        + str(gt[1])
        + ", "
        + str(gt[5] * -1)
        + ", WGS-84, units=Degrees}"
    )

    # This creates the coordinate system string
    # hard-coded replacement of wkt crs could probably be improved, though should be the same for all EMIT datasets
    csstring = '{ GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]] }'
    # List data variables (typically reflectance/radiance)
    var_names = list(xr_ds.data_vars)

    # Loop through variable names
    for var in var_names:
        # Define output filename
        output_name = os.path.join(output_dir, xr_ds.attrs["granule_id"] + "_" + var)

        nbands = 1
        if len(xr_ds[var].data.shape) > 2:
            nbands = xr_ds[var].data.shape[2]

        # Start building metadata
        metadata = {
            "lines": xr_ds[var].data.shape[0],
            "samples": xr_ds[var].data.shape[1],
            "bands": nbands,
            "interleave": interleave,
            "header offset": 0,
            "file type": "ENVI Standard",
            "data type": envi_typemap[str(xr_ds[var].data.dtype)],
            "byte order": 0,
            "data ignore value": -9999,
        }

        for key in list(xr_ds.attrs.keys()):
            if key == "summary":
                metadata["description"] = xr_ds.attrs[key]
            elif key not in ["geotransform", "spatial_ref"]:
                metadata[key] = f"{{ {xr_ds.attrs[key]} }}"

        # List all variables in dataset (including coordinate variables)
        meta_vars = list(xr_ds.variables)

        # Add band parameter information to metadata (ie wavelengths/obs etc.)
        for m in meta_vars:
            if m == "wavelengths" or m == "radiance_wl":
                metadata["wavelength"] = np.array(xr_ds[m].data).astype(str).tolist()
            elif m == "fwhm" or m == "radiance_fwhm":
                metadata["fwhm"] = np.array(xr_ds[m].data).astype(str).tolist()
            elif m == "good_wavelengths":
                metadata["good_wavelengths"] = (
                    np.array(xr_ds[m].data).astype(int).tolist()
                )
            elif m == "observation_bands":
                metadata["band names"] = np.array(xr_ds[m].data).astype(str).tolist()
            elif m == "mask_bands":
                if var == "band_mask":
                    metadata["band names"] = [
                        "packed_bands_" + bn
                        for bn in np.arange(285 / 8).astype(str).tolist()
                    ]
                else:
                    metadata["band names"] = (
                        np.array(xr_ds[m].data).astype(str).tolist()
                    )
            if "wavelength" in list(metadata.keys()) and "band names" not in list(
                metadata.keys()
            ):
                metadata["band names"] = metadata["wavelength"]

        # Add CRS/mapinfo if xarray dataset has been orthorectified
        if (
            "Orthorectified" in xr_ds.attrs.keys()
            and xr_ds.attrs["Orthorectified"] == "True"
        ):
            metadata["coordinate system string"] = csstring
            metadata["map info"] = mapinfo

        # Write Variables as ENVI Output
        envi_ds = envi.create_image(
            envi_header(output_name), metadata, ext=extension, force=overwrite
        )
        mm = envi_ds.open_memmap(interleave="bip", writable=True)

        dat = xr_ds[var].data

        if len(dat.shape) == 2:
            dat = dat.reshape((dat.shape[0], dat.shape[1], 1))

        mm[...] = dat

    # Create GLT Metadata/File
    if glt_file == True:
        # Output Name
        glt_output_name = os.path.join(
            output_dir, xr_ds.attrs["granule_id"] + "_" + "glt"
        )

        # Write GLT Metadata
        glt_metadata = metadata

        # Remove Unwanted Metadata
        glt_metadata.pop("wavelength", None)
        glt_metadata.pop("fwhm", None)

        # Replace Metadata
        glt_metadata["lines"] = xr_ds["glt_x"].data.shape[0]
        glt_metadata["samples"] = xr_ds["glt_x"].data.shape[1]
        glt_metadata["bands"] = 2
        glt_metadata["data type"] = envi_typemap["int32"]
        glt_metadata["band names"] = ["glt_x", "glt_y"]
        glt_metadata["coordinate system string"] = csstring
        glt_metadata["map info"] = mapinfo

        # Write GLT Outputs as ENVI File
        glt_ds = envi.create_image(
            envi_header(glt_output_name), glt_metadata, ext=extension, force=overwrite
        )
        mmglt = glt_ds.open_memmap(interleave="bip", writable=True)
        mmglt[...] = np.stack(
            (xr_ds["glt_x"].values, xr_ds["glt_y"].values), axis=-1
        ).astype("int32")
