# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:37:17 2025

@author: aiglsede
"""


import os
import rasterio



def strip_geoinfo_folder(input_folder, output_folder=None, overwrite=False):
    """
    Strips geoinfo from all .tif files in a folder.

    Parameters:
    - input_folder: folder containing GeoTIFF files
    - output_folder: where to save stripped files (required if overwrite=False)
    - overwrite: if True, overwrites original files
    """
    if not overwrite and output_folder is None:
        raise ValueError("You must specify an output folder if overwrite=False.")

    if not overwrite:
        os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if fname.lower().endswith(".tif"):
            in_path = os.path.join(input_folder, fname)
            out_path = in_path if overwrite else os.path.join(output_folder, fname)

            with rasterio.open(in_path) as src:
                data = src.read()
                dtype = src.dtypes[0]
                count = src.count
                height = src.height
                width = src.width

            # Build minimal metadata (no transform, CRS, tags)
            stripped_meta = {
                'driver': 'GTiff',
                'width': width,
                'height': height,
                'count': count,
                'dtype': dtype,
            }

            with rasterio.open(out_path, 'w', **stripped_meta) as dst:
                dst.write(data)

            print(f"Stripped: {fname}")

    print("All files processed.")


#strip_geoinfo_folder(ref_folder, overwrite=True)


