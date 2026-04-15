# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:49:49 2025

@author: aiglsede
"""

import os
import rasterio
import numpy as np

def copy_geoinfo_from_reference(model_folder, reference_folder, output_folder=None, overwrite=True):
    """
    Transfers geo-information (CRS, transform, etc.) from reference '_mask.tif' files
    to model result .tif files with matching base names.

    Parameters:
        model_folder (str): folder containing model output .tif files
        reference_folder (str): folder containing reference '_mask.tif' files
        output_folder (str, optional): folder to save updated files if not overwriting
        overwrite (bool): whether to overwrite model files (default: True)
    """
    os.makedirs(output_folder, exist_ok=True) if output_folder and not overwrite else None

    for fname in os.listdir(model_folder):
        if not fname.lower().endswith(".tif") or fname.lower().endswith("_mask.tif"):
            continue

        model_path = os.path.join(model_folder, fname)
        ref_name = fname.replace(".tif", "_mask.tif")
        ref_path = os.path.join(reference_folder, ref_name)

        if not os.path.exists(ref_path):
                        continue

        # Read model data
        with rasterio.open(model_path) as model_src:
            data = model_src.read()
            count = model_src.count
            dtype = model_src.dtypes[0]

        # Read reference metadata
        with rasterio.open(ref_path) as ref_src:
            meta = ref_src.meta.copy()
            meta.update({
                'count': count,
                'dtype': dtype
            })

        # Output path
        if overwrite:
            out_path = model_path
        else:
            out_path = os.path.join(output_folder, fname)

        # Write with reference geoinfo
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(data)
