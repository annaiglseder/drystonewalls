# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:12:25 2025

@author: aiglsede
"""
import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window


def crop_rasters_by_pixels(input_folder, output_folder, n_px):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]
    
    for fname in files:
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)
        
        with rasterio.open(in_path) as src:
            height, width = src.height, src.width
            # Define the window (crop box)
            window = Window(
                n_px, n_px,
                width - 2 * n_px,
                height - 2 * n_px
            )
            # Read cropped window
            data = src.read(window=window)
            # Update metadata
            meta = src.meta.copy()
            meta.update({
                'height': data.shape[1],
                'width': data.shape[2],
                'transform': rasterio.windows.transform(window, src.transform)
            })
            # Write cropped raster
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(data)
        print(f"{fname} cropped and saved to {out_path}")



def normalize_rasters(input_folder, file_pattern, min_val, max_val, output_folder):
    """
    Normalize raster values to [0,1] given a min/max, and save to new folder.
    
    Parameters
    ----------
    input_folder : str
        Path to folder with input rasters.
    file_pattern : str
        Glob-style pattern to match files (e.g., '*slope.tif').
    min_val : float
        Minimum value for normalization.
    max_val : float
        Maximum value for normalization.
    output_folder : str
        Path to output folder. Created if not existing.
    """
    
    # create output folder if missing
    os.makedirs(output_folder, exist_ok=True)

    # find files
    file_list = glob.glob(os.path.join(input_folder, file_pattern))
    if not file_list:
        print("No files found matching pattern:", file_pattern)
        return

    for fpath in file_list:
        with rasterio.open(fpath) as src:
            arr = src.read(1).astype(float)
            profile = src.profile

            # normalize
            norm_arr = (arr - min_val) / (max_val - min_val)
            norm_arr = np.clip(norm_arr, 0, 1)  # keep within [0,1]

            # update profile
            profile.update(dtype=rasterio.float32)

            # build output path
            out_path = os.path.join(output_folder, os.path.basename(fpath))

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(norm_arr.astype(rasterio.float32), 1)

        print(f"Processed {os.path.basename(fpath)} to {out_path}")


## Check if Files have NoData values (edge effects)

dtm_folder = ".../dtm"
pattern = "*0.25.tif"
out_txt = os.path.join(dtm_folder, "files_with_nodata.txt")

files = glob.glob(os.path.join(dtm_folder, pattern))
files_with_nodata = []

for file in files:
    with rasterio.open(file) as src:
        arr = src.read(1, masked=True)  # Read as masked array
        if arr.mask.any():  # True if any NoData in the raster
            files_with_nodata.append(os.path.basename(file))

with open(out_txt, "w") as f:
    for name in files_with_nodata:
        f.write(name + "\n")

print(f"Files with NoData written to {out_txt}")

# check the files with no data and see how large the cropping should be

dtm_folder_cropped = "../dtm_cropped"
n_px = 50


# crop files

crop_rasters_by_pixels(dtm_folder, dtm_folder_cropped, n_px)

# normalize data

min_val = 150
max_val = 800
dtm_norm_folder = ".../dtm_norm"


normalize_rasters(dtm_folder_cropped, pattern, min_val, max_val, dtm_norm_folder)















