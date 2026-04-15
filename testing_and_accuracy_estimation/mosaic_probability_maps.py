# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:53:11 2025

@author: aiglsede
"""


import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import from_bounds


'''
input_folder = "D:/PHOTO/SEMONA/NAWA/UNET/250804_results_unet_enet_b7_rgb_ep12/test_probs_georef"
output_path = "D:/PHOTO/SEMONA/NAWA/UNET/250804_results_unet_enet_b7_rgb_ep12/test_probs_mosaic/3405_0.tif"
name_pattern = "3405_0"
method = "min"
'''


def mosaic_rasters_stat(input_folder, output_path, stat="mean", nodata_value=None, name_pattern="*.tif"):
    """
    Mosaic single-band rasters using a specified statistic (mean, min, max) for overlapping pixels.
    Handles partial overlap and different extents. Assumes same CRS/resolution.
    
    Parameters:
        input_folder (str): Folder containing raster files.
        output_path (str): Path to write the mosaic.
        file_pattern (str): Pattern to match raster files (default '*.tif').
        nodata_value (float or int or None): Value to use for nodata (default: take from rasters).
        stat_method (str): Statistic for overlap: 'mean', 'min', or 'max'
    """
    stat = stat.lower()
    if stat not in ["mean", "min", "max"]:
        raise ValueError("stat_method must be one of 'mean', 'min', 'max'")

    files = glob.glob(os.path.join(input_folder, name_pattern))
    if not files:
        raise ValueError("No files found matching pattern in input folder.")

    srcs = [rasterio.open(f) for f in files]
    all_bounds = [src.bounds for src in srcs]

    min_x = min(b.left for b in all_bounds)
    min_y = min(b.bottom for b in all_bounds)
    max_x = max(b.right for b in all_bounds)
    max_y = max(b.top for b in all_bounds)

    res_x, res_y = srcs[0].res
    crs = srcs[0].crs
    if nodata_value is None:
        nodata = srcs[0].nodata if srcs[0].nodata is not None else np.nan
    else:
        nodata = nodata_value

    out_width = int(np.ceil((max_x - min_x) / res_x))
    out_height = int(np.ceil((max_y - min_y) / res_y))
    out_transform = from_bounds(min_x, min_y, max_x, max_y, out_width, out_height)

    # Prepare a stack for all rasters, filled with np.nan
    stack = []
    for src in srcs:
        arr = src.read(1, out_shape=(src.height, src.width), resampling=Resampling.nearest)
        arr_masked = np.where(arr == src.nodata, np.nan, arr)
        window = rasterio.windows.from_bounds(
            *src.bounds, 
            transform=out_transform,
            width=out_width,
            height=out_height
        )
        row_off = int(window.row_off)
        col_off = int(window.col_off)
        h, w = arr_masked.shape
        temp = np.full((out_height, out_width), np.nan, dtype=np.float32)
        temp[row_off:row_off+h, col_off:col_off+w] = arr_masked
        stack.append(temp)

    stack = np.stack(stack)  # Shape: (num_rasters, out_height, out_width)

    with np.errstate(invalid='ignore', divide='ignore'):
        if stat == "mean":
            result = np.nanmean(stack, axis=0)
        elif stat == "min":
            result = np.nanmin(stack, axis=0)
        elif stat == "max":
            result = np.nanmax(stack, axis=0)

    # Set nodata for empty pixels
    result = np.where(np.isnan(result), nodata, result)

    profile = srcs[0].profile.copy()
    profile.update(
        height=out_height,
        width=out_width,
        transform=out_transform,
        nodata=nodata,
        dtype=np.float32
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result.astype(np.float32), 1)

    for src in srcs:
        src.close()
# Example usage:
# mosaic_rasters_stat(pred_dir_tiles, output_file, stat="mean")


# rasterio merge only supports min, max (no mean!)
def mosaic_rasters_merge(input_folder, output_path, stat="min", nodata_value=None, name_pattern="*.tif"):
    """
    Mosaic rasters with rasterio.merge using min or max for overlap.
    Fast, efficient, and handles partial overlaps.
    """
    stat = stat.lower()
    if stat not in ["min", "max"]:
        raise ValueError("stat_method must be 'min' or 'max'")

    # 1. List raster files
    files = glob.glob(os.path.join(input_folder, name_pattern))
    if not files:
        raise ValueError("No files found matching pattern in input folder.")

    # 2. Open sources
    srcs = [rasterio.open(f) for f in files]

    # 3. Merge using the selected method
    mosaic, out_trans = merge(srcs, method=stat)

    # 4. Prepare output profile
    out_meta = srcs[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0]
    })

    # 5. Write output
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(mosaic)

    for src in srcs:
        src.close()

# mosaic_rasters_merge(pred_dir_tiles, output_file, stat="max")

'''
def mosaic_with_pyramids_numpy(input_folder, output_path, method='max', nodata_value=None, name_pattern=None):
    """
    Mosaics all .tif files in a folder using a NumPy statistical method for overlapping areas,
    and builds overviews (pyramids) for the output.

    Supported methods: 'min', 'max', 'mean', 'sum', 'median'
    Optional: Only mosaic files starting with name_pattern.
    """
    tif_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith('.tif') and (name_pattern is None or f.startswith(name_pattern))
    ]
    if not tif_files:
        print("No .tif files found in input folder matching pattern.")
        return

    method = method.lower()
    supported = ['min', 'max', 'mean', 'sum', 'median']
    if method not in supported:
        raise ValueError(f"Supported methods: {supported}")

    # Open datasets
    srcs = [rasterio.open(fp) for fp in tif_files]

    # Determine nodata value
    if nodata_value is None:
        nodata_value = srcs[0].nodata
        if nodata_value is None:
            dtype = srcs[0].dtypes[0]
            nodata_value = np.nan if "float" in dtype else 0

    # Stack all rasters (N, H, W)
    stacked, out_trans = merge(srcs, method='first', nodata=nodata_value)
    data = stacked.astype(np.float32)
    data[data == nodata_value] = np.nan

    # Aggregation
    if method == "max":
        mosaic = np.nanmax(data, axis=0)
    elif method == "min":
        mosaic = np.nanmin(data, axis=0)
    elif method == "sum":
        mosaic = np.nansum(data, axis=0)
    elif method == "mean":
        mosaic = np.nanmean(data, axis=0)
    elif method == "median":
        mosaic = np.nanmedian(data, axis=0)

    # Put back nodata
    mosaic = np.where(np.isnan(mosaic), nodata_value, mosaic).astype(srcs[0].dtypes[0])

    out_meta = srcs[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[0],
        "width": mosaic.shape[1],
        "transform": out_trans,
        "crs": srcs[0].crs,
        "nodata": nodata_value
    })

    # Write the mosaic raster to disk
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic[np.newaxis, :, :])
        # Build pyramids (overviews)
        decimation_factors = [2, 4, 8, 16, 32]
        dest.build_overviews(decimation_factors, resampling=rasterio.enums.Resampling.nearest)
        dest.update_tags(ns='rio_overview', resampling='nearest')

    for src in srcs:
        src.close()
    print(f"Mosaic with pyramids saved to {output_path} using numpy method={method}, nodata={nodata_value}")

# Example usage:
# mosaic_with_pyramids_numpy("input_folder", "output/mosaic.tif", method="mean", name_pattern="tile_")


'''