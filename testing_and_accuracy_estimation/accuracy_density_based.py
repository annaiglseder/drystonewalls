# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 17:29:37 2025

@author: aiglsede
"""

import os
import rasterio
import numpy as np
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiLineString
import geopandas as gpd

import numpy as np
import rasterio
from rasterio import Affine
from scipy.ndimage import convolve
from math import sqrt


def aggregate_sum(input_path, output_path, target_pixel_size):
    """
    Aggregates a binary raster to a coarser grid by summing the input values in each block.
    Output raster overlays exactly with the original, as long as the target pixel size is an exact multiple of the original.
    """
    with rasterio.open(input_path) as src:
        data = src.read(1)
        orig_x, orig_y = src.res
        height, width = src.height, src.width

        # Require integer factors for aggregation
        if (target_pixel_size % orig_x != 0) or (target_pixel_size % orig_y != 0):
            raise ValueError(
                f"Target pixel size ({target_pixel_size}) must be a multiple of original pixel size ({orig_x}, {orig_y})"
            )
        factor_x = int(target_pixel_size / orig_x)
        factor_y = int(target_pixel_size / orig_y)

        # Check that dimensions fit perfectly
        if height % factor_y != 0 or width % factor_x != 0:
            raise ValueError(
                f"Input raster dimensions ({height}, {width}) must be divisible by block factors ({factor_y}, {factor_x})"
            )
        new_height = height // factor_y
        new_width = width // factor_x

        # Only use the data that fits exactly
        data_core = data[:new_height * factor_y, :new_width * factor_x]

        # Block sum
        data_reshaped = data_core.reshape(
            new_height, factor_y, new_width, factor_x
        )
        aggregated = data_reshaped.sum(axis=(1, 3)).astype(np.float32)

        # New transform, exactly aligned with original
        new_transform = src.transform * rasterio.Affine.scale(factor_x, factor_y)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": aggregated.shape[0],
            "width": aggregated.shape[1],
            "transform": new_transform,
            "dtype": "float32"
        })
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(aggregated, 1)

    return output_path

                  
def neighborhood_weight_sum(input_path, output_path, pixel_size):
    # Read as float32!
    with rasterio.open(input_path) as src:
        arr = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        meta.update(dtype='float32', nodata=None)  # Ensure float32, no int/nodata confusion

    arr = (arr == 1).astype(np.float32)  # ensure binary 0/1, float for math

    # Define weights
    kernel = np.array([
        [pixel_size * sqrt(2) / 2, pixel_size/2, pixel_size * sqrt(2) / 2],
        [pixel_size/2,             0,           pixel_size/2],
        [pixel_size * sqrt(2) / 2, pixel_size/2, pixel_size * sqrt(2) / 2]
    ], dtype=np.float32)

    # Weighted sum
    sum_weights = convolve(arr, kernel, mode='constant', cval=0.0)

    # Output only for 1-pixels
    output = np.where(arr == 1, sum_weights, 0).astype(np.float32)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(output, 1)
       


def density_eval_raster(
    ref_folder, 
    parent_folder,
    model_pattern,
    target_pixel_size,
    stats = ["min", "max", "mean"],
    prob_ths = ["0.1", "0.2", "0.3", "0.4", "0.5"],
    pixel_size = 0.25
):
    """
    For each file and each stat/prob_th, resample model skeletons to target pixel size,
    save density rasters, and compute difference rasters.
    Reference rasters are only resampled.
    
    ref_folder = ref_dir_sv
    parent_folder = wd
    model_pattern = "test_pred_skel_vec"
    target_pixel_size = 10

    """
    ref_files = [f for f in os.listdir(ref_folder) if f.lower().endswith('.tif')]

    # Create output folder for reference densities (just once)
    ref_out_folder = os.path.join(parent_folder, f"test_ref_dens_{target_pixel_size}")
    os.makedirs(ref_out_folder, exist_ok=True)
                
    ref_sum = os.path.join(ref_folder, "length")
    os.makedirs(ref_sum, exist_ok=True)  

    # --- Compute reference densities ---
    for ref_file in ref_files:
        #ref_file = ref_files[0]
        ref_path = os.path.join(ref_folder, ref_file)
    
        ref_sum_path = os.path.join(ref_sum, ref_file)  
        neighborhood_weight_sum(ref_path, ref_sum_path, pixel_size)
            
        ref_density_path = os.path.join(ref_out_folder, ref_file)
        aggregate_sum(ref_sum_path, ref_density_path, target_pixel_size)

    # --- Now loop over stats/prob_th for model outputs ---
    for stat in stats:
        #stat = "mean"
        for prob_th in prob_ths:
            # prob_th = 0.5
            pred_folder = os.path.join(parent_folder, f"{model_pattern}_{stat}_{prob_th}")
            pred_sum = os.path.join(parent_folder, f"{model_pattern}_{stat}_{prob_th}", "length")
 
            pred_out_folder = os.path.join(parent_folder, f"test_pred_dens_{target_pixel_size}")
            diff_out_folder = os.path.join(parent_folder, f"test_diff_dens_{target_pixel_size}")

            os.makedirs(pred_out_folder, exist_ok=True)
            os.makedirs(diff_out_folder, exist_ok=True)
            os.makedirs(pred_sum, exist_ok=True)

            for ref_file in ref_files:
                #ref_file = ref_files[0]
                pred_path = os.path.join(pred_folder, ref_file.replace("_mask", ""))  # or adjust as needed

                pred_sum_path = os.path.join(pred_sum, ref_file.replace("_mask", ""))
                neighborhood_weight_sum(pred_path, pred_sum_path, pixel_size)

                ref_density_path = os.path.join(ref_out_folder, ref_file)
                pred_density_path = os.path.join(pred_out_folder, f"{ref_file[:len(ref_file)-9]}_{stat}_{prob_th}.tif")
                diff_density_path = os.path.join(diff_out_folder, f"{ref_file[:len(ref_file)-9]}_{stat}_{prob_th}.tif")

                aggregate_sum(pred_sum_path, pred_density_path, target_pixel_size)

                # Subtract model from reference
                with rasterio.open(ref_density_path) as ref_ds, rasterio.open(pred_density_path) as pred_ds:
                    ref_data = ref_ds.read(1).astype(np.float32)
                    pred_data = pred_ds.read(1).astype(np.float32)
                    diff_data = ref_data - pred_data
                    diff_meta = ref_ds.meta.copy()
                    diff_meta.update({'dtype': 'float32'})
                    with rasterio.open(diff_density_path, "w", **diff_meta) as dst:
                        dst.write(diff_data, 1)


def combine_raster_tiles_by_pattern(input_folders, output_folder, stats, prob_ths):
    """
    For each stat and prob_th, mosaics all matching raster tiles across input_folders.
    Output files are named: {input_folder_name}_{stat}_{prob_th}.tif
    Nodata is np.nan (float32).
    """
    os.makedirs(output_folder, exist_ok=True)

    for stat in stats:
        for prob_th in prob_ths:
            pattern = f"{stat}_{prob_th}"
            for input_folder in input_folders:
                folder_name = os.path.basename(os.path.normpath(input_folder))
                # Collect all tiles ending with {stat}_{prob_th}.tif
                tif_files = [
                    os.path.join(input_folder, f)
                    for f in os.listdir(input_folder)
                    if f.lower().endswith(f"{pattern}.tif")
                ]
                if not tif_files:
                    print(f"No files for {folder_name} {pattern}")
                    continue

                # Open and read rasters
                srcs = [rasterio.open(fp) for fp in tif_files]
                # Mosaic with float32, nodata=np.nan
                mosaic, out_transform = merge(srcs, method="first", nodata=np.nan)
                mosaic = mosaic.astype(np.float32)
                mosaic[np.isnan(mosaic)] = np.nan  # Ensure nodata is nan

                out_meta = srcs[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform,
                    "crs": srcs[0].crs,
                    "nodata": np.nan,
                    "dtype": "float32"
                })

                out_name = f"{folder_name}_{stat}_{prob_th}.tif"
                out_path = os.path.join(output_folder, out_name)
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(mosaic)
                for src in srcs:
                    src.close()
                print(f"Wrote {out_path}")


  