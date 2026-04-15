# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 17:29:37 2025

@author: aiglsede
"""

import os
import rasterio
import numpy as np
from rasterio.merge import merge
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
       
'''
    density_eval_raster( 
        tile_dir,
        "bin_vec",
        dens,
        pixel_size = 0.25
    )
    
'''


def density_eval_raster(
    parent_folder,
    model_pattern,
    target_pixel_size,
    pixel_size=0.25
):
    """
    For each model raster:
      1) Ensure its dimensions are divisible by the aggregation block factor (target_pixel_size / pixel_size).
         If not, trim trailing rows/cols to the nearest divisible size and use the trimmed copy.
      2) Compute neighborhood-weighted length sum.
      3) Aggregate to target pixel size to get density.
    """
    pred_folder = os.path.join(parent_folder, f"{model_pattern}")
    pred_sum = os.path.join(parent_folder, f"{model_pattern}", f"length_{target_pixel_size}")
    pred_out_folder = os.path.join(parent_folder, f"{model_pattern}", f"dens_{target_pixel_size}")

    os.makedirs(pred_sum, exist_ok=True)
    os.makedirs(pred_out_folder, exist_ok=True)

    pred_files = [f for f in os.listdir(pred_folder) if f.lower().endswith('.tif')]

    # block factor: how many original pixels per target pixel
    block = target_pixel_size / float(pixel_size)
    if abs(round(block) - block) > 1e-6:
        raise ValueError(
            f"target_pixel_size/pixel_size must be an integer. Got {target_pixel_size}/{pixel_size}={block}"
        )
    block = int(round(block))

    for pred_file in pred_files:
        pred_path = os.path.join(pred_folder, pred_file)

        # --- ensure divisibility; trim if necessary ---
        use_path = pred_path
        with rasterio.open(pred_path) as src:
            height, width = src.height, src.width

            new_h = (height // block) * block
            new_w = (width  // block) * block

            if (new_h != height) or (new_w != width):
                # Trim trailing rows/cols (bottom/right). Keep same origin/transform/CRS.
                data = src.read(1, masked=False)
                trimmed = data[:new_h, :new_w]

                profile = src.profile.copy()
                profile.update({
                    "height": new_h,
                    "width": new_w,
                })
                # predictor: 3 for float, 2 for int; keep dtype as-is
                np_dtype = np.dtype(profile["dtype"])
                profile["predictor"] = 3 if np.issubdtype(np_dtype, np.floating) else 2

                base, ext = os.path.splitext(pred_path)
                use_path = f"{base}_trim{ext}"

                with rasterio.open(use_path, "w", **profile) as dst:
                    dst.write(trimmed, 1)

                print(f"[INFO] Trimmed {pred_file}: ({height},{width}) -> ({new_h},{new_w}) using block {block}")

        # --- compute sums and densities using the (possibly) trimmed path ---
        pred_sum_path = os.path.join(pred_sum, os.path.basename(use_path))
        neighborhood_weight_sum(use_path, pred_sum_path, pixel_size)

        pred_density_path = os.path.join(pred_out_folder, os.path.basename(use_path))
        aggregate_sum(pred_sum_path, pred_density_path, target_pixel_size)



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


  