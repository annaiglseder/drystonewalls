# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 20:11:50 2025

@author: aiglsede
"""

import os
import time
from binarize_and_skeletonize import process_prob_raster_to_vector
from accuracy_density_based import density_eval_raster
from mosaic_probability_maps import mosaic_rasters_stat
from raster_stats import raster_stats_sum_and_threshold



mask_dir = r"P:\Projects\21_Semona_reloaded\07_Work_Data\Use_Cases\NAWA\Trockensteinmauern\_backup_from_opals01\ancillary_geodata\FI_2022_WCH_mask\buf_6px"

# make sure to buffer all tiles inwards to have no overlap, but also no gaps
tile_dir =  r"P:\Projects\21_Semona_reloaded\07_Work_Data\Use_Cases\NAWA\Trockensteinmauern\area_wide_application\dtm_cropped_normalized_250808_enet_b7"

th = 0.5
stat = "mean"
model_pattern = "masked_minlen_1.5_th0.5"
min_len = 1.5


# buffer inwards to get rid of overlap

tile_dir_masked = os.path.join(tile_dir, model_pattern)
os.makedirs(tile_dir_masked, exist_ok=True)

# skeletonize and vectorize

tiles = [f for f in os.listdir(tile_dir) if f.endswith(f".tif")]
masks = [f for f in os.listdir(mask_dir) if f.endswith(f"buf6px.tif")]

if len(tiles) != len(masks):
    raise ValueError("Number of input tiles does not match number of mask tiles")
  

for tile, mask in zip(tiles, masks):
    
    start = time.time()

    tile_path = os.path.join(tile_dir, tile)
    mask_path = os.path.join(mask_dir, mask)
    
    process_prob_raster_to_vector(
        tile_path,
        threshold=th,
        output_folder=tile_dir_masked,
        #pixel_size=1.0,
        #simplify_tolerance=None,
        min_length=min_len,
        #input_is_skeleton=False,
        prune_spurs_len=0.0,
        mask_path= mask_path,           # path to mask raster with identical grid as input
        invert_mask=True          # True -> exclude non-zero mask pixels
    )
        
    end = time.time()
    
    print(f"Processing {tile} took {end-start} seconds.")
    
densities = [25, 100]

for dens in densities:
    density_eval_raster( 
        tile_dir,
        model_pattern,
        dens,
        pixel_size = 0.25
    )

for dens in densities:
    mosaic_rasters_stat(
        os.path.join(tile_dir, model_pattern, f"dens_{dens}"), 
        os.path.join(tile_dir, model_pattern, f"dens_{dens}.tif"),  
        stat="mean", 
        nodata_value=None, 
        name_pattern="*.tif")

 
file_for_stats = ".../dens_100.tif"
total, sum_above, count_above, max_val = raster_stats_sum_and_threshold(
    file_for_stats,
    threshold=100
)
print(f"Sum of all pixels: {total}")
print(f"Sum >= 100: {sum_above}")
print(f"Count >= 100: {count_above}")
print(f"Max value: {max_val}")


