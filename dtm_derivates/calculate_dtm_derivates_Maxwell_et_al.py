# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:21:20 2025

@author: aiglsede
"""

import os
from calculate_tpi_from_scratch import process_dtm_tpi, absolute_raster
from calculate_slope_with_modules import calculate_slope_gdal
from normalize_and_calculate_square_root_of_slope import normalize_and_square_root_of_raster
from create_rgb_composite_and_normalize import create_rgb_composite



dtm_path = ".../dtm.tif"


output_folder = ".../dtm_derivatives_Maxwell"
# Check if the folder exists, and create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


### CALCULATING TPI RASTERS

# Circular neighborhood with radius=25.
process_dtm_tpi(dtm_path, output_folder, kernel_type='circle', radius=25)

# Ring neighborhood with inner_radius=1 and outer_radius=5.
process_dtm_tpi(dtm_path, output_folder, kernel_type='ring', inner_radius=1, outer_radius=5)

# Create absolute rasters
absolute_raster(os.path.join(output_folder, "dtm_tpi_circle_radius25.tif"))
absolute_raster(os.path.join(output_folder, "dtm_tpi_ring_inner_radius1_outer_radius5.tif"))


### CALCULATING SLOPE RASTER

calculate_slope_gdal(dtm_path, os.path.join(output_folder, "dtm_slope.tif"), scale=1.0, slope_format='degree', algorithm='Horn')
normalize_and_square_root_of_raster(os.path.join(output_folder, "dtm_slope.tif"), os.path.join(output_folder, "dtm_slope_sqrt.tif"), min_value=0, max_value=45)


### CREATE RGB COMPOSITE

red = os.path.join(output_folder, "dtm_slope_sqrt.tif")
green = os.path.join(output_folder, "dtm_tpi_circle_radius25.tif")
blue = os.path.join(output_folder, "dtm_tpi_ring_inner_radius1_outer_radius5.tif")
out_path =  os.path.join(output_folder, "dtm_tpi_circle_radius25_abs_tpi_ring_inner_radius1_outer_radius5_abs.tif")


out_rgb = create_rgb_composite(
    red, green, blue,
    red_norm_method='fixed', red_norm_params={'min': 0, 'max': 1},
    green_norm_method='fixed', green_norm_params={'min': 0, 'max': 1},
    blue_norm_method='fixed', blue_norm_params={'min': 0, 'max': 0.3},
    out_path=out_path
)


### CLIP TO SAMPLE PLOTS





