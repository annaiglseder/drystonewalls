# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:53:03 2025

@author: aiglsede
"""

from reference_data.create_tiles_for_model import extract_tiles_from_binary_masks, extract_tiles_from_rgb_rasters, extract_tiles_from_greyscale_rasters
from reference_data.rasterize_reference_data_to_samples import rasterize_lines_for_polygons
from reference_data.cut_sample_plots_from_feature_raster import clip_raster_with_shapes


tile_numbers = [3405, 3506, 3507]

for tile_number in tile_numbers:
    
    raster_file_names = [f"NOE_LAZE02_T_{tile_number}_DTM_0.25_rgb_slope_sqrt_tpi_circle_radius25_abs_tpi_ring_inner_radius1_outer_radius5_abs.tif", 
                         f"NOE_LAZE02_T_{tile_number}_DTM_0.25.tif"]
    
    raster_types = ["rgb", "dtm"]
    
    for raster_file_name, raster_type in zip(raster_file_names, raster_types): 
                          
        shapefile_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_shp"
        raster_file = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/dtm_derivates/{raster_file_name}"
        output_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_input_features/{raster_type}"
    
        clip_raster_with_shapes(shapefile_folder, raster_file, output_folder)



buffer_pixels = 1
tile_size = 256
overlap = 0.5
norm_min = 150
norm_max = 800

tile_numbers = [3405, 3506, 3507]
#tile_number = 3507
for tile_number in tile_numbers:
    
    lines_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_mapped_TSM"
    polygons_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_shp"
    reference_raster_path = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/dtm_derivates/NOE_LAZE02_T_{tile_number}_DTM_0.25.tif"
        
    binary_masks_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_masks"
    
    rgb_rasters_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_input_features/rgb"
    greyscale_rasters_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_input_features/dtm"
    
    rgb_outf = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/model_data/rgb"
    dtm_outf = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/model_data/dtm"
    
    
    rasterize_lines_for_polygons(lines_folder, polygons_folder, reference_raster_path, binary_masks_folder, buffer_pixels)
    
    extract_tiles_from_binary_masks(binary_masks_folder, rgb_outf)
    extract_tiles_from_binary_masks(binary_masks_folder, dtm_outf)
    
    extract_tiles_from_rgb_rasters(rgb_rasters_folder, rgb_outf, tile_size, overlap)
    extract_tiles_from_greyscale_rasters(greyscale_rasters_folder, dtm_outf, tile_size, overlap, norm_min, norm_max)
    
    
    
'''

# ANOTHER RGB FILE

for tile_number in tile_numbers:
    
    raster_file_names = [f"NOE_LAZE02_T_{tile_number}_DTM_0.25_rvt_slrm_ldom_mhs.tif"]
    
    raster_types = ["rgb2"]
    
    for raster_file_name, raster_type in zip(raster_file_names, raster_types): 
                          
        shapefile_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_shp"
        raster_file = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/dtm_derivates/{raster_file_name}"
        output_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_input_features/{raster_type}"
    
        clip_raster_with_shapes(shapefile_folder, raster_file, output_folder)

'''

buffer_pixels = 1
tile_size = 256
overlap = 0.5


tile_numbers = [3405, 3506, 3507]
#tile_number = 3507
for tile_number in tile_numbers:
    
    lines_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_mapped_TSM"
    polygons_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_shp"
    reference_raster_path = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/dtm_derivates/NOE_LAZE02_T_{tile_number}_DTM_0.25.tif"
        
    binary_masks_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_masks"
    
    rgb_rasters_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_input_features/rgb2"
    
    rgb_outf = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/model_data/rgb2"
    
    rasterize_lines_for_polygons(lines_folder, polygons_folder, reference_raster_path, binary_masks_folder, buffer_pixels)
    
    extract_tiles_from_binary_masks(binary_masks_folder, rgb_outf)
    
    extract_tiles_from_rgb_rasters(rgb_rasters_folder, rgb_outf, tile_size, overlap)

    
  
    
