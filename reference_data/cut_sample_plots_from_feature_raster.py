# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 09:53:16 2025

@author: aiglsede
"""

import os
import glob
import rasterio
import fiona
from rasterio.mask import mask

def clip_raster_with_shapes(shapefile_folder, raster_file, output_folder):
    """
    Clips a raster using square polygons from shapefiles in a given folder.
    
    Parameters:
    shapefile_folder (str): Path to the folder containing shapefiles.
    raster_file (str): Path to the raster file.
    output_folder (str): Path to the output folder.
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    shapefiles = glob.glob(os.path.join(shapefile_folder, "*.shp"))
    
    with rasterio.open(raster_file) as src:
        raster_crs = src.crs if src.crs else "EPSG:31256"
        
        for shapefile in shapefiles:
            with fiona.open(shapefile, "r") as shapefile_src:
                shape_crs = shapefile_src.crs if shapefile_src.crs else "EPSG:31256"
                features = [feature["geometry"] for feature in shapefile_src]
                
                try:
                    if shape_crs != raster_crs:
                        print(f"Reprojecting {shapefile} from {shape_crs} to {raster_crs}")
                    
                    out_image, out_transform = mask(src, features, crop=True)
                    out_meta = src.meta.copy()
                    
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "crs": raster_crs
                    })
                    
                    output_filename = os.path.splitext(os.path.basename(shapefile))[0] + ".tif"
                    substr = "_sample"
                    output_filename = output_filename.replace(substr, "")
                    output_path = os.path.join(output_folder, output_filename)
                    
                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(out_image)
                    
                    print(f"Saved clipped raster: {output_path}")
                except Exception as e:
                    print(f"Error processing {shapefile}: {e}")
                    
'''
tile_number = 3405
raster_file_name = f"NOE_LAZE02_T_{tile_number}_DTM_0.25_rgb_slope_sqrt_tpi_circle_radius25_abs_tpi_ring_inner_radius1_outer_radius5_abs.tif"
raster_file_name = f"NOE_LAZE02_T_{tile_number}_DTM_0.25.tif"

                  
shapefile_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_shp"
raster_file = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/dtm_derivates/{raster_file_name}"
output_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_number}/reference_tiles_input_features"

clip_raster_with_shapes(shapefile_folder, raster_file, output_folder)
'''












