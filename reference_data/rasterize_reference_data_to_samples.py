# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:35:00 2025

@author: aiglsede
"""

import os
import glob

import fiona
import rasterio
import rasterio.features
import rasterio.windows
from shapely.geometry import shape, mapping
import numpy as np

def rasterize_lines_for_polygons(
    lines_folder,
    polygons_folder,
    reference_raster_path,
    output_folder,
    buffer_pixels=0  # New parameter: buffer in pixels (default: no buffer)
):
    """
    Rasterizes line features within each polygon area.

    Parameters:
      lines_folder (str): Path to folder containing line shapefiles.
      polygons_folder (str): Path to folder containing polygon shapefiles (each with one 250x250 m polygon).
      reference_raster_path (str): Path to a big raster file whose grid (resolution, transform, extent, crs) defines the output.
      output_folder (str): Folder where output binary mask rasters will be saved.
      
    The output files will be named like: sample_mask_0.tif, sample_mask_1.tif, etc.
    
    NOTE: All datasets are assumed to use the same CRS (EPSG:31256).
    """
    # Create the output folder if it does not exist.
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the reference raster to get grid properties.
    with rasterio.open(reference_raster_path) as ref_src:
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        # Pixel size in map units (assume square pixels)
        pixel_size = ref_transform.a  # X pixel size

    
    # Read in all line features from the lines folder.
    line_shapefile_paths = glob.glob(os.path.join(lines_folder, "*.shp"))
    line_geoms = []
    for shp_path in line_shapefile_paths:
        with fiona.open(shp_path, "r") as src:
            for feat in src:
                # Skip if geometry is None
                geom_data = feat.get("geometry")
                if geom_data is None:
                    continue
    
                try:
                    geom = shape(geom_data)
                    line_geoms.append(geom)
                except AttributeError:
                    # Skip features that cause the AttributeError
                    continue
                
    # Process each polygon shapefile.
    polygon_shapefile_paths = glob.glob(os.path.join(polygons_folder, "*.shp"))
    for poly_path in polygon_shapefile_paths:
        # Open the polygon shapefile (assumed to contain one polygon feature).
        with fiona.open(poly_path, "r") as poly_src:
            # Get the first (and assumed only) feature.
            feat = next(iter(poly_src))
            polygon_geom = shape(feat["geometry"])
        
        # Get the bounding box of the polygon.
        minx, miny, maxx, maxy = polygon_geom.bounds

        # Determine the window (pixel subset) of the reference raster that covers the polygon.
        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=ref_transform)
        # Ensure that the window's offsets and shape are integer values.
        window = window.round_offsets().round_shape()
        out_shape = (int(window.height), int(window.width))
        # Compute the affine transform for the window.
        window_transform = rasterio.windows.transform(window, ref_transform)
        
        # Calculate buffer in map units
        buffer_dist = buffer_pixels * pixel_size

        shapes_to_rasterize = []
        for line in line_geoms:
            if line.intersects(polygon_geom):
                clipped = line.intersection(polygon_geom)
                if not clipped.is_empty:
                    # Buffer the line, if requested
                    if buffer_pixels > 0:
                        # Note: buffer(0) returns the original geometry
                        buffered = clipped.buffer(buffer_dist)
                        shapes_to_rasterize.append((mapping(buffered), 1))
                    else:
                        shapes_to_rasterize.append((mapping(clipped), 1))
        
        # If there are shapes to rasterize, use them; otherwise, create an empty mask.
        if shapes_to_rasterize:
            mask_array = rasterio.features.rasterize(
                shapes=shapes_to_rasterize,
                out_shape=out_shape,
                transform=window_transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )
        else:
            # No intersections: produce an empty (all zeros) array.
            mask_array = np.zeros(out_shape, dtype=np.uint8)
        
        # Create the output file name.
        # For example, if the polygon shapefile is "sample_0.shp", the output will be "sample_0_mask.tif".
        base_name = os.path.basename(poly_path)
        name_no_ext, _ = os.path.splitext(base_name)
        cleaned_name = lambda s: s.replace("_sample", "") if "_sample" in s else s
        name_no_ext = cleaned_name(name_no_ext)
        out_name = f"{name_no_ext}.tif"
        out_path = os.path.join(output_folder, out_name)
        
        # Prepare the output raster profile.
        out_profile = {
            "driver": "GTiff",
            "height": out_shape[0],
            "width": out_shape[1],
            "count": 1,
            "dtype": "uint8",
            "crs": ref_crs,
            "transform": window_transform,
            "nodata": 0,
        }
        
        # Write the binary mask raster.
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(mask_array, 1)
        
        print(f"Saved mask: {out_path}")

'''
tile_no = 3507

lines_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_no}/canny_edge"
polygons_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_no}/reference_tiles_shp"
reference_raster_path = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_no}/dtm_derivates/NOE_LAZE02_T_{tile_no}_DTM_0.25.tif"
output_folder = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/{tile_no}/reference_tiles_masks"
buffer_pixels = 0



rasterize_lines_for_polygons(lines_folder, polygons_folder, reference_raster_path, output_folder, buffer_pixels)

'''




# Example usage:
# rasterize_lines_for_polygons("path/to/lines_folder",
#                              "path/to/polygons_folder",
#                              "path/to/reference_raster.tif",
#                              "path/to/output_folder")


