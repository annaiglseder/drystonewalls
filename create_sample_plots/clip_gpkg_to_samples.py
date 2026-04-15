# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:03:25 2025

@author: aiglsede
"""
import os
import geopandas as gpd

def clip_geopackage_with_shapefiles(geopackage_path, shapefile_folder):
    """
    Clips linear features from a single-layer geopackage using polygon shapefiles from a folder.
    Only retains clipped geometries that are linestrings.
    
    Parameters:
        geopackage_path (str): Path to the geopackage file containing linear features.
        shapefile_folder (str): Folder containing polygon shapefiles for clipping.
    """
    # Read the single layer from the geopackage
    linear_gdf = gpd.read_file(geopackage_path)
    
    # Iterate over each shapefile in the provided folder
    for file in os.listdir(shapefile_folder):
        if file.endswith(".shp"):
            shapefile_path = os.path.join(shapefile_folder, file)
            polygon_gdf = gpd.read_file(shapefile_path)
            
            # Reproject polygon shapefile to match the CRS of the linear features if needed
            if polygon_gdf.crs != linear_gdf.crs:
                polygon_gdf = polygon_gdf.to_crs(linear_gdf.crs)
            
            # Clip the linear features using the polygon geometry
            clipped = gpd.clip(linear_gdf, polygon_gdf)
            
            # Filter out non-linestring geometries (e.g., POINT)
            valid_types = ["LineString", "MultiLineString"]
            clipped = clipped[clipped.geom_type.isin(valid_types)]
            
            if clipped.empty:
                print(f"No valid linear features after clipping for {file}.")
                continue
            
            # Construct the output file name and path in the same folder as the geopackage
            base_name = os.path.splitext(file)[0]
            output_filename = f"{base_name}_canny_succession.shp"
            output_path = os.path.join(os.path.dirname(geopackage_path), output_filename)
            
            # Save the clipped result as a new shapefile
            clipped.to_file(output_path)
            print(f"Saved clipped file to: {output_path}")


# Example usage:
      
canny = ".../canny_edges.gpkg"
sample_tiles = ".../reference_tiles_shape"


clip_geopackage_with_shapefiles(canny, sample_tiles)
