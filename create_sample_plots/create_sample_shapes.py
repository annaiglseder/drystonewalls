# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:51:18 2025

@author: aiglsede
"""

import random
import geopandas as gpd
from shapely.geometry import box, Polygon, mapping
import rasterio
import os
import re
import fiona


def create_random_samples(raster_file, output_dir, num_samples, side_length=250):
    """
    Create random polygon samples within a raster's bounds and save them as shapefiles.
    
    Parameters:
    - raster_file: Path to the input raster file.
    - output_dir: Directory to save the shapefiles.
    - num_samples: Number of random samples to create.
    - side_length: Side length of each sample polygon (default is 250 meters).
    """
    # Open the raster to get bounds and CRS
    with rasterio.open(raster_file) as src:
        raster_bounds = src.bounds
        raster_crs = src.crs

    # Calculate the range of coordinates for random sampling
    xmin, ymin, xmax, ymax = raster_bounds

    samples = []
    for _ in range(num_samples):
        # Randomly select a bottom-left corner for the polygon
        x_start = random.uniform(xmin, xmax - side_length)
        y_start = random.uniform(ymin, ymax - side_length)
        
        # Create a square polygon
        x_end = x_start + side_length
        y_end = y_start + side_length
        sample_polygon = box(x_start, y_start, x_end, y_end)
        samples.append(sample_polygon)

    # Create a GeoDataFrame for the samples
    gdf = gpd.GeoDataFrame(geometry=samples, crs=raster_crs)
    
    # Get tile number out of raster file name
    basename = os.path.basename(raster_file)
    # Use regex to search for a 4-digit number
    match = re.search(r'(\d{4})', basename)

    # Save each polygon as a separate shapefile
    for i, polygon in enumerate(gdf.geometry):
        output_path = f"{output_dir}/{match.group(1)}_sample_{i + 1}.shp"
        single_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_crs)
        single_gdf.to_file(output_path, driver="ESRI Shapefile")
        print(f"Sample saved: {output_path}")


def create_non_overlapping_random_samples(raster_file, output_dir, num_samples, side_length=250, max_attempts=10000):
    """
    Create non-overlapping random polygon samples within a raster's bounds and save them as shapefiles.
    
    Parameters:
    - raster_file: Path to the input raster file.
    - output_dir: Directory to save the shapefiles.
    - num_samples: Number of random samples to create.
    - side_length: Side length of each sample polygon (default is 250 meters).
    - max_attempts: Maximum attempts to generate non-overlapping samples.
    """
    # Open the raster to get bounds and CRS
    with rasterio.open(raster_file) as src:
        raster_bounds = src.bounds
        raster_crs = src.crs

    xmin, ymin, xmax, ymax = raster_bounds

    samples = []
    attempts = 0

    while len(samples) < num_samples and attempts < max_attempts:
        # Randomly select a bottom-left corner ensuring the square fits in the raster
        x_start = random.uniform(xmin, xmax - side_length)
        y_start = random.uniform(ymin, ymax - side_length)
        candidate = box(x_start, y_start, x_start + side_length, y_start + side_length)
        
        # Check for overlap with any previously accepted samples
        if any(candidate.intersects(existing) for existing in samples):
            attempts += 1
            continue
        
        # No overlap found; add the candidate to the list
        samples.append(candidate)
        attempts += 1

    if len(samples) < num_samples:
        print(f"Warning: Only {len(samples)} non-overlapping samples could be generated after {attempts} attempts.")
    
    # Get tile number out of raster file name
    basename = os.path.basename(raster_file)
    # Use regex to search for a 4-digit number
    match = re.search(r'(\d{4})', basename)


    # Save each polygon as a separate shapefile
    for i, polygon in enumerate(samples):
        output_path = f"{output_dir}/{match.group(1)}_sample_{i + 1}.shp"
        single_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=raster_crs)
        single_gdf.to_file(output_path, driver="ESRI Shapefile")
        print(f"Sample saved: {output_path}")



def create_individual_tile(upper_left, tile_size, epsg_code, out_file):

    upper_right = (upper_left[0]+tile_size, upper_left[1])
    lower_right = (upper_left[0]+tile_size, upper_left[1]-tile_size)
    lower_left = (upper_left[0], upper_left[1]-tile_size)
    
    coords = [
        upper_left,
        upper_right,
        lower_right,
        lower_left,
        upper_left]

    # Create the polygon
    polygon = Polygon(coords)
    
    # Define the schema for the shapefile
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }

    
    # Write the shapefile
    with fiona.open(
        out_file,
        mode='w',
        driver='ESRI Shapefile',
        schema=schema,
        crs=f"EPSG:{epsg_code}",  # Set the projection to EPSG:31256
    ) as layer:
        layer.write({
            'geometry': mapping(polygon),  # Convert the polygon to GeoJSON format
            'properties': {'id': 1},      # Add an identifier
        })

 
# Example usage
raster_file = ".../raster.tif"
output_dir = ".../reference_tiles_shape"
num_samples = 10

#create_random_samples(raster_file, output_dir, num_samples)
create_non_overlapping_random_samples(raster_file, output_dir, num_samples)

   
