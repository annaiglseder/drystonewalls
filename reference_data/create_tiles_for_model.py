# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:54:10 2025

@author: aiglsede
"""
import os
import glob
import csv
import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio import Affine
from rasterio.enums import Compression
from rasterio.windows import Window
from PIL import Image


def extract_tiles_from_binary_masks_png(binary_masks_folder, output_tiles_folder, csv_output_path, tile_size=256, overlap=0.5):
    """
    Extracts overlapping training tiles from binary mask TIFF files.
    
    Each binary mask is split into overlapping tiles of size tile_size x tile_size pixels,
    with the specified overlap fraction (e.g. 0.5 for 50% overlap). For example, if each binary mask 
    is 1000×1000 pixels, this will yield 49 tiles (7 columns × 7 rows). 
    The tiles are saved as PNG files with the naming convention:
        <sample>_<tile_number>.png 
    (e.g. sample_0_1.png, sample_0_2.png, …, sample_0_49.png).
    
    Additionally, for every tile the georeferenced coordinates (x, y) of its top‐left corner 
    are written to a CSV file along with the tile’s file path.
    
    Parameters:
      binary_masks_folder (str): Folder containing the binary mask TIFF files.
      output_tiles_folder (str): Folder where the output PNG tile files will be saved.
      csv_output_path (str): Path to the CSV file to record tile file paths and corner coordinates.
      tile_size (int): Tile width (and height) in pixels (default is 250).
      overlap (float): Fractional overlap between tiles (default is 0.5 for 50% overlap).
      
    NOTE:
      - This function assumes that the input masks are single-band and have georeferencing.
      - For a 50% overlap with 250×250 tiles, the step is 125 pixels. (In a 1000×1000 image,
        you will get 7 steps in each direction: (1000 - 250)/125 + 1 = 7.)
    """
    # Create output folder if it does not exist.
    os.makedirs(output_tiles_folder, exist_ok=True)
    
    # Open the CSV file for writing.
    with open(csv_output_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header row.
        csv_writer.writerow(["tile_file", "x_coordinate", "y_coordinate"])
        
        # List all binary mask files (assumed to be .tif files).
        mask_files = glob.glob(os.path.join(binary_masks_folder, "*.tif"))
        
        for mask_file in mask_files:
            with rasterio.open(mask_file) as src:
                # Read the first band (assumes a single-band binary mask).
                mask_array = src.read(1)
                transform = src.transform
                height, width = mask_array.shape
            
            # Compute the step size in pixels.
            step = int(tile_size * (1 - overlap))
            
            # Create naming
            base_name = os.path.basename(mask_file)
            name_no_ext = os.path.splitext(base_name)[0]
            sample_name = name_no_ext[:-5] if name_no_ext.endswith("_mask") else name_no_ext
            
            tile_coords = []  # Store (i, j) for all tile top-lefts
            
            # Standard grid
            for i in range(0, height - tile_size + 1, step):
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((i, j))

            # Rightmost column (if not exactly covered)
            if (width - tile_size) % step != 0:
                last_j = width - tile_size
                for i in range(0, height - tile_size + 1, step):
                    tile_coords.append((i, last_j))
            
            # Bottommost row (if not exactly covered)
            if (height - tile_size) % step != 0:
                last_i = height - tile_size
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((last_i, j))
            
            # Bottom-right corner (if both right and bottom are missed)
            if (width - tile_size) % step != 0 and (height - tile_size) % step != 0:
                tile_coords.append((height - tile_size, width - tile_size))

            # Remove any duplicate coordinates
            tile_coords = list(set(tile_coords))
            
            # Loop over the image in steps (row-wise and column-wise).
            tile_count = 1
            
            for i, j in sorted(tile_coords):
                # Extract the tile from the mask array.
                tile = mask_array[i:i+tile_size, j:j+tile_size]
                
                # Construct the output PNG file name.
                tile_filename = f"{sample_name}_{tile_count}_mask.tif"
                tile_path = os.path.join(output_tiles_folder, tile_filename)
                
                # --- Optional scaling: ---
                # Multiply by 255 so that binary 1 becomes 255 (white) for easier visualization.
                # If you prefer to keep the raw binary values (0 and 1), remove the multiplication.
                tile_to_save = (tile).astype(np.uint8)
                
                # Save the tile as a PNG image.
                img = Image.fromarray(tile_to_save)
                img.save(tile_path)
                
                # Compute the georeferenced coordinates of the top-left corner of the tile.
                # Rasterio’s transform.xy converts pixel coordinates (row, col) to geospatial coordinates.
                x_corner, y_corner = xy(transform, i, j, offset='ul')
                
                # Write the tile file path and its top-left corner coordinates to the CSV.
                csv_writer.writerow([tile_path, x_corner, y_corner])
                tile_count += 1
                


def extract_tiles_from_binary_masks(
        binary_masks_folder, output_tiles_folder, tile_size=256, overlap=0.5):
    """
    Extracts overlapping training tiles from binary mask TIFF files.
    Each tile is saved as a GeoTIFF with georeferencing.
    """
    os.makedirs(output_tiles_folder, exist_ok=True)
    
    mask_files = glob.glob(os.path.join(binary_masks_folder, "*.tif"))
        
    for mask_file in mask_files:
        with rasterio.open(mask_file) as src:
            height, width = src.height, src.width
            src_crs = src.crs
            src_transform = src.transform

            step = int(tile_size * (1 - overlap))
            base_name = os.path.basename(mask_file)
            name_no_ext = os.path.splitext(base_name)[0]
            sample_name = name_no_ext[:-5] if name_no_ext.endswith("_mask") else name_no_ext

            tile_coords = []
            for i in range(0, height - tile_size + 1, step):
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((i, j))
            if (width - tile_size) % step != 0:
                last_j = width - tile_size
                for i in range(0, height - tile_size + 1, step):
                    tile_coords.append((i, last_j))
            if (height - tile_size) % step != 0:
                last_i = height - tile_size
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((last_i, j))
            if (width - tile_size) % step != 0 and (height - tile_size) % step != 0:
                tile_coords.append((height - tile_size, width - tile_size))

            tile_coords = list(set(tile_coords))
            tile_count = 1

            for i, j in sorted(tile_coords):
                window = Window(j, i, tile_size, tile_size)
                tile = src.read(1, window=window)
                tile_transform = src.window_transform(window)
                
                tile_filename = f"{sample_name}_{tile_count}_mask.tif"
                tile_path = os.path.join(output_tiles_folder, tile_filename)
                
                with rasterio.open(
                    tile_path,
                    'w',
                    driver='GTiff',
                    height=tile.shape[0],
                    width=tile.shape[1],
                    count=1,
                    dtype=tile.dtype,
                    crs=src_crs,
                    transform=tile_transform,
                    compress='lzw'
                ) as dst:
                    dst.write(tile, 1)
                
                tile_count += 1



                    
 
def extract_tiles_from_rgb_rasters_png(rgb_rasters_folder, output_tiles_folder, csv_output_path, tile_size=256, overlap=0.5):
    """
    Extracts overlapping training tiles from three-band (RGB) raster TIFF files.
    
    Each RGB raster is split into overlapping tiles of size tile_size x tile_size pixels,
    with the specified overlap fraction (e.g., 0.5 for 50% overlap). The tiles are saved as RGB PNG files.
    
    Additionally, for every tile, the georeferenced coordinates (x, y) of its top-left corner 
    are written to a CSV file along with the tile’s file path.
    
    Parameters:
      rgb_rasters_folder (str): Folder containing the RGB raster TIFF files.
      output_tiles_folder (str): Folder where the output PNG tile files will be saved.
      csv_output_path (str): Path to the CSV file to record tile file paths and corner coordinates.
      tile_size (int): Tile width (and height) in pixels (default is 250).
      overlap (float): Fractional overlap between tiles (default is 0.5 for 50% overlap).
    """
    # Create output folder if it does not exist.
    os.makedirs(output_tiles_folder, exist_ok=True)
    
    # Open the CSV file for writing.
    with open(csv_output_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header row.
        csv_writer.writerow(["tile_file", "x_coordinate", "y_coordinate"])
        
        # List all raster files (assumed to be .tif files).
        raster_files = glob.glob(os.path.join(rgb_rasters_folder, "*.tif"))
        
        for raster_file in raster_files:
            with rasterio.open(raster_file) as src:
                # Read all three bands (RGB)
                rgb_array = src.read([1, 2, 3])  # Read three bands
                transform = src.transform
                height, width = rgb_array.shape[1], rgb_array.shape[2]  # Extract height and width from shape
                
                # Replace NaN and negative values before normalization
                rgb_array = np.nan_to_num(rgb_array, nan=0.0, posinf=1.0, neginf=0.0)
                rgb_array[rgb_array < 0] = 0  # Ensure no negative values
                
                '''
                # Normalize each band independently to match original color distribution
                for i in range(3):
                    band_min = rgb_array[i].min()
                    band_max = rgb_array[i].max()
                    if band_max > band_min:
                        rgb_array[i] = (rgb_array[i] - band_min) / (band_max - band_min)
                '''
                                
                # Normalize the values to 0-255 (assuming the data is in 0-1 range)
                rgb_array = (rgb_array * 255).clip(0, 255).astype(np.uint8)
            
            # Compute the step size in pixels.
            step = int(tile_size * (1 - overlap))
            
            tile_count = 1  # tile numbering starts at 1
            
            # Derive the sample’s base name from the raster file.
            base_name = os.path.basename(raster_file)
            name_no_ext = os.path.splitext(base_name)[0]

            # --- ADDED: Edge-covering tile coordinate logic ---
            tile_coords = []

            # Standard grid
            for i in range(0, height - tile_size + 1, step):
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((i, j))

            # Rightmost column (if not exactly covered)
            if (width - tile_size) % step != 0:
                last_j = width - tile_size
                for i in range(0, height - tile_size + 1, step):
                    tile_coords.append((i, last_j))
            
            # Bottommost row (if not exactly covered)
            if (height - tile_size) % step != 0:
                last_i = height - tile_size
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((last_i, j))
            
            # Bottom-right corner (if both right and bottom are missed)
            if (width - tile_size) % step != 0 and (height - tile_size) % step != 0:
                tile_coords.append((height - tile_size, width - tile_size))

            # Remove any duplicate coordinates
            tile_coords = list(set(tile_coords))
            # --- END ADDED ---

            # --- CHANGED: Loop over tile_coords instead of fixed grid ---
            for i, j in sorted(tile_coords):
                # Extract the tile from the RGB array.
                tile = rgb_array[:, i:i+tile_size, j:j+tile_size]  # Extract RGB tile
                
                # Convert from (bands, height, width) to (height, width, bands) for saving.
                tile = np.moveaxis(tile, 0, -1)
                
                # Construct the output PNG file name.
                tile_filename = f"{name_no_ext}_{tile_count}.png"
                tile_path = os.path.join(output_tiles_folder, tile_filename)
                
                # Save the tile as an RGB PNG image.
                img = Image.fromarray(tile)
                img.save(tile_path)
                
                # Compute the georeferenced coordinates of the top-left corner of the tile.
                x_corner, y_corner = xy(transform, j, i, offset='ul')  # (col, row) order!
                
                # Write the tile file path and its top-left corner coordinates to the CSV.
                csv_writer.writerow([tile_path, x_corner, y_corner])
                
                tile_count += 1


def extract_tiles_from_rgb_rasters(
        rgb_rasters_folder, output_tiles_folder, tile_size=256, overlap=0.5):
    """
    Extracts overlapping tiles from three-band (RGB) raster TIFF files.
    Each tile is saved as a GeoTIFF with georeferencing.
    """
    os.makedirs(output_tiles_folder, exist_ok=True)
    raster_files = glob.glob(os.path.join(rgb_rasters_folder, "*.tif"))
    
    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            height, width = src.height, src.width
            src_crs = src.crs
            #src_transform = src.transform

            # Read all three bands (RGB)
            #rgb_array = src.read([1, 2, 3])
            
            '''
            # Clean any NaNs/negative values before normalization
            rgb_array = np.nan_to_num(rgb_array, nan=0.0, posinf=1.0, neginf=0.0)
            rgb_array[rgb_array < 0] = 0  # No negative values
            
            # Scale to 0-255 for uint8 (if your source data is not 0-1, adjust as needed)
            rgb_array = (rgb_array * 255).clip(0, 255).astype(np.uint8)
            '''
            
            step = int(tile_size * (1 - overlap))
            base_name = os.path.basename(raster_file)
            name_no_ext = os.path.splitext(base_name)[0]

            tile_coords = []
            for i in range(0, height - tile_size + 1, step):
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((i, j))
            if (width - tile_size) % step != 0:
                last_j = width - tile_size
                for i in range(0, height - tile_size + 1, step):
                    tile_coords.append((i, last_j))
            if (height - tile_size) % step != 0:
                last_i = height - tile_size
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((last_i, j))
            if (width - tile_size) % step != 0 and (height - tile_size) % step != 0:
                tile_coords.append((height - tile_size, width - tile_size))

            tile_coords = list(set(tile_coords))
            tile_count = 1

            for i, j in sorted(tile_coords):
                window = Window(j, i, tile_size, tile_size)
                tile = src.read([1, 2, 3], window=window)
                tile_transform = src.window_transform(window)

                tile_filename = f"{name_no_ext}_{tile_count}.tif"
                tile_path = os.path.join(output_tiles_folder, tile_filename)

                with rasterio.open(
                    tile_path,
                    'w',
                    driver='GTiff',
                    height=tile.shape[1],
                    width=tile.shape[2],
                    count=3,
                    dtype=tile.dtype,
                    crs=src_crs,
                    transform=tile_transform,
                    compress='lzw'
                ) as dst:
                    dst.write(tile)
                tile_count += 1


def extract_tiles_from_greyscale_rasters_png(greyscale_rasters_folder, output_tiles_folder, csv_output_path, tile_size=256, overlap=0.5, norm_min=0, norm_max=255):
    """
    Extracts overlapping training tiles from single-band (greyscale) raster TIFF files (e.g., DTM).
    
    Each raster is split into overlapping tiles of size tile_size x tile_size pixels,
    with the specified overlap fraction (e.g., 0.5 for 50% overlap). The data are normalized 
    using the provided norm_min and norm_max values before scaling to 0-255 and saving as greyscale PNG files.
    
    Additionally, for every tile, the georeferenced coordinates (x, y) of its top-left corner 
    are written to a CSV file along with the tile’s file path.
    
    Parameters:
      greyscale_rasters_folder (str): Folder containing the greyscale raster TIFF files.
      output_tiles_folder (str): Folder where the output PNG tile files will be saved.
      csv_output_path (str): Path to the CSV file to record tile file paths and corner coordinates.
      tile_size (int): Tile width (and height) in pixels (default is 250).
      overlap (float): Fractional overlap between tiles (default is 0.5 for 50% overlap).
      norm_min (float): Minimum value for normalization.
      norm_max (float): Maximum value for normalization.
    """
    # Create output folder if it does not exist.
    os.makedirs(output_tiles_folder, exist_ok=True)
    
    # Open the CSV file for writing.
    with open(csv_output_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header row.
        csv_writer.writerow(["tile_file", "x_coordinate", "y_coordinate"])
        
        # List all raster files (assumed to be .tif files).
        raster_files = glob.glob(os.path.join(greyscale_rasters_folder, "*.tif"))
        
        for raster_file in raster_files:
            with rasterio.open(raster_file) as src:
                # Read the single band (greyscale)
                dtm_array = src.read(1)
                transform = src.transform
                height, width = dtm_array.shape
                
                # Replace NaN and infinite values; clip to the provided normalization range.
                dtm_array = np.nan_to_num(dtm_array, nan=norm_min, posinf=norm_max, neginf=norm_min)
                dtm_array = np.clip(dtm_array, norm_min, norm_max)
                
                # Normalize the values to the range 0-255.
                dtm_normalized = (dtm_array - norm_min) / (norm_max - norm_min)
                #dtm_normalized = (dtm_normalized * 255).clip(0, 255).astype(np.uint8)
            
            # Compute the step size in pixels.
            step = int(tile_size * (1 - overlap))
            
            tile_count = 1  # tile numbering starts at 1
            
            # Derive the sample’s base name from the raster file.
            base_name = os.path.basename(raster_file)
            name_no_ext = os.path.splitext(base_name)[0]
            
            # --- ADDED: Edge-covering tile coordinate logic ---
            tile_coords = []

            # Standard grid
            for i in range(0, height - tile_size + 1, step):
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((i, j))

            # Rightmost column (if not exactly covered)
            if (width - tile_size) % step != 0:
                last_j = width - tile_size
                for i in range(0, height - tile_size + 1, step):
                    tile_coords.append((i, last_j))
            
            # Bottommost row (if not exactly covered)
            if (height - tile_size) % step != 0:
                last_i = height - tile_size
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((last_i, j))
            
            # Bottom-right corner (if both right and bottom are missed)
            if (width - tile_size) % step != 0 and (height - tile_size) % step != 0:
                tile_coords.append((height - tile_size, width - tile_size))

            # Remove any duplicate coordinates
            tile_coords = list(set(tile_coords))
            # --- END ADDED ---

            # --- CHANGED: Loop over tile_coords instead of fixed grid ---
            for i, j in sorted(tile_coords):
                # Extract the tile from the normalized array.
                tile = dtm_normalized[i:i+tile_size, j:j+tile_size]
                
                # Construct the output PNG file name.
                tile_filename = f"{name_no_ext}_{tile_count}.png"
                tile_path = os.path.join(output_tiles_folder, tile_filename)
                
                # Save the tile as a greyscale PNG image.
                img = Image.fromarray(tile, mode='L')
                img.save(tile_path)
                
                # Compute the georeferenced coordinates of the top-left corner of the tile.
                x_corner, y_corner = xy(transform, j, i, offset='ul')
                
                # Write the tile file path and its top-left corner coordinates to the CSV.
                csv_writer.writerow([tile_path, x_corner, y_corner])
                
                tile_count += 1

def extract_tiles_from_greyscale_rasters(
        greyscale_rasters_folder, output_tiles_folder, tile_size=256, overlap=0.5, norm_min=0, norm_max=255):
    """
    Extracts overlapping training tiles from single-band (greyscale) raster TIFF files.
    Each tile is saved as a GeoTIFF with georeferencing.
    """
    os.makedirs(output_tiles_folder, exist_ok=True)
    raster_files = glob.glob(os.path.join(greyscale_rasters_folder, "*.tif"))
        
    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            dtm_array = src.read(1)
            src_crs = src.crs
            src_transform = src.transform
            height, width = dtm_array.shape
            
            # Replace NaN and infinite values; clip to the provided normalization range.
            dtm_array = np.nan_to_num(dtm_array, nan=norm_min, posinf=norm_max, neginf=norm_min)
            dtm_array = np.clip(dtm_array, norm_min, norm_max)
            
            # Normalize to 0-255
            dtm_normalized = (dtm_array - norm_min) / (norm_max - norm_min)
            #dtm_normalized = (dtm_normalized * 255).clip(0, 255).astype(np.uint8)

            step = int(tile_size * (1 - overlap))
            base_name = os.path.basename(raster_file)
            name_no_ext = os.path.splitext(base_name)[0]
            
            tile_coords = []
            for i in range(0, height - tile_size + 1, step):
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((i, j))
            if (width - tile_size) % step != 0:
                last_j = width - tile_size
                for i in range(0, height - tile_size + 1, step):
                    tile_coords.append((i, last_j))
            if (height - tile_size) % step != 0:
                last_i = height - tile_size
                for j in range(0, width - tile_size + 1, step):
                    tile_coords.append((last_i, j))
            if (width - tile_size) % step != 0 and (height - tile_size) % step != 0:
                tile_coords.append((height - tile_size, width - tile_size))

            tile_coords = list(set(tile_coords))
            tile_count = 1

            for i, j in sorted(tile_coords):
                window = Window(j, i, tile_size, tile_size)
                tile = dtm_normalized[i:i+tile_size, j:j+tile_size]
                tile_transform = src.window_transform(window)
                
                tile_filename = f"{name_no_ext}_{tile_count}.tif"
                tile_path = os.path.join(output_tiles_folder, tile_filename)
                
                with rasterio.open(
                    tile_path,
                    'w',
                    driver='GTiff',
                    height=tile.shape[0],
                    width=tile.shape[1],
                    count=1,
                    dtype=tile.dtype,
                    crs=src_crs,
                    transform=tile_transform,
                    compress='lzw'
                ) as dst:
                    dst.write(tile, 1)
                
                tile_count += 1


