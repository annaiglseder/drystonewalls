# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:54:19 2025

@author: aiglsede
"""

import os
import numpy as np
import rasterio
import scipy.ndimage as nd

def compute_tpi(dtm, kernel_type='square', **kwargs):
    """
    Compute the Topographic Position Index (TPI) for a DTM array.
    
    TPI is defined as:
        TPI = pixel value - (mean value of the defined neighborhood)

    The neighborhood is defined by a kernel, which can be a square, a circle, or a ring.
    The center pixel is excluded from the neighborhood.

    Parameters
    ----------
    dtm : numpy.ndarray
        2D array representing the digital terrain model.
    kernel_type : str, optional
        Neighborhood type: 'square', 'circle', or 'ring'.
    **kwargs : dict
        Additional parameters to define the neighborhood:
          - For 'square':
                half_size : int (default=1)
                (defines a (2*half_size+1) x (2*half_size+1) window)
          - For 'circle':
                radius : float (default=1.0)
          - For 'ring':
                inner_radius : float (default=1.0)
                outer_radius : float (default=2.0)

    Returns
    -------
    tpi : numpy.ndarray
        2D array of TPI values.
    """
    if kernel_type == 'square':
        half_size = kwargs.get('half_size', 1)
        kernel = np.ones((2 * half_size + 1, 2 * half_size + 1), dtype=float)
        kernel[half_size, half_size] = 0.0  # exclude the center
    elif kernel_type == 'circle':
        radius = kwargs.get('radius', 1.0)
        r = int(np.ceil(radius))
        y, x = np.ogrid[-r:r+1, -r:r+1]
        mask = np.sqrt(x**2 + y**2) <= radius
        mask[r, r] = False  # exclude the center
        kernel = mask.astype(float)
    elif kernel_type == 'ring':
        inner_radius = kwargs.get('inner_radius', 1.0)
        outer_radius = kwargs.get('outer_radius', 2.0)
        r = int(np.ceil(outer_radius))
        y, x = np.ogrid[-r:r+1, -r:r+1]
        dist = np.sqrt(x**2 + y**2)
        mask = (dist >= inner_radius) & (dist <= outer_radius)
        mask[r, r] = False  # exclude the center
        kernel = mask.astype(float)
    else:
        raise ValueError("Invalid kernel_type. Choose 'square', 'circle', or 'ring'.")

    # Convolve to get local sum and count of contributing neighbors
    local_sum = nd.convolve(dtm, kernel, mode='reflect')
    count = nd.convolve(np.ones_like(dtm), kernel, mode='reflect')
    with np.errstate(divide='ignore', invalid='ignore'):
        local_mean = np.where(count > 0, local_sum / count, np.nan)
    tpi = dtm - local_mean
    return tpi

def process_dtm_tpi(dtm_path, output_folder, kernel_type='square', **kwargs):
    """
    Process a DTM TIFF file to compute its Topographic Position Index (TPI) and save the result as a new TIFF.
    
    The output TIFF will have the same spatial grid, metadata, and CRS as the input.
    Its filename is constructed by inserting "_tpi_{kernel_type}_{kwargs}" before the ".tif" extension.

    Parameters
    ----------
    dtm_path : str
        File path to the input DTM TIFF.
    kernel_type : str, optional
        Neighborhood type to use ('square', 'circle', or 'ring').
    **kwargs : dict
        Additional parameters for the neighborhood:
            - For 'square': half_size (int, default=1)
            - For 'circle': radius (float, default=1.0)
            - For 'ring': inner_radius (float, default=1.0) and outer_radius (float, default=2.0)
    """
    # Read the DTM and its metadata from the TIFF file.
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)  # Assumes a single-band DTM.
        profile = src.profile

    # Compute the TPI.
    tpi = compute_tpi(dtm, kernel_type, **kwargs)

    # Update the profile for the output (ensure data type consistency).
    profile.update(dtype=tpi.dtype, count=1)

    # Construct the output filename.
    file_name, ext = os.path.splitext(os.path.basename(dtm_path))
    #base, ext = os.path.splitext(dtm_path)
    params_str = "_".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
    #params_str = "radius25"
    out_filename = os.path.join(output_folder, f"{file_name}_tpi_{kernel_type}")
    if params_str:
        out_filename += f"_{params_str}"
    out_filename += ext

    # Write the TPI to the new TIFF file.
    with rasterio.open(out_filename, 'w', **profile) as dst:
        dst.write(tpi, 1)

    print(f"TPI saved to: {out_filename}")
    
    
def absolute_raster(input_file_path):
    """
    Reads a raster TIFF file, computes the absolute (positive) values of each pixel,
    and writes the result to a new file with '_absolute' added before the .tif extension.
    
    Parameters:
        input_file_path (str): The file path to the input TIFF raster.
    
    Returns:
        str: The file path to the output TIFF raster with absolute values.
    """
    # Create the output file path by inserting '_absolute' before the file extension.
    base, ext = os.path.splitext(input_file_path)
    output_file_path = f"{base}_abs{ext}"
    
    # Open the input raster file.
    with rasterio.open(input_file_path) as src:
        # Read all bands from the raster.
        data = src.read()  # Shape: (bands, rows, cols)
        
        # Compute the absolute value for every pixel.
        data_abs = np.abs(data)
        
        # Copy the metadata to use for writing the output file.
        meta = src.meta.copy()
    
    # Write the absolute values to the new raster file.
    with rasterio.open(output_file_path, 'w', **meta) as dst:
        dst.write(data_abs)
    
    return output_file_path



