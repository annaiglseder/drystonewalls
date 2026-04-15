# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:20:15 2025

@author: aiglsede
"""

import numpy as np
import rasterio


def create_rgb_composite(red_tif, green_tif, blue_tif,
                         red_norm_method='auto', red_norm_params=None,
                         green_norm_method='auto', green_norm_params=None,
                         blue_norm_method='auto', blue_norm_params=None,
                         out_path=None):
    """
    Creates an RGB composite from three TIFF files with individual normalization options for each channel.
    
    Parameters
    ----------
    red_tif : str
        File path to the red channel (TIFF file).
    green_tif : str
        File path to the green channel (TIFF file).
    blue_tif : str
        File path to the blue channel (TIFF file).
    red_norm_method : str, optional
        Normalization method for the red channel. Options are:
          - 'auto': Use the image's actual min and max values.
          - 'quantile': Use quantiles. Expects red_norm_params to provide keys 'qmin' and 'qmax'
                        (default values are 0.02 and 0.98 if not provided).
          - 'fixed': Use a fixed range. Expects red_norm_params to provide keys 'min' and 'max'.
        Default is 'auto'.
    red_norm_params : dict, optional
        Dictionary with additional parameters for normalizing the red channel.
    green_norm_method : str, optional
        Normalization method for the green channel (see red_norm_method for options).
    green_norm_params : dict, optional
        Dictionary with additional parameters for normalizing the green channel.
    blue_norm_method : str, optional
        Normalization method for the blue channel (see red_norm_method for options).
    blue_norm_params : dict, optional
        Dictionary with additional parameters for normalizing the blue channel.
    out_path : str, optional
        If provided, the RGB composite is written as a TIFF file to this path.
        If not provided, the function returns the composite as a numpy array.
    
    Returns
    -------
    If out_path is provided, returns the output file path.
    Otherwise, returns a numpy array of shape (3, height, width) with float32 values in [0, 1].
    """
    # Helper function to normalize a single image
    def normalize(image, method='auto', params=None):
        if params is None:
            params = {}
        if method == 'auto':
            vmin, vmax = np.nanmin(image), np.nanmax(image)
        elif method == 'quantile':
            # Use provided quantiles or defaults (e.g., 2% and 98%)
            qmin = params.get('qmin', 0.02)
            qmax = params.get('qmax', 0.98)
            vmin, vmax = np.quantile(image, qmin), np.quantile(image, qmax)
        elif method == 'fixed':
            # Use fixed min and max if provided
            vmin = params.get('min', np.nanmin(image))
            vmax = params.get('max', np.nanmax(image))
        else:
            raise ValueError("Unsupported normalization method. Choose 'auto', 'quantile', or 'fixed'.")
    
        # Avoid division by zero
        if vmax - vmin == 0:
            norm_image = np.zeros_like(image, dtype=np.float32)
        else:
            norm_image = (image - vmin) / (vmax - vmin)
        # Ensure values are within [0, 1]
        return np.clip(norm_image, 0, 1).astype(np.float32)
    
    # Read the red channel and store metadata for later use
    with rasterio.open(red_tif) as src:
        red_data = src.read(1)
        meta = src.meta.copy()
    
    # Read the green channel
    with rasterio.open(green_tif) as src:
        green_data = src.read(1)
    
    # Read the blue channel
    with rasterio.open(blue_tif) as src:
        blue_data = src.read(1)
    
    # Ensure all images have the same dimensions
    if red_data.shape != green_data.shape or red_data.shape != blue_data.shape:
        raise ValueError("All input images must have the same dimensions.")
    
    # Normalize each channel individually using the specified method and parameters
    red_norm = normalize(red_data, red_norm_method, red_norm_params)
    green_norm = normalize(green_data, green_norm_method, green_norm_params)
    blue_norm = normalize(blue_data, blue_norm_method, blue_norm_params)
    
    # Stack the normalized bands into an RGB composite array
    rgb = np.stack([red_norm, green_norm, blue_norm])
    
    # If an output path is provided, write the composite to a new TIFF file
    if out_path:
        meta.update(count=3, dtype=rasterio.float32)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(rgb)
        return out_path
    else:
        return rgb














