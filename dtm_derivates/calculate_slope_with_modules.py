# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:46:25 2025

@author: aiglsede
"""

from osgeo import gdal

def calculate_slope_gdal(input_dem_path, output_slope_path, scale=1.0, slope_format='degree', algorithm='Horn'):
    """
    Calculate slope from a DEM using GDAL's DEMProcessing.

    Parameters:
        input_dem_path (str): Path to the input DEM file.
        output_slope_path (str): Path where the output slope raster will be saved.
        scale (float): Z-factor to convert elevation units to horizontal units (default is 1.0).
        slope_format (str): Output slope format. Options are 'degree' or 'percent' (default is 'degree').
        algorithm (str): Slope algorithm to use. Common options include 'Horn' and 'ZevenbergenThorne' (default is 'Horn').

    Returns:
        None. The function writes the output raster to output_slope_path.
    """
    # Open the input DEM
    ds = gdal.Open(input_dem_path)
    if ds is None:
        raise RuntimeError(f"Unable to open the DEM file: {input_dem_path}")

    # Use GDAL's DEMProcessing to compute the slope
    result = gdal.DEMProcessing(
        output_slope_path,
        ds,
        'slope',
        scale=scale,
        slopeFormat=slope_format,
        alg=algorithm
    )

    if result is None:
        raise RuntimeError("Error processing slope. Check input DEM and parameters.")

    print(f"Slope calculation completed. Output saved to {output_slope_path}")



