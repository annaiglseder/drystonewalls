# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:40:05 2025

@author: aiglsede
"""

from osgeo import gdal
import numpy as np
import os


dtm = ".../dtm.tif"
wd_out = "../aspect"


os.makedirs(wd_out, exist_ok=True)
name, ext = os.path.splitext(os.path.basename(dtm))
aspect = os.path.join(wd_out, f"{name}_aspect{ext}")
sin =  os.path.join(wd_out, f"{name}_aspect_sin{ext}")
cos =  os.path.join(wd_out, f"{name}_aspect_cos{ext}")

gdal.DEMProcessing(
    aspect,
    dtm,
    "aspect",           
    computeEdges=True,  
)

# Open the aspect raster using GDAL
aspect_dataset = gdal.Open(aspect)
aspect_band = aspect_dataset.GetRasterBand(1)
aspect_val = aspect_band.ReadAsArray()

# GDAL metadata
geotransform = aspect_dataset.GetGeoTransform()
projection = aspect_dataset.GetProjection()

# Convert aspect values from degrees to radians (GDAL stores aspect in degrees)
aspect_radians = np.radians(aspect_val)

# Calculate the cosine and sine of the aspect
aspect_cosine = np.cos(aspect_radians)
aspect_sine = np.sin(aspect_radians)

# Function to save a raster
def save_raster(output_file, data, reference_dataset):
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(output_file, reference_dataset.RasterXSize, reference_dataset.RasterYSize, 1, gdal.GDT_Float32)
    out_raster.SetGeoTransform(reference_dataset.GetGeoTransform())  
    out_raster.SetProjection(reference_dataset.GetProjection())     
    out_raster.GetRasterBand(1).WriteArray(data)
    out_raster.FlushCache()  
    out_raster = None  

# Save the cosine of the aspect as a new GeoTIFF
save_raster(cos, aspect_cosine, aspect_dataset)
save_raster(sin, aspect_sine, aspect_dataset)

