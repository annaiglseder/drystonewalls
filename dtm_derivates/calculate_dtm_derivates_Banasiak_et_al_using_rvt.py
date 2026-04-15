# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:44:29 2025

@author: aiglsede
"""

import os
os.chdir("H:/scripts/semona/trockensteinmauern/dtm_derivates")
import rasterio
import rvt.vis
import rvt.default
import matplotlib.pyplot as plt
import numpy as np
from create_rgb_composite_and_normalize import create_rgb_composite
from rasterio.windows import from_bounds


'''
SLRM (Simple Local Relief Model—radius for trend assessment 20 pixels, 8 bits), 
LD (Local Dominanceminimum/maximum pixel radius 10/20) and 
an analytical hillshading (Z-factor = 4) were created.


get code from here: https://github.com/EarthObservation/RVT_py/blob/master/examples/rvt_vis_example.ipynb


'''


#dtm_path = r"D:\PHOTO\SEMONA\NAWA\tsm_reference_data\3405\dtm_derivates\NOE_LAZE02_T_3405_DTM_0.25.tif"
dtm_path = ".../dtm.tif"
slrm_path = ".../rvt_slrm.tif"
local_dom_path = ".../rvt_ldom.tif"
multi_hillshade_path = ".../rvt_mhs.tif"
multi_hillshade_single_path = ".../rvt_mhs_single.tif"

rgb_comp = ".../rvt_slrm_ldom_mhs.tif"

dict_dem = rvt.default.get_raster_arr(dtm_path)
dem_arr = dict_dem["array"]  # numpy array of DEM
dem_resolution = dict_dem["resolution"]
dem_res_x = dem_resolution[0]  # resolution in X direction
dem_res_y = dem_resolution[1]  # resolution in Y direction
dem_no_data = dict_dem["no_data"]

#plt.imshow(dem_arr, cmap='gray')

# SLRM (simple local relief model)

radius_cell = 20  # radius to consider in pixels (not in meters)
slrm_arr = rvt.vis.slrm(dem=dem_arr, radius_cell=radius_cell, ve_factor=1, no_data=dem_no_data)

#plt.imshow(slrm_arr, cmap='gray')

#slrm_path = r"../test_data/TM1_564_146_slrm.tif"
rvt.default.save_raster(src_raster_path=dtm_path, out_raster_path=slrm_path, out_raster_arr=slrm_arr,
                        no_data=np.nan, e_type=6)


# LD (local dominance)

min_rad = 10  # minimum radial distance
max_rad = 20  # maximum radial distance
rad_inc = 1  # radial distance steps in pixels
angular_res = 15 # angular step for determination of number of angular directions
observer_height = 1.7  # height at which we observe the terrain
local_dom_arr = rvt.vis.local_dominance(dem=dem_arr, min_rad=min_rad, max_rad=max_rad, rad_inc=rad_inc, angular_res=angular_res,
                                       observer_height=observer_height, ve_factor=1,
                                       no_data=dem_no_data)

#plt.imshow(local_dom_arr, cmap='gray')

#local_dom_path = r"../test_data/TM1_564_146_local_dominance.tif"
rvt.default.save_raster(src_raster_path=dtm_path, out_raster_path=local_dom_path, out_raster_arr=local_dom_arr,
                        no_data=np.nan, e_type=6)


# Multidir Hillshade

nr_directions = 16  # Number of solar azimuth angles (clockwise from North) (number of directions, number of bands)
sun_elevation = 45  # Solar vertical angle (above the horizon) in degrees
multi_hillshade_arr = rvt.vis.multi_hillshade(dem=dem_arr, resolution_x=dem_res_x, resolution_y=dem_res_y,
                                              nr_directions=nr_directions, sun_elevation=sun_elevation, ve_factor=1,
                                              no_data=dem_no_data)

#plt.imshow(multi_hillshade_arr[0], cmap='gray')  # plot first direction where solar azimuth = 22.5 (360/16=22.5)



#multi_hillshade_path = r"../test_data/TM1_564_146_multi_hillshade.tif"
rvt.default.save_raster(src_raster_path=dtm_path, out_raster_path=multi_hillshade_path, out_raster_arr=multi_hillshade_arr,
                        no_data=np.nan, e_type=6)



# create mean of multiple bands


with rasterio.open(multi_hillshade_path) as src:
    # Read all three bands into a 3D array (bands, rows, cols)
    data = src.read()  # Shape: (3, height, width)

    mean_data = data.mean(axis=0)

    profile = src.profile
    profile.update(count=1, dtype=mean_data.dtype)

    with rasterio.open(multi_hillshade_single_path, 'w', **profile) as dst:
        dst.write(mean_data, 1)


# cut to same overlap


rasters = [slrm_path, local_dom_path, multi_hillshade_single_path]  # Update these paths!
# ---------------------------

# 1. Read all bounds and transforms
infos = []
for path in rasters:
    with rasterio.open(path) as src:
        infos.append({
            "path": path,
            "bounds": src.bounds,
            "crs": src.crs,
            "transform": src.transform,
            "dtype": src.dtypes[0],
            "count": src.count,
            "nodata": src.nodata,
            "profile": src.profile
        })

# 2. Check all have the same CRS
crs_set = {info["crs"] for info in infos}
if len(crs_set) > 1:
    raise ValueError("Input rasters do not have the same CRS!")

# 3. Compute intersection (overlapping area)
xmin = max(info["bounds"].left for info in infos)
ymin = max(info["bounds"].bottom for info in infos)
xmax = min(info["bounds"].right for info in infos)
ymax = min(info["bounds"].top for info in infos)
overlap_bounds = (xmin, ymin, xmax, ymax)

# 4. Cut and overwrite each raster
for info in infos:
    with rasterio.open(info["path"]) as src:
        # Get window for overlap
        window = from_bounds(*overlap_bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()  # ensure integer window
        
        # Read data
        data = src.read(window=window)
        
        # Update transform and shape
        new_transform = src.window_transform(window)
        new_profile = info["profile"].copy()
        new_profile.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": new_transform
        })
        
    # Overwrite original file
    with rasterio.open(info["path"], 'w', **new_profile) as dst:
        dst.write(data)

print("All rasters have been cropped to their overlapping area.")


'''
out_file = create_rgb_composite(
   slrm_path, local_dom_path, multi_hillshade_single_path,
   red_norm_method='quantile', red_norm_params={'qmin': 0.02, 'qmax': 0.98},
   green_norm_method='quantile', green_norm_params={'qmin': 0.02, 'qmax': 0.98},
   blue_norm_method='quantile', blue_norm_params={'qmin': 0.02, 'qmax': 0.98},
   out_path= rgb_comp
)
'''

out_file = create_rgb_composite(
   slrm_path, local_dom_path, multi_hillshade_single_path,
   red_norm_method='auto',
   green_norm_method='auto',
   blue_norm_method='auto',
   out_path= rgb_comp
)


