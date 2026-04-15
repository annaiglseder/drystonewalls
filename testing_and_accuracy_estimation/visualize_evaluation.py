# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 16:11:28 2025

@author: aiglsede
"""

import os
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import split, linemerge
import numpy as np
from tqdm import tqdm
import glob
import pandas as pd

def combine_shapefiles_by_folder(
    parent_folder,
    folder_pattern,
    shapefile_pattern='[0-9][0-9][0-9][0-9]*.shp',
    output_dir=None
):
    """
    Combine shapefiles from subfolders matching a pattern.
    The output shapefile for each folder is named: <4digitprefix>_<folder_suffix>.shp

    Parameters:
    - parent_folder: str, path to the main folder
    - folder_pattern: str, starting pattern of the subfolders (e.g. 'prefix_')
    - shapefile_pattern: str, glob pattern for shapefiles (default: 4 digits at start)
    - output_dir: str, directory to save merged shapefiles (default: '<parent_folder>/combined_out')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if output_dir is None:
        output_dir = os.path.join(parent_folder, 'combined_out')
    os.makedirs(output_dir, exist_ok=True)

    for subfolder in os.listdir(parent_folder):
        if subfolder.startswith(folder_pattern):
            #subfolder = 'test_ref_skel_vec_mask'
            folder_suffix = subfolder[len(folder_pattern):]
            folder_path = os.path.join(parent_folder, subfolder)
            shapefiles = glob.glob(os.path.join(folder_path, shapefile_pattern))
            if not shapefiles:
                print(f"No shapefiles found in {folder_path}")
                continue

            gdfs = []
            for shp in shapefiles:
                gdf = gpd.read_file(shp)
                gdfs.append(gdf)
            combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

            basename = os.path.basename(shapefiles[0])
            prefix = basename[:4]
            out_name = f"{prefix}_{folder_suffix}.shp"
            out_path = os.path.join(output_dir, out_name)
            combined_gdf.to_file(out_path)
            print(f"Combined shapefile saved: {out_path}")
            
            

def cut_linestring(line, max_length):
    """Cut a LineString into shorter segments, max segment length = max_length."""
    if line.length <= max_length:
        return [line]
    points = [line.interpolate(d) for d in np.arange(0, line.length, max_length)]
    points.append(line.interpolate(line.length))
    return [LineString([points[i], points[i+1]]) for i in range(len(points)-1)]

def explode_to_segments(gdf, max_length):
    """Explode lines to max segment length."""
    new_geoms = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.type == 'LineString':
            lines = cut_linestring(geom, max_length)
        elif geom.type == 'MultiLineString':
            lines = []
            for part in geom.geoms:
                lines.extend(cut_linestring(part, max_length))
        else:
            continue
        for l in lines:
            r = row.copy()
            r.geometry = l
            new_geoms.append(r)
    return gpd.GeoDataFrame(new_geoms, columns=gdf.columns, crs=gdf.crs)

def segment_proximity_evaluation(
    reference_fp, model_dir, output_dir, 
    max_segment_length, buffer_distances,
    stat_list = ["min", "max", "mean"], prob_ths=["0.1", "0.2", "0.3", "0.4", "0.5"],
    verbose=True
):
    """
    For each model output, evaluate proximity of reference segments to model, and vice versa.
    Create for each model file:
        - Reference file with buffer columns (True/False for each buffer)
        - Model file with buffer columns (True/False for each buffer)
    Both ref and model are cut to max_segment_length before evaluation.

    Args:
        reference_fp (str): path to reference shapefile
        model_dir (str): path to dir with model shapefiles
        output_dir (str): dir to save outputs
        max_segment_length (float): maximum segment length (units = CRS units)
        buffer_distances (list of float): list of buffer distances (same units)
        stat_list (list): ["min", ...]
        prob_ths (list): ["0.1", ...]
        verbose (bool)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read reference
    ref_gdf = gpd.read_file(reference_fp)
    # Clip reference to segments
    ref_segments = explode_to_segments(ref_gdf, max_segment_length)
    ref_name = os.path.splitext(os.path.basename(reference_fp))[0]

    for stat in stat_list:
        for prob in prob_ths:
            model_fp = None
            # find file in model_dir with {stat}_{prob}
            for f in os.listdir(model_dir):
                if f.endswith(".shp") and stat in f and str(prob) in f:
                    model_fp = os.path.join(model_dir, f)
                    break
            if model_fp is None:
                if verbose: print(f"Missing model for {stat} {prob}")
                continue

            # Read and segment model
            model_gdf = gpd.read_file(model_fp)
            model_segments = explode_to_segments(model_gdf, max_segment_length)

            # Buffer model segments (for ref evaluation)
            model_buffers = {dist: model_segments.buffer(dist) for dist in buffer_distances}
            # Buffer ref segments (for model evaluation)
            ref_buffers = {dist: ref_segments.buffer(dist) for dist in buffer_distances}

            # Prepare output DataFrames
            ref_out = ref_segments.copy()
            model_out = model_segments.copy()
            
            # For each buffer: For each ref segment, does it intersect any model buffer?
            for dist in buffer_distances:
                colname = f"mbuf_{dist:.2f}"
                buffer_union = model_buffers[dist].unary_union
                ref_out[colname] = ref_out.geometry.apply(lambda x: x.intersects(buffer_union))
            
            # For each buffer: For each model segment, does it intersect any ref buffer?
            for dist in buffer_distances:
                colname = f"rbuf_{dist:.2f}"
                buffer_union = ref_buffers[dist].unary_union
                model_out[colname] = model_out.geometry.apply(lambda x: x.intersects(buffer_union))
            
            # Save reference output
            out_ref_fp = os.path.join(
                output_dir, f"{ref_name}_{stat}_{prob}.shp"
            )
            ref_out.to_file(out_ref_fp)
            
            # Save model output
            base_model_name = os.path.splitext(os.path.basename(model_fp))[0]
            out_model_fp = os.path.join(
                output_dir, f"{base_model_name}.shp"
            )
            model_out.to_file(out_model_fp)
            if verbose:
                print(f"Saved: {out_ref_fp} and {out_model_fp}")