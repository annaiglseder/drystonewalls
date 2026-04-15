# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 23:05:34 2025

@author: aiglsede
"""


import os
import math
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
import fiona
import geopandas as gpd
from shapely.geometry import box
import numpy as np


def rasterize_gpkgs_with_pairwise_refs(
    gpkg_folder,
    ref_raster_folder,
    out_folder,
    attribute="tsm_prob",
    layer=None,                 # set a layer name, else first layer in GPKG
    all_touched=False,
    dtype="float32",
    nodata=np.nan,
    merge_alg="replace"         # "replace" or "add" (portable across Rasterio versions)
):
    os.makedirs(out_folder, exist_ok=True)

    # portable merge algs
    merge_alg_map = {
        "replace": MergeAlg.replace,
        "add":     MergeAlg.add,
    }
    if merge_alg not in merge_alg_map:
        raise ValueError(f"merge_alg must be one of {list(merge_alg_map)}")

    for name in os.listdir(gpkg_folder):
        if not name.lower().endswith(".gpkg"):
            continue

        gpkg_path = os.path.join(gpkg_folder, name)
        base = os.path.splitext(name)[0]
        ref_path = os.path.join(ref_raster_folder, f"{base}_mask.tif")

        if not os.path.exists(ref_path):
            print(f"[WARN] No reference raster for {name} at {ref_path}. Skipping.")
            continue

        # --- lock output to the reference grid ---
        with rasterio.open(ref_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
            ref_bounds = ref.bounds

        rio_dtype = np.dtype(dtype).name
        out_meta = {
            "driver": "GTiff",
            "height": ref_height,
            "width": ref_width,
            "count": 1,
            "dtype": rio_dtype,
            "crs": ref_crs,
            "transform": ref_transform,
            "nodata": nodata,
            "compress": "deflate",
            "predictor": 2 if "float" in rio_dtype else 3,
            "tiled": True,
            "blockxsize": 512,   # multiples of 16
            "blockysize": 512,
        }

        # pick layer
        lyr = layer
        if lyr is None:
            try:
                layers = fiona.listlayers(gpkg_path)
                if not layers:
                    print(f"[WARN] No layers in {gpkg_path}, skipping.")
                    continue
                lyr = layers[0]
            except Exception as e:
                print(f"[WARN] Could not list layers for {gpkg_path}: {e}")
                continue

        # read vector
        try:
            gdf = gpd.read_file(gpkg_path, layer=lyr)
        except Exception as e:
            print(f"[WARN] Failed to read {gpkg_path} (layer '{lyr}'): {e}")
            continue

        # Prepare shapes (clip to ref bbox to avoid spill & speed up)
        shapes = []
        if not gdf.empty:
            if attribute not in gdf.columns:
                print(f"[WARN] {gpkg_path}:{lyr} missing '{attribute}'. Filling with 0.")
                gdf[attribute] = 0.0

            if gdf.crs is None:
                print(f"[WARN] {gpkg_path}:{lyr} has no CRS. Assuming reference CRS.")
                gdf = gdf.set_crs(ref_crs)
            elif gdf.crs != ref_crs:
                gdf = gdf.to_crs(ref_crs)

            ref_bbox = box(*ref_bounds)

            # quick reject: if nothing intersects the bbox, skip heavy ops
            try:
                if not gdf.total_bounds:
                    gdf = gdf.iloc[0:0]
                else:
                    minx, miny, maxx, maxy = gdf.total_bounds
                    if (maxx < ref_bounds.left or minx > ref_bounds.right or
                        maxy < ref_bounds.bottom or miny > ref_bounds.top):
                        gdf = gdf.iloc[0:0]
            except Exception:
                pass

            if not gdf.empty:
                try:
                    gdf = gdf.clip(ref_bbox)
                except Exception:
                    gdf = gdf[gdf.geometry.intersects(ref_bbox)]

                gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
                gdf = gdf[~gdf[attribute].isnull()]

                if not gdf.empty:
                    shapes = [(geom, float(val)) for geom, val in zip(gdf.geometry, gdf[attribute])]

        # Initialize full grid with nodata (guarantees identical extent to reference)
        out_arr = np.full((ref_height, ref_width), nodata, dtype=rio_dtype)

        if shapes:
            out_arr = rasterize(
                shapes=shapes,
                out_shape=(ref_height, ref_width),
                transform=ref_transform,
                fill=nodata,
                all_touched=all_touched,
                dtype=rio_dtype,
                merge_alg=merge_alg_map[merge_alg],
            )

            # guard: integer dtype cannot use NaN nodata
            if np.issubdtype(np.dtype(dtype), np.integer) and isinstance(nodata, float) and math.isnan(nodata):
                raise ValueError("Integer dtype cannot use NaN as nodata. Choose an integer nodata value.")

        out_path = os.path.join(out_folder, f"{base}.tif" if layer is None else f"{base}_{lyr}.tif")
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_arr, 1)

        print(f"[OK] {os.path.basename(out_path)} written (extent = {ref_bounds})")

def rasterize_geopackages(
    gpkg_folder,
    reference_raster,
    out_folder,
    attribute="tsm_prob",
    all_touched=False,
    dtype="float32",
    nodata=np.nan,
    merge_alg="replace",      # "replace" or "add" (broadly supported)
    layer=None
):
    os.makedirs(out_folder, exist_ok=True)

    # --- Read reference grid metadata (sets the **exact** output extent) ---
    with rasterio.open(reference_raster) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds   # (left, bottom, right, top)

    # Output profile is locked to reference grid
    rio_dtype = np.dtype(dtype).name
    out_meta = {
        "driver": "GTiff",
        "height": ref_height,
        "width": ref_width,
        "count": 1,
        "dtype": rio_dtype,
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2 if "float" in rio_dtype else 3,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    # Merge algs available on all Rasterio versions
    merge_alg_map = {
        "replace": MergeAlg.replace,
        "add":     MergeAlg.add,
    }
    if merge_alg not in merge_alg_map:
        raise ValueError(f"merge_alg must be one of {list(merge_alg_map)}")

    ref_bbox_geom = box(*ref_bounds)

    # --- Iterate over GPKGs ---
    for name in os.listdir(gpkg_folder):
        if not name.lower().endswith(".gpkg"):
            continue
        gpkg_path = os.path.join(gpkg_folder, name)

        # Decide layer
        lyr = layer
        if lyr is None:
            try:
                layers = fiona.listlayers(gpkg_path)
                if not layers:
                    print(f"[WARN] No layers in {gpkg_path}, skipping.")
                    continue
                lyr = layers[0]
            except Exception as e:
                print(f"[WARN] Could not list layers for {gpkg_path}: {e}")
                continue

        # Read vectors
        try:
            gdf = gpd.read_file(gpkg_path, layer=lyr)
        except Exception as e:
            print(f"[WARN] Failed to read {gpkg_path} (layer '{lyr}'): {e}")
            continue

        if gdf.empty:
            shapes = []
        else:
            # Ensure attribute
            if attribute not in gdf.columns:
                print(f"[WARN] {gpkg_path}:{lyr} missing '{attribute}'. Filling with 0.")
                gdf[attribute] = 0.0

            # CRS handling
            if gdf.crs is None:
                print(f"[WARN] {gpkg_path}:{lyr} has no CRS. Assuming reference CRS.")
                gdf = gdf.set_crs(ref_crs)
            elif gdf.crs != ref_crs:
                gdf = gdf.to_crs(ref_crs)

            # Clip to reference raster extent (guarantees no spill & speeds up)
            try:
                gdf = gdf.clip(ref_bbox_geom)
            except Exception:
                gdf = gdf[gdf.geometry.intersects(ref_bbox_geom)]

            # Clean
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
            gdf = gdf[~gdf[attribute].isnull()]

            shapes = [(geom, float(val)) for geom, val in zip(gdf.geometry, gdf[attribute])]

        # Prepare output array filled with nodata over the **entire** ref extent
        out_arr = np.full((ref_height, ref_width), nodata, dtype=rio_dtype)

        # Rasterize into that fixed grid
        if shapes:
            out_arr = rasterize(
                shapes=shapes,
                out_shape=(ref_height, ref_width),
                transform=ref_transform,
                fill=nodata,
                all_touched=all_touched,
                dtype=rio_dtype,
                merge_alg=merge_alg_map[merge_alg],
            )

            # Guard: integer dtype cannot use NaN nodata
            if np.issubdtype(np.dtype(dtype), np.integer) and isinstance(nodata, float) and math.isnan(nodata):
                raise ValueError("Integer dtype cannot use NaN as nodata. Choose an integer nodata value.")

        # Write (same spatial extent as reference by design)
        base = os.path.splitext(name)[0]
        out_name = f"{base}.tif" if layer is None else f"{base}_{lyr}.tif"
        out_path = os.path.join(out_folder, out_name)
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_arr, 1)

        print(f"[OK] Wrote {out_path} (extent locked to reference grid)")




def fill_missing_rasters_like_ref(
    mask_folder,
    data_folder,
    mask_suffix="_mask.tif",
    data_ext=".tif",
    override_dtype="float32",      # e.g., "float32"; if None, copy mask dtype
    override_nodata=np.nan,      # e.g., np.nan or -9999; if None, copy mask nodata
    compress="deflate",
    blocksize=256,
):
    """
    For every <name>{mask_suffix} in mask_folder, ensure <name>{data_ext} exists in data_folder.
    If missing, create a raster with identical grid (CRS, transform, width, height) and fill with nodata.

    - If override_dtype/override_nodata are None, copy dtype/nodata from the mask.
    - If mask has no nodata and dtype is integer, uses -9999 by default.
      If mask has no nodata and dtype is float, uses np.nan by default.
    """
    os.makedirs(data_folder, exist_ok=True)
    made, skipped = 0, 0

    for fname in os.listdir(mask_folder):
        if not fname.lower().endswith(mask_suffix.lower()):
            continue

        base = fname[:-len(mask_suffix)]  # strip the exact mask suffix
        mask_path = os.path.join(mask_folder, fname)
        out_name = base + data_ext
        out_path = os.path.join(data_folder, out_name)

        if os.path.exists(out_path):
            skipped += 1
            continue

        # Read mask grid
        with rasterio.open(mask_path) as src:
            ref_crs = src.crs
            ref_transform = src.transform
            ref_width = src.width
            ref_height = src.height
            mask_dtype = src.dtypes[0]
            mask_nodata = src.nodata

        # Decide dtype / nodata
        dtype = override_dtype if override_dtype is not None else mask_dtype
        np_dtype = np.dtype(dtype)

        if override_nodata is not None:
            nodata = override_nodata
        else:
            if mask_nodata is not None:
                nodata = mask_nodata
            else:
                # sensible defaults if mask has no nodata
                if np.issubdtype(np_dtype, np.floating):
                    nodata = np.nan
                else:
                    nodata = -9999  # integer fallback

        # Guard: NaN only valid for float dtypes
        if isinstance(nodata, float) and np.isnan(nodata) and not np.issubdtype(np_dtype, np.floating):
            raise ValueError("NaN nodata requested for an integer dtype. Choose a numeric integer nodata (e.g. -9999).")

        # Create an all-nodata array
        fill_value = nodata
        # For float NaN, np.full works fine; for ints, ensure type matches
        out_arr = np.full((ref_height, ref_width), fill_value, dtype=np_dtype)

        # Build profile
        profile = {
            "driver": "GTiff",
            "height": ref_height,
            "width": ref_width,
            "count": 1,
            "dtype": np_dtype.name,
            "crs": ref_crs,
            "transform": ref_transform,
            "nodata": nodata,
            "compress": compress,
            "tiled": True,
            "blockxsize": blocksize,
            "blockysize": blocksize,
            # predictor 2 for float, 3 for integer works well with deflate
            #"predictor": 2 if np.issubdtype(np_dtype, np.floating) else 3,
        }

        # Write it
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_arr, 1)

        made += 1
        print(f"[OK] Created {out_path} (like {mask_path})")

    print(f"Done. Created: {made}, already existed: {skipped}")




'''
import os
import math
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from shapely.geometry import box
import fiona
import geopandas as gpd




def rasterize_geopackages(
    gpkg_folder,
    reference_raster,
    out_folder,
    attribute="tsm_prob",
    all_touched=False,
    dtype="float32",          # use float (safer for probabilities)
    nodata=np.nan,            # NaN for float; set e.g. -9999 if you prefer
    merge_alg="replace",      # options: replace, add, max, min
    layer=None                # set a layer name if your GPKGs have multiple layers
):
    os.makedirs(out_folder, exist_ok=True)

    # --- Read reference grid metadata ---
    with rasterio.open(reference_raster) as ref:
        ref_meta = ref.meta.copy()
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    # force output dtype & nodata (decoupled from reference raster’s dtype)
    rio_dtype = getattr(np, dtype).name
    out_meta = {
        "driver": "GTiff",
        "height": ref_height,
        "width": ref_width,
        "count": 1,
        "dtype": rio_dtype,
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2 if "float" in rio_dtype else 3,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    # pick merge algorithm
    merge_alg_map = {
        "replace": MergeAlg.replace,
        "add":     MergeAlg.add#,
        #"max":     MergeAlg.max,
        #"min":     MergeAlg.min,
    }
    if merge_alg not in merge_alg_map:
        raise ValueError(f"merge_alg must be one of {list(merge_alg_map)}")

    # --- Iterate over GeoPackages ---
    for name in os.listdir(gpkg_folder):
        if not name.lower().endswith(".gpkg"):
            continue
        gpkg_path = os.path.join(gpkg_folder, name)

        # Determine layer to read
        lyr = layer
        if lyr is None:
            try:
                layers = fiona.listlayers(gpkg_path)
                if not layers:
                    print(f"[WARN] No layers in {gpkg_path}, skipping.")
                    continue
                lyr = layers[0]  # take first layer by default
            except Exception as e:
                print(f"[WARN] Could not list layers for {gpkg_path}: {e}")
                continue

        # Load vector
        try:
            gdf = gpd.read_file(gpkg_path, layer=lyr)
        except Exception as e:
            print(f"[WARN] Failed to read {gpkg_path} (layer '{lyr}'): {e}")
            continue

        if gdf.empty:
            print(f"[INFO] {gpkg_path}:{lyr} has no features. Writing empty raster.")
        else:
            # Ensure attribute exists
            if attribute not in gdf.columns:
                print(f"[WARN] {gpkg_path}:{lyr} missing '{attribute}'. "
                      f"Filling with 0.")
                gdf[attribute] = 0.0

            # Reproject to match reference CRS
            if gdf.crs is None:
                print(f"[WARN] {gpkg_path}:{lyr} has no CRS. Assuming same as reference.")
                gdf.set_crs(ref_crs, inplace=True)
            elif gdf.crs != ref_crs:
                gdf = gdf.to_crs(ref_crs)

            # Drop rows with null geometries or null attribute
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
            gdf = gdf[~gdf[attribute].isnull()]

        # Prepare shapes iterable: (geometry, value)
        shapes = [(geom, float(val)) for geom, val in zip(gdf.geometry, gdf[attribute])] if not gdf.empty else []

        # Create output filename
        base = os.path.splitext(name)[0]
        out_name = f"{base}.tif" if layer is None else f"{base}_{lyr}.tif"
        out_path = os.path.join(out_folder, out_name)

        # Rasterize
        out_arr = np.full((ref_height, ref_width), nodata, dtype=rio_dtype)

        if shapes:
            out_arr = rasterize(
                shapes=shapes,
                out_shape=(ref_height, ref_width),
                transform=ref_transform,
                fill=nodata,
                all_touched=all_touched,
                dtype=rio_dtype,
                merge_alg=merge_alg_map[merge_alg],
            )

            # for integer nodata with NaN requested: ensure integer nodata
            if np.issubdtype(getattr(np, dtype), np.integer) and (isinstance(nodata, float) and math.isnan(nodata)):
                raise ValueError("Integer dtype cannot use NaN as nodata. Pick an integer nodata value.")

        # Write raster
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_arr, 1)

        print(f"[OK] Wrote {out_path}")

# ---- Example call ----
# rasterize_geopackages(
#     gpkg_folder=r"D:/data/walls_gpkg",
#     reference_raster=r"D:/data/grid/reference.tif",
#     out_folder=r"D:/data/walls_rasters",
#     attribute="tsm_prob",
#     all_touched=False,   # set True if you want any-touch behavior
#     dtype="float32",
#     nodata=np.nan,
#     merge_alg="replace", # or "max" if you want highest prob where overlaps
#     layer=None           # set to a layer string if needed
# )


def rasterize_gpkgs_with_pairwise_refs(
    gpkg_folder,
    ref_raster_folder,
    out_folder,
    attribute="tsm_prob",
    layer=None,                 # set to a layer name, else first layer in GPKG
    all_touched=False,
    dtype="float32",
    nodata=np.nan,
    merge_alg="replace"         # "replace" | "add" | "max" | "min"
):
    os.makedirs(out_folder, exist_ok=True)

    merge_alg_map = {
        "replace": MergeAlg.replace,
        "add":     MergeAlg.add#,
        #"max":     MergeAlg.max,
        #"min":     MergeAlg.min,
    }
    if merge_alg not in merge_alg_map:
        raise ValueError(f"merge_alg must be one of {list(merge_alg_map)}")

    # walk gpkg folder
    for name in os.listdir(gpkg_folder):
        if not name.lower().endswith(".gpkg"):
            continue

        gpkg_path = os.path.join(gpkg_folder, name)
        base = os.path.splitext(name)[0]
        ref_path = os.path.join(ref_raster_folder, f"{base}_mask.tif")

        if not os.path.exists(ref_path):
            print(f"[WARN] No reference raster for {name} at {ref_path}. Skipping.")
            continue

        # open reference raster -> grid/CRS/transform/extent
        with rasterio.open(ref_path) as ref:
            ref_profile = ref.profile.copy()
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
            ref_bounds = ref.bounds

        # output profile (override dtype, nodata, compression)
        rio_dtype = np.dtype(dtype).name
        out_meta = {
            "driver": "GTiff",
            "height": ref_height,
            "width": ref_width,
            "count": 1,
            "dtype": rio_dtype,
            "crs": ref_crs,
            "transform": ref_transform,
            "nodata": nodata,
            "compress": "deflate",
            "predictor": 2 if "float" in rio_dtype else 3,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }

        # pick layer
        lyr = layer
        if lyr is None:
            try:
                layers = fiona.listlayers(gpkg_path)
                if not layers:
                    print(f"[WARN] No layers in {gpkg_path}, skipping.")
                    continue
                lyr = layers[0]
            except Exception as e:
                print(f"[WARN] Could not list layers for {gpkg_path}: {e}")
                continue

        # read vector
        try:
            gdf = gpd.read_file(gpkg_path, layer=lyr)
        except Exception as e:
            print(f"[WARN] Failed to read {gpkg_path} (layer '{lyr}'): {e}")
            continue

        if gdf.empty:
            print(f"[INFO] {gpkg_path}:{lyr} has no features. Writing empty raster.")
        else:
            # ensure attribute
            if attribute not in gdf.columns:
                print(f"[WARN] {gpkg_path}:{lyr} missing '{attribute}'. Filling with 0.")
                gdf[attribute] = 0.0

            # CRS handling
            if gdf.crs is None:
                print(f"[WARN] {gpkg_path}:{lyr} has no CRS. Assuming reference CRS.")
                gdf = gdf.set_crs(ref_crs)
            elif gdf.crs != ref_crs:
                gdf = gdf.to_crs(ref_crs)

            # fast bbox clip to ref extent to reduce work
            bbox_geom = box(*ref_bounds)
            try:
                gdf = gdf.clip(bbox_geom)
            except Exception:
                # fallback if clip fails for any odd geometry issues
                gdf = gdf[gdf.geometry.intersects(bbox_geom)]

            # clean
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
            gdf = gdf[~gdf[attribute].isnull()]

        # shapes iterable
        shapes = [(geom, float(val)) for geom, val in zip(gdf.geometry, gdf[attribute])] if not gdf.empty else []

        # rasterize
        out_arr = np.full((ref_height, ref_width), nodata, dtype=rio_dtype)

        if shapes:
            out_arr = rasterize(
                shapes=shapes,
                out_shape=(ref_height, ref_width),
                transform=ref_transform,
                fill=nodata,
                all_touched=all_touched,
                dtype=rio_dtype,
                merge_alg=merge_alg_map[merge_alg],
            )

            # guard: integer dtype cannot use NaN nodata
            if np.issubdtype(np.dtype(dtype), np.integer) and (isinstance(nodata, float) and math.isnan(nodata)):
                raise ValueError("Integer dtype cannot use NaN as nodata. Pick an integer nodata value.")

        # write out (basename of gpkg)
        out_path = os.path.join(out_folder, f"{base}.tif" if layer is None else f"{base}_{lyr}.tif")
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_arr, 1)

        print(f"[OK] {os.path.basename(out_path)} written.")

# ---- Example call ----
# rasterize_gpkgs_with_pairwise_refs(
#     gpkg_folder=r"D:/data/walls_gpkg",
#     ref_raster_folder=r"D:/data/reference_grids",
#     out_folder=r"D:/data/walls_rasters",
#     attribute="tsm_prob",
#     layer=None,             # or provide a layer name
#     all_touched=False,
#     dtype="float32",
#     nodata=np.nan,
#     merge_alg="max"         # e.g., keep highest probability where overlaps
# )

'''