# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 21:15:25 2025

@author: aiglsede
"""

import os
import time
import numpy as np
import rasterio
from rasterio.transform import Affine
from shapely.geometry import LineString, mapping
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
import fiona


# 8-neighborhood offsets
_OFF = np.array([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)], dtype=int)
# kernel to count 8-neighbors
_K = np.ones((3,3), dtype=np.int8)

def _prune_spurs(skel, steps):
    """Remove degree-1 endpoints iteratively for 'steps' iterations (fast)."""
    if steps <= 0: 
        return skel
    s = skel.copy()
    for _ in range(steps):
        deg = convolve(s.astype(np.int8), _K, mode='constant', cval=0) - s
        # endpoints == 1 (and currently skeleton)
        rm = (deg == 1) & (s == 1)
        if not rm.any():
            break
        s[rm] = 0
    return s

def _trace_lines(skel, transform, simplify_tolerance=None, min_length=None):
    """
    Vectorize skeleton by degree-based tracing:
    - endpoints: deg==1
    - junctions: deg>=3
    - chains: deg==2
    """
    H, W = skel.shape
    s = (skel.astype(bool))
    if not s.any():
        return []

    # degree image: neighbors count (exclude center)
    deg = convolve(s.astype(np.int8), _K, mode='constant', cval=0) - s.astype(np.int8)

    visited = np.zeros_like(s, dtype=bool)

    def to_xy(rr, cc):
        # pixel centers
        xs, ys = [], []
        for r, c in zip(rr, cc):
            x, y = transform * (c + 0.5, r + 0.5)
            xs.append(x); ys.append(y)
        return list(zip(xs, ys))

    lines = []

    # helper to follow a chain from (r,c) toward a neighbor (nr,nc)
    def walk_chain(r0, c0, nr, nc):
        rr = [r0, nr]; cc = [c0, nc]
        pr, pc = r0, c0
        cr, ccurr = nr, nc
        visited[r0, c0] = True
        visited[nr, nc] = True
        while True:
            # stop at endpoint or junction (deg != 2)
            if deg[cr, ccurr] != 2:
                break
            # pick the next neighbor that is skeleton and not the previous
            found = False
            for dr, dc in _OFF:
                r2, c2 = cr + dr, ccurr + dc
                if 0 <= r2 < H and 0 <= c2 < W and s[r2, c2] and not (r2 == pr and c2 == pc):
                    pr, pc = cr, ccurr
                    cr, ccurr = r2, c2
                    rr.append(cr); cc.append(ccurr)
                    visited[cr, ccurr] = True
                    found = True
                    break
            if not found:
                break
        return np.array(rr, dtype=int), np.array(cc, dtype=int)

    # 1) start from all endpoints
    er, ec = np.where((s) & (deg == 1))
    for r, c in zip(er, ec):
        if visited[r, c]:
            continue
        # go to its single neighbor
        nbr = None
        for dr, dc in _OFF:
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < H and 0 <= c2 < W and s[r2, c2]:
                nbr = (r2, c2)
                break
        if nbr is None:
            continue
        rr, cc = walk_chain(r, c, nbr[0], nbr[1])
        if rr.size >= 2:
            xy = to_xy(rr, cc)
            line = LineString(xy)
            if simplify_tolerance and simplify_tolerance > 0:
                line = line.simplify(simplify_tolerance, preserve_topology=False)
            if (min_length is None or line.length >= min_length) and line.length > 0:
                lines.append(line)

    # 2) start from junction neighbors to capture small loops/no-endpoint bits
    jr, jc = np.where((s) & (deg >= 3))
    for r, c in zip(jr, jc):
        for dr, dc in _OFF:
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < H and 0 <= c2 < W and s[r2, c2] and not visited[r2, c2]:
                rr, cc = walk_chain(r, c, r2, c2)
                if rr.size >= 2:
                    xy = to_xy(rr, cc)
                    line = LineString(xy)
                    if simplify_tolerance and simplify_tolerance > 0:
                        line = line.simplify(simplify_tolerance, preserve_topology=False)
                    if (min_length is None or line.length >= min_length) and line.length > 0:
                        lines.append(line)

    return lines


def process_prob_raster_to_vector(
    raster_path,
    threshold=0.5,
    output_folder=None,
    pixel_size=1.0,
    simplify_tolerance=None,
    min_length=None,
    input_is_skeleton=False,
    prune_spurs_len=0.0,
    # --- NEW ---
    mask_path=None,           # path to mask raster with identical grid as input
    invert_mask=True          # True -> exclude non-zero mask pixels
):
    """
    Fast vectorizer with optional (inverted) mask support.

    If mask_path is given, it must match the input raster's width/height/transform/CRS.
    When invert_mask=True (default), pixels where mask != 0 are excluded (set to 0).
    """
    t0 = time.time()

    # paths
    input_dir, filename = os.path.split(raster_path)
    name, _ = os.path.splitext(filename)
    if output_folder is None:
        output_folder = os.path.join(input_dir, "bin_vec")
    os.makedirs(output_folder, exist_ok=True)
    skeleton_raster_path = os.path.join(output_folder, f"{name}.tif")
    vector_output_path   = os.path.join(output_folder, f"{name}.gpkg")  # GPKG by default

    # read input
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        transform = src.transform if src.transform is not None else Affine.identity()
        crs = src.crs
        height, width = src.height, src.width

    # fallback transform if missing geoinfo
    if transform == Affine.identity():
        print(f"Warning: No geoinfo in {raster_path}, assuming pixel size {pixel_size}.")
        transform = Affine.translation(0, 0) * Affine.scale(pixel_size, -pixel_size)
        crs = None

    # --- NEW: apply mask (same grid as input) ---
    if mask_path is not None:
        with rasterio.open(mask_path) as msrc:
            m = msrc.read(1)
            # quick sanity checks
            if (msrc.width != width or msrc.height != height or msrc.transform != transform or msrc.crs != crs):
                raise ValueError("Mask grid mismatch: mask must have identical size, transform, and CRS as the input raster.")
        # build keep-mask (True = keep data)
        if invert_mask:
            # exclude non-zero mask pixels
            keep = (m == 0)
        else:
            # keep only non-zero mask pixels
            keep = (m != 0)
        # apply by zeroing excluded pixels (works for both prob rasters & skeleton inputs)
        # if arr is float and has NaNs, comparison with threshold will handle them later
        arr = np.where(keep, arr, 0)

    # binarize/skeletonize or accept given skeleton
    if input_is_skeleton:
        skeleton = (arr > 0).astype(np.uint8)
    else:
        binary = (arr >= threshold).astype(np.uint8)
        skeleton = skeletonize(binary.astype(bool)).astype(np.uint8) if binary.any() \
                   else np.zeros_like(binary, dtype=np.uint8)

    # spur pruning (meters -> pixel steps)
    if prune_spurs_len and prune_spurs_len > 0:
        px = abs(transform.a) if transform.a != 0 else pixel_size
        steps = int(round(prune_spurs_len / max(px, 1e-9)))
        if steps > 0:
            skeleton = _prune_spurs(skeleton, steps)

    # write skeleton (for QA)
    meta = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "uint8", "transform": transform, "nodata": 0
    }
    if crs:
        meta["crs"] = crs
    with rasterio.open(skeleton_raster_path, "w", **meta) as dst:
        dst.write(skeleton, 1)

    # vectorize
    t1 = time.time()
    lines = _trace_lines(skeleton, transform, simplify_tolerance, min_length)
    t2 = time.time()

    # stream write with Fiona
    schema = {"geometry": "LineString", "properties": {}}
    crs_wkt = crs.to_wkt() if crs else None
    with fiona.open(
        vector_output_path, mode="w",
        driver="GPKG", schema=schema,
        crs_wkt=crs_wkt, layer=name
    ) as dst:
        for geom in lines:
            dst.write({"geometry": mapping(geom), "properties": {}})

    t3 = time.time()
    print(f"[Timing] read+prep: {t1 - t0:.2f}s | vectorize: {t2 - t1:.2f}s | write: {t3 - t2:.2f}s | total: {t3 - t0:.2f}s")

    return skeleton_raster_path, vector_output_path

