# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:50:41 2025

@author: aiglsede
"""

import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def _ensure_nodata(profile):
    """Return (nodata_value, updated_profile_if_changed)."""
    nodata = profile.get("nodata", None)
    if nodata is not None:
        return nodata, profile
    # Choose a sensible nodata if none is set
    dtype = np.dtype(profile["dtype"])
    profile = profile.copy()
    if np.issubdtype(dtype, np.floating):
        profile["nodata"] = np.nan
    else:
        profile["nodata"] = 0
    return profile["nodata"], profile

def mask_tiles_with_rasters_inplace(
    tiles_folder,
    mask_paths,
    combine="and",
    invert=True,
    resampling=Resampling.nearest,
):
    """
    Apply one or more mask rasters to every .tif tile in a folder, overwriting files.

    Parameters
    ----------
    tiles_folder : str | Path
        Folder containing .tif tiles to be masked.
    mask_paths : str | Path | list[str|Path]
        One or many mask rasters (0 => remove, >0 => keep). Different CRS/transform ok.
    combine : {'and','or'}
        How multiple masks are combined: 'and' (intersection) or 'or' (union).
    invert : bool | list[bool]
        Invert mask logic (keep<->remove). If list, must match number of masks.
    resampling : rasterio.warp.Resampling
        Resampling for mask reprojection (default nearest).
    """
    tiles_folder = Path(tiles_folder)
    if isinstance(mask_paths, (str, Path)):
        mask_paths = [mask_paths]
    mask_paths = [Path(p) for p in mask_paths]
    if not mask_paths:
        raise ValueError("mask_paths must not be empty.")

    # Normalize invert to a list matching mask_paths
    if isinstance(invert, bool):
        invert_flags = [invert] * len(mask_paths)
    else:
        if len(invert) != len(mask_paths):
            raise ValueError("invert list must have same length as mask_paths.")
        invert_flags = [bool(x) for x in invert]

    tif_files = [p for p in tiles_folder.iterdir() if p.suffix.lower() == ".tif"]
    for tile_path in tif_files:
        with rasterio.open(tile_path) as src:
            tile_profile = src.profile.copy()
            nodata, tile_profile = _ensure_nodata(tile_profile)
            tile_data = src.read(masked=False)

            # Initialize combined boolean "keep" mask
            if combine == "and":
                keep = np.ones((src.height, src.width), dtype=bool)
                comb_op = np.logical_and
            elif combine == "or":
                keep = np.zeros((src.height, src.width), dtype=bool)
                comb_op = np.logical_or
            else:
                raise ValueError("combine must be 'and' or 'or'.")

            # Merge masks
            for mpath, inv in zip(mask_paths, invert_flags):
                with rasterio.open(mpath) as msrc:
                    # Reproject mask to tile grid
                    mdata = msrc.read(1, masked=False)
                    dst = np.zeros((src.height, src.width), dtype=mdata.dtype)
                    reproject(
                        source=mdata,
                        destination=dst,
                        src_transform=msrc.transform,
                        src_crs=msrc.crs,
                        dst_transform=src.transform,
                        dst_crs=src.crs,
                        resampling=resampling,
                    )

                    # Build boolean keep for this mask:
                    # value > 0 => keep; value == 0 or equals mask nodata => remove
                    m_nodata = msrc.nodata
                    this_keep = dst > 0
                    if m_nodata is not None and not np.isnan(m_nodata):
                        this_keep &= (dst != m_nodata)

                    if inv:
                        this_keep = ~this_keep

                    keep = comb_op(keep, this_keep)

            # Apply combined mask to all bands: where not keep => set to nodata
            if np.issubdtype(np.dtype(tile_profile["dtype"]), np.floating) and np.isnan(nodata):
                # float with NaN nodata
                tile_data[:, ~keep] = np.nan
            else:
                tile_data[:, ~keep] = nodata

        # Overwrite the file with masked data
        with rasterio.open(tile_path, "w", **tile_profile) as dst:
            dst.write(tile_data)

# --- Example ---
if __name__ == "__main__":
    mask_tiles_with_rasters_inplace(
        tiles_folder=r"PATH\TO\tiles",
        mask_paths=[r"PATH\TO\mask1.tif", r"PATH\TO\mask2.tif"],
        combine="and",          # 'and' = intersection, 'or' = union
        invert=[False, True],   # invert second mask only (optional)
    )