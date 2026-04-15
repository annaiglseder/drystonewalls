# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 17:39:55 2025

@author: aiglsede
"""

import rasterio
import numpy as np

def raster_stats_sum_and_threshold(raster_path, threshold, nodata=None):
    """
    Calculate:
      - sum of all valid pixel values
      - sum of pixel values >= threshold
      - count of pixels >= threshold
      - max pixel value

    Parameters
    ----------
    raster_path : str
        Path to the raster file.
    threshold : float
        Threshold for selecting pixels.
    nodata : float, optional
        If None, uses the raster's own NoData value.

    Returns
    -------
    total_sum : float
        Sum of all valid pixel values.
    sum_above : float
        Sum of pixel values >= threshold.
    count_above : int
        Number of pixels >= threshold.
    max_val : float
        Maximum valid pixel value.
    """
    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True)  # masked array ignores NoData
        if nodata is not None:
            arr = np.ma.masked_equal(arr, nodata)

    total_sum = float(arr.sum())

    mask_above = arr >= threshold
    sum_above = float(arr[mask_above].sum())
    count_above = int(mask_above.sum())

    max_val = float(arr.max())

    return total_sum, sum_above, count_above, max_val