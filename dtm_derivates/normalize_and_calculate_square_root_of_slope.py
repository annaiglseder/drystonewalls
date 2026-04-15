# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:17:24 2025

@author: aiglsede
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:03:20 2025

@author: aiglsede
"""

import numpy as np
import rasterio
import rasterio.plot
import rasterio.mask
import json

def calculate_raster_statistics(input_file):
    """Calculate statistics for the raster file."""
    with rasterio.open(input_file) as src:
        data = src.read(1, masked=True)  # Read first band, masked for NoData
        
        stats = {
            'min': float(np.nanmin(data)),
            'max': float(np.nanmax(data)),
            'mean': float(np.nanmean(data)),
            'median': float(np.nanmedian(data)),
            'quantiles': {
                'q2': float(np.nanpercentile(data, 2)),
                'q10': float(np.nanpercentile(data, 10)),
                'q25': float(np.nanpercentile(data, 25)),
                'q50': float(np.nanpercentile(data, 50)),
                'q75': float(np.nanpercentile(data, 75)),
                'q90': float(np.nanpercentile(data, 90)),
                'q98': float(np.nanpercentile(data, 98)),
            }
        }
        
        # Print stats
        print(json.dumps(stats, indent=4))
        
        return stats


def normalize_and_square_root_of_raster(input_file, output_file, min_value=0, max_value=90):
    """Clip outliers, normalize to 0-1, and apply square root transformation."""
    
    with rasterio.open(input_file) as src:
        profile = src.profile  # Save profile for output
        data = src.read(1, masked=True)  # Read first band, masked for NoData

        # 1. Clip values (outliers)
        clipped = np.clip(data, min_value, max_value)

        # 2. Normalize to 0-1 range
        normalized = (clipped - min_value) / (max_value - min_value)
        normalized = np.clip(normalized, 0, 1)  # Ensure within [0,1]

        # 3. Square root transformation
        sqrt_data = np.sqrt(normalized)

        # 4. Save to output raster
        profile.update(
            dtype='float32',
            compress='lzw',
            count=1,
            nodata=np.nan
        )
        
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(sqrt_data.astype(np.float32), 1)

        print(f"Processed raster saved to {output_file}")

def normalize_raster(input_file, output_file, min_value, max_value):
    """Clip outliers, normalize to 0-1, and apply square root transformation."""
    
    with rasterio.open(input_file) as src:
        profile = src.profile  # Save profile for output
        data = src.read(1, masked=True)  # Read first band, masked for NoData

        # 1. Clip values (outliers)
        clipped = np.clip(data, min_value, max_value)

        # 2. Normalize to 0-1 range
        normalized = (clipped - min_value) / (max_value - min_value)
        normalized = np.clip(normalized, 0, 1)  # Ensure within [0,1]

        # 3. Save to output raster
        profile.update(
            dtype='float32',
            compress='lzw',
            count=1,
            nodata=np.nan
        )
        
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(normalized.astype(np.float32), 1)

        print(f"Processed raster saved to {output_file}")


