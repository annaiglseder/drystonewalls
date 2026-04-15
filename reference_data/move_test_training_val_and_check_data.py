# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:02:41 2025

@author: aiglsede
"""

import os
import re
import shutil
from pathlib import Path
import rasterio
import numpy as np

raster_type = ["rgb", "dtm"]
raster_type = ['rgb2']
for raster in raster_type:
    
    # ==== USER PARAMETERS ======
    input_dirs = [
        f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/3405/model_data/{raster}",
        f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/3506/model_data/{raster}",
        f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/3507/model_data/{raster}"
    ]
    output_test = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/model/{raster}/test"
    output_val = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/model/{raster}/val"
    output_train = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/model/{raster}/train"
    error_log_path = f"D:/PHOTO/SEMONA/NAWA/tsm_reference_data/model/{raster}/error_files.txt"
    # ===========================
    
    # Compile filename patterns
    pat_folder2 = [
        re.compile(r"^3506_1_"),
        re.compile(r"^3506_9_"),
        re.compile(r"^3507_4"),
        re.compile(r"^3705_7"),
    ]
    
    def check_mask_valid(data):
        return np.isin(data, [0, 1]).all()
    
    def check_float_valid(data):
        return np.logical_and(data >= 0, data <= 1).all()
    
    def is_nan(data):
        return np.isnan(data).any()
    
    with open(error_log_path, "w") as log:
        for input_dir in input_dirs:
            for file_path in Path(input_dir).glob("*.tif"):
                fname = file_path.name
    
                # Destination selection
                if fname.startswith("3405"):
                    dest_folder = output_test
                elif any(pat.match(fname) for pat in pat_folder2):
                    dest_folder = output_val
                else:
                    dest_folder = output_train
    
                # Raster validation
                valid = True
                try:
                    with rasterio.open(file_path) as src:
                        data = src.read(1)  # Read first band
                        if is_nan(data):
                            valid = False
                        elif fname.endswith("_mask.tif"):
                            if not check_mask_valid(data):
                                valid = False
                        else:
                            if not check_float_valid(data):
                                valid = False
                except Exception as e:
                    valid = False  # File is not a valid raster
    
                if valid:
                    Path(dest_folder).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(file_path), str(Path(dest_folder) / fname))
                    print(f"Moved: {fname} -> {dest_folder}")
                else:
                    log.write(str(file_path) + "\n")
                    print(f"INVALID: {file_path} (logged)")
    
    print("Done.")
