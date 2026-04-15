# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 16:30:11 2025

@author: aiglsede
"""

import os
import glob
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import numpy as np

def unique_starting_strings(folder_path, n_chars=4):
    """
    Returns a sorted list of unique starting strings of files in a folder.

    Parameters:
        folder_path (str): Path to the folder.
        n_chars (int): Number of characters to use from the start of each filename.

    Returns:
        List[str]: Sorted list of unique starting strings.
    """
    strings = set()
    for fname in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, fname)):
            strings.add(fname[:n_chars])
    return sorted(strings)

# Example usage:
# unique_starting_strings("P:/Projects/yourfolder", n_chars=6)

