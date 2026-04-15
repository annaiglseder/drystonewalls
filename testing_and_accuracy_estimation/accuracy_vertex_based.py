# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:37:03 2025

@author: aiglsede
"""
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist

def extract_vertices_from_file(path):
    """Extract all (x, y) vertex coordinates from a vector file."""
    gdf = gpd.read_file(path)
    coords = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            coords.extend(list(geom.coords))
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                coords.extend(list(line.coords))
    return np.array(coords)



def evaluate_vector_proximity(model_path, reference_path, threshold):
    """
    Evaluate spatial proximity between vertices in two vector files.

    Parameters:
        model_path (str): path to model output vector file
        reference_path (str): path to reference vector file
        threshold (float): max allowed distance between matched vertices

    Returns:
        dict with precision, recall, f1 score, IoU, and confusion matrix
    """
    model_points = extract_vertices_from_file(model_path)
    ref_points = extract_vertices_from_file(reference_path)

    # Handle edge cases: one or both are empty
    if len(ref_points) == 0 and len(model_points) == 0:
        TP = FP = FN = TN = 0
    elif len(ref_points) == 0:
        TP = FN = TN = 0
        FP = len(model_points)
    elif len(model_points) == 0:
        TP = FP = TN = 0
        FN = len(ref_points)
    else:
        # Normal case
        D = cdist(ref_points, model_points)
        FN = np.sum(D.min(axis=1) > threshold)
        TP = np.sum(D.min(axis=0) <= threshold)
        FP = np.sum(D.min(axis=0) > threshold)
        TN = 0

    # Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    confusion = np.array([[TP, FN], [FP, TN]])

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "confusion_matrix": confusion
    }


def evaluate_vector_proximity_folder(model_folder, reference_folder, threshold, ref_suffix="_mask"):
    """
    Evaluate vector proximity for all shapefiles in a folder against corresponding reference shapefiles.

    Parameters:
        model_folder (str): folder with model output .shp files
        reference_folder (str): folder with ground truth .shp files
        threshold (float): distance threshold (map units)
        ref_suffix (str): suffix for reference files (e.g., '_mask')

    Returns:
        List of result dicts for each file.
    """
    from pathlib import Path

    model_folder = Path(model_folder)
    reference_folder = Path(reference_folder)

    results = []

    for model_file in model_folder.glob("*.shp"):
        base_name = model_file.stem
        ref_name = f"{base_name}{ref_suffix}.shp"
        ref_file = reference_folder / ref_name

        if not ref_file.exists():
            #print(f" Skipping {model_file.name}: no matching reference {ref_file.name}")
            continue

        try:
            #print(f"Evaluating: {model_file.name} vs {ref_file.name}")
            result = evaluate_vector_proximity(str(model_file), str(ref_file), threshold)
            result["file"] = model_file.name
            results.append(result)
        except Exception as e:
            print(f" Error processing {model_file.name}: {e}")

    return results







