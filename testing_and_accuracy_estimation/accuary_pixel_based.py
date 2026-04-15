# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 12:57:57 2025

@author: aiglsede
"""

import os
import numpy as np
import rasterio
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix



def evaluate_skeleton_pair(
    model_file,
    reference_folder,
    ref_suffix="_mask"
):
    """
    Evaluate pixel-based accuracy between a model skeleton raster and its matching reference.

    Parameters:
    - model_file: path to the model skeleton raster (.tif)
    - reference_folder: folder containing reference skeleton rasters
    - ref_suffix: suffix appended to the base filename for the reference file

    Returns:
    - dict with precision, recall, F1, IoU, and confusion matrix (or None if failed)
    """

    model_dir, model_name = os.path.split(model_file)
    base_name, ext = os.path.splitext(model_name)
    ref_name = f"{base_name}{ref_suffix}{ext}"
    ref_file = os.path.join(reference_folder, ref_name)

    if not os.path.exists(ref_file):
        print(f" Reference file not found: {ref_name}")
        return None

    # Load rasters
    with rasterio.open(model_file) as model_src:
        model_raster = model_src.read(1)

    with rasterio.open(ref_file) as ref_src:
        ref_raster = ref_src.read(1)

    # Check shape match
    if model_raster.shape != ref_raster.shape:
        print(f" Shape mismatch: {model_name} vs {ref_name}")
        return None

    # Flatten and binarize
    y_pred = (model_raster > 0).astype(np.uint8).flatten()
    y_true = (ref_raster > 0).astype(np.uint8).flatten()

    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "filename": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "confusion_matrix": cm
    }




def evaluate_skeleton_folder(
    model_folder,
    reference_folder,
    ref_suffix="_mask"
):
    """
    Evaluate all skeleton raster pairs in a folder against their references.

    Returns:
        List of result dicts, and a total confusion matrix.
    """
    results = []
    total_cm = np.array([[0, 0], [0, 0]])

    model_files = [f for f in os.listdir(model_folder) if f.endswith(".tif")]

    for model_name in model_files:
        model_path = os.path.join(model_folder, model_name)

        result = evaluate_skeleton_pair(
            model_file=model_path,
            reference_folder=reference_folder,
            ref_suffix=ref_suffix
        )

        if result is not None:
            results.append(result)
            total_cm += result["confusion_matrix"]

    return results, total_cm


'''
example use for the folder function

results, total_cm = evaluate_skeleton_folder(
    model_folder="P:/data/model_skeletons",
    reference_folder="P:/data/reference_skeletons",
    ref_suffix="_mask"
)

print(f"\n✅ Processed {len(results)} files.")

# Print averages
if results:
    print("Average Precision:", np.mean([r["precision"] for r in results]))
    print("Average Recall:   ", np.mean([r["recall"] for r in results]))
    print("Average F1 Score: ", np.mean([r["f1_score"] for r in results]))
    print("Average IoU:      ", np.mean([r["iou"] for r in results]))

# Print total confusion matrix
tn, fp, fn, tp = total_cm.ravel()
print("\n📊 Confusion Matrix (summed over all images):")
print("            Pred 0   Pred 1")
print(f"True 0:     {tn:7d}  {fp:7d}")
print(f"True 1:     {fn:7d}  {tp:7d}")


'''
