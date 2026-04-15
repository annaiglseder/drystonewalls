# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 09:35:02 2025

@author: aiglsede
"""

# change dir to work with local scripts

import os

import numpy as np
import time
import sys
import pandas as pd
import shutil

from binarize_and_skeletonize import process_prob_raster_to_vector
from accuary_pixel_based import evaluate_skeleton_folder
from accuracy_vertex_based import evaluate_vector_proximity_folder
from copy_geoinfo_from_reference import copy_geoinfo_from_reference
from mosaic_probability_maps import mosaic_rasters_stat, mosaic_rasters_merge
from helpers import unique_starting_strings
from visualize_evaluation import combine_shapefiles_by_folder, segment_proximity_evaluation
from accuracy_density_based import density_eval_raster, combine_raster_tiles_by_pattern
from rasterize_gpkg import rasterize_gpkgs_with_pairwise_refs, fill_missing_rasters_like_ref
from masking_rasters import mask_tiles_with_rasters_inplace

start = time.time()


wd_path = ".../eval_models"
model_folder = "251210_sam2_unext_23_valid_dtm"
wd = os.path.join(wd_path, model_folder)    


def main_evaluation(
    wd, stats, prob_ths, dist_ths, ref_dir_sv, pred_dir, process_prob_raster_to_vector, 
    evaluate_skeleton_folder, evaluate_vector_proximity_folder, output_csv=os.path.join(wd,"summary_metrics.csv")
):
    records = []
    for stat in stats:
        for prob_th in prob_ths:
            result_row = {"stat": stat, "prob_th": prob_th}
            
            # --- Logging to txt file ---
            log = os.path.join(wd, f"log_{stat}_{prob_th}.txt")
            log_file = open(log, 'w')
            sys.stdout = log_file  # Redirect all print() output
            start = time.time()
            
            pred_dir_sv = os.path.join(wd, f"test_pred_skel_vec_{stat}_{prob_th}")
            os.makedirs(pred_dir_sv, exist_ok=True)
            for filename in os.listdir(os.path.join(pred_dir, stat)):
                if not filename.lower().endswith(".tif"):
                    continue
                file_path = os.path.join(pred_dir, stat, filename)
                #process_prob_raster_to_vector(file_path, threshold=prob_th, output_folder=pred_dir_sv, pixel_size=1.0)
                process_prob_raster_to_vector(file_path, 
                                              threshold=prob_th, 
                                              output_folder=pred_dir_sv, 
                                              pixel_size=1.0, 
                                              simplify_tolerance=simplify_tolerance, 
                                              min_length=min_length, 
                                              input_is_skeleton=input_is_skeleton, 
                                              prune_spurs_len=prune_spurs_len)
            
            # --- Pixel-based metrics ---
            px_results, px_total_cm = evaluate_skeleton_folder(
                model_folder=pred_dir_sv,
                reference_folder=ref_dir_sv,
                ref_suffix="_mask"
            )
            tn, fp, fn, tp = px_total_cm.ravel()
            p_pre = tp/(tp+fp) if (tp+fp) > 0 else 0
            p_rec = tp/(tp+fn) if (tp+fn) > 0 else 0
            p_f1  = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
            p_iou = tp/(tp+fp+fn) if (tp+fp+fn) > 0 else 0
            result_row.update({
                "p_pre": p_pre, "p_rec": p_rec, "p_f1": p_f1, "p_iou": p_iou,
                "p_fp": fp, "p_fn": fn, "p_tp": tp
            })

            print(f"\n Pixel-based comparison, probability threshold: {prob_th}")
            '''
            # Print averages
            
            # issue with average: empty tiles are going in with 0 for the metrics and lower the values
            print("Average Precision:", f'{np.mean([r["precision"] for r in px_results]):.3f}')
            print("Average Recall:   ", f'{np.mean([r["recall"] for r in px_results]):.3f}')
            print("Average F1 Score: ", f'{np.mean([r["f1_score"] for r in px_results]):.3f}')
            print("Average IoU:      ", f'{np.mean([r["iou"] for r in px_results]):.3f}')
            '''
            print("\n Confusion Matrix (summed over all tiles):")
            print("             Pred 0    Pred 1     sum")
            print(f"True 0:     {tn:7d}  {fp:7d}     {fp+tn:7d}")
            print(f"True 1:     {fn:7d}  {tp:7d}     {fn+tp:7d}")
            print("\n Summed Metrics:")
            print(f"Precision:  {p_pre:.3f}")
            print(f"Recall:     {p_rec:.3f}")
            print(f"F1:         {p_f1:.3f}")
            print(f"IoU:        {p_iou:.3f}")
            print("\n")
            print("################################################################")
            
            print(f"\n Vector-based comparison, probability threshold: {prob_th}")
            # --- Vector-based metrics for each buffer ---
            for dist_th in dist_ths:
                ver_results = evaluate_vector_proximity_folder(
                    model_folder=pred_dir_sv,
                    reference_folder=ref_dir_sv,
                    threshold=dist_th,
                    ref_suffix="_mask"
                )
                precisions = [r["precision"] for r in ver_results]
                recalls = [r["recall"] for r in ver_results]
                f1s = [r["f1_score"] for r in ver_results]
                ious = [r["iou"] for r in ver_results]
                total_cm = np.sum([r["confusion_matrix"] for r in ver_results], axis=0)
                TP, FN = total_cm[0]
                FP, TN = total_cm[1]
                v_pre = TP/(TP+FP) if (TP+FP) > 0 else 0
                v_rec = TP/(TP+FN) if (TP+FN) > 0 else 0
                v_f1  = (2*TP)/(2*TP+FP+FN) if (2*TP+FP+FN) > 0 else 0
                v_iou = TP/(TP+FP+FN) if (TP+FP+FN) > 0 else 0

                # Add to row with buffer-specific column names
                result_row.update({
                    f"v_{dist_th}_pre": v_pre, f"v_{dist_th}_rec": v_rec,
                    f"v_{dist_th}_f1": v_f1,  f"v_{dist_th}_iou": v_iou,
                    f"v_{dist_th}_fp": FP,    f"v_{dist_th}_fn": FN,
                    f"v_{dist_th}_tp": TP
                })

                print("~~~~~~~~~~~~~~~~~~~~~")
                print(f"\n Buffer: {dist_th} m")  
                '''
                print("\n Average Metrics:")
                # issue with average: empty tiles are going in with 0 for the metrics and lower the values
                print(f"Precision: {np.mean(precisions):.3f}")
                print(f"Recall:    {np.mean(recalls):.3f}")
                print(f"F1:        {np.mean(f1s):.3f}")
                print(f"IoU:        {np.mean(ious):.3f}")
                '''
                print("\n Summed Confusion Matrix:")
                print("             Pred 0   Pred 1     sum")
                print(f"True 0:     {TN:7d}  {FP:7d}    {TN+FP:7d}")
                print(f"True 1:     {FN:7d}  {TP:7d}    {FN+TP:7d}")
                print("\n Summed Metrics:")
                print(f"Precision:  {v_pre:.3f}")
                print(f"Recall:     {v_rec:.3f}")
                print(f"F1:         {v_f1:.3f}")
                print(f"IoU:        {v_iou:.3f}")
                print("~~~~~~~~~~~~~~~~~~~~~")
                print("\n")
            
            end = time.time()
            print(f"\n Total elapsed time: {end - start:.2f} seconds")
            sys.stdout = sys.__stdout__
            log_file.close()
            records.append(result_row)
    
    # Build DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(wd, output_csv), index=False)
    print("Summary metrics saved to:", os.path.join(wd, output_csv))
    return df




pred_dir_orig = os.path.join(wd,"test_probs")
pred_dir_gpkg = os.path.join(wd,"test_gpkg_pred")
ref_dir_orig = os.path.join(wd,"test_ref_masks")


px_size = 0.25
stats = ["mean", "max", "min"]
prob_ths = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
dist_ths_px = [1, 2, 4, 6, 8, 10]
max_seg = 1
raster_sizes = [10, 25]

simplify_tolerance=None
min_length=1.5
input_is_skeleton=False     # <- NEW
prune_spurs_len=0.0

input_gpkg=False
masking=True

mask_file_forest = ".../forest_mask.tif"
mask_file_builtup = ".../built_up_area_mask.tif"

mask_file_list=[None, mask_file_forest, mask_file_builtup, [mask_file_forest, mask_file_builtup]]
mask_name_list=["no_mask", "forest_mask", "builtup_mask", "combined_mask"]

dist_ths = [x * px_size for x in dist_ths_px]

if masking is True:
    for mask_file, mask_name in zip(mask_file_list, mask_name_list):
        #mask_file = [mask_file_forest, mask_file_builtup]
        #mask_name = "combined_mask"
        #mask_file = mask_file_builtup
        #mask_name = "builtup_mask"
        #mask_file = mask_file_forest
        #mask_name = "forest_mask"
        
        wd_mask = os.path.join(wd, mask_name)
        wd_orig = wd
        wd = wd_mask
        
        os.makedirs(wd, exist_ok=True)
                       
        # create files and folders

        # logging preprocessing      
        log = os.path.join(wd, "log_preprocessing.txt")       
        log_file = open(log, 'w')
        sys.stdout = log_file  # Redirect all print() output
        start = time.time()

        # create new dirs

        pred_dir_tiles = os.path.join(wd,"test_probs_georef")
        pred_dir = os.path.join(wd, "test_probs_mosaic")
        ref_dir = os.path.join(wd, "test_ref_mosaic")
        temp_dir = os.path.join(wd, "_temp")

        # dirs for skeletonized and vectorized pred and ref tiles

        ref_dir_sv = os.path.join(wd, "test_ref_skel_vec_mask")

        os.makedirs(pred_dir_tiles, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(ref_dir, exist_ok=True)
        os.makedirs(ref_dir_sv, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # prepare reference data - this can be outcommented once done for the reference data

        # get tile names for mosaicing

        tiles = unique_starting_strings(ref_dir_orig, 7)
        tiles = [s[:-1] if s.endswith("_") else s for s in tiles]


        ### ref tiles ###

        # mosaic

        for tile in tiles:
            output_file = os.path.join(ref_dir, f'{tile}_mask.tif')
            mosaic_rasters_merge(ref_dir_orig, output_file, stat="max", nodata_value=None, name_pattern=f'{tile}_*') 
        
        if mask_file is not None:
            mask_tiles_with_rasters_inplace(ref_dir, mask_file)
            
        # skeletonize and vectorize reference data
        
        for filename in os.listdir(ref_dir):
        
            if not filename.lower().endswith("mask.tif"):
                continue
        
            file_path = os.path.join(ref_dir, filename)
            #threshold_bin = 0.5
            process_prob_raster_to_vector(file_path, 
                                          threshold=0.5, 
                                          output_folder=ref_dir_sv, 
                                          pixel_size=1.0, 
                                          simplify_tolerance=simplify_tolerance, 
                                          min_length=min_length, 
                                          input_is_skeleton=input_is_skeleton, 
                                          prune_spurs_len=prune_spurs_len)
        
        
        # add geoinfo to model output
        
        
        if input_gpkg is True:
            os.makedirs(pred_dir_orig, exist_ok=True)
            rasterize_gpkgs_with_pairwise_refs(
                pred_dir_gpkg,
                ref_dir_orig,
                pred_dir_orig,
                attribute="tsm_prob",
                all_touched=False,
                dtype="float32",          # use float (safer for probabilities)
                nodata=np.nan,            # NaN for float; set e.g. -9999 if you prefer
                merge_alg="replace",      # options: replace, add, max, min
                layer=None                # set a layer name if your GPKGs have multiple layers
            )
            
            fill_missing_rasters_like_ref(
                mask_folder=ref_dir_orig,
                data_folder=pred_dir_orig

            )
            
            # mosaic prob map with different stats
            
            for stat in stats:
                #stat = "mean"
                pred_dir_stat = os.path.join(pred_dir, stat)
                os.makedirs(pred_dir_stat, exist_ok=True)
            
                for tile in tiles:
                    #tile = tiles[0]
                    output_file = os.path.join(pred_dir_stat, f'{tile}.tif') 
                    if stat in ["min", "max"]:
                        mosaic_rasters_merge(pred_dir_orig, output_file, stat=stat, nodata_value=None, name_pattern=f'{tile}_*')
                    else:
                        mosaic_rasters_stat(pred_dir_orig, output_file, stat=stat, nodata_value=None, name_pattern=f'{tile}_*')
                        
                if mask_file is not None:
                    mask_tiles_with_rasters_inplace(pred_dir_stat, mask_file)
                    
                    
                    
                    
        if input_gpkg is False:
            copy_geoinfo_from_reference(pred_dir_orig, ref_dir_orig, output_folder=pred_dir_tiles, overwrite=False)
        
            # mosaic prob map with different stats
            
            for stat in stats:
                #stat = "mean"
                pred_dir_stat = os.path.join(pred_dir, stat)
                os.makedirs(pred_dir_stat, exist_ok=True)
            
                for tile in tiles:
                    #tile = tiles[0]
                    output_file = os.path.join(pred_dir_stat, f'{tile}.tif') 
                    if stat in ["min", "max"]:
                        mosaic_rasters_merge(pred_dir_tiles, output_file, stat=stat, nodata_value=None, name_pattern=f'{tile}_*')
                    else:
                        mosaic_rasters_stat(pred_dir_tiles, output_file, stat=stat, nodata_value=None, name_pattern=f'{tile}_*')

                if mask_file is not None:
                    mask_tiles_with_rasters_inplace(pred_dir_stat, mask_file)
                    
        
        end = time.time()
        print(f"\n Total elapsed time: {end - start:.2f} seconds")
        
        sys.stdout = sys.__stdout__
        log_file.close()
        

        df_eval = main_evaluation(wd, stats, prob_ths, dist_ths, ref_dir_sv, pred_dir, process_prob_raster_to_vector, evaluate_skeleton_folder, evaluate_vector_proximity_folder, output_csv=os.path.join(wd,"summary_metrics.csv"))
        #df_eval = pd.read_csv(os.path.join(wd,"summary_metrics.csv"))
        
        
        # Get min/max of the v_1.5_f1 column
        min_f1 = df_eval["v_1.5_f1"].min()
        max_f1 = df_eval["v_1.5_f1"].max()
        
        # Get row(s) where max occurs (there could be ties, so .iloc[0] for first match)
        max_row = df_eval[df_eval["v_1.5_f1"] == max_f1].iloc[0]
        max_f1_stat = max_row["stat"]
        max_f1_prob = max_row["prob_th"]
        
        # shift all other folders in a _temp folder
        
        for stat in stats:
            for prob_th in prob_ths:
                if (stat != max_f1_stat or prob_th != max_f1_prob):
                    src1 = os.path.join(wd, f"test_pred_skel_vec_{stat}_{prob_th}")
                    src2 = os.path.join(wd, f"log_{stat}_{prob_th}.txt")
                    try:
                        shutil.move(src1, temp_dir)
                    except shutil.Error as e:
                        print(f"Ignored error for {src1}: {e}")
                    except FileExistsError as e:
                        print(f"Ignored FileExistsError for {src1}: {e}")
                    try:
                        shutil.move(src2, temp_dir)
                    except shutil.Error as e:
                        print(f"Ignored error for {src2}: {e}")
                    except FileExistsError as e:
                        print(f"Ignored FileExistsError for {src2}: {e}")
        
        stats = [max_f1_stat]
        prob_ths = [max_f1_prob]
        #dist_ths_px = [6]
        
        
        # combine shape files
        
        combine_shapefiles_by_folder(
            parent_folder=wd,
            folder_pattern='test_pred_skel_vec_',
            output_dir=os.path.join(wd, "shp_comb_pred")
        )
                
        combine_shapefiles_by_folder(
            parent_folder=wd,
            folder_pattern='test_ref_skel_vec_',
            output_dir=os.path.join(wd, "shp_comb_ref")
        )    
        
        
        
        '''
        # add attributes to shapes - all files
        
        segment_proximity_evaluation(
            reference_fp=next((os.path.join(os.path.join(wd, "shp_comb_ref"), f) for f in os.listdir(os.path.join(wd, "shp_comb_ref")) if f.lower().endswith('.shp')), None),
            model_dir=os.path.join(wd, "shp_comb_pred"),
            output_dir=os.path.join(wd, "shp_comb_eval"),
            max_segment_length=max_seg,
            buffer_distances=dist_ths,
            stat_list = stats,
            prob_ths = prob_ths
        )
        '''
        
        # select optimal parameter and do finer segment proximity mapping
        
        segment_proximity_evaluation(
            reference_fp=next((os.path.join(os.path.join(wd, "shp_comb_ref"), f) for f in os.listdir(os.path.join(wd, "shp_comb_ref")) if f.lower().endswith('.shp')), None),
            model_dir=os.path.join(wd, "shp_comb_pred"),
            output_dir=os.path.join(wd, "shp_comb_eval"),
            max_segment_length=max_seg,
            buffer_distances=dist_ths,
            stat_list = stats,
            prob_ths = prob_ths
        )
        
        # create len rasters
        
        for raster_size in raster_sizes:
            
            density_eval_raster(
                ref_folder = ref_dir_sv, 
                parent_folder = wd,
                model_pattern = "test_pred_skel_vec",
                target_pixel_size = raster_size,
                stats = stats,
                prob_ths = prob_ths,
                pixel_size = px_size
            )
              
            combine_raster_tiles_by_pattern(
                input_folders=[
                    os.path.join(wd, f"test_diff_dens_{raster_size}"),
                    os.path.join(wd, f"test_pred_dens_{raster_size}"),
                    os.path.join(wd, f"test_ref_dens_{raster_size}")
                ],
                output_folder=os.path.join(wd, f"test_dens_mosaic_{raster_size}"),
                stats=stats,
                prob_ths=prob_ths
            )
            
            
        wd = wd_orig
        
else:    
    
    # create files and folders
    
    # logging preprocessing      
    log = os.path.join(wd, "log_preprocessing.txt")       
    log_file = open(log, 'w')
    sys.stdout = log_file  # Redirect all print() output
    start = time.time()
    
    # create new dirs
    
    pred_dir_tiles = os.path.join(wd,"test_probs_georef")
    pred_dir = os.path.join(wd, "test_probs_mosaic")
    ref_dir = os.path.join(wd, "test_ref_mosaic")
    temp_dir = os.path.join(wd, "_temp")
    
    # dirs for skeletonized and vectorized pred and ref tiles
    
    ref_dir_sv = os.path.join(wd, "test_ref_skel_vec_mask")
    
    os.makedirs(pred_dir_tiles, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(ref_dir_sv, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # prepare reference data - this can be outcommented once done for the reference data
    
    # get tile names for mosaicing
    
    tiles = unique_starting_strings(ref_dir_orig, 7)
    tiles = [s[:-1] if s.endswith("_") else s for s in tiles]
    
    
    ### ref tiles ###
    
    # mosaic
    
    for tile in tiles:
        output_file = os.path.join(ref_dir, f'{tile}_mask.tif')
        mosaic_rasters_merge(ref_dir_orig, output_file, stat="max", nodata_value=None, name_pattern=f'{tile}_*')    
    
    # skeletonize and vectorize reference data
    
    for filename in os.listdir(ref_dir):
    
        if not filename.lower().endswith("mask.tif"):
            continue
    
        file_path = os.path.join(ref_dir, filename)
        #threshold_bin = 0.5
        process_prob_raster_to_vector(file_path, 
                                      threshold=0.5, 
                                      output_folder=ref_dir_sv, 
                                      pixel_size=1.0, 
                                      simplify_tolerance=simplify_tolerance, 
                                      min_length=min_length, 
                                      input_is_skeleton=input_is_skeleton, 
                                      prune_spurs_len=prune_spurs_len)
    
    

    
    if input_gpkg is True:
        os.makedirs(pred_dir_orig, exist_ok=True)
        rasterize_gpkgs_with_pairwise_refs(
            pred_dir_gpkg,
            ref_dir_orig,
            pred_dir_orig,
            attribute="tsm_prob",
            all_touched=False,
            dtype="float32",          # use float (safer for probabilities)
            nodata=np.nan,            # NaN for float; set e.g. -9999 if you prefer
            merge_alg="replace",      # options: replace, add, max, min
            layer=None                # set a layer name if your GPKGs have multiple layers
        )
        
        fill_missing_rasters_like_ref(
            mask_folder=ref_dir_orig,
            data_folder=pred_dir_orig

        )
    
        # mosaic prob map with different stats
        
        for stat in stats:
            #stat = "mean"
            pred_dir_stat = os.path.join(pred_dir, stat)
            os.makedirs(pred_dir_stat, exist_ok=True)
        
            for tile in tiles:
                #tile = tiles[0]
                output_file = os.path.join(pred_dir_stat, f'{tile}.tif') 
                if stat in ["min", "max"]:
                    mosaic_rasters_merge(pred_dir_orig, output_file, stat=stat, nodata_value=None, name_pattern=f'{tile}_*')
                else:
                    mosaic_rasters_stat(pred_dir_orig, output_file, stat=stat, nodata_value=None, name_pattern=f'{tile}_*')
    
        
    
    if input_gpkg is False:
        copy_geoinfo_from_reference(pred_dir_orig, ref_dir_orig, output_folder=pred_dir_tiles, overwrite=False)
    
        # mosaic prob map with different stats
        
        for stat in stats:
            #stat = "mean"
            pred_dir_stat = os.path.join(pred_dir, stat)
            os.makedirs(pred_dir_stat, exist_ok=True)
        
            for tile in tiles:
                #tile = tiles[0]
                output_file = os.path.join(pred_dir_stat, f'{tile}.tif') 
                if stat in ["min", "max"]:
                    mosaic_rasters_merge(pred_dir_tiles, output_file, stat=stat, nodata_value=None, name_pattern=f'{tile}_*')
                else:
                    mosaic_rasters_stat(pred_dir_tiles, output_file, stat=stat, nodata_value=None, name_pattern=f'{tile}_*')
                
    
    end = time.time()
    print(f"\n Total elapsed time: {end - start:.2f} seconds")
    
    sys.stdout = sys.__stdout__
    log_file.close()
    

    
    df_eval = main_evaluation(wd, stats, prob_ths, dist_ths, ref_dir_sv, pred_dir, process_prob_raster_to_vector, evaluate_skeleton_folder, evaluate_vector_proximity_folder)
    #df_eval = pd.read_csv(os.path.join(wd,"summary_metrics.csv"))
    
    
    # Get min/max of the v_1.5_f1 column
    min_f1 = df_eval["v_1.5_f1"].min()
    max_f1 = df_eval["v_1.5_f1"].max()
    
    # Get row(s) where max occurs (there could be ties, so .iloc[0] for first match)
    max_row = df_eval[df_eval["v_1.5_f1"] == max_f1].iloc[0]
    max_f1_stat = max_row["stat"]
    max_f1_prob = max_row["prob_th"]
    
    # shift all other folders in a _temp folder
    
    for stat in stats:
        for prob_th in prob_ths:
            if (stat != max_f1_stat or prob_th != max_f1_prob):
                src1 = os.path.join(wd, f"test_pred_skel_vec_{stat}_{prob_th}")
                src2 = os.path.join(wd, f"log_{stat}_{prob_th}.txt")
                try:
                    shutil.move(src1, temp_dir)
                except shutil.Error as e:
                    print(f"Ignored error for {src1}: {e}")
                except FileExistsError as e:
                    print(f"Ignored FileExistsError for {src1}: {e}")
                try:
                    shutil.move(src2, temp_dir)
                except shutil.Error as e:
                    print(f"Ignored error for {src2}: {e}")
                except FileExistsError as e:
                    print(f"Ignored FileExistsError for {src2}: {e}")
    
    stats = [max_f1_stat]
    prob_ths = [max_f1_prob]
    #dist_ths_px = [6]
    
    
    # combine shape files
    
    combine_shapefiles_by_folder(
        parent_folder=wd,
        folder_pattern='test_pred_skel_vec_',
        output_dir=os.path.join(wd, "shp_comb_pred")
    )
            
    combine_shapefiles_by_folder(
        parent_folder=wd,
        folder_pattern='test_ref_skel_vec_',
        output_dir=os.path.join(wd, "shp_comb_ref")
    )    
    
    
    
    '''
    # add attributes to shapes - all files
    
    segment_proximity_evaluation(
        reference_fp=next((os.path.join(os.path.join(wd, "shp_comb_ref"), f) for f in os.listdir(os.path.join(wd, "shp_comb_ref")) if f.lower().endswith('.shp')), None),
        model_dir=os.path.join(wd, "shp_comb_pred"),
        output_dir=os.path.join(wd, "shp_comb_eval"),
        max_segment_length=max_seg,
        buffer_distances=dist_ths,
        stat_list = stats,
        prob_ths = prob_ths
    )
    '''
    
    # select optimal parameter and do finer segment proximity mapping
    
    segment_proximity_evaluation(
        reference_fp=next((os.path.join(os.path.join(wd, "shp_comb_ref"), f) for f in os.listdir(os.path.join(wd, "shp_comb_ref")) if f.lower().endswith('.shp')), None),
        model_dir=os.path.join(wd, "shp_comb_pred"),
        output_dir=os.path.join(wd, "shp_comb_eval"),
        max_segment_length=max_seg,
        buffer_distances=dist_ths,
        stat_list = stats,
        prob_ths = prob_ths
    )
    
    # create len rasters
    
    for raster_size in raster_sizes:
        
        density_eval_raster(
            ref_folder = ref_dir_sv, 
            parent_folder = wd,
            model_pattern = "test_pred_skel_vec",
            target_pixel_size = raster_size,
            stats = stats,
            prob_ths = prob_ths,
            pixel_size = px_size
        )
          
        combine_raster_tiles_by_pattern(
            input_folders=[
                os.path.join(wd, f"test_diff_dens_{raster_size}"),
                os.path.join(wd, f"test_pred_dens_{raster_size}"),
                os.path.join(wd, f"test_ref_dens_{raster_size}")
            ],
            output_folder=os.path.join(wd, f"test_dens_mosaic_{raster_size}"),
            stats=stats,
            prob_ths=prob_ths
        )
    
    
print('everything finished here')




