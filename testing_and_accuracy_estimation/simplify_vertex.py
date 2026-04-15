# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 14:51:07 2025

@author: aiglsede
"""

#------------------------------------------------------------------------------------------------------------------------#
#### Functions to process vector (LineString roads) ####
#------------------------------------------------------------------------------------------------------------------------#

from shapely.geometry import LineString, Point, MultiLineString
from shapely.strtree import STRtree
import shapely
import geopandas as gpd

def get_allpoints(geom, all_vertices=True):
    coords = list(geom.coords)
    if all_vertices:
        return [Point(p) for p in coords]  # Use all points
    else:
        return [Point(coords[0]), Point(coords[-1])]  # Use only end points

def build_endpoint_index(gdf, all_vertices=True):
    endpoints = []
    geom_to_feature_map = []

    for fid, geom in enumerate(gdf.geometry):
        if geom.geom_type == "LineString":
            for pt in get_allpoints(geom, all_vertices):
                endpoints.append(pt)
                geom_to_feature_map.append(fid)  # Optional: track which feature it came from

    index = STRtree(endpoints)
    return index, endpoints, geom_to_feature_map

def snap_endpoints(gdf, tolerance=5., all_vertices=True):
    index, endpoints, _ = build_endpoint_index(gdf, all_vertices)
    snapped_geoms = []
    processed = []
    for geom in gdf.geometry:
        if geom.geom_type != "LineString":
            snapped_geoms.append(geom)
            continue

        coords = list(geom.coords)
        for i in [0, -1]:  # snap start and end only
            pt = Point(coords[i])
            match_indices = index.query(pt.buffer(tolerance))

            nearest = None
            nearest_dist = float('inf')
            on_line = False
            for idx in match_indices:
                candidate = endpoints[idx]
                if pt.equals(candidate):
                    continue  # skip exact match
                if idx in processed:
                    continue  # skip modified verices
                for co in coords:
                    if Point(co).equals(candidate):
                        on_line = True
                        break
                if on_line:
                    continue

                dist = pt.distance(candidate)
                if dist < nearest_dist:
                    nearest = candidate
                    nearest_dist = dist

            if nearest and nearest_dist <= tolerance:
                coords[i] = (nearest.x, nearest.y)
                processed.append(index.query(pt)[0])

        snapped_geoms.append(LineString(coords))
    snapped_geoms = shapely.ops.linemerge(shapely.MultiLineString(snapped_geoms))
    return snapped_geoms

def remove_short_lines(lines, min_size=5.):
    if lines.geom_type == "MultiLineString":
        lines_filtered = [line for line in list(lines.geoms) if line.length> min_size]
        return MultiLineString(lines_filtered)
    else:
        return lines
   
def save_gpkg(multi_lines, path, layer_name="skid_roads", year ="YYYY-MM-DD", crs=None, mode="w"):
    
    if isinstance(multi_lines, list):
        multi_lines = MultiLineString(multi_lines)

    lines = []
    lengths = []
    years = []
    for line in multi_lines.geoms:
        lines.append(line)
        lengths.append(line.length)
        years.append(year)
    
    gdf = gpd.GeoDataFrame({
        "length": lengths,
        "year": years
    }, geometry=lines, crs=crs)     
    gdf.to_file(filename=path, layer=layer_name, driver="GPKG",mode=mode)


# ----------------------
# Use Case Processing shapefile
# ----------------------

INPUT_FILE = r"D:\PHOTO\SEMONA\NAWA\UNET\250804_results_unet_enet_b7_rgb_ep12\_test_simplify_linestrings\3405_0.shp"       # your predicted road shapefile
OUTPUT_FILE = r"D:\PHOTO\SEMONA\NAWA\UNET\250804_results_unet_enet_b7_rgb_ep12\_test_simplify_linestrings\3405_0_smpl.shp"

SNAP_TOLERANCE = 1                    # meters
SIMPLIFY_TOLERANCE = 1
MIN_SEGMENT_LENGTH = 3                # meters

gdf = gpd.read_file(INPUT_FILE)
gdf = shapely.remove_repeated_points(gdf.geometry)
gdf = gdf.set_crs(epsg=31256, allow_override=True)
snapped = snap_endpoints(gdf, tolerance=SNAP_TOLERANCE, all_vertices=True)
simplified = shapely.simplify(snapped, tolerance=SIMPLIFY_TOLERANCE, preserve_topology=True)
filtered_lines = remove_short_lines(simplified, MIN_SEGMENT_LENGTH)

save_gpkg(filtered_lines,OUTPUT_FILE, "TSM", "2018-04-09", gdf.crs, mode="w")