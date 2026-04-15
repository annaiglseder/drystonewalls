# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:35:32 2025

@author: aiglsede
"""

import opals
from opals import Info, Cell, Types, Grid, View, FillGaps, AddInfo, Import
import os
import time
from datetime import datetime
import sys


# this workflow creates DTMs of a classified point cloud with an hierachical iterative approach using opals
# https://opals.geo.tuwien.ac.at/html/stable/index.html

# the output are two dtms (0.25 m and 0.5 m)

# set wd
wd = ".../odm"
tile = "ALS_point_cloud.odm"


odm = os.path.join(wd, tile)


start_time = time.time()

tmp_1 = f'{odm[:-4]}_min01.odm'
tmp_2 = f'{odm[:-4]}_min3.odm'
tmp_3 = f'{odm[:-4]}_min3_tr.tif'
tmp_4 = f'{odm[:-4]}_min3_tr_fg.tif'
tmp_5 = f'{odm[:-4]}_DTM2.tif'
tmp_6 = f'{odm[:-4]}_DTM2_fg.tif'
tmp_7 = f'{odm[:-4]}_DTM3.tif'
tmp_8 = f'{odm[:-4]}_DTM3_fg.tif'
tmp_9 = f'{odm[:-4]}_DTM4.tif'
tmp_10 = f'{odm[:-4]}_DTM5.tif'
dtm = f'{odm[:-4]}_DTM_0.25.tif'
dtm05 = f'{odm[:-4]}_DTM_0.5.tif'


tmp = [tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10]


# thinning - lowest last echo within 0.1 m cells as basis for DTM calculation & filter ground points
cell = Cell.Cell()
cell.inFile = odm
cell.outFile = tmp_1
cell.cellSize = 0.1
cell.feature = "min"
cell.globals.points_in_memory = 100000000
cell.commons.nbThreads = 4
cell.globals.points_in_memory = 1000000000
cell.globals.create_option = ['BLOCKXSIZE=4096', 'BLOCKYSIZE=4096',
                              'COMPRESS=LZW', 'BIGTIFF=IF_SAFER', 'TFW=NO', 'TILED=YES',
                              'PROFILE=GDALGeoTIFF', 'DECIMAL_PRECISION=4']
#filter = f"class[Ground]"
filter = f"echo[last] and class[Ground]"
cell.filter = filter
cell.run(reset=True)

#creating a DTM using a hierarchical iterativ approach
cell = Cell.Cell()
cell.inFile = tmp_1
cell.outFile = tmp_2
cell.cellSize = 3
cell.feature = "quantile:0.05"
cell.globals.points_in_memory = 100000000
cell.commons.nbThreads = 4
cell.globals.points_in_memory = 1000000000
cell.globals.create_option = ['BLOCKXSIZE=4096', 'BLOCKYSIZE=4096',
                              'COMPRESS=LZW', 'BIGTIFF=IF_SAFER', 'TFW=NO', 'TILED=YES',
                              'PROFILE=GDALGeoTIFF', 'DECIMAL_PRECISION=4']
cell.run(reset=True)

grid = Grid.Grid()
grid.inFile = tmp_2
grid.outFile = tmp_3
grid.gridSize = 0.5
grid.interpolation = opals.Types.GridInterpolator.delaunayTriangulation
grid.searchRadius = 5
grid.commons.nbThreads = 4
grid.globals.points_in_memory = 1000000000
grid.globals.create_option = ['BLOCKXSIZE=4096', 'BLOCKYSIZE=4096',
                              'COMPRESS=LZW', 'BIGTIFF=IF_SAFER', 'TFW=NO', 'TILED=YES',
                              'PROFILE=GDALGeoTIFF', 'DECIMAL_PRECISION=4']
grid.run(reset=True)

fillgaps = FillGaps.FillGaps()
fillgaps.inFile = tmp_3
fillgaps.outFile = tmp_4
fillgaps.method = opals.Types.FillMethod.triangulation
fillgaps.run(reset=True)

addinfo = AddInfo.AddInfo()
addinfo.inFile = tmp_1
addinfo.gridFile = tmp_4
addinfo.attribute = "normalizedz = z-r[0]"
addinfo.run(reset=True)

grid = Grid.Grid()
grid.inFile = tmp_1
grid.outFile = tmp_5
grid.gridSize = 0.5
grid.interpolation = opals.Types.GridInterpolator.movingPlanes
grid.searchRadius = 3
grid.neighbours = 50
filter = f"generic[normalizedz<1 and normalizedz>-2]"
grid.filter = filter
grid.commons.nbThreads = 4
grid.globals.points_in_memory = 1000000000
grid.globals.create_option = ['BLOCKXSIZE=4096', 'BLOCKYSIZE=4096',
                              'COMPRESS=LZW', 'BIGTIFF=IF_SAFER', 'TFW=NO', 'TILED=YES',
                              'PROFILE=GDALGeoTIFF', 'DECIMAL_PRECISION=4']
grid.run(reset=True)

fillgaps = FillGaps.FillGaps()
fillgaps.inFile = tmp_5
fillgaps.outFile = tmp_6
fillgaps.method = opals.Types.FillMethod.triangulation
fillgaps.run(reset=True)

addinfo = AddInfo.AddInfo()
addinfo.inFile = tmp_1
addinfo.gridFile = tmp_6
addinfo.attribute = "normalizedz = z-r[0]"
addinfo.searchRadius = 10
addinfo.run(reset=True)

grid = Grid.Grid()
grid.inFile = tmp_1
grid.outFile = tmp_7
grid.gridSize = 0.25
grid.interpolation = opals.Types.GridInterpolator.movingPlanes
grid.searchRadius = 2
grid.neighbours = 10
filter = f"generic[normalizedz<0.35]"
grid.filter = filter
grid.commons.nbThreads = 4
grid.globals.points_in_memory = 1000000000
grid.globals.create_option = ['BLOCKXSIZE=4096', 'BLOCKYSIZE=4096',
                              'COMPRESS=LZW', 'BIGTIFF=IF_SAFER', 'TFW=NO', 'TILED=YES',
                              'PROFILE=GDALGeoTIFF', 'DECIMAL_PRECISION=4']
grid.run(reset=True)

fillgaps = FillGaps.FillGaps()
fillgaps.inFile = tmp_7
fillgaps.outFile = tmp_8
fillgaps.method = opals.Types.FillMethod.triangulation
fillgaps.run(reset=True)

addinfo = AddInfo.AddInfo()
addinfo.inFile = tmp_1
addinfo.gridFile = tmp_8
addinfo.attribute = "normalizedz = z-r[0]"
addinfo.searchRadius = 10
addinfo.run(reset=True)

grid = Grid.Grid()
grid.inFile = tmp_1
grid.outFile = tmp_9
grid.gridSize = 0.25
grid.interpolation = opals.Types.GridInterpolator.movingPlanes
grid.searchRadius = 1
grid.neighbours = 10
filter = f"generic[normalizedz<0.2]"
grid.filter = filter
grid.commons.nbThreads = 4
grid.globals.points_in_memory = 1000000000
grid.globals.create_option = ['BLOCKXSIZE=4096', 'BLOCKYSIZE=4096',
                              'COMPRESS=LZW', 'BIGTIFF=IF_SAFER', 'TFW=NO', 'TILED=YES',
                              'PROFILE=GDALGeoTIFF', 'DECIMAL_PRECISION=4']
grid.run(reset=True)

fillgaps = FillGaps.FillGaps()
fillgaps.inFile = tmp_9
fillgaps.outFile = dtm
fillgaps.method = opals.Types.FillMethod.triangulation
fillgaps.run(reset=True)

grid = Grid.Grid()
grid.inFile = tmp_1
grid.outFile = tmp_10
grid.gridSize = 0.5
grid.interpolation = opals.Types.GridInterpolator.movingPlanes
grid.searchRadius = 1
grid.neighbours = 10
filter = f"generic[normalizedz<0.2]"
grid.filter = filter
grid.commons.nbThreads = 4
grid.globals.points_in_memory = 1000000000
grid.globals.create_option = ['BLOCKXSIZE=4096', 'BLOCKYSIZE=4096',
                              'COMPRESS=LZW', 'BIGTIFF=IF_SAFER', 'TFW=NO', 'TILED=YES',
                              'PROFILE=GDALGeoTIFF', 'DECIMAL_PRECISION=4']
grid.run(reset=True)

fillgaps = FillGaps.FillGaps()
fillgaps.inFile = tmp_10
fillgaps.outFile = dtm05
fillgaps.method = opals.Types.FillMethod.triangulation
fillgaps.run(reset=True)


#imp = Import.Import()
#imp.inFile = dtm
#imp.outFile = dtm_odm
#imp.run(reset=True)

for f in tmp:
    try:
        os.remove(f)
    except FileNotFoundError:
        continue

end_time = time.time()

elapsed_time = end_time - start_time

#with open(processed, 'a') as file:
#    file.write(f"{odm}\t {datetime.now()}\t processing time: {elapsed_time} seconds\n")

