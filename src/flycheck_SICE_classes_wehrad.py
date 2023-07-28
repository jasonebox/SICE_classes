#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:41:27 2023

@author: jason
"""
import rasterio
from rasterio.mask import mask
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn import svm

# ## change to your system's login name to change dir for local work
if os.getlogin() == "jason":
    base_path = "/Users/jason/Dropbox/S3/SICE_classes/"
if os.getlogin() == "adrien":
    base_path = "/home/adrien/EO-IO/SICE_classes/"
os.chdir(base_path)

# relative paths
path_raw = "./SICE_rasters/"
path_ROI = "./ROIs/"


def read_S3(fn):
    test_file = Path(fn)
    # print(fn)
    r = np.zeros((5424, 2959)) * np.nan
    if test_file.is_file():
        rx = rasterio.open(fn)
        r = rx.read(1)
        r[r > 1] = np.nan
    else:
        print("no file")
    return r


# bands to consider for classification
bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_21"]
n_bands = len(bands)

ni = 5424
nj = 2959  # all greenland raster dimensions

# initialise array to contain band data
Xs = np.zeros(((n_bands, ni, nj)))

# load bands into a 3 D array
for i, band in enumerate(bands):
    print("reading " + band)
    fn = path_raw + "2019-08-02_" + band + ".tif"
    r = read_S3(fn)
    Xs[i, :, :] = r

# %% test multi band array loading
plt.imshow(Xs[3, :, :])
# %%
# load lables into a 3 D array

rois = ["bright_ice", "dark_ice", "dry_snow", "flooded_snow", "melted_snow", "red_snow"]

n_features = len(rois)

LABELS = np.zeros(((n_features, ni, nj)))

for i, roi in enumerate(rois):
    print(roi, path_ROI + roi + ".shp")
    ROI_label_gdf = gpd.read_file(path_ROI + roi + ".shp")
    ROI_label = ROI_label_gdf.to_crs("epsg:3413").iloc[0].geometry

    temp = rasterio.open(fn)
    masked = rasterio.mask.mask(temp, [ROI_label], nodata=np.nan)[0][0, :, :]
    LABELS[i, :, :] = masked

    print(roi, np.nanmean(masked), np.nanstd(masked))

    do_histogram = 1
    if do_histogram:
        temp = masked[np.isfinite(masked)]
        plt.hist(temp)  # plt.imshow(r)
        plt.title(roi)
        plt.show()
# %% test multi label array loading
plt.imshow(LABELS[2, :, :])

# %% format inputs and labels

labels = []
S3_data_for_labels_all = []

for i, roi in enumerate(rois):
    mask = np.isfinite(LABELS[i, :, :])

    S3_data_for_labels = np.vstack(
        [read_S3(f"{path_raw}2019-08-02_{band}.tif")[mask] for band in bands]
    ).T

    S3_data_for_labels_all.append(S3_data_for_labels)

    labels.append(np.repeat(i, S3_data_for_labels.shape[0]))

labels_for_svm = np.hstack(
    labels
)  # 1D array (size = nb of pixels labelled, values are label numbers)
inputs_for_svm = np.vstack(
    S3_data_for_labels_all
)  # 2D array (size = nb of S3 bands * nb of pixels labelled, values are the S3 reflectances)

no_nan_mask = np.where(np.sum(~np.isnan(inputs_for_svm), axis=1) == 4)[0]

inputs_for_svm = inputs_for_svm[no_nan_mask, :]
labels_for_svm = labels_for_svm[no_nan_mask]

# %% train SVM


# %%

clf = svm.SVC(decision_function_shape="ovo")
clf.fit(inputs_for_svm, labels_for_svm)
