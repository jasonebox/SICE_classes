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
import xarray as xr
from rasterio.transform import Affine
from pyproj import CRS as CRSproj
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def opentiff(filename):
    
    "Input: Filename of GeoTIFF File "
    "Output: xgrid,ygrid, data paramater of Tiff, the data projection"
   
    da = xr.open_rasterio(filename)
    proj = CRSproj.from_string(da.crs)


    transform = Affine(*da.transform)
    elevation = np.array(da.variable[0],dtype=np.float32)
    nx,ny = da.sizes['x'],da.sizes['y']
    x,y = np.meshgrid(np.arange(nx,dtype=np.float32), np.arange(ny,dtype=np.float32)) * transform

    da.close()
   
    return x,y,elevation,proj

def exporttiff(x,y,z,crs,filename):
    
    "Input: xgrid,ygrid, data paramater, the data projection, export path, name of tif file"
   
    resx = (x[0,1] - x[0,0])
    resy = (y[1,0] - y[0,0])
    transform = Affine.translation((x.ravel()[0]),(y.ravel()[0])) * Affine.scale(resx, resy)
    
    if resx == 0:
        resx = (x[0,0] - x[1,0])
        resy = (y[0,0] - y[0,1])
        transform = Affine.translation((y.ravel()[0]),(x.ravel()[0])) * Affine.scale(resx, resy)
    
    # with rio.open(
    #     path,
    #     'w',
    #     driver='GTiff',
    #     height=z.shape[0],
    #     width=z.shape[1],
    #     count=1,
    #     compress='lzw',
    #     dtype=z.dtype,
    #     # dtype=rasterio.uint8,
    #     crs=crs,
    #     transform=transform,
    #     ) as dst:
    #         dst.write(z, 1)
    with rasterio.open(
    filename,
    'w',
    driver='GTiff',
    height=z.shape[0],
    width=z.shape[1],
    count=1,
    compress='lzw',
    dtype=z.dtype,
    crs=crs,
    transform=transform,
    ) as dst:
        dst.write(z, 1)
    
    dst.close()
    
    return None 


# ## change to your system's login name to change dir for local work
if os.getlogin() == "jason":
    base_path = "/Users/jason/Dropbox/S3/SICE_classes/"
if os.getlogin() == "adrien":
    base_path = "/home/adrien/EO-IO/SICE_classes/"

# !! Rasmus paths
# current_path = os.getcwd()
# base_path = os.path.abspath('..')

os.chdir(base_path)

# relative paths
path_raw = "./SICE_rasters/"
path_ROI = "./ROIs/"
path_Figs = "./Figs/"

raster_path = base_path + path_raw

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

datex='2019-08-02'

ni = 5424
nj = 2959  # all greenland raster dimensions

band = bands[0]

# initialise array to contain band data
Xs = np.zeros(((n_bands, ni, nj)))

# load bands into a 3 D array
for i, band in enumerate(bands):
    print("reading " + band)
    fn = path_raw +  datex+ "_" + band + ".tif"
    r = read_S3(fn)
    Xs[i, :, :] = r

# %% test multi band array loading
plt.imshow(Xs[1, :, :])
plt.axis("off")
plt.show()
# %%
# load lables into a 3 D array

rois = ["dry_snow", "melted_snow", "flooded_snow","red_snow","bright_ice", "dark_ice"]

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
plt.imshow(LABELS[1, :, :])
plt.axis("off")
plt.show()

# %% format inputs and labels and make test dataset

labels = []
S3_data_for_labels_all = []

# Training Data with Label
for i, roi in enumerate(rois):
    mask = np.isfinite(LABELS[i, :, :])

    S3_data_for_labels = np.vstack(
        [read_S3(f"{path_raw}{datex}_{band}.tif")[mask] for band in bands]
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


### Splitting dataset into Training and Test set ###
# When there is more labeled data, it would be good to have a another date as test set #

data_train, data_test, label_train, label_test \
     = train_test_split(inputs_for_svm, labels_for_svm, test_size=0.10, random_state=42) 

### Maybe introduce robust statistics ### 
# ml_est = sum(w_i * d_i) / sum(w_i)
# if samples are gaussian w_i = 1 
### reweighting algorithm using Huber weights, until convergence ###

def huber_w(w,d,sigma):
    
    break_p = 1.345  
    ml_est = sum(w * d) / sum(w)
    eps = (d - ml_est) / sigma 
    
    w[eps > break_p] = break_p/eps[eps > break_p]
    w[abs(eps) <= break_p] = 1
    w[eps < -break_p] = -break_p/eps[eps < -break_p]
    
    return w 


no_i = np.arange(50) #Number of iterations
w_all = np.ones_like(data_train)

for n in np.arange(n_features):
    for b in np.arange(len(bands)):
        w = w_all[:,b][label_train == n]
        d = data_train[:,b][label_train == n]
        sigma = np.std(d)
        for i in no_i:
            w = huber_w(w,d,sigma)
        w_all[:,b][label_train == n] = w 

w_samples = np.array([np.nanmean(w) for w in w_all])        
 

# %% Find Regularization Parameter C
# takes several minutes

alpha = np.arange(1,10,0.1)

n_alpha=len(alpha)

lloss_svc = np.ones_like(alpha)

for i,a in enumerate(alpha):
  print(f'Finding Solution for Alpha Value no. {i} {n_alpha-i}')
  C = 1 / a
  clf = svm.SVC(C = C, decision_function_shape="ovo",probability = True)
  clf.fit(data_train, label_train,sample_weight=w_samples)  
  label_prob = clf.predict_proba(data_test)
  lloss_svc[i] = log_loss(label_test,label_prob)
  
  #score_svc[i] = clf.sore(data_test,label_test)

#%% plot result
alpha = np.arange(1,10,0.1)

plt.figure(figsize=(14,8))
ax=plt.gca()
ax.plot(alpha,lloss_svc)

lloss_min_val = np.nanmin(lloss_svc)
alp_lloss = alpha[lloss_svc==lloss_min_val]
lloss_min = lloss_svc[lloss_svc==lloss_min_val]

ax.scatter(alp_lloss,lloss_min,c='red',s=12)

plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.xlabel('Alpha Value',fontsize = 26)
plt.ylabel('svc Log Loss',fontsize = 26)

plt.show()

# scores = defaultdict(list)
# 
#     clf.fit(X_train, y_train)
#     y_prob = clf.predict_proba(X_test)
#     y_pred = clf.predict(X_test)
#     scores["Classifier"].append(name)

#     for metric in [brier_score_loss, log_loss, roc_auc_score]:
#         score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
#         scores[score_name].append(metric(y_test, y_prob[:, 1]))

#     for metric in [precision_score, recall_score, f1_score]:
#         score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
#         scores[score_name].append(metric(y_test, y_pred))

#     score_df = pd.DataFrame(scores).set_index("Classifier")
#     score_df.round(decimals=3)

# score_df

# %% train SVM

alpha = 0

if alpha == 0:
    C = 1
else:
    C = 1 / alpha

clf = svm.SVC(C = C, decision_function_shape="ovo",probability = True)
clf.fit(data_train, label_train,sample_weight=w_samples)

model_params = clf.get_params()


# %% Test SVM 

score_svc = clf.score(data_test,label_test)
label_prob = clf.predict_proba(data_test)
lloss_svc = log_loss(label_test,label_prob)

print(f"Accuracy of Model {score_svc}")
print(f"Log Loss of Model {lloss_svc}")

# %% Data to Classify
# takes ~15 sec
S3_data_for_predict_all = []


# Data to predict a Label

x_grid,y_grid,dummy,proj = opentiff(f"{raster_path}{datex}_{band}.tif")


### Because we are predicting on the same date as the training data , ### 
### i made mask to remove the part which is labeled already ### 
### If we predict on another date we do not this mask! ###

mask_predict = ~np.isfinite(LABELS[:, :, :])
mask_predict = np.array([[all(mask_predict[:,n,m]) for m in np.arange(nj)] for n in np.arange(ni)])

x_coor = x_grid[mask_predict]
y_coor = y_grid[mask_predict]

S3_data_for_predict = np.vstack(
     [read_S3(f"{path_raw}{datex}_{band}.tif")[mask_predict] for band in bands]
 ).T


no_nan_mask = np.where(np.sum(~np.isnan(S3_data_for_predict), axis=1) == 4)[0]
S3_data_for_predict_all =  S3_data_for_predict[no_nan_mask, :]

x_coor = x_coor[no_nan_mask]
y_coor = y_coor[no_nan_mask]

# %% Model Predict
# takes ~10 minutes
labels_predict = clf.predict(S3_data_for_predict_all)

# %% Regridding to SICE Grid and saving as geotiff

tree = KDTree(np.c_[x_coor.ravel(),y_coor.ravel()])   
datagrid = np.ones_like(x_grid) * np.nan

for i,(xmid,ymid) in enumerate(zip(x_grid.ravel(),y_grid.ravel())):     
        
        dd, ii = tree.query([xmid,ymid],k = 1,p = 2)
        ii = ii[dd<500]
        
        if len(ii) > 0:
            datagrid.ravel()[i] = labels_predict.ravel()[ii]

file = raster_path + os.sep +  datex +'labels_v2.tif'
exporttiff(x_grid,y_grid,datagrid,proj,file)
# %% plot classes raster
# classes=read_S3(f"{file}")

classes=datagrid
nams=rois

fs=12

plt.close()
fig, ax = plt.subplots(figsize=(10, 10))

co=150
palette = np.array([
            [0,0,0], # NaN
            [255, 255, 255], # 5 dry snow
            [255,200,200],   # 3, melted snow
            [100,100,250], # 4 flooded snow
            [255, 0, 0], # 6 red snow
            [co,co,co],   # 2 bright bare ice
            [200,100,200],   # 1 dark bare ice
            # [255,165, 0], # 6 red ice
            ]
                    )  # white

classesx=classes.copy()
classesx+=1
classesx[np.isnan(classesx)]=0
RGB=palette[classesx.astype(int)]
cntr=ax.imshow(RGB)

plt.axis('off')

mult=0.6
xx0=0.6 ; yy0=0.04 ; dy=0.02 ; cc=0
for i,nam in enumerate(nams):
    plt.text(xx0, yy0+dy*cc,nam,
              color=palette[i+1]/255,
              transform=ax.transAxes, fontsize=fs*mult,ha="left")
    cc+=1

cc=0
xx0=0.015 ; yy0=0.955
xx0=0.62 ; yy0=0.18
mult=0.8
color_code='k'
props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
plt.text(xx0, yy0, datex,
        fontsize=fs*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5

ly='p'
if ly == 'x':plt.show()

if ly == 'p':
    band='classes'
    # opath='/Users/jason/0_dat/S3/opendap/Figs/'+region_name+'/'
    os.system('mkdir -p '+path_Figs)
    figname=path_Figs+datex+'_classes_SVC.png' 
    plt.savefig(figname, bbox_inches='tight', dpi=200, facecolor='k')
    os.system('open '+figname)