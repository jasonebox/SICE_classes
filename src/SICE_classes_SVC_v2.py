#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:41:27 2023

@authors: Jason, Rasmus, Adrien, Jasper

issues:
    For Jasper: see !! 
    for Rasmus:
    - make relative paths smarter
    - have code integrate better with Thredds, to not have to DL what files are needed locally
    - see !! was 4 now is n_bands (currently 3), lines ~505 and ~339
    - want to re-insert the band data that is masked out in the final classification
        adjust code to not clip data that's outside the training set
    - better results for more training data, e.g. different SZA, sza
        in the training, this code wants to load more than one date
    - how to feed in a different date for the prediction? in this case, monthly means, see !! below ~line 496
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from pathlib import Path
from sklearn import svm
import xarray as xr
import rasterio
from rasterio.mask import mask
from rasterio.transform import Affine
from pyproj import CRS as CRSproj
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import time


st_all = time.time()

def normalisedx(write_out,fn_band_A,fn_band_B,ofile):
    # normalised difference index
    test_file = Path(fn_band_A)
    if test_file.is_file():
        # print(fn_band_A)
        # print(fn_band_B)
        band_Ax = rasterio.open(fn_band_A)
        profile=band_Ax.profile
        band_A=band_Ax.read(1)
    
        band_Bx = rasterio.open(fn_band_B)
        profile=band_Bx.profile
        band_B=band_Bx.read(1)

        #(band 8-band6)/(band 8 + band6)
        normalised=(band_A-band_B)/(band_A+band_B)
        # normalised[normalised<-0.2]=np.nan
        # normalised[normalised>0.2]=np.nan
        if write_out:
            # resx = (x[1] - x[0])
            # resy = (y[1] - y[0])
            # transform = Affine.translation((x[0]),(y[0])) * Affine.scale(resx, resy)
            with rasterio.Env():
                with rasterio.open(ofile, 'w', **profile) as dst:
                    dst.write(normalised, 1)
            # with rasterio.open(
            #     ofile,
            #     'w', #**profile,
            #     driver='GTiff',
            #     height=normalised.shape[0],
            #     width=normalised.shape[1],
            #     count=1,
            #     compress='lzw',
            #     dtype=normalised.dtype,
            #     # dtype=rasterio.uint8,
            #     crs=PolarProj,
            #     transform=transform,
            #     ) as dst:
            #         dst.write(normalised, 1)
    else:
        print('file missing')
    return normalised

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

def ratio_image(write_out,fn_band_A,fn_band_B,ofile):
    
    rat=np.zeros((5424, 2959))*np.nan

    test_file = Path(fn_band_A)
    if test_file.is_file():
        band_Ax = rasterio.open(fn_band_A)
        profile=band_Ax.profile
        band_A=band_Ax.read(1)
    
        band_Bx = rasterio.open(fn_band_B)
        profile=band_Bx.profile
        band_B=band_Bx.read(1)
        
        rat=band_A/band_B
            
        if write_out:
            with rasterio.Env():
                with rasterio.open(ofile, 'w', **profile) as dst:
                    dst.write(rat, 1)
    return rat

def read_S3(fn):
    test_file = Path(fn)
    # print(fn)
    r = np.zeros((5424, 2959)) * np.nan
    if test_file.is_file():
        print("reading " + fn)
        rx = rasterio.open(fn)
        r = rx.read(1)
        r[r > 1] = np.nan
    else:
        print("no file")
    return r

from skimage import exposure # maybe add this to the import packages block at the top?

def RGBx(f_Red,f_Green,f_Blue, out_file):
    red=read_S3(f_Red)
    gre=read_S3(f_Green)
    blu=read_S3(f_Blue)
    
    vred=red<0
    vgre=gre<0
    vblu=blu<0
    red[vred]=np.nan
    red[vgre]=np.nan
    red[vblu]=np.nan
    gre[vred]=np.nan
    gre[vgre]=np.nan
    gre[vblu]=np.nan
    blu[vred]=np.nan
    blu[vgre]=np.nan
    blu[vblu]=np.nan
    
    vred=red>1
    vgre=gre>1
    vblu=blu>1
    red[vred]=np.nan
    red[vgre]=np.nan
    red[vblu]=np.nan
    gre[vred]=np.nan
    gre[vgre]=np.nan
    gre[vblu]=np.nan
    blu[vred]=np.nan
    blu[vgre]=np.nan
    blu[vblu]=np.nan

    # v=((red<0)or(gre<0)or(blu<0))
    # red[v]=np.nan
    # gre[gre<0]=np.nan
    # blu[blu<0]=np.nan
#                v=np.where(red >=0 and red <=1)
    img = np.dstack((red,gre,blu))  # stacks 3 h x w arraLABELS -> h x w x 3
    # img[land] = exposure.adjust_log(img[land], 1.)
#                    # Gamma
    img = exposure.adjust_gamma(img, 2)
    
    # export as geoTIFF:
    bands = [red, gre, blu]
    
    red_o = rasterio.open(f_Red)
    meta = red_o.meta.copy()
    meta.update({"count": 3,
                 "nodata": -9999,
                 "compress": "lzw"})
    
    with rasterio.open(out_file, "w", **meta) as dest:
        for band, src in enumerate(bands, start=1):
            dest.write(src, band)
    
    return img    


# ## change to your system's login name to change dir for local work
if os.getlogin() == "jason":
    base_path = "/Users/jason/Dropbox/S3/SICE_classes/"
if os.getlogin() == "adrien":
    base_path = "/home/adrien/EO-IO/SICE_classes/"
if os.getlogin() == "rasmus":
# !! Rasmus paths
    current_path = os.getcwd()
    base_path = os.path.abspath('..')
if os.getlogin() == "Jasper":
    base_path = "E:/Jasper/Denmark/GEUS/SICE_classes/"

os.chdir(base_path)

# relative paths
path_raw = "./SICE_rasters/"
# path_raw = "/Users/jason/0_dat/S3/opendap/"
path_ROI = "./ROIs/"
path_Figs = "./Figs/"

raster_path = base_path + path_raw

# bands to consider for classification
bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_21"]
bands = ["r_TOA_02", "r_TOA_NDXI_0806", "r_TOA_NDXI_1110", "r_TOA_NDXI_0802", "r_TOA_21"] ; version_name='5bands_3NDXI'
bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_08", "r_TOA_10", "r_TOA_11", "r_TOA_21"] ; version_name='7bands_02_04_06_08_10_11_21'
bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_08", "r_TOA_21"] ; version_name='5bands_02_04_06_08_21'
# bands = ["r_TOA_02", "r_TOA_06", "r_TOA_08", "r_TOA_21"] ; version_name='4bands_02_06_08_21'
n_bands = len(bands)

region_name='Greenland'
datex='2019-08-02'; year='2019'
# datex='2021-07-30'; year='2021'
# datex='2017-07-28' ; year='2017'

#!! other dates
# datex = "2017-07-12"; year = "2017"
# datex = "2020-07-22"; year = "2020"
# datex='2022-07-31'; year='2022'

show_plots=1



# !! modify somehow to feed N dates 
# dates=['2019-08-02','2017-07-28']
# dates=['2019-08-02']

# for datex in dates:

do_generate_rasters=1

if do_generate_rasters:
    # for red snow
    normalised=normalisedx(1,f"{path_raw}{region_name}/{year}/{datex}_r_TOA_06.tif",
          f"{path_raw}{region_name}/{year}/{datex}_r_TOA_08.tif",
          f"{path_raw}{region_name}/{year}/{datex}_r_TOA_NDXI_0608.tif")

    # normalised=normalisedx(1,f"{path_raw}{region_name}/{year}/{datex}_r_TOA_08.tif",
    #       f"{path_raw}{region_name}/{year}/{datex}_r_TOA_02.tif",
    #       f"{path_raw}{region_name}/{year}/{datex}_r_TOA_NDXI_0802.tif")

    # for flooded areas
    ratio_BRx=ratio_image(1,f"{path_raw}{region_name}/{year}/{datex}_r_TOA_02.tif",
          f"{path_raw}{region_name}/{year}/{datex}_r_TOA_08.tif",
          f"{path_raw}{region_name}/{year}/{datex}_r_TOA_0802.tif")

    ratio_BRx=ratio_image(1,f"{path_raw}{region_name}/{year}/{datex}_r_TOA_06.tif",
          f"{path_raw}{region_name}/{year}/{datex}_r_TOA_08.tif",
          f"{path_raw}{region_name}/{year}/{datex}_r_TOA_0806.tif")

temp=RGBx(f"{path_raw}{region_name}/{year}/{datex}_r_TOA_08.tif",
  f"{path_raw}{region_name}/{year}/{datex}_r_TOA_06.tif",
  f"{path_raw}{region_name}/{year}/{datex}_r_TOA_02.tif",
  f"{path_raw}{region_name}/{year}/{datex}_r_TOA_RGB.tif")
if show_plots:
    plt.imshow(temp)
    plt.axis("Off")
    
#%%
ni = 5424 ; nj = 2959  # all greenland raster dimensions

# initialise array to contain stack of input rasters
Xs = np.zeros(((n_bands, ni, nj)))

# load bands into a 3 D array
for i, band in enumerate(bands):
    fn = path_raw+region_name+'/'+year+'/'+datex+ "_" + band + ".tif"
    r = read_S3(fn)
    Xs[i, :, :] = r


# %% test multi band array loading
if show_plots:
    print('test multi band array loading')
    band_choice_index=4
    plt.imshow(Xs[band_choice_index, :, :])#,vmin=-0.05,vmax=0.05)
    plt.axis("off")
    plt.title(bands[band_choice_index])
    plt.colorbar()
    plt.show()
# %% load lables into a 3 D array
print('load lables into a 3 D array')

features = ["dry_snow", "melted_snow", "flooded_snow","red_snow","bright_ice", "dark_ice"]

n_features = len(features)

LABELS = np.zeros(((n_features, ni, nj)))

for i, feature in enumerate(features):
    ROI_label_gdf = gpd.read_file(f"{path_ROI}{region_name}/{year}/{datex}/{feature}.shp")
    ROI_label = ROI_label_gdf.to_crs("epsg:3413").iloc[0].geometry

    band='r_TOA_NDXI_0806'
    band='r_TOA_02'
    fn = f"{path_raw}{region_name}/{year}/{datex}_{band}.tif" #path_raw+region_name+'/'+year+'/'+datex+ "_" + band + ".tif"
    temp = rasterio.open(fn)
    masked = rasterio.mask.mask(temp, [ROI_label], nodata=np.nan)[0][0, :, :]
    LABELS[i, :, :] = masked

    print(feature)
    print('     mean',np.nanmean(masked),'median',np.nanmedian(masked),'stdev',np.nanstd(masked))

    do_histogram = 1
    if do_histogram and show_plots:
        temp = masked[np.isfinite(masked)]
        plt.hist(temp)  # plt.imshow(r)
        plt.title(feature+' OLCI: '+band)
        plt.show()


# %% test multi label array loading
if show_plots:
    print('test multi label array loading')
    label_choice_index=1
    plt.imshow(LABELS[label_choice_index, :, :])
    plt.axis("off")
    plt.title(features[label_choice_index])
    plt.colorbar()
    plt.show()


# %% format inputs and labels and make test dataset
print('format inputs and labels and make test dataset, takes some seconds')
labels = []
S3_data_for_labels_all = []

# Training Data with Label
for i, roi in enumerate(features):
    mask = np.isfinite(LABELS[i, :, :])

    S3_data_for_labels = np.vstack(
        [read_S3(f"{path_raw}{region_name}/{year}/{datex}_{band}.tif")[mask] for band in bands]
    ).T
    
    S3_data_for_labels_all.append(S3_data_for_labels)

    labels.append(np.repeat(i, S3_data_for_labels.shape[0]))

labels_for_svm = np.hstack(
    labels
)  # 1D array (size = nb of pixels labelled, values are label numbers)
inputs_for_svm = np.vstack(
    S3_data_for_labels_all
)  # 2D array (size = nb of S3 bands * nb of pixels labelled, values are the S3 reflectances)

# !! is this kosher?
no_nan_mask = np.where(np.sum(~np.isnan(inputs_for_svm), axis=1) == n_bands)[0]

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
    # Huber weights, by variance for each band to allow higher prediction skill

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

do_parameter_C=0

if do_parameter_C:
    st = time.time()

    print('Find Regularization Parameter C')
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
 
    elapsed_time = time.time() - st
    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

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
# takes a few sec
print('train SVM, takes a few sec')
alpha = 0

if alpha == 0:
    C = 1
else:
    C = 1 / alpha

clf = svm.SVC(C = C, decision_function_shape="ovo",probability = True)
clf.fit(data_train, label_train,sample_weight=w_samples)

model_params = clf.get_params()


# %% Test SVM 

print('Test SVM')
score_svc = clf.score(data_test,label_test)
label_prob = clf.predict_proba(data_test)
lloss_svc = log_loss(label_test,label_prob)

print(f"Accuracy of Model {score_svc}")
print(f"Log Loss of Model {lloss_svc}")

# %% Data to Classify
# takes ~20 sec
print('Data to Classify, takes ~20 sec')
st = time.time()

S3_data_for_predict_all = []


# Data to predict a Label
x_grid,y_grid,dummy,proj = opentiff(f"{path_raw}{region_name}/{year}/{datex}_{band}.tif")


### Because we are predicting on the same date as the training data , ### 
### i made mask to remove the part which is labeled already ### 
### If we predict on another date we do not this mask! ###

mask_predict = ~np.isfinite(LABELS[:, :, :])
mask_predict = np.array([[all(mask_predict[:,n,m]) for m in np.arange(nj)] for n in np.arange(ni)])

x_coor = x_grid[mask_predict]
y_coor = y_grid[mask_predict]

#!! predicting on say another date
S3_data_for_predict = np.vstack(
      [read_S3(f"{path_raw}{region_name}/{year}/{datex}_{band}.tif")[mask_predict] for band in bands]
     # [read_S3(f"{path_raw}monthly/{band}_{year}07.tif")[mask_predict] for band in bands] # !!
 ).T

# !! is this kosher?
no_nan_mask = np.where(np.sum(~np.isnan(S3_data_for_predict), axis=1) == n_bands)[0]
S3_data_for_predict_all =  S3_data_for_predict[no_nan_mask, :]

x_coor = x_coor[no_nan_mask]
y_coor = y_coor[no_nan_mask]

elapsed_time = time.time() - st
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
# %% Model Predict
# takes 2-3 minutes
st = time.time()

print('Model Predict, takes 2-3 minutes')
labels_predict = clf.predict(S3_data_for_predict_all)

elapsed_time = time.time() - st
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# %% Regridding to SICE Grid and saving as geotiff
# takes ~12 minutes
# !! optimise?? live with it until it stops us

st = time.time()

print('Regridding to SICE Grid and saving as geotiff,  takes ~12 minutes')
tree = KDTree(np.c_[x_coor.ravel(),y_coor.ravel()])   
classes = np.ones_like(x_grid) * np.nan

for i,(xmid,ymid) in enumerate(zip(x_grid.ravel(),y_grid.ravel())):     
        
        dd, ii = tree.query([xmid,ymid],k = 1,p = 2)
        ii = ii[dd<500]
        
        if len(ii) > 0:
            classes.ravel()[i] = labels_predict.ravel()[ii]

file = raster_path + os.sep +  datex +'_labels_'+version_name+'.tif'
exporttiff(x_grid,y_grid,classes,proj,file)

elapsed_time = time.time() - st
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# %% plot classes raster
# classes=read_S3(f"{file}")
print('plot classes raster')
nams=features

if show_plots==0:
    ly='p'

fs=12 # fontsize

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

props = dict(boxstyle='round', facecolor='k', alpha=1,edgecolor='w')
plt.text(xx0, yy0, datex,
        fontsize=fs*mult,color='w',bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5

if ly == 'x':plt.show()

if ly == 'p':
    band='classes'
    # opath='/Users/jason/0_dat/S3/opendap/Figs/'+region_name+'/'
    os.system('mkdir -p '+path_Figs)
    figname=path_Figs+datex+'_classes_SVM'+version_name+'.png' 
    plt.savefig(figname, bbox_inches='tight', dpi=200, facecolor='k')
    os.system('open '+figname)

elapsed_time = time.time() - st_all
print('Execution time all:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
