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

# ## change to your system's login name to change dir for local work
if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/S3/SICE_classes/'
os.chdir(base_path)

# relative paths
path_raw='./SICE_rasters/'
path_ROI='./ROIs/'

def read_S3(fn):
    test_file = Path(fn)
    r=np.zeros((5424, 2959))*np.nan
    if test_file.is_file():
        rx = rasterio.open(fn)
        r=rx.read(1)
        r[r>1]=np.nan
    else:
        print('no file')
    return r

# bands to consider for classification
bands=['r_TOA_02','r_TOA_04','r_TOA_06','r_TOA_21']
n_bands=len(bands)

ni=5424 ; nj=2959 # all greenland raster dimensions

# initialise array to contain band data
Xs=np.zeros(((n_bands,ni,nj)))

# load bands into a 3 D array
for i,band in enumerate(bands):
    print('reading '+band)
    fn=path_raw+'2019-08-02_'+band+'.tif'
    r=read_S3(fn)
    Xs[i,:,:]=r

# #%% test multi band array loading
# plt.imshow(Xs[3,:,:])

# load lables into a 3 D array

rois=['bright_ice',
    'dark_ice',
    'dry_snow',
    'flooded_snow',
    'melted_snow',
    'red_snow']

n_features=len(rois)
#%%
LABELS=np.zeros(((n_features,ni,nj)))

for i,roi in enumerate(rois):
    print(roi,path_ROI+roi+'.shp')
    ROI_label_gdf = gpd.read_file(path_ROI+roi+'.shp')
    ROI_label = ROI_label_gdf.to_crs('epsg:3413').iloc[0].geometry
    
    temp = rasterio.open(fn)
    masked = rasterio.mask.mask(temp,[ROI_label], nodata=np.nan)[0][0, :, :]
    LABELS[i,:,:]=masked
    
    print(roi,np.nanmean(masked),np.nanstd(masked))
    
    do_histogram=1
    if do_histogram:
        temp=masked[np.isfinite(masked)]
        plt.hist(temp)# plt.imshow(r)
        plt.title(roi)
        plt.show()
#%% test multi label array loading
plt.imshow(LABELS[2,:,:])
    #%% https://scikit-learn.org/stable/modules/svm.html
    # from sklearn import tree

    # masked2=masked.copy()

    # masked2[~np.isfinite(masked2)]=0.
    # temp2=r.copy()
    # temp2[~np.isfinite(temp2)]=0.

    # print(np.shape(masked2))
    # print(np.shape(temp2))
#%%
y = LABELS
X = Xs
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
X, y = make_classification(n_features=n_features, random_state=0)
clf = make_pipeline(StandardScaler(),LinearSVC(dual="auto", random_state=0, tol=1e-5))
clf.fit(X, y)
#%%



        
        # #%%
        
        # fs=18
        # plt.close()
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(temp)
        # plt.axis('off')
        # # plt.title(datex)


        # # plt.text(xx0, yy0+dy*cc,datex,
        # #          color='k',
        # #           transform=ax.transAxes, fontsize=fs*mult,ha="left")
        # cc=0
        # xx0=0.03 ; yy0=0.955
        # mult=0.8
        # color_code='k'
        # props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
        # plt.text(xx0, yy0, datex,
        #         fontsize=fs*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes) ; cc+=1.5

        # # du_color_bar=0
        
        # # if du_color_bar:
        # #     cbax = ax.inset_axes([1.04, 0.01, 0.05, 0.4], transform=ax.transAxes)
        # #     # cbax.ax.set_label(fontsize=12)
    
        # #     # cbax.set_title('vertical\nwinds,\nm/s',fontsize=font_size,c='k',ha='center')
        # #     # clb=plt.colorbar(cntr, ax=ax, cax=cbax, shrink=0.7,orientation='vertical')#,extend=extend)            
        # #     # clb.ax.set_title(band+'\n',fontsize=12,ha='left')
        # #     cbar = fig.colorbar(cntr,ax=ax, cax=cbax, ticks=[0,1,2,3], orientation='vertical')
        # #     cbar.ax.tick_params(labelsize=12)
        # #     cbar.ax.set_xticklabels(['nan', 'dark ice', 'melted snow','bare ice','dry snow','red surface'])  # horizontal colorbar

        # # ax.set_facecolor('k')
        # # plt.margins(0,0)
        # ax.patch.set_edgecolor('black')  

        # ax.patch.set_linewidth('1')  

            
        # ly='p'
        
        # if ly == 'x':plt.show()

        # if ly == 'p':
        #     band='classes'
        #     # opath='/Users/jason/0_dat/S3/opendap/Figs/'+region_name+'/'
        #     opath='/Users/jason/0_dat/S3/opendap/'+region_name+'/'+year+'/'
        #     os.sLABELStem('mkdir -p '+opath)
        #     figname=opath+datex+'_RGB.png' 
        #     plt.savefig(figname, bbox_inches='tight', dpi=300, facecolor='k')
        #     os.sLABELStem('open '+figname)
