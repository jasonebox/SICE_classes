#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 06:23:44 2023

@author: jason

issues:
    compute monthly means on Thredds?
    
outputs npy arrays and figures of SICE3 data

['ANG', 'AOD_550', 'O3_SICE', 'al', 'albedo_bb_planar_sw', 'albedo_bb_spherical_sw', 'albedo_spectral_planar_01',
 'albedo_spectral_planar_02', 'albedo_spectral_planar_03', 'albedo_spectral_planar_04', 'albedo_spectral_planar_05',
 'albedo_spectral_planar_06', 'albedo_spectral_planar_07', 'albedo_spectral_planar_08', 'albedo_spectral_planar_09',
 'albedo_spectral_planar_10', 'albedo_spectral_planar_11', 'albedo_spectral_planar_16', 'albedo_spectral_planar_17',
 'albedo_spectral_planar_18', 'albedo_spectral_planar_19', 'albedo_spectral_planar_20', 'albedo_spectral_planar_21',
 'cloud_mask', 'crs', 'cv1', 'cv2', 'factor', 'grain_diameter', 'isnow', 'lat', 'lon', 'r0', 'rBRR_01', 'rBRR_02',
 'rBRR_03', 'rBRR_04', 'rBRR_05', 'rBRR_06', 'rBRR_07', 'rBRR_08', 'rBRR_09', 'rBRR_10', 'rBRR_11', 'rBRR_16', 
 'rBRR_17', 'rBRR_18', 'rBRR_19', 'rBRR_20', 'rBRR_21', 'r_TOA_01', 'r_TOA_02', 'r_TOA_03', 'r_TOA_04', 'r_TOA_05',
 'r_TOA_06', 'r_TOA_07', 'r_TOA_08', 'r_TOA_09', 'r_TOA_10', 'r_TOA_11', 'r_TOA_12', 'r_TOA_13', 'r_TOA_14', 'r_TOA_15',
 'r_TOA_16', 'r_TOA_17', 'r_TOA_18', 'r_TOA_19', 'r_TOA_20', 'r_TOA_21', 'saa', 'snow_specific_surface_area', 'sza',
 'threshold', 'vaa', 'vza']
"""

import os
import pandas as pd
# from datetime import datetime 
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.basemap import Basemap
import numpy as np
# import xarray as xr
import os.path
# from glob import glob
import rasterio
from datetime import date, timedelta
from rasterio.transform import Affine

fs=24 ; th=1
# plt.rcParams['font.sans-serif'] = ['Georgia']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "grey"
plt.rcParams["font.size"] = fs
#params = {'legend.fontsize': 20,
#          'legend.handlelength': 2}
plt.rcParams['legend.fontsize'] = fs*0.8

def datesx(date0,date1):
    # difference between current and previous date
    delta = timedelta(days=1)
    # store the dates between two dates in a list
    dates = []
    while date0 <= date1:
        # add current date to list by converting  it to iso format
        dates.append(date0.isoformat())
        # increment start date by timedelta
        date0 += delta
    print('Dates between', date0, 'and', date1)
    print(dates)
    return dates

raw_data_path='/Users/jason/0_dat/S3/opendap/Greenland/'

doys=pd.to_datetime(np.arange(1,366), format='%j')
months=doys.strftime('%b')
months_int=doys.strftime('%m').astype(int)
ni=5424
nj=2959


years=np.arange(2017,2018).astype(str)
# years=np.arange(2022,2023).astype(str)
# years=np.arange(2018,2019).astype(str)
# years=np.arange(2023,2024).astype(str)
# years=np.arange(2017,2024).astype(str)

month='07'
do_cum=0
cum_name='' ; cum_name2='cumulative'

ni = 5424 ; nj = 2959  # all greenland raster dimensions

if do_cum:
    plotvar=np.zeros((ni,nj))*np.nan
    cum_name='_cumu'
    cum_name2='cumulative'

for year in years:
    dates=datesx(date(int(year), 8, 21),date(int(year), 8, 21))
    dates=datesx(date(int(year), 7, 1),date(int(year), 8, 11))
    dates=datesx(date(int(year), 7, 14),date(int(year), 8, 4))
    dates=datesx(date(int(year), 7, 1),date(int(year), 7, 31))
    
                    
    varnam='r_TOA_21' ; lo=0 ; hi=1 ; extend='both' ; units='unitless'
    bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_08", "r_TOA_21"]
    bands = ["r_TOA_21"]
    for band in bands:
        cc=0
        # varnam='albedo_bb_planar_sw' ; lo=0.3 ; hi=0.85 ; extend='both' ; units='unitless'
        # varnam='AOD_550' ; lo=0. ; hi=0.25 ; extend='both' ; units='unitless'
        raster_stack=[]
        for i,datex in enumerate(dates):
            # if i==0:
            # if i>=0:
            infile=f"{raw_data_path}{year}/{datex}_{band}.tif"
            test_file = Path(infile)
            if test_file.is_file():
                # datex=pd.to_datetime(file.split('/')[-1][0:10]).strftime('%Y-%m-%d')
                print(f'{band} {datex}')
            
                file = rasterio.open(infile)
                # profile_S3=SWIR1x.profile
                r=file.read(1)
                raster_stack.append(r)
            else:
                cc+=1
                print(f'{band} missing date {datex}, count of missing: {cc}')
        
        raster_stack=np.array(raster_stack)
        
        #!! average within N std is imperfect, WTF to do?
        mean_month=np.nanmean(raster_stack,axis=0)
        
        # print(np.shape(mean_month))
        count_month=np.count_nonzero(~np.isnan(raster_stack),axis=0)
        mean_month[count_month<4]=np.nan

        # print(np.shape(count_month))
        
        # plt.imshow(count_month)
        # plt.colorbar()
        # plt.axis('off')
        # #%%
        plotvar=mean_month
        
        ofile=f'/Users/jason/Dropbox/S3/SICE_classes/SICE_rasters/Greenland/monthly/{band}_{year}{month}.tif'
        temp = rasterio.open('/Users/jason/Dropbox/S3/SICE_classes/SICE_rasters/Greenland/2019/2019-08-02_r_TOA_21.tif')
        profile=temp.profile

        with rasterio.Env():
            with rasterio.open(ofile, 'w', **profile) as dst:
                dst.write(mean_month, 1)
        # write_geotiff(ni,nj,mean_month,ofile)
#%%
do_plot=1

if do_plot:
    plt.close()
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10*ni/nj))
    
    if varnam=='grain_diameter':
        cm = plt.cm.magma                        
        cm.set_over('r')
    else:
        cm = plt.cm.viridis
        cm.set_under('purple')
        cm.set_over('orange')
        
    c=plotvar


    pp=plt.imshow(c,cmap=cm,
                  vmin=lo,vmax=hi)
    
    plt.axis('Off')
    # plt.colorbar()
    
    # ax = plt.gca()     
    
    xx0=0.99
    yy0x=0.18
    dyx=0.4
    
    
    # ---------------------    
    # --------------------- colorbar location
    cbaxes = fig.add_axes([xx0-0.04, yy0x, 0.03, dyx]) 
    
    cbar = plt.colorbar(pp,orientation='vertical',cax=cbaxes,extend=extend)
    # # --------------------- colorbar location
    # xx0=0.6 ; yy0x=0.1 ; dxy=0.4
    # cbaxes = fig.add_axes([xx0-0.04, yy0x, 0.015, dxy]) 
    
    # cbar = plt.colorbar(ax,orientation='vertical',format="%d",cax=cbaxes)
    # # cbar = plt.colorbar(ax,orientation='vertical')

    # mult=1
    # yy0=yy0x+0.45 ; dy2=-0.03 ; cc=0
    
    cc=0
    
    if do_cum:
        units_title=varnam+'\n'+cum_name2+',\n'+units
    else:
        units_title=units
    
    plt.text(1.02, 0.96,datex+'\n'+varnam, fontsize=fs*1.2,
             transform=ax.transAxes, color='k') ; cc+=1. 
    
    plt.text(xx0+0.15, yy0x+dyx+0.04,units_title,ha='center', fontsize=fs,
             transform=ax.transAxes, color='k') ; cc+=1. 
    
    plt.text(1.01, 0.005,'Sentinel-3 SICEv3\nGEUS, ESA NoR', fontsize=fs,
             transform=ax.transAxes, color='b') ; cc+=1. 
    
    ly='x'
    
    if ly == 'x':
         plt.show() 
    
    DPI=100
     
    if ly == 'p':
         figpath='/Users/jason/0_dat/S3/opendap/Figs/'+varnam+'/'
         os.system('mkdir -p '+figpath)
         if do_cum:
             figpath='/Users/jason/0_dat/S3/Figs/'+varnam+'/'+cum_name2+'/'
         os.system('mkdir -p '+figpath)
         figname=datex+cum_name
         plt.savefig(figpath+figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
