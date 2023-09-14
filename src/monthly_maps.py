# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:21:53 2023

@author: rabni
"""


import argparse
from pyproj import CRS
import sys 
import logging
import time
import datetime
import glob
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from rasterio.transform import Affine
import rasterio
import warnings
import xarray as xr
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from multiprocessing import set_start_method,get_context

if sys.version_info < (3, 4):
    raise "must use python 3.6 or greater"
    
if not os.path.exists("logs"):
        os.makedirs("logs")
        
logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f'logs/monthlymaps_{time.strftime("%Y_%m_%d",time.localtime())}.log'),
            logging.StreamHandler()
        ])

def parse_arguments():
        parser = argparse.ArgumentParser(description='Date range excicuteable for the CARRA2 Module')
        parser.add_argument("-mo","--month", type=str,help="Please input the month(s) you want to process in this format [YYYY-MM]")
        parser.add_argument("-c","--cores", type=int,default=4,help="Please input the number of cores you want to use")
        args = parser.parse_args()
        return args
    
    
def class_val(cl): 
    class_w = [1,0,5,3,4,2]
    return class_w[int(cl)]

def class_transform(matrix):
    matrix[~np.isnan(matrix)] = np.array(list(map(class_val,matrix[~np.isnan(matrix)])))
    return matrix

def remove_noise(arr):
    
    unique_elements, counts = np.unique(arr, return_counts=True)
    mask = np.isin(arr,unique_elements[counts < 3])
    arr[mask] = np.nan
    return arr

def multimaps(month,base_folder):
    
    #logging.info("Processing: " + month)
    print(f"Processing: {month}")
    
    data_folder = base_folder + os.sep + 'output'
    output_folder = base_folder + os.sep + 'monthlymaps'
    
    print(f'data_folder: {data_folder}')
    month_s = month.replace('-','_')
    files = glob.glob(data_folder + os.sep + f"{month_s}*.tif")
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    start = 1
    
    no_f = len(files)
    
    data = []
    
    print(f'number of files: {no_f}')
    
    print('loading files, class transforming, and saving into matrix')
    for i,f in enumerate(files): 
        
        x,y,z,crs = opentiff(f)
        #z = class_transform(z)
        
        if start == 1:
            data = np.tile(z * np.nan, (len(files), 1, 1))
            start = 0
            
        data[i,:,:] = z
    
    if len(data[:,0,0]) > 26:
        
        print('computing monthly maps.....')
        l,m,n = np.shape(data)
        
        data =  np.transpose(np.array([[remove_noise(data[:,i,j]) for i in range(m)] for j in range(n)]))
        
        mergemedian = np.nanmedian(data, axis=0) 
        mergemax = np.nanmax(data,axis=0)
        mergemin = np.nanmin(data,axis=0)
        
        
        name_median = month + "_SICE_Classes_monthlymedian_v2.tif"  
        exporttiff(x,y,mergemedian,CRS.from_string("+init=EPSG:3413"),output_folder,name_median)
        name_max = month + "_SICE_Classes_monthlymax_v2.tif"  
        exporttiff(x,y,mergemax,CRS.from_string("+init=EPSG:3413"),output_folder,name_max)
        name_min = month + "_SICE_Classes_monthlymin_v2.tif"  
        exporttiff(x,y,mergemin,CRS.from_string("+init=EPSG:3413"),output_folder,name_min)
        
        print(f"{month} has been exported")
        #logging.info(f"{month} has been exported")
    else: 
        #logging.info(f"{month} is missing data, please process rest of the dates in that month")
        print(f"{month} is missing data, please process rest of the dates in that month")

def exporttiff(x,y,z,crs,path,filename):
    
    "Input: xgrid,ygrid, data paramater, the data projection, export path, name of tif file"
    
    resx = (x[0,1] - x[0,0])
    resy = (y[1,0] - y[0,0])
    transform = Affine.translation((x.ravel()[0]),(y.ravel()[0])) * Affine.scale(resx, resy)
    
    if resx == 0:
        resx = (x[0,0] - x[1,0])
        resy = (y[0,0] - y[0,1])
        transform = Affine.translation((y.ravel()[0]),(x.ravel()[0])) * Affine.scale(resx, resy)
    
    with rasterio.open(
    path + os.sep + filename,
    'w',
    driver='GTiff',
    height=z.shape[0],
    width=z.shape[1],
    count=1,
    dtype=z.dtype,
    crs=crs,
    transform=transform,
    ) as dst:
        dst.write(z, 1)
    
    dst.close()
    
    return None 
 
def opentiff(filename):
    
   "Input: Filename of GeoTIFF File "
   "Output: xgrid,ygrid, data paramater of Tiff, the data projection"
   
   da = xr.open_rasterio(filename)
   proj = CRS.from_string(da.crs)
   
   transform = Affine(*da.transform)
   elevation = np.array(da.variable[0],dtype=np.float32)
   nx,ny = da.sizes['x'],da.sizes['y']
   x,y = transform * np.meshgrid(np.arange(nx,dtype=np.float32), np.arange(ny,dtype=np.float32))
   da.close()
   
   return x,y,elevation,proj


if __name__ == "__main__":
    
    args = parse_arguments()
    base_folder = os.path.abspath("..")
    data_folder = base_folder + os.sep + 'output'
    #thisyear = datetime.date.today().year
    #months = [str(y)+m for m in [args.month] for y in list(np.arange(1982,thisyear))]
    
    
    logging.info("Number of Months: " + str(len(args.month)))
    
    monthlyfolder = base_folder + os.sep + "monthlymaps"
    
    if not os.path.exists(monthlyfolder):
        os.mkdir(monthlyfolder)
   
    base_folder_list = [base_folder for i in range(len(args.month))] 
    
    set_start_method("spawn")
    
    with get_context("spawn").Pool(args.cores) as p:       
            p.starmap(multimaps,zip(args.month,base_folder_list))
    
    logging.info("Processing Done!")
    