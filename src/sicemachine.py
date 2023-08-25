# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:47:18 2023

@author: rabni
"""

import pandas as pd
import os
import numpy as np
import geopandas as gpd
from sklearn import svm
import xarray as xr
import rasterio as rio
from rasterio.transform import Affine
from pyproj import CRS as CRSproj
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import glob
import json
import datetime
import geopandas
from matplotlib import path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def huber_w(w,d,sigma):
    # Huber weights, by variance for each band to allow higher prediction skill

    break_p = 1.345  
    ml_est = sum(w * d) / sum(w)
    eps = (d - ml_est) / sigma 
    
    w[eps > break_p] = break_p/eps[eps > break_p]
    w[abs(eps) <= break_p] = 1
    w[eps < -break_p] = -break_p/eps[eps < -break_p]
    
    return w 

def date_format(date_string):
       try:
           datetime.date.fromisoformat(date_string)
       except ValueError:
           return 'err'
       return "OK"

class ClassifierSICE():
    def __init__(self):
            self.src_folder = os.getcwd()
            self.base_folder = os.path.abspath('..')
            self.training_bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_08", "r_TOA_21"]
            
    def get_training_data(self,polar = None):
        
        '''Imports training from thredds server using OPeNDAP.
        The training dates,area and features are defined by the shapefiles in the /labels folder
        
        Parameters
        ----------
        self :
            
        polar:
            
          
        Returns
        -------
        dict
            dictionarty of training data
        '''
        
        shp_files = glob.glob(self.base_folder + os.sep + '**' + os.sep + '**.shp', recursive=True)
        training_dates = np.unique([d.split(os.sep)[-2].replace('-','_') for d in shp_files])
        dataset_ids = ['sice_500_' + d + '.nc' for d in training_dates]
        regions = ([d.split(os.sep)[-4] for d in shp_files])
        features = np.unique([d.split(os.sep)[-1][:-4] for d in shp_files])
   
        #ds_ref = xr.open_dataset(f'https://thredds.geus.dk/thredds/dodsC/SICE_Greenland_500m/{ref_DATASET}')
        training_data = {}
                
        print(f"Training Dates {training_dates}")
        
        for d,ref,re in zip(training_dates,dataset_ids,regions):     
            print(f"Getting Training Data for {d}")
            training_data[d] = {}
            ds = xr.open_dataset(f'https://thredds.geus.dk/thredds/dodsC/SICE_{re}_500m/{ref}')
            shp_files_date = [s for s in shp_files if d in s.replace('-','_')]
            
            for s,f in zip(shp_files_date,features):
                label_gdf = gpd.read_file(s).to_crs("epsg:3413")
                label_shps = json.loads(label_gdf.exterior.to_json())['features']
                
                x = np.array(ds[self.training_bands[0]].x)
                y = np.array(ds[self.training_bands[0]].y)
                xgrid,ygrid = np.meshgrid(x,y)
                mask = (np.ones_like(xgrid) * False).astype(bool)
                
                for ls in label_shps:
                    x_poly, y_poly = map(list, zip(*ls['geometry']['coordinates']))
                    p = path.Path(np.column_stack((x_poly,y_poly)))
                    idx_poly = p.contains_points(np.column_stack((xgrid.ravel(),ygrid.ravel())))
                    mask.ravel()[idx_poly] = True
                
                training_data[d][f] = {k:np.array(ds[k])[mask] for k in self.training_bands}
                #training_data[d][f] = {k:np.array(ds[k].where(mask))[mask] for k in self.training_bands}
                
                ds.close()
           
        return training_data
                
    def train_svm(self,training_data = None,c = 1,weights = True):
        
        if not training_data:
            training_data = self.get_training_data()
            
        t_days = list(training_data.keys())
        
        if len(t_days) > 2:
            testing_date = t_days[-1]
            print(f'Splitting Training Dates, Removing Date: {testing_date} for Testing')            
        else: 
            testing_date = None
        
        features = list(training_data[t_days[0]].keys())
        n_features = len(features)
        n_bands = len(self.training_bands)
            
        train_data = []
        train_label = []
        
       
        for f_int,f in enumerate(features):
            data = [np.array([training_data[d][f][b] for b in self.training_bands]).T for d in t_days if d != testing_date]
            data_stack = np.vstack([arr for arr in data])
            train_data.append(data_stack)
            
            label = (np.ones_like(data_stack[:,0]) * f_int).reshape(-1,1)            
            train_label.append(label)
        
        train_data = np.vstack([arr for arr in train_data])
        train_label = np.vstack([arr for arr in train_label]).ravel() 
        
        if testing_date is not None: 
            
            test_data = []
            test_label = []
            
            for f_int,f in enumerate(features):
                data = [np.array([training_data[testing_date][f][b] for b in self.training_bands]).T]
                data_stack = np.vstack([arr for arr in data])
                test_data.append(data_stack)
                
                label = (np.ones_like(data_stack[:,0]) * f_int).reshape(-1,1)
                test_label.append(label)
                
            test_data = np.vstack([arr for arr in test_data])
            test_label = np.vstack([arr for arr in test_label]).ravel() 
            
       
        if testing_date is None:
            train_data, test_data, train_label, test_label \
                 = train_test_split(train_data, train_label, test_size=0.10, random_state=42) 
            
        if weights:
            no_i = np.arange(50) #Number of iterations
            w_all = np.ones_like(train_data)
            
            for n in np.arange(n_features):
                for b in np.arange(n_bands):
                    w = w_all[:,b][train_label  == n]
                    d = train_data[:,b][train_label == n]
                    sigma = np.std(d)
                    for i in no_i:
                        w = huber_w(w,d,sigma)
                    w_all[:,b][train_label == n] = w 
                    
            w_samples = np.array([np.nanmean(w) for w in w_all])        
        else:
            w_samples = np.ones_like(train_label)
        
        
        model = svm.SVC(C = c, decision_function_shape="ovo",probability = True)
        model.fit(train_data, train_label,sample_weight=w_samples)
        
        data_split_svm = {}
        
        for i,f in enumerate(features): 
            data_split_svm[f] = {'train_data' : train_data[train_label==i],'train_label' : train_label[train_label==i],\
                                 'test_data' : test_data[test_label==i],'test_label' : test_label[test_label==i]}        
                
        data_split_svm['meta'] = {'testing_date' : testing_date}        
        
        return model,data_split_svm

    def test_svm(self,model=None, data_split = None):
        
        if data_split is None:
                model,data_split = self.train_svm()
            
        print('Test SVM for each Class \n')
        
        meta = data_split['meta']['testing_date']
        
        if meta is not None: 
            print(f"The model is being tested on an independent date: {meta}")
        
        classes = list(data_split.keys())
        
        for cl in classes:
            
            if cl !='meta':
                
                print(f'Test Results for Class {cl}:')
                data_test = data_split[cl]['test_data']
                label_test = data_split[cl]['test_label']
                
                #Predicting on Test Data:
                
                labels_pred = model.predict(data_test)
                cm = confusion_matrix(labels_pred, label_test)
                ac = np.round(accuracy_score(labels_pred,label_test),3)
                
                print(f"Accuracy of Predicting {cl}: {ac}")
                print(f"Confusion Matrix of {cl}: \n {cm} \n")
                
                
                for l in list(np.unique(labels_pred)):
                    
                    no_l_p = len(labels_pred[labels_pred==l])
                    label_name_prd = classes[int(l)] 
                    label_name_cor = cl
                    
                    print(f'Model Classified {label_name_prd} {no_l_p} times, the Correct Class was {label_name_cor} \n')
                
                
        return 
    
    def get_prediction_data(self,dates_to_predict):
        
        if dates_to_predict is None:
            print("Please Specify a Date to Predict")
            return None
        
        for d in dates_to_predict:
            msg = date_format(d)
            if msg == 'err':
                print(f"Incorrect date format for {d}, should be YYYY-MM-DD")
                return None
        
        dataset_ids = ['sice_500_' + d.replace('-','_') + '.nc' for d in dates_to_predict]
        prediction_data = {}
        for d,ref in zip(dates_to_predict,dataset_ids): 
            ds = xr.open_dataset(f'https://thredds.geus.dk/thredds/dodsC/SICE_Greenland_500m/{ref}')
            prediction_data[d] = {k:np.array(ds[k]) for k in self.training_bands}
            x = np.array(ds[self.training_bands[0]].x)
            y = np.array(ds[self.training_bands[0]].y)
            crs = CRSproj.from_string("+init=EPSG:3413")
            xgrid,ygrid = np.meshgrid(x,y)
            prediction_data[d]['meta'] = {'x' : xgrid, 'y' : ygrid,'crs' : crs}
            ds.close()
        
        return prediction_data
        
    def predict_svm(self,dates_to_predict,model=None,export=None):
        
        print('Loading Bands for Prediction Dates:')
        prediction_data = self.get_prediction_data(dates_to_predict)
        
        if prediction_data is None:
            return
        
        if model is None:
                print('Training Model:')
                model,data_split = self.train_svm()
       
        p_days = list(prediction_data.keys())
       
        if not os.path.exists(self.base_folder + os.sep + "output"):
            os.mkdir(self.base_folder + os.sep + "output")
       
        for d in p_days:
            print('Predicting Classes...:')
            data = np.array([prediction_data[d][b] for b in self.training_bands])
            xgrid = prediction_data[d]['meta']['x']
            ygrid = prediction_data[d]['meta']['y']
            crs = prediction_data[d]['meta']['crs']
            
            mask = ~np.isnan(data[0,:,:])
            data_masked = data[:,mask].T
            
            labels_predict = model.predict(data_masked)
            labels_grid = np.ones_like(xgrid) * np.nan
            labels_grid[mask] = labels_predict
            
            if not export or export in ["tiff","tif","geotiff"]:
                f_name = self.base_folder + os.sep + "output" + os.sep + d.replace('-','_') + '_SICE_surface_classes.tif'
                self._export_as_tiff(xgrid,ygrid,labels_grid,crs,f_name)   
        print('Done')
    
    def _export_as_tiff(self,x,y,z,crs,filename):
        
        "Input: xgrid,ygrid, data paramater, the data projection, export path, name of tif file"
       
        resx = (x[0,1] - x[0,0])
        resy = (y[1,0] - y[0,0])
        transform = Affine.translation((x.ravel()[0]),(y.ravel()[0])) * Affine.scale(resx, resy)
        
        if resx == 0:
            resx = (x[0,0] - x[1,0])
            resy = (y[0,0] - y[0,1])
            transform = Affine.translation((y.ravel()[0]),(x.ravel()[0])) * Affine.scale(resx, resy)
            
        with rio.open(
        filename,
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
    
    