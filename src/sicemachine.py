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
import matplotlib.pyplot as plt
import warnings
import colorsys
import random
from multiprocessing import set_start_method,get_context
warnings.filterwarnings("ignore", category=FutureWarning)


def compute_weighted_mean(w,d):
    return sum(w * d) / sum(w)

def tukey_w(w,d,sigma):
    # Huber weights, by variance for each band to allow higher prediction skill

    break_p = 4.685
    ml_est = sum(w * d) / sum(w)
    eps = (d - ml_est) / sigma 
    
    w[abs(eps) <= break_p] = (1-(eps[abs(eps) <= break_p]/break_p)**2)**2
    w[abs(eps) > break_p] = 0
    
    return w 


def huber_w(w,d,sigma):
    # Huber weights, by variance for each band to allow higher prediction skill

    break_p = 1.345  
    ml_est = sum(w * d) / sum(w)
    eps = (d - ml_est) / sigma 
    
    w[eps > break_p] = break_p/eps[eps > break_p]
    w[abs(eps) <= break_p] = 1
    w[eps < -break_p] = -break_p/eps[eps < -break_p]
    
    return w 

def generate_diverging_colors_hex(num_colors, center_color='#808080'):
    colors = []
    for i in range(num_colors):
        if i < num_colors // 2:
            hue = random.uniform(0.7, 1.0)  # Warm colors
        else:
            hue = random.uniform(0.0, 0.3)  # Cool colors
        saturation = random.uniform(0.5, 1.0)
        value = random.uniform(0.5, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append('#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    colors.insert(num_colors // 2, center_color)
    return colors

def date_format(date_string):
       try:
           datetime.date.fromisoformat(date_string)
       except ValueError:
           return 'err'
       return "OK"
   
def freedman_bins(df): 
    quartiles = df.quantile([0.25, 0.75])
    iqr = quartiles.loc[0.75] - quartiles.loc[0.25]
    n = len(df)
    h = 2 * iqr * n**(-1/3)
    bins = (df.max() - df.min())/h 
    if np.isnan(np.array(bins)):
        bins = 2 
    return int(np.ceil(bins))
    
class ClassifierSICE():
    def __init__(self):
            self.src_folder = os.getcwd()
            self.base_folder = os.path.abspath('..')
            self.training_bands = ["r_TOA_02", "r_TOA_04", "r_TOA_06", "r_TOA_08", "r_TOA_21"]
            self.classes = ['dark_ice','bright_ice','red_ice','flooded_snow','melted_snow','dry_snow']
            
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
        features =  np.unique([f for f in features if len(f.split('_'))==2]) # checking if thera are more than one shp file per class
        
        
        #ds_ref = xr.open_dataset(f'https://thredds.geus.dk/thredds/dodsC/SICE_Greenland_500m/{ref_DATASET}')
        training_data = {}
                
        print(f"Training Dates {training_dates}")
        
        for d,ref,re in zip(training_dates,dataset_ids,regions):     
            print(f"Getting Training Data for {d}")
            training_data[d] = {}
            ds = xr.open_dataset(f'https://thredds.geus.dk/thredds/dodsC/SICEvEDC_500m/Greenland/{ref}')
            shp_files_date = [s for s in shp_files if d in s.replace('-','_')]
            
            for f in self.classes:
                
                shp = [s for s in shp_files_date if f.split('_')[0] in s]
                
                label_shps = []
                
                for s in shp: 
                    label_gdf = gpd.read_file(s).to_crs("epsg:3413")
                    shps = json.loads(label_gdf.exterior.to_json())['features']
                    label_shps.append(shps)
                   
                label_shps = [item for sublist in label_shps for item in sublist]    
                
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
    
    def plot_training_data(self,training_data=None):
        
        if not training_data:
            training_data = self.get_training_data()
        
        t_days = list(training_data.keys())
        features = list(training_data[t_days[0]].keys())
        alpha_value = 0.4
        #center_color = (0.5, 0.5, 0.5)  # Adjust the center color as needed
        color_multi = generate_diverging_colors_hex(len(t_days))
        
        pdf_all_no_w = {k:[] for k in self.training_bands}
        pdf_all_t_w = {k:[] for k in self.training_bands}
        for f_int,f in enumerate(features):
            data_all = []
            for i,d in enumerate(t_days):
                data = np.array([training_data[d][f][b] for b in self.training_bands]).T
                data[data>1] = np.nan
                data[data<0.001] = np.nan
                dates = np.ones_like(data[:,0]) * i
                data_w_dates = np.column_stack((data, dates))
                data_all.append(data_w_dates)
            
            data_all = np.vstack([arr for arr in data_all])
            df_col = self.training_bands + ['date']
            df_data = pd.DataFrame(data_all,columns=[df_col])
            
            column_names = [d[0] for d in df_data.columns]
            num_rows = -(-len(column_names) // 2)
            fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(22, 12), gridspec_kw={'hspace': 0.5})
            axes = axes.flatten()
            
            for i,col in enumerate(column_names):
                if 'date' not in col:
                    ax = axes[i]
                   
                    x = np.array(df_data[col]).ravel()
                    mu = np.nanmean(x)
                    sigma = np.nanstd(x)
                    std_mask = (abs(mu-x)<100*sigma)
                    x = np.sort(x[std_mask])
                    x = x[~np.isnan(x)]
                    
                    mu = np.nanmean(x)
                    sigma = np.nanstd(x)
                    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                         np.exp(-0.5 * ((x - mu) / sigma)**2))
                    
                    
                    col_class = np.ones_like(y) * f_int
                    pdf_stack = np.array([x,y,col_class]).T
                    pdf_all_no_w[col].append(pdf_stack)
                    
                    w = np.ones_like(x)
                    no_i = np.arange(50)
                    for ite in no_i:
                        w = tukey_w(w, x, sigma)
                        
                    mu = compute_weighted_mean(w, x)
                    
                    x_weigted = x[w>0]
                    
                    y_weighted = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                         np.exp(-0.5 * (1 / sigma * (x_weigted - mu))**2)) 
                    
                    #y_weighted = y_weighted / np.nanmax(y_weighted)
                    
                    #x_range = np.linspace(np.nanmin(x),np.nanmax(x),num=20)
                    
                    
                    col_class = np.ones_like(y_weighted) * f_int
                    pdf_stack = np.array([x_weigted,y_weighted,col_class]).T
                    pdf_all_t_w[col].append(pdf_stack)
                    
                    ax.plot(x,y, color ='red',linewidth=6,\
                            label='Gaussian pdf of combined training data',zorder=1)
                    ax.plot(x,y, color ='black',linewidth=7,\
                            zorder=0)
                        
                    for date_id in np.unique(df_data['date']):
                      
                        mask = df_data['date']==date_id
                        date_df = df_data[col][mask.squeeze()]
                        date_data_std = np.nanstd(date_df)
                        date_name = t_days[int(date_id)]
                        #print(f'Band {col} sigma of {f} at {date_name}: {date_data_std}')
                        bins = freedman_bins(date_df)
                        ax.hist(date_df, bins=bins, alpha=1, density=True,zorder=-1,\
                                          edgecolor='black', linewidth=1.2,histtype='step')
                        ax.hist(date_df, bins=bins, alpha=alpha_value, density=True,\
                                label=f'{date_name}', color=color_multi[int(date_id)],zorder=-2)
                        
                        
                    ax.set_title(f'Band: {col}',fontsize=20)
                    ax.set_ylabel('Density Count',fontsize=20)
                    ax.set_xlabel('Reflectance',fontsize=20)
                    ax.tick_params(labelsize=16)
                    ax.legend()
                    
                    #### Add Combined Dist. ####
                    
                  
            #if len(column_names) % 2 == 1:
            #    fig.delaxes(axes[-1])  
            fig.delaxes(axes[-1])
            plt.suptitle(f'Training Data Band Distributions of Class {f}', fontsize=30)  # Add a single title
            #plt.tight_layout()  # Adjust layout to make space for the title
            plt.show()
            
            
        num_rows = -(-len(self.training_bands) // 2)
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(22, 12), gridspec_kw={'hspace': 0.5})
        axes = axes.flatten()
        color_multi = generate_diverging_colors_hex(len(features))
        for i,toa in enumerate(self.training_bands):
            data = pdf_all_t_w[toa]
            ax = axes[i]
            for j,cl in enumerate(data):
                
                x = cl[:,0]
                y = cl[:,1]
                class_int = int(np.unique(cl[:,2]))
                class_name = features[class_int]
                
                ax.plot(x,y, color = color_multi[class_int],linewidth=6,\
                        label=f'{class_name}',zorder=1)
                ax.plot(x,y, color ='black',linewidth=7,\
                        zorder=0)
            
            ax.set_title(f'Band: {toa}',fontsize=20)
            ax.set_ylabel('Density',fontsize=20)
            ax.set_xlabel('Reflectance',fontsize=20)
            ax.tick_params(labelsize=16)
            ax.legend()
            
        if len(column_names) % 2 == 1:
            fig.delaxes(axes[-1])  
        fig.delaxes(axes[-1])      
        plt.suptitle('Gaussian PDF of all classes - With Tukey BiWeights', fontsize=30)  # Add a single title
        #plt.tight_layout()  # Adjust layout to make space for the title
        plt.show()
        
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(22, 12), gridspec_kw={'hspace': 0.5})
        axes = axes.flatten()
        color_multi = generate_diverging_colors_hex(len(features))
        for i,toa in enumerate(self.training_bands):
            data = pdf_all_no_w[toa]
            ax = axes[i]
            for j,cl in enumerate(data):
                
                x = cl[:,0]
                y = cl[:,1]
                class_int = int(np.unique(cl[:,2]))
                class_name = features[class_int]
                
                ax.plot(x,y, color = color_multi[class_int],linewidth=6,\
                        label=f'{class_name}',zorder=1)
                ax.plot(x,y, color ='black',linewidth=7,\
                        zorder=0)
            
            ax.set_title(f'Band: {toa}',fontsize=20)
            ax.set_ylabel('Density',fontsize=20)
            ax.set_xlabel('Reflectance',fontsize=20)
            ax.tick_params(labelsize=16)
            ax.legend()
            
        if len(column_names) % 2 == 1:
            fig.delaxes(axes[-1])  
        fig.delaxes(axes[-1])      
        plt.suptitle('Gaussian PDF of all classes - No ML Estimation', fontsize=30)  # Add a single title
        #plt.tight_layout()  # Adjust layout to make space for the title
        plt.show()
                    
        return
                
    def train_svm(self,training_data=None,c=1,weights=True):
        
        if not training_data:
            training_data = self.get_training_data()
            
        t_days = list(training_data.keys())
        
        if len(t_days) > 2:
            testing_date = t_days[1]
                     
        else:
            testing_date = None
        
        features = list(training_data[t_days[0]].keys())
        n_features = len(features)
        n_bands = len(self.training_bands)
            
        train_data = []
        train_label = []
        
        print('Formatting Training Data')
        
        for f_int,f in enumerate(features):
            data = [np.array([training_data[d][f][b] for b in self.training_bands]).T for d in t_days if d != testing_date]
            data_stack = np.vstack([arr for arr in data])
            
            data_stack = np.array([dd for dd in data_stack[:] if not np.isnan(dd).any()])
            data_stack = np.array([dd for dd in data_stack[:] if len(dd[dd==0])==0])
              
            train_data.append(data_stack)
            
            label = (np.ones_like(data_stack[:,0]) * f_int).reshape(-1,1)            
            train_label.append(label)
        
        train_data = np.vstack([arr for arr in train_data])
        train_label = np.vstack([arr for arr in train_label]).ravel() 
        
        if testing_date is not None: 
            
            print(f'Splitting Training Dates, Removing Date: {testing_date} for Testing')  
            test_data = []
            test_label = []
            
            for f_int,f in enumerate(features):
                data = [np.array([training_data[testing_date][f][b] for b in self.training_bands]).T]
                data_stack = np.vstack([arr for arr in data])
                #print(np.shape(data_stack))
                data_stack = np.array([dd for dd in data_stack[:] if not np.isnan(dd).any()])
                
                data_stack = np.array([dd for dd in data_stack[:] if len(dd[dd==0])==0])
                
                #print(np.shape(data_stack))
                test_data.append(data_stack)
                
                label = (np.ones_like(data_stack[:,0]) * f_int).reshape(-1,1)
                test_label.append(label)
                
            test_data = np.vstack([arr for arr in test_data])
            test_label = np.vstack([arr for arr in test_label]).ravel() 
            
            
       
        if testing_date is None:
            train_data, test_data, train_label, test_label \
                 = train_test_split(train_data, train_label, test_size=0.10, random_state=42) 
            
        
        
        if weights:
            print('Computing Weights')
            no_i = np.arange(50) #Number of iterations
            w_all = np.ones_like(train_data)
            
            for n in np.arange(n_features):
                for b in np.arange(n_bands):
                    w = w_all[:,b][train_label  == n]
                    d = train_data[:,b][train_label == n]
                    sigma = np.std(d)
                    for i in no_i:
                        w = tukey_w(w,d,sigma)
                    w_all[:,b][train_label == n] = w 
                    
            w_samples = np.array([np.nanmean(w) for w in w_all])        
        else:
            w_samples = np.ones_like(train_label)
        
        print('Training Model....')
        model = svm.SVC(C = c, decision_function_shape="ovo")
        model.fit(train_data, train_label,sample_weight=w_samples)
        print('Done')
        
        data_split_svm = {}
        print('Splitting dataset')
        for i,f in enumerate(features): 
            data_split_svm[f] = {'train_data' : train_data[train_label==i],'train_label' : train_label[train_label==i],\
                                 'test_data' : test_data[test_label==i],'test_label' : test_label[test_label==i]}        
                
        data_split_svm['meta'] = {'testing_date' : testing_date}        
        
        return model,data_split_svm

    def test_svm(self,model=None, data_split=None, export_error=None):
        
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
                data_train = data_split[cl]['train_data']
                
                #Predicting on Test Data:
                labels_pred = model.predict(data_test)
                cm = confusion_matrix(labels_pred, label_test)
                ac = np.round(accuracy_score(labels_pred,label_test),3)
                
                print(f"Plotting Band Distribution in class {cl}")
                alpha_value = 0.35
                num_bins = 10
                den = False
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
                color_multi = ['#FFA500', '#FF8C00', '#FFD700', '#FF6347', '#FFA07A', '#FF4500']
            
                #if len([l for l in labels_pred if l not in label_test]) > 0:    
                print(f"Plotting Band Distribution of Predicted Label(s) in Class {cl}: \n")
                l_mask = np.array([True if l not in label_test else False for l in labels_pred])
                bad_labels = data_test[l_mask,:]
                bad_labels_cl = labels_pred[l_mask]
                good_labels = data_test[~l_mask,:]
                
                bad_labels_val = np.column_stack((bad_labels, bad_labels_cl))
                bad_labes_col = self.training_bands + ['label']
                bad_labels = pd.DataFrame(bad_labels_val,columns=[bad_labes_col])
                good_labels = pd.DataFrame(good_labels,columns=[self.training_bands])
                train_labels = pd.DataFrame(data_train,columns=[self.training_bands])
                
                # Get column names from the DataFrames
                column_names = good_labels.columns
                num_rows = -(-len(column_names) // 2) 
                
                # Creating overlapping histogram plots for all columns from both DataFrames
                fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 12))
                axes = axes.flatten()
                for i, col in enumerate(column_names):
                    ax = axes[i]
                    
                    bins = freedman_bins(good_labels[col])
                    
                    good_labels[col].hist(ax=ax, bins=bins, alpha=alpha_value,\
                                          label='Correct Labelled Test Data', color=colors[0],\
                                          histtype='barstacked', density=den)
                        
                    good_labels[col].hist(ax=ax, bins=bins, alpha=1, density=den,\
                                          edgecolor='black', linewidth=1.2,histtype='step')
                    """
                    bad_labels[col].hist(ax=ax, bins=num_bins, alpha=alpha_value,\
                                         label='Bad Labelled Test Data',  color=colors[1],\
                                         histtype='barstacked', density=True)
                    """
                    
                    for class_id in np.unique(bad_labels['label']):
                        
                        if len(bad_labels)/len(good_labels) < 0.01: 
                            pass
                            #class_df = bad_labels[col]
                        else:
                            mask = bad_labels['label']==class_id
                            class_df = bad_labels[col][mask.squeeze()]
                            class_name = classes[int(class_id)]
                            bins = freedman_bins(class_df)
                         
                            ax.hist(class_df, bins=bins, alpha=alpha_value, density=den,\
                                    label=f'Test Data Labelled Wrongly as {class_name}', color=color_multi[int(class_id)])
                            ax.hist(class_df, bins=bins, alpha=1, density=den,\
                                                 edgecolor='black', linewidth=1.2,histtype='step')
                    
                    bins = freedman_bins(train_labels[col])
              
                    train_labels[col].hist(ax=ax, bins=bins, alpha=alpha_value,\
                                         label='Traning Data', color=colors[2],\
                                         histtype='barstacked', density=den)
                    train_labels[col].hist(ax=ax, bins=bins, alpha=1, density=den,\
                                         edgecolor='black', linewidth=1.2,histtype='step')
                
                    #bad_labels[col].hist(ax=ax, bins=num_bins, alpha=1, density=True,\
                    #                     edgecolor='black', linewidth=1.2,histtype='step')
                   
                    
                    ax.set_title(f'Band: {col}',fontsize=20)
                    ax.legend()
                    
                if len(column_names) % 2 == 1:
                    fig.delaxes(axes[-1])    
                    
                plt.suptitle(f'Band Distributions of Predicted Class {cl}', fontsize=20)  # Add a single title
                plt.tight_layout()  # Adjust layout to make space for the title
                plt.show()
                                        
                        
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
                print(f"Incorrect date format for {d}, should be [YYYY-MM-DD] in a list!")
                return None
        
        dataset_ids = ['sice_500_' + d.replace('-','_') + '.nc' for d in dates_to_predict]
        prediction_data = {}
        for d,ref in zip(dates_to_predict,dataset_ids): 
            print(f'Loading {d} ......')
            try:
                ds = xr.open_dataset(f'https://thredds.geus.dk/thredds/dodsC/SICEvEDC_500m/Greenland/{ref}')
                prediction_data[d] = {k:np.array(ds[k]) for k in self.training_bands}
                x = np.array(ds[self.training_bands[0]].x)
                y = np.array(ds[self.training_bands[0]].y)
                crs = CRSproj.from_string("+init=EPSG:3413")
                xgrid,ygrid = np.meshgrid(x,y)
                prediction_data[d]['meta'] = {'x' : xgrid, 'y' : ygrid,'crs' : crs}
                ds.close()
            except: 
                print(f'{d} does not exist on the thredds server')
        return prediction_data
        
    def predict_svm(self,dates_to_predict,cor=10,model=None,export=None):
        
        self.model = model
        
        if self.model is None:
                print('Training Model:')
                self.model,data_split = self.train_svm()
       
        print('Loading Bands for Prediction Dates:')
        self.prediction_data = self.get_prediction_data(dates_to_predict)
        
        if self.prediction_data is None:
            return
       
        p_days = list(self.prediction_data.keys())
       
        if not os.path.exists(self.base_folder + os.sep + "output"):
            os.mkdir(self.base_folder + os.sep + "output")
        
        for d in p_days:
            self._predict_for_date(d)
        
        # print('initializing multiprocessing:')
        # set_start_method("spawn")
        
        
        # with get_context("spawn").Pool(cor) as p:     
        #         p.starmap(self._predict_for_date,zip(p_days))
        #         p.close()
        #         p.join()
        #         print("Done with multiprocessing")
       
    
    def _predict_for_date(self,date):

        print(f'Predicting Classes for {date}.....')
        data = np.array([self.prediction_data[date][b] for b in self.training_bands])
        xgrid = self.prediction_data[date]['meta']['x']
        ygrid = self.prediction_data[date]['meta']['y']
        crs = self.prediction_data[date]['meta']['crs']
        
        mask = ~np.isnan(data[0,:,:])
        data_masked = data[:,mask].T
        
        labels_predict = self.model.predict(data_masked)
        #labels_prob = model.predict_proba(data_masked)
        labels_grid = np.ones_like(xgrid) * np.nan
        labels_grid[mask] = labels_predict
        print(f'Done for {date}')
        
        f_name = self.base_folder + os.sep + "output" + os.sep + date.replace('-','_') + '_SICE_surface_classes.tif'
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
    
    