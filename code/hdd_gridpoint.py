
"""
Purpose - This script uses the previous fall SIC, SST, and Z70 indices to predict
          the UK winter electricity consumption using the leave-one-year out method.

Input - 2D array: Observed UK seasonal electricity consumption [Year, Month]
        1D time series (predictors): SEP Z70 index, SEP SST index, OCT SIC index [YR]
"""
#########################################
# Import Packages, may need to install some
#########################################
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from scipy import signal
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter, detrend

import mat73
mat = mat73.loadmat('ERA5NAODJFdata1979to2020MVdn2tr3E3r2forSam.mat')
new_predictors = mat['PCx']
new_NAO = signal.detrend(np.array(mat['grdbNAO197912to202002DJF']))

#Leave-one-out cross-validation helper function to be called later
def lv1out(x0,y0):
  reg=LinearRegression()
  ys=np.zeros(len(y0))
  for i in range(len(y0)):
    x1=x0[i,:]          #testing data point
    x=np.delete(x0,i,axis=0) #construct the training dataset
    y=np.delete(y0,i,axis=0)
    reg.fit(x,y)        #train the model with x and y

    #reg.predict expects 2D array. array.reshape(1,-1) is used to reshape x1
    y1=reg.predict(x1.reshape(1, -1))  #predict y for the testing point
    ys[i]=y1[:]         #ys is the predicted time series of the predictand
  return ys


#updated
new_sic = signal.detrend(np.array(new_predictors[0]))
new_sst = signal.detrend(np.array(new_predictors[1]))
new_Z70hPa = signal.detrend(np.array(new_predictors[2]))


#Load UK temp data
index = signal.detrend(np.load('./hdd_gridpoint.npy'))

#Initialize the predicted array
yhat = np.zeros((40,))

#Loop performs the leave-one-out cross-validation
#X and y are the training set, with a testing year (i) excluded.
#yhat is the time series of predicted blocking frequency
ct=0
X=np.column_stack((new_sic[:-1],new_sst[:-1],new_Z70hPa[:-1]))
lat_range, lon_range = index.shape[0:2]
corr_hdd = np.full((128, 512), -1.0)
p_hdd = np.full((128, 512), -1.0)
for lat in range(lat_range):
    for lon in range(lon_range):
        Y=np.array(index[lat][lon])
        yhat=lv1out(X,Y)
        #Calculate the Pearson R between the observed and predicted time series of blocking
        corr,p = stats.pearsonr(savgol_filter(yhat, 5, 2),Y)
        if np.isnan(corr):
            corr,p=0,-1
        
        corr_hdd[lat][lon] = corr
        p_hdd[lat][lon] = p
np.save('p_hdd', p_hdd)
np.save('corr_hdd', corr_hdd)