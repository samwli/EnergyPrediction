#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 03:05:11 2020

@author: samli
"""


#Read 3D array (x,y,t)
########
import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from scipy import signal
from leave_one_out import lv1out

#input data
filein=['./UK_T2m.txt',
    './UK_TDD2m.txt',
    './UK_ws10m.txt',
	'./NAO-DJF.txt',
    './seasonal_elec.txt']
titles=['(a) Temp','(b) TempDD','(c) Wind Speed', '(d) NAO', '(e) Electricity']

#Load in the predictors
sic = np.load('./SIC.npy')		#SIC INDEX
sst = np.load('./SST.npy')		#SST INDEX
Z70hPa = np.load('./Z70hPa.npy')		#Z70hPa INDEX

#fix range
sic = signal.detrend(sic[:34])
sst = signal.detrend(sst[:34])
Z70hPa = signal.detrend(Z70hPa[:34])



#############################################
#Initialize for plotting
plt.close("all")
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10,10))

yr = np.arange(1979,2013,1)
norml=1
for i in range(len(filein)):
    
    #Read 3D data
    axx = axes[i]
    dat = pd.read_csv(filein[i],sep='\s+')
    dat = dat.val
    dat = signal.detrend(dat)
    
    #Initialize the predicted array
    yhat = np.zeros(dat.shape)
    
    #Loop performs the leave-one-out cross-validation
    #X and y are the training set, with a testing year (i) excluded.
    #yhat is the time series of predicted blocking frequency
    ct=0
    X=np.column_stack((sic,sst,Z70hPa))
    Y=dat
    yhat=lv1out(X,Y)
    
    #normalization
    if norml==1:
      dat=(dat-np.mean(dat))/np.std(dat)
      yhat=(yhat-np.mean(yhat))/np.std(yhat)
      
    
    #Plot the predicted and observed index
    
    axx.plot(yr,dat,'b')
    axx.plot(yr,yhat,'r')
    #plt.xlabel('Year')
    #plt.ylabel('% Change')
    #plt.legend(['Observed', 'Predicted (r = '+str(round(corr1,2))+')'])#,'Perfect NAO (r = '+str(round(corr1,2))+')'],loc = 4,ncol=3)
    #plt.xlim([1979,2012])
    #plt.tight_layout()
    #plt.ylim([15,25])
    axx.set_title(titles[i],fontsize=13,loc='left')
    fmt='pdf'
    
    print(stats.pearsonr(dat,yhat))
   
#add a super title
plt.suptitle('My Super Title',fontsize=16)
#adjust the panels
plt.subplots_adjust(top=0.92,bottom=0.08,left=0.05,right=.95,hspace=0.35,wspace=0.10)
plt.show(block=False)
