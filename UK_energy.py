"""
Purpose - This script uses the previous fall SIC, SST, and Z70 indices to predict
          the UK winter electricity consumption using the leave-one-year out method.

Input - 2D array: Observed UK seasonal electricity consumption [Year, Month]
        1D time series (predictors): SEP Z70 index, SEP SST index, OCT SIC index [YR]
        
"""
#########################################
# Import Pakcages, may need to install some
#########################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from scipy import signal
from leave_one_out import lv1out

#Load in the predictors
sic = np.load('./SIC.npy')		#SIC INDEX
sst = np.load('./SST.npy')		#SST INDEX
Z70hPa = np.load('./Z70hPa.npy')		#Z70hPa INDEX

#fix range
sic = signal.detrend(sic[:34])
sst = signal.detrend(sst[:34])
Z70hPa = signal.detrend(Z70hPa[:34])

#Load UK electricity consumption data
elec = pd.read_csv('./seasonal_elec.txt',sep='\s+')
elec = (elec.DJF)
elec = signal.detrend(elec)

#Load NAO data
NAO = pd.read_csv('./NAO-DJF.txt',sep='\s+')
NAO = (NAO.DJF)
NAO = signal.detrend(NAO)

#Load UK temp data
temperature_code = ['min', 'mean', 'max']
vname = temperature_code[1]
temp = np.load('./UK_'+vname+'_temp.npy')
temp= signal.detrend(temp)

#set predictand
predictand_code=[elec, NAO, temp]
index = predictand_code[0]

#Initialize the predicted array
yhat = np.zeros(index.shape)

#Loop performs the leave-one-out cross-validation
#X and y are the training set, with a testing year (i) excluded.
#yhat is the time series of predicted blocking frequency
ct=0
X=np.column_stack((sic,sst,Z70hPa))
Y=index
yhat=lv1out(X,Y)

#Calculate the Pearson R between the observed and predicted time series of blocking
corr1,p1 = stats.pearsonr(index[:],yhat)
corr2,p2 = stats.pearsonr(NAO[:],temp)
corr3,p3 = stats.pearsonr(elec[:],NAO)
corr4,p4 = stats.pearsonr(elec[:],temp)
print(corr1,p1)
print(corr2,p2)
print(corr3,p3)
print(corr4,p4)

#normalization
norml=1
if norml==1:
  yhat=(yhat-np.mean(yhat))/np.std(yhat)
  index=(index-np.mean(index))/np.std(index)
  elec=(elec-np.mean(elec))/np.std(elec)
  temp=(temp-np.mean(temp))/np.std(temp)
  NAO=(NAO-np.mean(NAO))/np.std(NAO)

#year range
yr = np.arange(1979,2013,1)

#Plot the predicted and observed index
plt.figure(figsize=(10,4))
plt.plot(yr,index,'b')
plt.plot(yr, yhat*1,'r')
plt.title('UK DJF Electricity Consumption')# (r = '+str(round(corr,2))+')')
plt.xlabel('Year')
plt.ylabel('% Change')
plt.legend(['Observed', 'Predicted (r = '+str(round(corr1,2))+')'])#,'Perfect NAO (r = '+str(round(corr1,2))+')'],loc = 4,ncol=3)
plt.xlim([1979,2012])
plt.tight_layout()
#plt.ylim([15,25])
fmt='pdf'

#Plot NAO and temperature correlation
plt.figure(figsize=(10,4))
plt.plot(yr, NAO,'b')
plt.plot(yr, temp,'g')
plt.title('UK DJF NAO and Temperature')# (r = '+str(round(corr,2))+')')
plt.xlabel('Year')
plt.ylabel('% Change')
plt.legend(['NAO', 'Temperature (r = '+str(round(corr2,2))+')']) #(r = '+str(round(corr,2))+')'])#,'Perfect NAO (r = '+str(round(corr1,2))+')'],loc = 4,ncol=3)
plt.xlim([1979,2012])
plt.tight_layout()
#plt.ylim([15,25])
fmt='pdf'

#Plot electricity consumption correlation with NAO and temperature
plt.figure(figsize=(10,4))
plt.plot(yr,elec,'r')
plt.plot(yr, NAO,'b')
plt.plot(yr, temp,'g')
plt.title('UK DJF Electricity, NAO, and Temperature')# (r = '+str(round(corr,2))+')')
plt.xlabel('Year')
plt.ylabel('% Change')
plt.legend(['Electricity', 'NAO (r = '+str(round(corr3,2))+')', 'Temperature (r = '+str(round(corr4,2))+')']) #(r = '+str(round(corr,2))+')'])#,'Perfect NAO (r = '+str(round(corr1,2))+')'],loc = 4,ncol=3)
plt.xlim([1979,2012])
plt.tight_layout()
#plt.ylim([15,25])
fmt='pdf'

plt.show(block=False)