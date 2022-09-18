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


#Load in the predictors
sic = np.load('./SIC.npy')		#SIC INDEX
sst = np.load('./SST.npy')		#SST INDEX
Z70hPa = np.load('./Z70hPa.npy')		#Z70hPa INDEX

#fix range
sic = signal.detrend(sic[:34])
sst = signal.detrend(sst[:34])
Z70hPa = signal.detrend(Z70hPa[:34])

#updated
new_sic = signal.detrend(np.array(new_predictors[0]))
new_sst = signal.detrend(np.array(new_predictors[1]))
new_Z70hPa = signal.detrend(np.array(new_predictors[2]))

#Load UK electricity consumption data
elec = pd.read_csv('./seasonal_elec.txt',sep='\s+')
elec = (elec.val)
elec = signal.detrend(elec)

#Load NAO data
NAO = pd.read_csv('./NAO-DJF.txt',sep='\s+')
NAO = (NAO.val)
NAO = signal.detrend(NAO)

#Load UK temp data
temperature_code = ['min', 'mean', 'max']
vname = temperature_code[1]
temp = np.load('./UK_'+vname+'_temp.npy')
temp= signal.detrend(temp)

#Load UK wind speed data
ws = pd.read_csv('./UK_ws10m.txt',sep='\s+')
ws = (ws.val)
ws = signal.detrend(ws)

#Load UK tdd data
tdd = pd.read_csv('./UK_TDD2m.txt',sep='\s+')
tdd = (tdd.val)
tdd = signal.detrend(tdd)

#Load preprocessed HDD data
HDD = signal.detrend(np.load('hdd.npy'))
index=HDD[:]

#set predictand
predictand_code=[elec, NAO, temp]
index = predictand_code[2]

#Initialize the predicted array
yhat = np.zeros(index.shape)

#Loop performs the leave-one-out cross-validation
#X and y are the training set, with a testing year (i) excluded.
#yhat is the time series of predicted blocking frequency
ct=0
X=np.column_stack((new_sic[:-1],new_sst[:-1],new_Z70hPa[:-1]))
Y=index
yhat=lv1out(X,Y)

index = index
yhat = elec

#Calculate the Pearson R between the observed and predicted time series of blocking
corr1,p1 = stats.pearsonr(yhat,index)
print(corr1, p1)
corr1,p1 = stats.pearsonr(savgol_filter(yhat, 5, 2),index)
#normalization
norml=1
if norml==1:
  yhat=(yhat-np.mean(yhat))/np.std(yhat)
  index=(index-np.mean(index))/np.std(index)
  #new_index=(new_index-np.mean(new_index))/np.std(new_index)
  elec=(elec-np.mean(elec))/np.std(elec)
  temp=(temp-np.mean(temp))/np.std(temp)
  NAO=(NAO-np.mean(NAO))/np.std(NAO)
  new_NAO=(new_NAO-np.mean(new_NAO))/np.std(new_NAO)

#year range
yr = np.arange(1979,1979+yhat[:].shape[0],1)

#Plot the predicted and observed index
plt.figure(figsize=(10,4))
plt.plot(yr,index[:],'red')
plt.plot(yr, savgol_filter(yhat, 5, 2)[:]*1,'red', linestyle='dashed')
#plt.plot(yr, elec,'k'1
plt.title('UK DJF HDD', fontsize = 18, loc = 'left')# (r = '+str(round(corr,2))+')')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Normalized Anomolies', fontsize=15)
plt.legend(['Observed HDD', 'Predicted HDD (r = '+str(round(corr1,2))+')', 'Elec. Consumption'])#,'Perfect NAO (r = '+str(round(corr1,2))+')'],loc = 4,ncol=3)
plt.xlim([1979,1979+yhat[:].shape[0]-1])
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.tight_layout()
#plt.ylim([15,25])
fmt='pdf'

plt.savefig('NAO.pdf')

plt.show(block=False)
