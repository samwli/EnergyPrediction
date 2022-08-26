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
	'./NAO-DJF.txt']
titles=['(a) Temp','(b) TempDD','(c) Wind Speed', '(d) NAO']


elec = pd.read_csv('./seasonal_elec.txt',sep='\s+')
elec = elec.val
elec = signal.detrend(elec)

#############################################
#Initialize for plotting
plt.close("all")
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10,10))

yr = np.arange(1979,2013,1)
norml=1
if norml==1:
     elec=(elec-np.mean(elec))/np.std(elec)
for i in range(len(filein)):
    
    #Read 3D data
    axx = axes[i]
    dat = pd.read_csv(filein[i],sep='\s+')
    dat = dat.val
    dat = signal.detrend(dat)
    
    #normalization
    if norml==1:
      dat=(dat-np.mean(dat))/np.std(dat)
    
    #Plot the predicted and observed index
    
    axx.plot(yr,elec,'b')
    axx.plot(yr,dat,'r')
    #plt.xlabel('Year')
    #plt.ylabel('% Change')
    #plt.legend(['Observed', 'Predicted (r = '+str(round(corr1,2))+')'])#,'Perfect NAO (r = '+str(round(corr1,2))+')'],loc = 4,ncol=3)
    #plt.xlim([1979,2012])
    #plt.tight_layout()
    #plt.ylim([15,25])
    axx.set_title(titles[i],fontsize=13,loc='left')
    fmt='pdf'
    
    print(stats.pearsonr(dat,elec))
   
#add a super title
plt.suptitle('My Super Title',fontsize=16)
#adjust the panels
plt.subplots_adjust(top=0.92,bottom=0.08,left=0.05,right=.95,hspace=0.35,wspace=0.10)
plt.show(block=False)
