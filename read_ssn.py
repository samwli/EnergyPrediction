#Read 3D array (x,y,t)
########
import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as pyplot
from scipy import stats
import pandas as pd
from scipy import signal
from leave_one_out import lv1out
import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

from mpl_toolkits.basemap import Basemap

#input data, iyr1-iyr2 is the data range
iyr1=1979
iyr2=2012
nyr=iyr2-iyr1+1

filein='./erai.T2m.DJF.mean.nc'

#latlon of the plotting domain for UK
lat1=50;lat2=59
lon1=-11;lon2=2

#Read data
ncFid = Dataset(filein,mode='r')
vname=list(ncFid.variables.keys())[3]
lon0=ncFid.variables[ "lon" ][:]
lat0=ncFid.variables[ "lat" ][:]

time=ncFid.variables[ "time" ][:]
units=ncFid.variables[ "time" ].units

#find the latlon indices for the domain of interest
j1=np.argmin(abs(lat0-lat1))
j2=np.argmin(abs(lat0-lat2))
i1=np.argmin(abs(lon0-lon1))
i2=np.argmin(abs(lon0-lon2))
nlat=j2-j1
nlon=i2-i1

#extract time info
nt=len(time)
datevar=num2date(time,units)
yrs=np.zeros(nt)
for i in range(nt):
  yrs[i]=datevar[i].year

D1=np.where((yrs==iyr1))[0][0]
D2=np.where((yrs==iyr2))[0][0]+1
dat0=ncFid.variables[vname][D1:D2,j1:j2,i1:i2]

ncFid.close()


#Load in the predictors
sic = np.load('./SIC.npy')		#SIC INDEX
sst = np.load('./SST.npy')		#SST INDEX
Z70hPa = np.load('./Z70hPa.npy')		#Z70hPa INDEX

#fix range
sic = signal.detrend(sic[:34])
sst = signal.detrend(sst[:34])
Z70hPa = signal.detrend(Z70hPa[:34])

corr=np.zeros(dat0[0,:,:].shape)
p=np.zeros(dat0[0,:,:].shape)


for j in range(nlat):
   for i in range(nlon):
      index=dat0[:,j,i]
      #Initialize the predicted array
      yhat = np.zeros(index.shape)
      #Loop performs the leave-one-out cross-validation
      #X and y are the training set, with a testing year (i) excluded.
      #yhat is the time series of predicted blocking frequency
      ct=0
      X=np.column_stack((sic,sst,Z70hPa))
      Y=index
      yhat=lv1out(X,Y)
      a,b=stats.pearsonr(index,yhat)
      corr[j,i]=a
      p[j,i]=b


#shift the data from 0-360E to -180 to 180E so that we can plot across the prime meridian
ic=np.argmin(abs(lon0-180))
tmp2d=np.zeros(corr.shape)
tmp2d[:,:ic]=corr[:,nlon-ic:]
tmp2d[:,ic:]=datm[:,:nlon-ic]
datm=tmp2d
tmp1d=np.zeros(lon0.shape)
tmp1d[:ic]=lon0[nlon-ic:]-360
tmp1d[ic:]=lon0[:nlon-ic]
lon0=tmp1d

#Plot the field variable
fig=pyplot.figure(
cmap = pyplot.get_cmap('RdBu_r')

m = Basemap(projection="cyl",llcrnrlat=lat1, llcrnrlon=lon1, 
            urcrnrlat=lat2,urcrnrlon=lon2)
m.drawcoastlines (linewidth=0.5, color="black")
m.drawmeridians(np.arange(lon1,lon2+1,2), labels = [0 ,0 ,0 ,1])
m.drawparallels(np.arange(lat1,lat2+1,2),  labels = [1 ,0 ,0 ,0])

x,y = np.meshgrid()
X,Y = m(x,y)
clevs = np.arange(0,1,0.01)	#contour intervals
im = m.contourf(x,y,corr,clevs,cmap=cmap,extend='both')
pyplot.title(vname+' (DJF)',fontsize=13,loc='left')
fig.colorbar(im,orientation='horizontal')

fmt="pdf"
#pyplot.savefig('test-plot.'+fmt,format=fmt,bbox_inches='tight')
pyplot.show(block=False)
