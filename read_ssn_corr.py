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

filein='./erai.ws10m.DJF.mean.nc'
filein0='./erai.LSMsfc.nc'

#latlon of the plotting domain for UK
lat1=50;lat2=59
lon1=-11;lon2=2

#Read data
ncFid = Dataset(filein,mode='r')
vname=list(ncFid.variables.keys())[3]
#dat0=ncFid.variables[vname][:]
lon0=ncFid.variables[ "lon" ][:]
lat0=ncFid.variables[ "lat" ][:]

nlon=len(lon0)
nlat=len(lat0)

time=ncFid.variables[ "time" ][:]
units=ncFid.variables[ "time" ].units

#find the latlon indices for the domain of interest
j1=np.argmin(abs(lat0-lat1))
j2=np.argmin(abs(lat0-lat2))
i1=np.argmin(abs(lon0-lon1))
i2=np.argmin(abs(lon0-lon2))

#extract time info
nt=len(time)
datevar=num2date(time,units)
yrs=np.zeros(nt)
for i in range(nt):
  yrs[i]=datevar[i].year

#create a cosine weighting function in the shape of (nday,j2-j1,i2-i1)
lons,lats=np.meshgrid(lon0[i1:i2],lat0[j1:j2])
cs=np.cos(np.deg2rad(lats))
cswt=np.reshape(np.tile(cs.flatten(),nyr),(nyr,j2-j1,i2-i1))

D1=np.where((yrs==iyr1))[0][0]
D2=np.where((yrs==iyr2))[0][0]+1
dat0=ncFid.variables[vname][D1:D2,j1:j2,i1:i2]

ncFid.close()

#Read land-sea mask
ncFid0 = Dataset(filein0,mode='r')
vname0=list(ncFid0.variables.keys())[2]
lmask=ncFid0.variables[vname0][j1:j2, i1:i2]
ncFid0.close()

#mask out data over ocean; lmask=0 over ocean and lmask=1 over land
lmask3D=np.broadcast_to(lmask,dat0.shape)
dat0[np.where(lmask3D<0.1)]=np.NaN

datm=np.nanmean(dat0,axis=(1,2))


#Load UK electricity consumption data
elec = pd.read_csv('./seasonal_elec.txt',sep='\s+')
elec = (elec.DJF)
elec = signal.detrend(elec)

#Load NAO data
NAO = pd.read_csv('./NAO-DJF.txt',sep='\s+')
NAO = (NAO.DJF)
NAO = signal.detrend(NAO)

corr,p=stats.pearsonr(datm,elec)
print(corr,p)

corr1,p1=stats.pearsonr(datm,NAO)
print(corr1,p1)


"""
#Load in the predictors
sic = np.load('./SIC.npy')		#SIC INDEX
sst = np.load('./SST.npy')		#SST INDEX
Z70hPa = np.load('./Z70hPa.npy')		#Z70hPa INDEX

#fix range
sic = signal.detrend(sic[:34])
sst = signal.detrend(sst[:34])
Z70hPa = signal.detrend(Z70hPa[:34])

yhat = np.zeros(datm.shape)
#Loop performs the leave-one-out cross-validation
#X and y are the training set, with a testing year (i) excluded.
#yhat is the time series of predicted blocking frequency
ct=0
X=np.column_stack((sic,sst,Z70hPa))
Y=datm
yhat=lv1out(X,Y)
a,b=stats.pearsonr(datm,yhat)
print(a,b)
"""