#This code reads the accumulated downward thermal radiation from the ERAI forecast datasets and 
#converts it to the right units by dividing it by the relevant time interval, save the data in 
#a netCDF file. See see Section 2.7 in
#https://www.ecmwf.int/en/elibrary/8174-era-interim-archive-version-20
#All operations are placed within loops to process daily data
#--------------------------------------
from netCDF4 import Dataset
import xarray as xr 
from calendar import monthrange
import numpy as np
import warnings
# Ignore invalid arcsin() in calculation
warnings.simplefilter("ignore", RuntimeWarning)

def degree_days(mins, maxs, threshold):
    """
    Use a sinusoidal approximation to estimate the number of Cooling Degree Days (CDD)
    above a given threshold, and Heating Degree-Days (HDD) below the same given threshold,
    using daily minimum and maximum temperatures.
    CDD is proxy for cooling/AC demand, HDD is proxy for heating demand
    mins and maxs are numpy arrays; threshold is in the same units.
    """

    # Useful quantities
    plus_over_2 = (mins + maxs)/2
    minus_over_2 = (maxs - mins)/2
    tbar = np.arccos((threshold - plus_over_2) / minus_over_2)

    # Set thresholds
    aboves = mins >= threshold
    belows = maxs <= threshold
    betweens = (mins < threshold) * (maxs > threshold)

    # Integral expressions for betweens (nans -> zero)
    Fcdd = 1/np.pi * np.nan_to_num((tbar * (plus_over_2 - threshold) + minus_over_2 * np.sin(tbar)))
    Fhdd = 1/np.pi * np.nan_to_num((np.pi - tbar) * (threshold - plus_over_2) + minus_over_2 * np.sin(tbar))

    # return CDD, HDD
    CDD = (aboves * (plus_over_2 - threshold)) + (belows * 0.) + (betweens * Fcdd)
    HDD = (aboves * 0.) + (belows * (threshold - plus_over_2)) + (betweens * Fhdd)
    return CDD, HDD

########################################################################
#Save data into a netCDF file
def save_netCDF(fnc,datm,lon1d,lat1d,vname,lname,YYYY,MM,DD):
  ncID = Dataset(fnc, 'w', format='NETCDF4')
  ncID.description = lname

  #set dimensions
  ncID.createDimension('time', None)
  ncID.createDimension('lon', len(lon1d))
  ncID.createDimension('lat', len(lat1d))

  #create new variables
  time = ncID.createVariable('time', 'f8', ('time',))
  lon = ncID.createVariable('lon', 'f4', ('lon',))
  lat = ncID.createVariable('lat', 'f4', ('lat',))
  field = ncID.createVariable(vname, 'f4', ('time', 'lat', 'lon',))

  #set attributes
  lat.units = "degrees_north"
  lat.long_name= "Latitude"
  lon.units = "degrees_east"
  lon.long_name= "Longitude"
  time.units = 'minutes since '+YYYY+'-'+MM+'-'+DD+' 00:00'
  time.long_name = "Time"

  #set data values
  lon[:] = lon1d
  lat[:] = lat1d
  time[0] = 0
  field[0,:,:] = datm

  ncID.close()

#################################################
"""
Time range: we will process for DJF from 1979 to 2019. 
Note that a winter season includes three consecutive months. 
For example, Dec 1979-Feb 1980 is denoted as winter 1979 or 1979 DJF, instead of 1980 DJF. 
nd: number of days per season; 
ny and nx: data dimension in the y and x directions: 0-90N and 0-360E
"""
iyr1=1979;iyr2=2018
nd=90;ny=128;nx=512

#variable name and long name 
vname1='CDD'
lname1='Cooling Degree Days derived from ERAI 2m Tmax and Tmin'
vname2='HDD'
lname2='Heating Degree Days derived from ERAI 2m Tmax and Tmin'

#initialize arrays. tmax and tmin are 3D arrays holding daily data for one season (90 days)
tmin=np.zeros((nd,ny,nx))
tmax=np.zeros((nd,ny,nx))

##################################################
#Read and Process data. dir0 and dir1 specify the input and output directories, respectively
dir0=''
dir1='./Temp/'
CDD_arr = np.full((128, 512, 40), -1.0)
HDD_arr = np.full((128, 512, 40), -1.0)
for iy in range(iyr1,iyr2+1):
  ii=0
  #read daily data in the month of Dec in Year "iy"
  imn1=12;imn2=12
  for im in range(imn1,imn2+1):
    nday=monthrange(iy, im)[1]
    for id in range(1,nday+1):
      tstr=str(iy)+str(im).zfill(2)+str(id).zfill(2)
      #read Tmax
      fnc_max=dir0+tstr[:4]+'/Tmax.ei.oper.fc.sfc.regn128sc.'+tstr+'.nc'
      ds=xr.open_dataset(fnc_max)
      tmax[ii]=ds.Tmax
      lon1D=ds.lon.values
      lat1D=ds.lat.values
      
      #read 12UTC data
      fnc_min=dir0+tstr[:4]+'/Tmin.ei.oper.fc.sfc.regn128sc.'+tstr+'.nc'
      ds=xr.open_dataset(fnc_min)
      tmin[ii]=ds.Tmin
      ii=ii+1
	  
  #read daily data in the months of Jan and Feb in Year "iy+1"	  
  imn1=1;imn2=2
  for im in range(imn1,imn2+1):
    if im == 2 and (iy+1)%4==0:
      nday = 28
    else:
      nday=monthrange(iy+1, im)[1]
    for id in range(1,nday+1):
      tstr=str(iy+1)+str(im).zfill(2)+str(id).zfill(2)
      #read Tmax
      fnc_max=dir0+tstr[:4]+'/Tmax.ei.oper.fc.sfc.regn128sc.'+tstr+'.nc'
      ds=xr.open_dataset(fnc_max)
      tmax[ii]=ds.Tmax

      #read 12UTC data
      fnc_min=dir0+tstr[:4]+'/Tmin.ei.oper.fc.sfc.regn128sc.'+tstr+'.nc'
      ds=xr.open_dataset(fnc_min)
      tmin[ii]=ds.Tmin
      ii=ii+1
	  
  
  #calculate degree days. If degree_days doesn't take 3D arrays as arguments, 
  #you need to loop over all grid points
  CDD_total = 0
  HDD_total = 0
  lat_range, long_range = tmin.shape[1], tmin.shape[2]
  #tmin = np.sum(tmin[:,44:56,:], axis = (1, 2))
  #tmax = np.sum(tmax, axis = (1, 2))
  
  for lat in range(lat_range):
    for long in range(long_range):
        #if (ds.lon.values[long] >= 0 and ds.lon.values[long] <= 2) or (ds.lon.values[long] >= 352 and ds.lon.values[long] <= 360):
            #if (ds.lat.values[lat] >= 50 and ds.lat.values[lat] <= 59):
        CDD, HDD=degree_days(mins = tmin[:,lat,long], maxs = tmax[:,lat,long], threshold = 293.15)
        
        CDD_arr[lat][long][iy-iyr1] = CDD.sum()
        HDD_arr[lat][long][iy-iyr1] = HDD.sum()
      
  
np.save('hdd_gridpoint', HDD_arr)
"""
  print(tmin)
  CDD, HDD=degree_days(mins = tmin[:], maxs = tmax[:], threshold = 293.4278)
  CDD_arr.append(CDD.sum())
  HDD_arr.append(HDD.sum())
"""
"""
  CDD, HDD=degree_days(mins = tmin, maxs = tmax, threshold = 18.0)
  print(CDD.shape)
  CDD_monthly = CDD.resample(time="M").sum()
  HDD_monthly = HDD.resample(time="M").sum() 
  
  #specify the output file name and write CDD to a netCDF file
  fnc=dir1+'/'+vname1+'.ei.oper.fc.sfc.regn128sc.'+tstr+'.nc'
  ff=save_netCDF(fnc,CDD_monthly,lon1D,lat1D,vname1,lname1,str(iy),str(im).zfill(2),str(id).zfill(2))
  
  #specify the output file name and write HDD to a netCDF file
  fnc=dir1+'/'+vname2+'.ei.oper.fc.sfc.regn128sc.'+tstr+'.nc'
  ff=save_netCDF(fnc,HDD_monthly,lon1D,lat1D,vname2,lname2,str(iy),str(im).zfill(2),str(id).zfill(2))
  print(tstr)

print('Normal End!')
"""