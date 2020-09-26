#Read 3D array (x,y,t)
########
import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as pyplot
from mpl_toolkits.basemap import Basemap 

#input data
filein=['./corr_temp.npy',
	'./corr_tempdd.npy',
	'./corr_winds.npy']
titles=['(a) Temp','(b) TempDD','(c) Wind Speed']

#latlon of the plotting domain 
lat1=30;lat2=85
lon1=-80;lon2=140

#############################################
#Initialize for plotting
pyplot.close("all")
fig, axes = pyplot.subplots(nrows=3, ncols=1, figsize=(8,6.5))
cmap = pyplot.get_cmap('RdBu_r')

for i in range(len(filein)):
   #Read 3D data
   dat = np.load(filein[i])
   lon0 = np.load('./lon0.npy')
   lat0 = np.load('./lat0.npy')
   nlon = 512
   ic=int(nlon/2)
   
   
   
   axx=axes.flat[i]		#choosing the subplot
   m = Basemap(projection="cyl",llcrnrlat=lat1, llcrnrlon=lon1, 
            urcrnrlat=lat2,urcrnrlon=lon2,ax=axx)
   m.drawcoastlines (linewidth=0.5, color="black")
   m.drawmeridians(np.arange(lon1,lon2+1,60), labels = [0 ,0 ,0 ,1])
   m.drawparallels(np.arange(lat1,lat2+1,20),  labels = [1 ,0 ,0 ,0])

   x,y = np.meshgrid(lon0,lat0)
   X,Y = m(x,y)
   clevs = np.arange(0.2,0.7,0.05)	#contour intervals
   im = m.contourf(X,Y,dat,clevs,cmap=cmap,extend='both')
   axx.set_title(titles[i],fontsize=13,loc='left')
#add a super title
pyplot.suptitle('My Super Title',fontsize=16)
#adjust the panels
pyplot.subplots_adjust(top=0.92,bottom=0.08,left=0.05,right=0.85,hspace=0.35,wspace=0.10)
#draw a super color bar
cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.8])
fig.colorbar(im,ax=axes.ravel().tolist(),orientation="vertical",cax=cbar_ax)
pyplot.show(block=False)
