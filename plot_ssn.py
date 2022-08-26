#Read 3D array (x,y,t)
########
import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap 

#input data
filein=['./corr_temp.npy',
	'./corr_tempdd.npy',
	'./corr_hdd.npy']
titles=['a) Temp','b) TDD','c) HDD']

#latlon of the plotting domain 
lat1=30;lat2=85 
lon1=-80;lon2=140

#############################################

#Initialize for plotting
#fig = pyplot(figsize=(8,6.5))
plt.figure(figsize=(8,4))
cmap = plt.get_cmap('OrRd')


#Read 3D data
dat = np.load(filein[2])

#only plot where p-values are significant
p_hdd = np.load('./p_hdd.npy')
dat = np.where(((p_hdd > 0) & (p_hdd <= 0.05)), dat, np.nan)
#create a blank array to pad HDD which is shape (128, 512). Other data has (256, 512)
zeros = np.full((128, 512), np.nan)
#other data has lat from -90 to 90, and HDD is 90 to 0. So we want to reverse HDD, and pad the beginning with our zeros array
dat = np.append(zeros, np.flip(dat), axis = 0)
#we want to reverse the halves of our latitude; to move 0:360 to -180:180
dat = np.append(dat[:,256:], dat[:,:256], axis = 1)
lon0 = np.load('./lon0.npy')
lat0 = np.load('./lat0.npy')
nlon = 512
ic=int(nlon/2)
   
   
   
#axx=axes.flat[i]		#choosing the subplot
m = Basemap(projection="cyl",llcrnrlat=lat1, llcrnrlon=lon1, 
         urcrnrlat=lat2,urcrnrlon=lon2)#,ax=axx)
m.drawcoastlines (linewidth=0.5, color="black")
m.drawmeridians(np.arange(lon1,lon2+1,60), labels = [0 ,0 ,0 ,1])
m.drawparallels(np.arange(lat1,lat2+1,20),  labels = [1 ,0 ,0 ,0])

x,y = np.meshgrid(lon0,lat0)
X,Y = m(x,y)
clevs = np.arange(0.275,0.625,0.025)	#contour intervals
im = m.contourf(X,Y,dat,clevs,cmap=cmap,extend='both')
#axx.set_title(titles[i],fontsize=15,loc='left')
#add a super title

#pyplot.suptitle('Gridpoint Prediction Skill',fontsize=18)
#adjust the panels
#plt.subplots_adjust(top=0.92,bottom=0.08,left=0.05,right=0.85,hspace=0.35,wspace=0.10)
#draw a super color bar
#cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.8])
plt.colorbar(im,orientation="horizontal")
#pyplot.show(block=False)
#fig.savefig('gridpoint.pdf')
plt.title('HDD Gridpoint Prediction Skill', fontsize = 18, loc = 'center')# (r = '+str(round(corr,2))+')')
plt.tight_layout()